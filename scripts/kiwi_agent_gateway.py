#!/usr/bin/env python3
"""Agent-facing visual grounding, map rendering, and navigation coordination.

This module is transport independent.  ``kiwi_agent_mcp.py`` exposes it over
MCP, while the existing image-navigation gallery can use the same state and
navigation manager without importing the MCP SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from kiwi_navigation_core import AStarPlanner, PathNotFound


CLIP_INDEX_FORMAT = "kiwi-clip-index-v1"
DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"


class AgentGatewayError(RuntimeError):
    """A safe, actionable failure that may be returned to an MCP caller."""

    def __init__(self, message: str, *, code: str = "invalid_request",
                 retryable: bool = False, suggested_tool: str | None = None,
                 details: dict | None = None):
        super().__init__(message)
        self.code = str(code)
        self.retryable = bool(retryable)
        self.suggested_tool = suggested_tool
        self.details = dict(details or {})

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
            "suggested_tool": self.suggested_tool,
            "details": self.details,
        }


class ImageEncoder(Protocol):
    model_name: str
    pretrained: str
    preprocessing_version: str

    def encode_images(self, paths: list[Path]) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class ImageAttachment:
    label: str
    mime_type: str
    path: Path | None = None
    data: bytes | None = None


@dataclass(frozen=True)
class GatewayResult:
    structured: dict
    images: tuple[ImageAttachment, ...] = ()


@dataclass
class StoredPreview:
    preview_id: str
    capture_ref: str
    max_travel_distance_m: float
    created_monotonic: float
    expires_monotonic: float
    plan: dict


@dataclass
class NavigationTrace:
    action_id: str
    capture_ref: str
    goal_pose: dict
    planned_route: list[dict]
    planned_path_distance_m: float | None
    started_at: float
    started_monotonic: float
    poses: list[dict]
    frames: list[dict]
    last_state: dict
    last_frame_monotonic: float | None = None
    finished_at: float | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentGatewayError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise AgentGatewayError(f"{name} must be finite")
    return number


def capture_ref(session_id: str, capture_id: int) -> str:
    return f"{session_id}:{int(capture_id)}"


def parse_capture_ref(value: str) -> tuple[str, int]:
    if not isinstance(value, str) or ":" not in value:
        raise AgentGatewayError(
            "capture_ref must be '<session_id>:<capture_id>'",
            code="invalid_capture_ref", suggested_tool="search_goal_images")
    session_id, raw_id = value.rsplit(":", 1)
    if not session_id or not raw_id.isdigit():
        raise AgentGatewayError(
            "capture_ref must be '<session_id>:<capture_id>'",
            code="invalid_capture_ref", suggested_tool="search_goal_images")
    return session_id, int(raw_id)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 0.0):
        raise AgentGatewayError("embedding model returned invalid vectors")
    return array / norms


class OpenClipEncoder:
    """Lazy OpenCLIP encoder so read-only robot tools start immediately."""

    preprocessing_version = "open-clip-default-v1"

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL,
                 pretrained: str = DEFAULT_CLIP_PRETRAINED,
                 device: str | None = None,
                 batch_size: int = 32):
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("OpenCLIP batch size must be positive")
        self._lock = threading.RLock()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import open_clip
                import torch
            except ImportError as exc:
                raise AgentGatewayError(
                    "visual search requires open_clip_torch; install "
                    "requirements-agent.txt") from exc
            device = self.device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif (hasattr(torch.backends, "mps") and
                      torch.backends.mps.is_available()):
                    device = "mps"
                else:
                    device = "cpu"
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained=self.pretrained)
                tokenizer = open_clip.get_tokenizer(self.model_name)
                model = model.to(device).eval()
            except Exception as exc:
                raise AgentGatewayError(
                    f"could not load OpenCLIP {self.model_name}/"
                    f"{self.pretrained}: {exc}") from exc
            self.device = device
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._torch = torch

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        self._load()
        if not paths:
            return np.empty((0, 0), dtype=np.float32)
        try:
            encoded = []
            for start in range(0, len(paths), self.batch_size):
                tensors = []
                for path in paths[start:start + self.batch_size]:
                    with Image.open(path) as image:
                        tensors.append(self._preprocess(image.convert("RGB")))
                batch = self._torch.stack(tensors).to(self.device)
                with self._torch.no_grad():
                    values = self._model.encode_image(batch)
                encoded.append(values.detach().float().cpu().numpy())
            return _normalize_rows(np.vstack(encoded))
        except AgentGatewayError:
            raise
        except Exception as exc:
            raise AgentGatewayError(f"could not embed goal images: {exc}") from exc

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        try:
            tokens = self._tokenizer([text]).to(self.device)
            with self._torch.no_grad():
                values = self._model.encode_text(tokens)
            return _normalize_rows(values.detach().float().cpu().numpy())[0]
        except AgentGatewayError:
            raise
        except Exception as exc:
            raise AgentGatewayError(f"could not embed search text: {exc}") from exc


class ClipPlaceIndex:
    """Incremental, checksum-keyed exact cosine index for one image map."""

    def __init__(self, encoder: ImageEncoder | None = None):
        self.encoder = encoder or OpenClipEncoder()
        self._lock = threading.RLock()
        self._cache_key = None
        self._capture_ids: list[int] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _paths(dataset) -> tuple[Path, Path]:
        root = dataset.manifest_path.parent
        return root / "clip-index-v1.json", root / "clip-index-v1.npz"

    def status(self, dataset=None) -> dict:
        metadata_path = None
        ready = bool(len(self._capture_ids))
        if dataset is not None:
            metadata_path, _ = self._paths(dataset)
        return {
            "ready": ready,
            "model": self.encoder.model_name,
            "pretrained": self.encoder.pretrained,
            "indexed_captures": len(self._capture_ids),
            "metadata_path": str(metadata_path) if metadata_path else None,
        }

    def _load_existing(self, dataset):
        metadata_path, vectors_path = self._paths(dataset)
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(vectors_path, allow_pickle=False) as archive:
                vectors = np.asarray(archive["embeddings"], dtype=np.float32)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return {}
        if (not isinstance(document, dict) or
                document.get("format") != CLIP_INDEX_FORMAT or
                document.get("session_id") != dataset.session_id or
                document.get("model") != self.encoder.model_name or
                document.get("pretrained") != self.encoder.pretrained or
                document.get("preprocessing") !=
                self.encoder.preprocessing_version):
            return {}
        records = document.get("captures")
        if (not isinstance(records, list) or vectors.ndim != 2 or
                len(records) != len(vectors)):
            return {}
        reused_by_checksum = {}
        for record, vector in zip(records, vectors):
            try:
                checksum = str(record["sha256"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(checksum) == 64 and np.isfinite(vector).all():
                reused_by_checksum[checksum] = vector
        return reused_by_checksum

    def _persist(self, dataset, checksums: dict[int, str],
                 capture_ids: list[int], embeddings: np.ndarray) -> None:
        metadata_path, vectors_path = self._paths(dataset)
        token = secrets.token_hex(6)
        metadata_tmp = metadata_path.with_name(
            f".{metadata_path.name}.{token}.tmp")
        vectors_tmp = vectors_path.with_name(
            f".{vectors_path.name}.{token}.tmp")
        document = {
            "format": CLIP_INDEX_FORMAT,
            "session_id": dataset.session_id,
            "model": self.encoder.model_name,
            "pretrained": self.encoder.pretrained,
            "preprocessing": self.encoder.preprocessing_version,
            "embedding_dimension": int(embeddings.shape[1]),
            "built_at": _utc_now(),
            "captures": [
                {"capture_id": capture_id, "sha256": checksums[capture_id]}
                for capture_id in capture_ids
            ],
        }
        try:
            with vectors_tmp.open("wb") as handle:
                np.savez_compressed(handle, embeddings=embeddings)
            metadata_tmp.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(vectors_tmp, vectors_path)
            os.replace(metadata_tmp, metadata_path)
        finally:
            for path in (metadata_tmp, vectors_tmp):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def ensure(self, dataset) -> None:
        with self._lock:
            stat_key = (
                str(dataset.manifest_path),
                dataset.manifest_path.stat().st_mtime_ns,
                tuple((item.capture_id, item.image_path.stat().st_mtime_ns)
                      for item in dataset.captures),
            )
            if stat_key == self._cache_key:
                return
            checksums = {
                item.capture_id: self._checksum(item.image_path)
                for item in dataset.captures
            }
            existing_by_checksum = self._load_existing(dataset)
            missing = [
                item for item in dataset.captures
                if checksums[item.capture_id] not in existing_by_checksum
            ]
            if missing:
                added = self.encoder.encode_images(
                    [item.image_path for item in missing])
                if len(added) != len(missing):
                    raise AgentGatewayError(
                        "embedding model returned the wrong number of images")
                for item, vector in zip(missing, added):
                    existing_by_checksum[checksums[item.capture_id]] = vector
            capture_ids = [item.capture_id for item in dataset.captures]
            if not capture_ids:
                raise AgentGatewayError("the active image map has no captures")
            dimensions = {np.asarray(existing_by_checksum[checksums[item]]).shape
                          for item in capture_ids}
            if len(dimensions) != 1:
                raise AgentGatewayError("image embeddings have inconsistent sizes")
            embeddings = _normalize_rows(np.vstack([
                existing_by_checksum[checksums[item]] for item in capture_ids]))
            self._persist(dataset, checksums, capture_ids, embeddings)
            self._capture_ids = capture_ids
            self._embeddings = embeddings
            self._cache_key = stat_key

    def search(self, dataset, query: str, top_n: int,
               diversify: bool = True) -> list[tuple[object, float]]:
        query = str(query).strip()
        if not query:
            raise AgentGatewayError("query must not be empty")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 8:
            raise AgentGatewayError("top_n must be an integer in [1, 8]")
        self.ensure(dataset)
        with self._lock:
            text_embedding = self.encoder.encode_text(query)
            if text_embedding.shape != self._embeddings.shape[1:]:
                raise AgentGatewayError(
                    "text and image embedding dimensions do not match")
            scores = self._embeddings @ text_embedding
            order = list(np.argsort(-scores))
            selected = []
            for index in order:
                candidate = dataset.capture(self._capture_ids[int(index)])
                if diversify and any(
                    math.hypot(candidate.x - prior.x, candidate.y - prior.y) < 0.35
                    and abs(math.atan2(
                        math.sin(candidate.yaw - prior.yaw),
                        math.cos(candidate.yaw - prior.yaw))) < math.radians(20.0)
                    for prior, _score in selected
                ):
                    continue
                selected.append((candidate, float(scores[int(index)])))
                if len(selected) >= top_n:
                    break
            if len(selected) < top_n:
                chosen = {item.capture_id for item, _score in selected}
                for index in order:
                    candidate = dataset.capture(self._capture_ids[int(index)])
                    if candidate.capture_id in chosen:
                        continue
                    selected.append((candidate, float(scores[int(index)])))
                    if len(selected) >= top_n:
                        break
            return selected


def _path_length(points) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _world_to_image(occupancy, point, crop) -> tuple[float, float]:
    row0, row1, col0, col1 = crop
    x, y = point
    column = (float(x) - occupancy.origin_x) / occupancy.resolution_m - col0
    row = (float(y) - occupancy.origin_y) / occupancy.resolution_m - row0
    return column, (row1 - row0 - 1) - row


def render_pose_map(occupancy, pose: dict, *, route=None, goal=None,
                    view: str = "full", radius_m: float | None = None,
                    map_age_s: float | None = None,
                    pose_age_s: float | None = None) -> tuple[bytes, dict]:
    """Render occupancy, robot cursor, heading, and optional route to PNG."""
    if view not in ("full", "local"):
        raise AgentGatewayError("view must be 'full' or 'local'")
    data = np.asarray(occupancy.data)
    height, width = data.shape
    if view == "local":
        radius = 2.5 if radius_m is None else _finite_number(radius_m, "radius_m")
        if radius <= 0.0:
            raise AgentGatewayError("radius_m must be positive")
        center_col = int(round((pose["x"] - occupancy.origin_x) /
                               occupancy.resolution_m))
        center_row = int(round((pose["y"] - occupancy.origin_y) /
                               occupancy.resolution_m))
        center_col = min(max(center_col, 0), width - 1)
        center_row = min(max(center_row, 0), height - 1)
        cells = max(1, int(math.ceil(radius / occupancy.resolution_m)))
        row0, row1 = max(0, center_row - cells), min(height, center_row + cells + 1)
        col0, col1 = max(0, center_col - cells), min(width, center_col + cells + 1)
    else:
        row0, row1, col0, col1 = 0, height, 0, width
    crop = row0, row1, col0, col1
    visible = data[row0:row1, col0:col1]
    gray = np.empty(visible.shape, dtype=np.uint8)
    gray[visible < 0] = 145
    known = visible >= 0
    gray[known] = np.clip(248 - visible[known] * 2.15, 28, 248).astype(np.uint8)
    rgb = np.repeat(np.flipud(gray)[:, :, None], 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")

    base_max = max(image.size)
    scale = min(6.0, max(0.1, 900.0 / max(1, base_max)))
    if scale != 1.0:
        image = image.resize(
            (max(1, int(round(image.width * scale))),
             max(1, int(round(image.height * scale)))),
            Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    def pixel(point):
        x, y = _world_to_image(occupancy, point, crop)
        return x * scale, y * scale

    route_points = [] if route is None else list(route)
    if len(route_points) >= 2:
        draw.line([pixel(point) for point in route_points],
                  fill=(45, 135, 255), width=max(3, int(3 * scale)), joint="curve")
    if goal is not None:
        gx, gy = pixel((goal["x"], goal["y"]))
        radius_px = max(6, int(5 * scale))
        draw.ellipse((gx - radius_px, gy - radius_px,
                      gx + radius_px, gy + radius_px),
                     fill=(230, 45, 165), outline=(255, 240, 250),
                     width=max(2, int(scale)))

    map_diagonal_m = math.hypot(width, height) * occupancy.resolution_m
    arrow_length_m = min(2.0, max(0.75, 0.12 * map_diagonal_m))
    start = pixel((pose["x"], pose["y"]))
    end_world = (
        pose["x"] + arrow_length_m * math.cos(pose["yaw"]),
        pose["y"] + arrow_length_m * math.sin(pose["yaw"]),
    )
    end = pixel(end_world)
    robot_radius = max(7, int(6 * scale))
    draw.ellipse((start[0] - robot_radius, start[1] - robot_radius,
                  start[0] + robot_radius, start[1] + robot_radius),
                 fill=(25, 220, 105), outline=(0, 35, 15),
                 width=max(2, int(2 * scale)))
    arrow_width = max(4, int(4 * scale))
    draw.line((start, end), fill=(0, 80, 35), width=arrow_width)
    screen_angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(12, int(10 * scale))
    left = (end[0] - head * math.cos(screen_angle - 0.55),
            end[1] - head * math.sin(screen_angle - 0.55))
    right = (end[0] - head * math.cos(screen_angle + 0.55),
             end[1] - head * math.sin(screen_angle + 0.55))
    draw.polygon((end, left, right), fill=(0, 80, 35))

    scale_bar_m = 1.0 if map_diagonal_m >= 2.0 else 0.5
    scale_bar_px = scale_bar_m / occupancy.resolution_m * scale
    margin = max(12, int(10 * scale))
    bar_y = image.height - margin
    draw.line((margin, bar_y, margin + scale_bar_px, bar_y),
              fill=(245, 195, 40), width=max(3, int(3 * scale)))
    draw.text((margin, max(0, bar_y - 18)), f"{scale_bar_m:g} m",
              fill=(245, 195, 40))

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    bounds = {
        "min_x": occupancy.origin_x + col0 * occupancy.resolution_m,
        "max_x": occupancy.origin_x + (col1 - 1) * occupancy.resolution_m,
        "min_y": occupancy.origin_y + row0 * occupancy.resolution_m,
        "max_y": occupancy.origin_y + (row1 - 1) * occupancy.resolution_m,
    }
    return output.getvalue(), {
        "view": view,
        "map_bounds": bounds,
        "resolution_m": occupancy.resolution_m,
        "keyframes": occupancy.keyframes,
        "arrow_length_m": arrow_length_m,
        "scale_bar_m": scale_bar_m,
        "pose_age_s": pose_age_s,
        "map_age_s": map_age_s,
        "image_width_px": image.width,
        "image_height_px": image.height,
    }


def _evenly_spaced(items: list, count: int) -> list:
    if not items or count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    if len(items) <= count:
        return list(items)
    return [items[round(index * (len(items) - 1) / (count - 1))]
            for index in range(count)]


def render_camera_contact_sheet(frames: list[dict], *, frame_count: int = 8,
                                brightness_gain: float = 1.0) -> tuple[bytes, list[dict]]:
    """Render evenly spaced action-camera frames without using simulator state."""
    selected = _evenly_spaced(frames, frame_count)
    columns = min(4, max(1, len(selected)))
    rows = max(1, int(math.ceil(max(1, len(selected)) / columns)))
    tile_width, camera_height, caption_height = 280, 210, 46
    gap, margin, header = 12, 22, 58
    width = margin * 2 + columns * tile_width + (columns - 1) * gap
    height = header + margin + rows * (camera_height + caption_height) + \
        (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 16), "Robot camera sequence", fill=(24, 34, 50))
    subtitle = (
        f"{len(selected)} evenly spaced frames"
        + ("" if abs(brightness_gain - 1.0) < 1e-9
           else f"  |  display brightness x{brightness_gain:g}")
    )
    draw.text((margin, 34), subtitle, fill=(83, 97, 116))
    metadata = []
    for index, frame in enumerate(selected):
        row, column = divmod(index, columns)
        x = margin + column * (tile_width + gap)
        y = header + row * (camera_height + caption_height + gap)
        draw.rounded_rectangle(
            (x, y, x + tile_width, y + camera_height + caption_height),
            radius=9, fill=(255, 255, 255), outline=(216, 222, 230), width=2)
        try:
            with Image.open(io.BytesIO(frame["jpeg"])) as source:
                camera = source.convert("RGB")
        except (KeyError, OSError, ValueError):
            camera = Image.new("RGB", (4, 3), (28, 32, 38))
        if abs(brightness_gain - 1.0) > 1e-9:
            camera = ImageEnhance.Brightness(camera).enhance(brightness_gain)
        camera.thumbnail((tile_width, camera_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_width, camera_height), (20, 24, 30))
        canvas.paste(camera, ((tile_width - camera.width) // 2,
                              (camera_height - camera.height) // 2))
        sheet.paste(canvas, (x, y))
        badge_radius = 14
        badge_x, badge_y = x + 18, y + 18
        draw.ellipse((badge_x - badge_radius, badge_y - badge_radius,
                      badge_x + badge_radius, badge_y + badge_radius),
                     fill=(255, 139, 36))
        draw.text((badge_x - 3, badge_y - 6), str(index + 1), fill="white")
        elapsed = float(frame.get("elapsed_s", 0.0))
        pose = frame.get("pose") or {}
        draw.text((x + 10, y + camera_height + 7),
                  f"{index + 1}  +{elapsed:.2f} s", fill=(27, 39, 56))
        draw.text((x + 10, y + camera_height + 24),
                  f"x {pose.get('x', 0.0):+.2f} m   y {pose.get('y', 0.0):+.2f} m",
                  fill=(99, 113, 132))
        metadata.append({
            "number": index + 1,
            "elapsed_s": elapsed,
            "sequence": frame.get("sequence"),
            "pose": dict(pose),
        })
    if not selected:
        draw.text((margin, header + 20),
                  "No camera frames were recorded for this action.",
                  fill=(120, 70, 35))
    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=True)
    return output.getvalue(), metadata


def _draw_dashed(draw, points, *, fill, width=3, dash=12, gap=8):
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        offset = 0.0
        while offset < length:
            stop = min(length, offset + dash)
            a = (start[0] + dx * offset / length,
                 start[1] + dy * offset / length)
            b = (start[0] + dx * stop / length,
                 start[1] + dy * stop / length)
            draw.line((a, b), fill=fill, width=width)
            offset += dash + gap


def render_navigation_trace_map(occupancy, actual_poses: list[dict],
                                planned_route: list[dict], goal_pose: dict,
                                selected_frames: list[dict]) -> tuple[bytes, dict]:
    """Plot a measured SLAM trajectory against its plan on live occupancy."""
    points = [
        (float(item["x"]), float(item["y"]))
        for item in [*actual_poses, *planned_route, goal_pose]
        if isinstance(item, dict) and "x" in item and "y" in item
    ]
    if not points:
        raise AgentGatewayError(
            "navigation trace has no map-frame poses",
            code="trace_not_ready", retryable=True,
            suggested_tool="get_navigation_status")
    pad_m = 0.55
    min_x = min(point[0] for point in points) - pad_m
    max_x = max(point[0] for point in points) + pad_m
    min_y = min(point[1] for point in points) - pad_m
    max_y = max(point[1] for point in points) + pad_m
    data = np.asarray(occupancy.data)
    height, width = data.shape
    col0 = max(0, int(math.floor(
        (min_x - occupancy.origin_x) / occupancy.resolution_m)))
    col1 = min(width, int(math.ceil(
        (max_x - occupancy.origin_x) / occupancy.resolution_m)) + 1)
    row0 = max(0, int(math.floor(
        (min_y - occupancy.origin_y) / occupancy.resolution_m)))
    row1 = min(height, int(math.ceil(
        (max_y - occupancy.origin_y) / occupancy.resolution_m)) + 1)
    if row1 <= row0 or col1 <= col0:
        row0, row1, col0, col1 = 0, height, 0, width
    crop = row0, row1, col0, col1
    visible = data[row0:row1, col0:col1]
    gray = np.empty(visible.shape, dtype=np.uint8)
    gray[visible < 0] = 205
    known = visible >= 0
    gray[known] = np.clip(
        250 - visible[known] * 2.2, 30, 250).astype(np.uint8)
    rgb = np.repeat(np.flipud(gray)[:, :, None], 3, axis=2)
    base = Image.fromarray(rgb, mode="RGB")
    scale = min(8.0, max(1.0, 900.0 / max(base.size)))
    base = base.resize((max(1, round(base.width * scale)),
                        max(1, round(base.height * scale))),
                       Image.Resampling.NEAREST)
    header = 54
    image = Image.new("RGB", (base.width, base.height + header), (247, 249, 252))
    image.paste(base, (0, header))
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), "Actual SLAM trajectory vs MCP plan", fill=(24, 34, 50))
    draw.line((16, 34, 48, 34), fill=(0, 168, 120), width=5)
    draw.text((54, 28), "actual SLAM", fill=(60, 76, 94))
    _draw_dashed(draw, [(154, 34), (188, 34)], fill=(53, 120, 229), width=4)
    draw.text((194, 28), "planned", fill=(60, 76, 94))

    def pixel(item):
        x, y = (item["x"], item["y"]) if isinstance(item, dict) else item
        px, py = _world_to_image(occupancy, (x, y), crop)
        return px * scale, py * scale + header

    planned_pixels = [pixel(item) for item in planned_route]
    actual_pixels = [pixel(item) for item in actual_poses]
    if len(planned_pixels) >= 2:
        _draw_dashed(draw, planned_pixels, fill=(53, 120, 229),
                     width=max(3, round(scale * 1.5)))
    if len(actual_pixels) >= 2:
        draw.line(actual_pixels, fill=(0, 168, 120),
                  width=max(4, round(scale * 2.0)), joint="curve")
    if actual_pixels:
        sx, sy = actual_pixels[0]
        draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7),
                     fill=(91, 102, 117), outline="white", width=2)
        fx, fy = actual_pixels[-1]
        draw.ellipse((fx - 9, fy - 9, fx + 9, fy + 9),
                     fill=(0, 168, 120), outline="white", width=2)
    gx, gy = pixel(goal_pose)
    draw.ellipse((gx - 13, gy - 13, gx + 13, gy + 13),
                 outline=(53, 120, 229), width=4)
    draw.ellipse((gx - 4, gy - 4, gx + 4, gy + 4), fill=(53, 120, 229))
    for frame in selected_frames:
        pose = frame.get("pose") or {}
        if "x" not in pose or "y" not in pose:
            continue
        x, y = pixel(pose)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11),
                     fill=(255, 139, 36), outline="white", width=2)
        draw.text((x - 3, y - 6), str(frame["number"]), fill="white")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), {
        "map_bounds": {
            "min_x": occupancy.origin_x + col0 * occupancy.resolution_m,
            "max_x": occupancy.origin_x + (col1 - 1) * occupancy.resolution_m,
            "min_y": occupancy.origin_y + row0 * occupancy.resolution_m,
            "max_y": occupancy.origin_y + (row1 - 1) * occupancy.resolution_m,
        },
        "resolution_m": occupancy.resolution_m,
        "image_width_px": image.width,
        "image_height_px": image.height,
    }


class KiwiAgentGateway:
    """Single stateful coordinator used by all seven agent tools."""

    def __init__(self, dataset_store, navigation, live,
                 *, clip_index: ClipPlaceIndex | None = None,
                 preview_ttl_s: float = 30.0,
                 max_action_distance_m: float = 5.0,
                 trace_frame_interval_s: float = 0.18,
                 watchdog_interval_s: float | None = None):
        self.dataset_store = dataset_store
        self.navigation = navigation
        self.live = live
        self.clip_index = clip_index or ClipPlaceIndex()
        self.preview_ttl_s = float(preview_ttl_s)
        self.max_action_distance_m = float(max_action_distance_m)
        self.trace_frame_interval_s = float(trace_frame_interval_s)
        if (not math.isfinite(self.preview_ttl_s) or self.preview_ttl_s <= 0.0 or
                not math.isfinite(self.max_action_distance_m) or
                self.max_action_distance_m <= 0.0 or
                not math.isfinite(self.trace_frame_interval_s) or
                self.trace_frame_interval_s < 0.0):
            raise ValueError("preview TTL and maximum action distance must be positive")
        self._lock = threading.RLock()
        self._previews: dict[str, StoredPreview] = {}
        self._traces: dict[str, NavigationTrace] = {}
        self._active_trace_id: str | None = None
        self._latest_pose: dict | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None
        if hasattr(live, "add_pose_callback"):
            live.add_pose_callback(self._observe_pose)
        if hasattr(live, "add_navigation_callback"):
            live.add_navigation_callback(self._observe_navigation_state)
        if hasattr(live, "add_mux_callback"):
            live.add_mux_callback(self._observe_mux_status)
        if hasattr(live, "add_camera_callback"):
            live.add_camera_callback(self._observe_camera)
        if watchdog_interval_s is not None:
            interval = float(watchdog_interval_s)
            if not math.isfinite(interval) or interval <= 0.0:
                raise ValueError("watchdog interval must be positive")
            self._watchdog_thread = threading.Thread(
                target=self._watch_safety,
                args=(interval,),
                name="kiwi-agent-readiness-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    def _active_trace(self) -> NavigationTrace | None:
        with self._lock:
            if self._active_trace_id is None:
                return None
            return self._traces.get(self._active_trace_id)

    def _observe_pose(self, pose: dict) -> None:
        self.navigation.observe_pose(pose)
        now = time.monotonic()
        wall_time = time.time()
        state = self.navigation.snapshot()
        with self._lock:
            self._latest_pose = dict(pose)
            trace = self._traces.get(state.get("action_id"))
            if trace is None:
                return
            trace.last_state = dict(state)
            if state.get("phase") in ("running", "stopping"):
                trace.poses.append({
                    "elapsed_s": max(0.0, now - trace.started_monotonic),
                    "wall_time": wall_time,
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "yaw": float(pose["yaw"]),
                })
                if len(trace.poses) > 4000:
                    trace.poses[:] = trace.poses[::2]
            elif trace.finished_at is None:
                trace.finished_at = wall_time

    def _observe_navigation_state(self, state: dict) -> None:
        self.navigation.observe_navigation_state(state)
        snapshot = self.navigation.snapshot()
        with self._lock:
            trace = self._traces.get(snapshot.get("action_id"))
            if trace is not None:
                trace.last_state = dict(snapshot)
            if (trace is not None and
                    snapshot.get("phase") not in ("running", "stopping") and
                    trace.finished_at is None):
                trace.finished_at = time.time()

    def _observe_mux_status(self, state: dict) -> None:
        self.navigation.observe_mux_status(state)

    def _observe_camera(self, payload: bytes) -> None:
        now = time.monotonic()
        state = self.navigation.snapshot()
        with self._lock:
            trace = self._traces.get(state.get("action_id"))
            pose = None if self._latest_pose is None else dict(self._latest_pose)
            if (trace is None or pose is None or
                    state.get("phase") not in ("running", "stopping") or
                    (trace.last_frame_monotonic is not None and
                     now - trace.last_frame_monotonic <
                     self.trace_frame_interval_s)):
                return
        from kiwi_image_map import decode_camera_sample
        sample = decode_camera_sample(payload)
        if sample is None:
            return
        with self._lock:
            trace = self._traces.get(state.get("action_id"))
            if trace is None:
                return
            trace.last_frame_monotonic = now
            trace.frames.append({
                "elapsed_s": max(0.0, now - trace.started_monotonic),
                "wall_time": time.time(),
                "sequence": sample.sequence,
                "pose": pose,
                "jpeg": sample.jpeg,
            })
            if len(trace.frames) > 160:
                trace.frames[:] = trace.frames[::2]

    def _watch_safety(self, interval_s: float) -> None:
        while not self._watchdog_stop.wait(interval_s):
            state = self.navigation.snapshot()
            if state.get("phase") not in ("running", "stopping"):
                continue
            try:
                dataset = self.dataset_store.snapshot()
                live = self.live.status(dataset.session_id)
            except Exception as exc:
                reason = f"readiness check failed: {exc}"
            else:
                if live["ready"]:
                    continue
                reason = f"safety interlock: {live['reason']}"
            try:
                self.navigation.stop(
                    action_id=state.get("action_id"), reason=reason)
            except (RuntimeError, ValueError):
                pass

    def close(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)

    def _dataset_and_capture(self, ref: str):
        dataset = self.dataset_store.snapshot()
        session_id, capture_id = parse_capture_ref(ref)
        if session_id != dataset.session_id:
            raise AgentGatewayError(
                f"capture_ref session {session_id!r} is not the active "
                f"session {dataset.session_id!r}",
                code="inactive_capture_session", retryable=True,
                suggested_tool="search_goal_images",
                details={
                    "capture_session_id": session_id,
                    "active_session_id": dataset.session_id,
                })
        try:
            destination = dataset.capture(capture_id)
        except KeyError as exc:
            raise AgentGatewayError(
                f"capture {capture_id} was not found",
                code="capture_not_found", retryable=True,
                suggested_tool="search_goal_images",
                details={"capture_id": capture_id}) from exc
        return dataset, destination

    @staticmethod
    def _pose_json(destination) -> dict:
        return {
            "frame": "map",
            "x": destination.x,
            "y": destination.y,
            "yaw_rad": destination.yaw,
            "yaw_deg": math.degrees(destination.yaw),
        }

    def get_robot_status(self) -> GatewayResult:
        dataset = self.dataset_store.snapshot()
        live = self.live.status(dataset.session_id)
        recovery = live.get("recovery")
        if (not live["ready"] and
                live.get("live_session_id") is not None and
                live.get("live_session_id") != dataset.session_id):
            recovery = {
                "code": "session_mismatch",
                "manifest_path": str(dataset.manifest_path),
                "expected_session_id": dataset.session_id,
                "live_session_id": live.get("live_session_id"),
                "action": (
                    "restart SLAM and image navigation with this same manifest"
                ),
            }
            live = {**live, "status_code": "session_mismatch",
                    "recovery": recovery}
        next_tool = (
            "search_goal_images" if live["ready"] else "get_robot_status")
        return GatewayResult({
            "ready": live["ready"],
            "reason": live["reason"],
            "session_id": dataset.session_id,
            "capture_count": len(dataset.captures),
            "live": live,
            "navigation": self.navigation.snapshot(),
            "visual_search": self.clip_index.status(dataset),
            "next_action": {
                "tool": next_tool,
                "reason": (
                    "choose a visual goal" if live["ready"]
                    else "resolve readiness before previewing motion"),
            },
            "recovery": recovery,
            "safety": {
                "unknown_cells_blocked": True,
                "max_action_distance_m": self.max_action_distance_m,
                "requires_preview": True,
                "single_action_only": True,
            },
        })

    def search_goal_images(self, query: str, top_n: int = 4,
                           diversify: bool = True) -> GatewayResult:
        dataset = self.dataset_store.snapshot()
        state = self.live.snapshot(dataset.session_id)
        pose = state.get("pose")
        ranked = self.clip_index.search(dataset, query, top_n, diversify)
        results = []
        images = []
        for rank, (destination, score) in enumerate(ranked, 1):
            distance = (None if pose is None else math.hypot(
                destination.x - pose["x"], destination.y - pose["y"]))
            ref = capture_ref(dataset.session_id, destination.capture_id)
            results.append({
                "capture_ref": ref,
                "rank": rank,
                "similarity": score,
                "similarity_semantics": "cosine ranking score; not confidence",
                "pose": self._pose_json(destination),
                "straight_line_distance_m": distance,
            })
            images.append(ImageAttachment(
                label=f"rank {rank}: {ref}", mime_type="image/jpeg",
                path=destination.image_path))
        return GatewayResult({
            "query": str(query).strip(),
            "session_id": dataset.session_id,
            "diversified": bool(diversify),
            "results": results,
        }, tuple(images))

    def get_pose_on_map(self, view: str = "full",
                        radius_m: float | None = None,
                        include_path: bool = True) -> GatewayResult:
        dataset = self.dataset_store.snapshot()
        state = self.live.snapshot(dataset.session_id)
        occupancy, pose = state.get("occupancy"), state.get("pose")
        if occupancy is None:
            raise AgentGatewayError("no live occupancy map is available")
        if pose is None:
            raise AgentGatewayError("no live SLAM pose is available")
        route = state.get("trajectory") if include_path else None
        navigation_state = state.get("navigation_state") or {}
        goal = navigation_state.get("goal") if include_path else None
        png, metadata = render_pose_map(
            occupancy, pose, route=route, goal=goal, view=view,
            radius_m=radius_m, map_age_s=state.get("map_age_s"),
            pose_age_s=state.get("pose_age_s"))
        structured = {
            "frame": "map",
            "pose": {
                **pose,
                "yaw_deg": math.degrees(pose["yaw"]),
            },
            "localization_quality": state.get("quality"),
            "include_path": bool(include_path),
            **metadata,
        }
        return GatewayResult(structured, (
            ImageAttachment("robot pose on occupancy map", "image/png", data=png),
        ))

    def _plan(self, destination, max_travel_distance_m: float) -> tuple[dict, bytes]:
        dataset = self.dataset_store.snapshot()
        state = self.live.snapshot(dataset.session_id)
        live_status = state["status"]
        blockers = []
        if not live_status["ready"]:
            blockers.append(live_status["reason"])
        if max_travel_distance_m > self.max_action_distance_m:
            blockers.append(
                f"requested distance exceeds the human-configured "
                f"{self.max_action_distance_m:g} m ceiling")
        occupancy, pose = state.get("occupancy"), state.get("pose")
        path = None
        planning_error = None
        soft_start_recovery = False
        if occupancy is None:
            planning_error = "no live occupancy map is available"
        elif pose is None:
            planning_error = "no live SLAM pose is available"
        else:
            try:
                planner = AStarPlanner(
                    occupancy,
                    inflation_radius_m=self.navigation.settings.inflation_radius,
                    occupied_threshold=65,
                    allow_unknown=False,
                )
                runtime_planner = AStarPlanner(
                    occupancy,
                    inflation_radius_m=getattr(
                        self.navigation.settings, "runtime_collision_radius",
                        self.navigation.settings.inflation_radius),
                    occupied_threshold=65,
                    allow_unknown=False,
                )
                start_xy = (pose["x"], pose["y"])
                soft_start_recovery = (
                    not planner.cell_is_free(planner.world_to_cell(start_xy))
                    and runtime_planner.cell_is_free(
                        runtime_planner.world_to_cell(start_xy)))
                path = planner.plan_with_start_recovery(
                    start_xy, (destination.x, destination.y), runtime_planner)
            except (PathNotFound, ValueError) as exc:
                planning_error = str(exc)
        if planning_error:
            blockers.append(planning_error)
        planned_distance = None if path is None else _path_length(path)
        if (planned_distance is not None and
                planned_distance > max_travel_distance_m + 1e-9):
            blockers.append(
                f"planned path {planned_distance:.3f} m exceeds the authorized "
                f"{max_travel_distance_m:.3f} m envelope")
        straight_line = None if pose is None else math.hypot(
            destination.x - pose["x"], destination.y - pose["y"])
        route = [] if path is None else [
            {"x": float(point[0]), "y": float(point[1])} for point in path]
        plan = {
            "capture_ref": capture_ref(dataset.session_id,
                                       destination.capture_id),
            "goal_pose": self._pose_json(destination),
            "straight_line_distance_m": straight_line,
            "planned_path_distance_m": planned_distance,
            "max_travel_distance_m": max_travel_distance_m,
            "estimated_duration_s": (
                None if planned_distance is None else
                planned_distance /
                max(1e-6, self.navigation.settings.max_linear_speed)),
            "safe_to_start": not blockers,
            "blockers": blockers,
            "soft_start_recovery": soft_start_recovery,
            "planning_inflation_radius_m": (
                self.navigation.settings.inflation_radius),
            "runtime_collision_radius_m": getattr(
                self.navigation.settings, "runtime_collision_radius",
                self.navigation.settings.inflation_radius),
            "tracking_buffer_m": max(
                0.0,
                self.navigation.settings.inflation_radius - getattr(
                    self.navigation.settings, "runtime_collision_radius",
                    self.navigation.settings.inflation_radius)),
            "route": route,
            "map_keyframes": (None if occupancy is None else occupancy.keyframes),
            "planned_at": _utc_now(),
        }
        if occupancy is None or pose is None:
            blank = Image.new("RGB", (640, 360), (45, 45, 45))
            draw = ImageDraw.Draw(blank)
            draw.text((24, 24), planning_error or "navigation inputs unavailable",
                      fill=(255, 230, 180))
            output = io.BytesIO()
            blank.save(output, format="PNG")
            return plan, output.getvalue()
        png, _metadata = render_pose_map(
            occupancy, pose,
            route=[(point["x"], point["y"]) for point in route],
            goal={"x": destination.x, "y": destination.y,
                  "yaw": destination.yaw},
            map_age_s=state.get("map_age_s"),
            pose_age_s=state.get("pose_age_s"))
        return plan, png

    def preview_image_goal(self, ref: str,
                           max_travel_distance_m: float) -> GatewayResult:
        dataset, destination = self._dataset_and_capture(ref)
        budget = _finite_number(
            max_travel_distance_m, "max_travel_distance_m")
        if budget <= 0.0:
            raise AgentGatewayError("max_travel_distance_m must be positive")
        plan, png = self._plan(destination, budget)
        now = time.monotonic()
        preview_id = secrets.token_urlsafe(12)
        plan.update({
            "preview_id": preview_id,
            "expires_in_s": self.preview_ttl_s,
            "expires_at": datetime.fromtimestamp(
                time.time() + self.preview_ttl_s,
                timezone.utc).isoformat(),
        })
        with self._lock:
            self._previews = {
                key: value for key, value in self._previews.items()
                if value.expires_monotonic > now
            }
            self._previews[preview_id] = StoredPreview(
                preview_id=preview_id,
                capture_ref=ref,
                max_travel_distance_m=budget,
                created_monotonic=now,
                expires_monotonic=now + self.preview_ttl_s,
                plan=dict(plan),
            )
        return GatewayResult(plan, (
            ImageAttachment(
                f"goal image {ref}", "image/jpeg", path=destination.image_path),
            ImageAttachment("planned route", "image/png", data=png),
        ))

    def navigate_to_image(self, preview_id: str) -> GatewayResult:
        now = time.monotonic()
        with self._lock:
            preview = self._previews.get(str(preview_id))
        if preview is None:
            raise AgentGatewayError(
                "preview_id was not found", code="preview_not_found",
                retryable=True, suggested_tool="preview_image_goal")
        if preview.expires_monotonic <= now:
            with self._lock:
                self._previews.pop(preview.preview_id, None)
            raise AgentGatewayError(
                "preview_id has expired; create a new preview",
                code="preview_expired", retryable=True,
                suggested_tool="preview_image_goal")
        _dataset, destination = self._dataset_and_capture(preview.capture_ref)
        plan, _png = self._plan(
            destination, preview.max_travel_distance_m)
        if not plan["safe_to_start"]:
            raise AgentGatewayError(
                "navigation revalidation failed: " + "; ".join(plan["blockers"]),
                code="safety_revalidation_failed", retryable=True,
                suggested_tool="preview_image_goal",
                details={"blockers": plan["blockers"]})
        action_id = secrets.token_urlsafe(12)
        now = time.monotonic()
        trace = NavigationTrace(
            action_id=action_id,
            capture_ref=preview.capture_ref,
            goal_pose=self._pose_json(destination),
            planned_route=list(plan["route"]),
            planned_path_distance_m=plan["planned_path_distance_m"],
            started_at=time.time(),
            started_monotonic=now,
            poses=[],
            frames=[],
            last_state={"phase": "starting", "action_id": action_id},
        )
        try:
            live_state = self.live.snapshot(_dataset.session_id)
            initial_pose = live_state.get("pose")
        except Exception:
            initial_pose = None
        if initial_pose is not None:
            trace.poses.append({
                "elapsed_s": 0.0,
                "wall_time": trace.started_at,
                "x": float(initial_pose["x"]),
                "y": float(initial_pose["y"]),
                "yaw": float(initial_pose["yaw"]),
            })
        with self._lock:
            if initial_pose is not None:
                self._latest_pose = dict(initial_pose)
            self._traces[action_id] = trace
            self._active_trace_id = action_id
            while len(self._traces) > 4:
                oldest = next(iter(self._traces))
                if oldest == action_id:
                    break
                self._traces.pop(oldest, None)
        try:
            self.navigation.start(
                destination,
                action_id=action_id,
                capture_ref=preview.capture_ref,
                max_travel_distance_m=preview.max_travel_distance_m,
            )
            with self._lock:
                trace.last_state = self.navigation.snapshot()
        except (RuntimeError, ValueError) as exc:
            with self._lock:
                self._traces.pop(action_id, None)
                if self._active_trace_id == action_id:
                    self._active_trace_id = None
            raise AgentGatewayError(str(exc)) from exc
        with self._lock:
            self._previews.pop(preview.preview_id, None)
        structured = {
            "action_id": action_id,
            "phase": "running",
            "capture_ref": preview.capture_ref,
            "goal_pose": self._pose_json(destination),
            "planned_path_distance_m": plan["planned_path_distance_m"],
            "max_travel_distance_m": preview.max_travel_distance_m,
            "distance_budget_remaining_m": preview.max_travel_distance_m,
            "message": "navigation started after fresh safety revalidation",
        }
        return GatewayResult(structured, (
            ImageAttachment(
                f"active goal {preview.capture_ref}", "image/jpeg",
                path=destination.image_path),
        ))

    def get_navigation_status(self, action_id: str | None = None,
                              wait_s: float = 0.0) -> GatewayResult:
        wait = _finite_number(wait_s, "wait_s")
        if not 0.0 <= wait <= 20.0:
            raise AgentGatewayError("wait_s must be in [0, 20]")
        deadline = time.monotonic() + wait
        initial = self.navigation.snapshot()
        while wait > 0.0 and time.monotonic() < deadline:
            state = self.navigation.snapshot()
            if state.get("phase") not in ("running", "stopping"):
                break
            if state != initial:
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        state = self.navigation.snapshot()
        if action_id is not None and state.get("action_id") != action_id:
            raise AgentGatewayError(
                f"action_id {action_id!r} is not the current or latest action",
                code="action_not_found", retryable=False,
                suggested_tool="get_robot_status")
        terminal = state.get("phase") not in ("running", "stopping")
        return GatewayResult({
            **state,
            "terminal": terminal,
            "retry_after_s": None if terminal else 0.5,
            "suggested_tool": (
                "get_navigation_report" if terminal
                else "get_navigation_status"),
        })

    def get_navigation_report(self, action_id: str | None = None,
                              frame_count: int = 8,
                              brightness_gain: float = 1.0) -> GatewayResult:
        """Return camera and live-SLAM visual evidence for an action."""
        if (isinstance(frame_count, bool) or not isinstance(frame_count, int) or
                not 2 <= frame_count <= 12):
            raise AgentGatewayError(
                "frame_count must be an integer in [2, 12]",
                code="invalid_frame_count")
        brightness = _finite_number(brightness_gain, "brightness_gain")
        if not 0.5 <= brightness <= 3.0:
            raise AgentGatewayError(
                "brightness_gain must be in [0.5, 3.0]",
                code="invalid_brightness_gain")
        navigation_state = self.navigation.snapshot()
        trace_id = action_id or navigation_state.get("action_id")
        with self._lock:
            if trace_id is None:
                trace_id = self._active_trace_id
            trace = self._traces.get(trace_id)
            if trace is None:
                raise AgentGatewayError(
                    "no recorded navigation trace was found for that action",
                    code="trace_not_found", retryable=False,
                    suggested_tool="get_navigation_status",
                    details={"action_id": trace_id})
            if navigation_state.get("action_id") == trace.action_id:
                trace.last_state = dict(navigation_state)
                if (navigation_state.get("phase") not in
                        ("running", "stopping") and
                        trace.finished_at is None):
                    trace.finished_at = time.time()
            poses = [dict(item) for item in trace.poses]
            frames = [dict(item) for item in trace.frames]
            goal_pose = dict(trace.goal_pose)
            route = [dict(item) for item in trace.planned_route]
            state = dict(trace.last_state)
            started_at = trace.started_at
            finished_at = trace.finished_at
            capture = trace.capture_ref
            planned_distance = trace.planned_path_distance_m
        dataset = self.dataset_store.snapshot()
        live_state = self.live.snapshot(dataset.session_id)
        occupancy = live_state.get("occupancy")
        if occupancy is None:
            raise AgentGatewayError(
                "no live occupancy map is available for the report",
                code="map_not_ready", retryable=True,
                suggested_tool="get_robot_status")
        contact_png, selected = render_camera_contact_sheet(
            frames, frame_count=frame_count, brightness_gain=brightness)
        map_png, map_metadata = render_navigation_trace_map(
            occupancy, poses, route, goal_pose, selected)
        actual_distance = 0.0
        for before, after in zip(poses, poses[1:]):
            actual_distance += math.hypot(
                after["x"] - before["x"], after["y"] - before["y"])
        if finished_at is not None:
            duration = max(0.0, finished_at - started_at)
        elif poses:
            duration = max(0.0, float(poses[-1]["elapsed_s"]))
        else:
            duration = 0.0
        final_pose = None if not poses else {
            key: poses[-1][key] for key in ("x", "y", "yaw")
        }
        structured = {
            "action_id": trace_id,
            "capture_ref": capture,
            "phase": state.get("phase"),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": duration,
            "pose_count": len(poses),
            "camera_frame_count": len(frames),
            "selected_frame_count": len(selected),
            "distance_traveled_m": state.get(
                "distance_traveled_m", actual_distance),
            "measured_trace_distance_m": actual_distance,
            "planned_path_distance_m": planned_distance,
            "remaining_path_m": state.get("remaining_path_m"),
            "cross_track_error_m": state.get("cross_track_error_m"),
            "stop_reason": state.get("stop_reason"),
            "navigator_message": state.get("navigator_message"),
            "logs": list(state.get("logs") or []),
            "goal_pose": goal_pose,
            "final_pose": final_pose,
            "camera_frames": selected,
            "map": map_metadata,
            "evidence_sources": ["camera/jpeg", "slam/pose", "slam/map"],
            "simulator_ground_truth_used": False,
            "message": (
                "actual trajectory is measured from live SLAM poses; "
                "simulator ground truth is not subscribed or read"
            ),
        }
        return GatewayResult(structured, (
            ImageAttachment(
                "robot camera contact sheet", "image/png", data=contact_png),
            ImageAttachment(
                "actual SLAM trajectory versus MCP plan", "image/png",
                data=map_png),
        ))

    def stop_navigation(self, action_id: str | None = None,
                        reason: str | None = None) -> GatewayResult:
        stop_reason = str(reason).strip() if reason is not None else ""
        try:
            stopped = self.navigation.stop(
                action_id=action_id,
                reason=stop_reason or "agent requested stop")
        except (RuntimeError, ValueError) as exc:
            raise AgentGatewayError(str(exc)) from exc
        return GatewayResult({
            "stopped": stopped,
            "navigation": self.navigation.snapshot(),
        })
