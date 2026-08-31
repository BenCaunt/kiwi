"""Pose-correlated camera capture storage and wire encoding for Kiwi SLAM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import struct
import threading
import time

import numpy as np
from PIL import Image

from kiwi_slam_core import Pose2, between, compose, wrap_angle


CAMERA_MAGIC = b"KVC1"
CAMERA_HEADER_BYTES = 32
IMAGE_CAPTURE_MAGIC = b"KIM1"
IMAGE_CAPTURE_HEADER = struct.Struct("<4sII")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class CameraHeader:
    """Metadata and encoded bytes from one ``KVC1`` camera message."""

    width: int
    height: int
    sequence: int
    sensor_time_us: int
    encoded: bytes


@dataclass(frozen=True)
class CameraSample:
    """An upright decoded camera frame and an upright JPEG copy."""

    width: int
    height: int
    sequence: int
    sensor_time_us: int
    pixels: np.ndarray
    jpeg: bytes


def parse_camera_header(payload: bytes) -> CameraHeader | None:
    """Validate a camera message without paying the cost of JPEG decoding."""
    payload = bytes(payload)
    if len(payload) < CAMERA_HEADER_BYTES or payload[:4] != CAMERA_MAGIC:
        return None
    header_len = struct.unpack_from("<H", payload, 10)[0]
    if header_len < CAMERA_HEADER_BYTES or header_len >= len(payload):
        return None
    width, height = struct.unpack_from("<HH", payload, 6)
    sequence = struct.unpack_from("<I", payload, 12)[0]
    sensor_time_us = struct.unpack_from("<Q", payload, 16)[0]
    encoded_length = struct.unpack_from("<I", payload, 24)[0]
    encoded = payload[header_len:]
    if encoded_length and encoded_length != len(encoded):
        return None
    return CameraHeader(
        width=int(width),
        height=int(height),
        sequence=int(sequence),
        sensor_time_us=int(sensor_time_us),
        encoded=encoded,
    )


def decode_camera_sample(payload: bytes) -> CameraSample | None:
    """Decode, orient, and JPEG-encode one camera topic payload."""
    header = parse_camera_header(payload)
    if header is None:
        return None
    try:
        with Image.open(io.BytesIO(header.encoded)) as image:
            upright = image.convert("RGB").transpose(Image.Transpose.ROTATE_180)
            pixels = np.array(upright)
            encoded = io.BytesIO()
            upright.save(encoded, format="JPEG", quality=90, optimize=False)
    except (OSError, ValueError):
        return None
    height, width = pixels.shape[:2]
    return CameraSample(
        width=int(width),
        height=int(height),
        sequence=header.sequence,
        sensor_time_us=header.sensor_time_us,
        pixels=pixels,
        jpeg=encoded.getvalue(),
    )


def encode_image_capture(metadata: dict, jpeg: bytes) -> bytes:
    """Encode a correlated image as JSON metadata followed by JPEG bytes."""
    metadata_bytes = json.dumps(
        metadata, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    jpeg = bytes(jpeg)
    return (
        IMAGE_CAPTURE_HEADER.pack(
            IMAGE_CAPTURE_MAGIC, len(metadata_bytes), len(jpeg)
        )
        + metadata_bytes
        + jpeg
    )


def decode_image_capture(payload: bytes) -> tuple[dict, bytes]:
    """Decode and validate a pose-correlated image topic payload."""
    payload = bytes(payload)
    if len(payload) < IMAGE_CAPTURE_HEADER.size:
        raise ValueError("image capture payload is truncated")
    magic, metadata_length, jpeg_length = IMAGE_CAPTURE_HEADER.unpack_from(payload)
    expected = IMAGE_CAPTURE_HEADER.size + metadata_length + jpeg_length
    if magic != IMAGE_CAPTURE_MAGIC or expected != len(payload):
        raise ValueError("invalid image capture envelope")
    metadata_end = IMAGE_CAPTURE_HEADER.size + metadata_length
    try:
        metadata = json.loads(
            payload[IMAGE_CAPTURE_HEADER.size:metadata_end].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid image capture metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("image capture metadata must be an object")
    jpeg = payload[metadata_end:]
    if not jpeg:
        raise ValueError("image capture has no image bytes")
    return metadata, jpeg


def _pose_dict(pose: Pose2) -> dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "yaw": float(pose.yaw)}


def _dict_pose(value: dict) -> Pose2:
    return Pose2(float(value["x"]), float(value["y"]), float(value["yaw"]))


def _validate_session_id(value: str) -> str:
    value = str(value)
    if not SESSION_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "image session id must use 1-128 letters, digits, '.', '_', or '-'"
        )
    return value


def discover_compatible_image_manifest(output_prefix, keyframes):
    """Find the newest largest image map that belongs to ``keyframes``.

    Prefix directories can contain manifests from maps that previously used
    the same filename. A compatible capture must point at a real keyframe and
    reproduce its saved map pose from raw odometry. This keeps auto-resume from
    silently importing images from an unrelated overwritten map.
    """
    keyframes = list(keyframes)
    keyframe_sessions = {keyframe.session_id for keyframe in keyframes}
    root = Path(f"{Path(output_prefix).expanduser()}.images")
    compatible = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = document.get("captures") if isinstance(document, dict) else None
        if (document.get("format") != ImageMapRecorder.FORMAT or
                document.get("frame") != "map" or
                not isinstance(records, list) or not records):
            continue
        valid = True
        anchored_records = 0
        for record in records:
            try:
                pose_time_s = float(record["pose_time_s"])
                raw_pose = _dict_pose(record["raw_pose"])
                saved_pose = _dict_pose(record["pose"])
                image_name = record["image"]
                image_path = (manifest_path.parent / image_name).resolve()
                image_path.relative_to(manifest_path.parent.resolve())
                if not image_path.is_file():
                    raise ValueError
                session_id = record.get("slam_session_id")
                nearest_keyframe = record.get("nearest_keyframe")
                if nearest_keyframe is None:
                    # A camera callback used to be able to append one last
                    # capture after shutdown reprojection. It still belongs to
                    # this image session, but has no keyframe association yet.
                    if (session_id is None or isinstance(session_id, bool) or
                            int(session_id) not in keyframe_sessions):
                        raise ValueError
                    if not all(math.isfinite(value) for value in (
                            pose_time_s, raw_pose.x, raw_pose.y, raw_pose.yaw,
                            saved_pose.x, saved_pose.y, saved_pose.yaw)):
                        raise ValueError
                    continue
                keyframe_index = int(nearest_keyframe)
                keyframe = keyframes[keyframe_index]
                if keyframe.index != keyframe_index:
                    raise ValueError
                projected = compose(
                    keyframe.pose, between(keyframe.raw_pose, raw_pose))
                disagreement = between(saved_pose, projected)
                translation = math.hypot(disagreement.x, disagreement.y)
                rotation = abs(wrap_angle(disagreement.yaw))
                if not all(math.isfinite(value) for value in (
                        pose_time_s, raw_pose.x, raw_pose.y, raw_pose.yaw,
                        saved_pose.x, saved_pose.y, saved_pose.yaw,
                        translation, rotation)):
                    raise ValueError
                if session_id is not None:
                    if int(session_id) != keyframe.session_id:
                        raise ValueError
                elif abs(keyframe.time_s - pose_time_s) > 3.0:
                    # Legacy captures have no explicit graph session, so time
                    # proximity remains part of their compatibility proof.
                    raise ValueError
                if (translation > 0.10 or
                        rotation > math.radians(5.0)):
                    raise ValueError
                anchored_records += 1
            except (IndexError, KeyError, TypeError, ValueError, OverflowError):
                valid = False
                break
        if valid and anchored_records:
            compatible.append((
                anchored_records, manifest_path.stat().st_mtime_ns,
                manifest_path.resolve(),
            ))
    if not compatible:
        return None
    compatible.sort(reverse=True, key=lambda item: item[:2])
    return compatible[0][2]


class ImageMapRecorder:
    """Persist pose-spaced images and maintain a future-CLIP-ready manifest."""

    FORMAT = "kiwi-image-map-v1"

    def __init__(
        self,
        output_prefix: str | Path,
        translation_spacing_m: float = 0.50,
        rotation_spacing_rad: float = math.radians(30.0),
        min_interval_s: float = 0.50,
        horizontal_fov_deg: float = 72.0,
        camera_height_m: float = 0.10,
        session_id: str | None = None,
        resume_manifest: str | Path | None = None,
        slam_session_id: int = 0,
        keyframe_sessions: dict[int, int] | None = None,
        read_only: bool = False,
    ):
        if translation_spacing_m < 0.0 or rotation_spacing_rad < 0.0:
            raise ValueError("image spacing must be non-negative")
        if min_interval_s < 0.0:
            raise ValueError("image minimum interval must be non-negative")
        if not 1.0 <= horizontal_fov_deg < 179.0:
            raise ValueError("camera horizontal FOV must be in [1, 179) degrees")
        self.translation_spacing_m = float(translation_spacing_m)
        self.rotation_spacing_rad = float(rotation_spacing_rad)
        self.min_interval_s = float(min_interval_s)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.camera_height_m = float(camera_height_m)
        self.slam_session_id = int(slam_session_id)
        if self.slam_session_id < 0:
            raise ValueError("SLAM session id must be nonnegative")
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.read_only = bool(read_only)
        if self.read_only and resume_manifest is None:
            raise ValueError("read-only image maps require a resumed manifest")
        self._session_id_explicit = session_id is not None
        self.session_id = _validate_session_id(
            session_id or datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"))
        root = Path(f"{Path(output_prefix)}.images")
        self.records: list[dict] = []
        self._last_pose: Pose2 | None = None
        self._last_capture_monotonic: float | None = None
        self._lock = threading.RLock()
        self._accepting_captures = True
        self._resumed = resume_manifest is not None
        self._active = not self._resumed
        if resume_manifest is not None:
            self.manifest_path = Path(resume_manifest).expanduser().resolve()
            self.image_dir = self.manifest_path.parent
            self._load_manifest(keyframe_sessions or {})
        else:
            image_dir = root / self.session_id
            suffix = 1
            while image_dir.exists():
                image_dir = root / f"{self.session_id}_{suffix}"
                suffix += 1
            if image_dir.name != self.session_id:
                self.session_id = image_dir.name
            self.image_dir = image_dir
            self.manifest_path = image_dir / "manifest.json"
            self.image_dir.mkdir(parents=True, exist_ok=False)
            self._write_manifest()

    def _load_manifest(self, keyframe_sessions):
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(
                f"cannot read image-map manifest: {self.manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid image-map manifest: {self.manifest_path}") from exc
        if (not isinstance(document, dict) or
                document.get("format") != self.FORMAT or
                document.get("frame") != "map"):
            raise ValueError("unsupported resumed image-map manifest")
        source_session_id = _validate_session_id(document.get("session_id", ""))
        if not self._session_id_explicit:
            self.session_id = source_session_id
        created_at = document.get("created_at")
        if isinstance(created_at, str) and created_at:
            self.created_at = created_at
        records = document.get("captures")
        if not isinstance(records, list):
            raise ValueError("resumed image-map captures must be an array")
        capture_ids = set()
        for index, value in enumerate(records):
            if not isinstance(value, dict):
                raise ValueError(f"image capture {index} must be an object")
            record = dict(value)
            capture_id = record.get("id")
            if (isinstance(capture_id, bool) or
                    not isinstance(capture_id, int) or capture_id < 0 or
                    capture_id in capture_ids):
                raise ValueError(f"image capture {index} has an invalid id")
            capture_ids.add(capture_id)
            image_name = record.get("image")
            if not isinstance(image_name, str) or not image_name:
                raise ValueError(f"image capture {capture_id} has no image")
            image_path = (self.image_dir / image_name).resolve()
            try:
                image_path.relative_to(self.image_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"image capture {capture_id} escapes its session directory"
                ) from exc
            if not image_path.is_file():
                raise ValueError(
                    f"image capture {capture_id} is missing: {image_path}")
            try:
                raw_pose = _dict_pose(record["raw_pose"])
                map_pose = _dict_pose(record["pose"])
                pose_time_s = float(record["pose_time_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"image capture {capture_id} has invalid pose metadata") from exc
            if not all(math.isfinite(value) for value in (
                    raw_pose.x, raw_pose.y, raw_pose.yaw,
                    map_pose.x, map_pose.y, map_pose.yaw, pose_time_s)):
                raise ValueError(
                    f"image capture {capture_id} pose metadata is not finite")
            graph_session = record.get("slam_session_id")
            if graph_session is None:
                nearest = record.get("nearest_keyframe")
                if nearest not in keyframe_sessions:
                    raise ValueError(
                        f"image capture {capture_id} has no saved keyframe session")
                graph_session = keyframe_sessions[nearest]
            elif isinstance(graph_session, bool):
                raise ValueError(
                    f"image capture {capture_id} has invalid SLAM session")
            graph_session = int(graph_session)
            if graph_session < 0:
                raise ValueError(
                    f"image capture {capture_id} has invalid SLAM session")
            nearest = record.get("nearest_keyframe")
            if (isinstance(nearest, int) and not isinstance(nearest, bool) and
                    nearest in keyframe_sessions and
                    keyframe_sessions[nearest] != graph_session):
                raise ValueError(
                    f"image capture {capture_id} keyframe session disagrees")
            old_record_session = record.get("session_id", source_session_id)
            if old_record_session != self.session_id:
                record.setdefault("source_session_id", str(old_record_session))
            record["session_id"] = self.session_id
            record["slam_session_id"] = graph_session
            self.records.append(record)
        if self.records:
            self._last_pose = _dict_pose(self.records[-1]["pose"])

    def activate(self):
        """Commit an in-memory resumed manifest after SLAM relocalizes."""
        with self._lock:
            if not self._active:
                self._active = True
                if not self.read_only:
                    self._write_manifest()

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def should_capture(self, pose: Pose2, monotonic_s: float | None = None) -> bool:
        """Return whether ``pose`` clears the configured translation/rotation gate."""
        if self.read_only:
            return False
        now = time.monotonic() if monotonic_s is None else float(monotonic_s)
        with self._lock:
            if not self._accepting_captures:
                return False
            if self._last_pose is None:
                return True
            if (
                self._last_capture_monotonic is not None
                and now - self._last_capture_monotonic < self.min_interval_s
            ):
                return False
            translation = math.hypot(
                pose.x - self._last_pose.x, pose.y - self._last_pose.y
            )
            rotation = abs(wrap_angle(pose.yaw - self._last_pose.yaw))
            return (
                translation >= self.translation_spacing_m
                or rotation >= self.rotation_spacing_rad
            )

    def capture(
        self,
        sample: CameraSample,
        raw_pose: Pose2,
        map_pose: Pose2,
        pose_time_s: float,
        wall_time_s: float | None = None,
        monotonic_s: float | None = None,
    ) -> tuple[dict, bytes] | None:
        """Record a sample if it clears the pose spacing gate."""
        if self.read_only:
            return None
        now = time.monotonic() if monotonic_s is None else float(monotonic_s)
        with self._lock:
            if not self._accepting_captures:
                return None
            if not self.should_capture(map_pose, now):
                return None
            capture_id = max(
                (int(record["id"]) for record in self.records), default=-1) + 1
            filename = f"{capture_id:06d}.jpg"
            if (self.image_dir / filename).exists():
                raise FileExistsError(
                    f"image capture path already exists: {filename}")
            intrinsics = self._intrinsics(sample.width, sample.height)
            record = {
                "id": capture_id,
                "session_id": self.session_id,
                "slam_session_id": self.slam_session_id,
                "image": filename,
                "time_s": float(time.time() if wall_time_s is None else wall_time_s),
                "pose_time_s": float(pose_time_s),
                "camera_sequence": int(sample.sequence),
                "camera_sensor_time_us": int(sample.sensor_time_us),
                "pose": _pose_dict(map_pose),
                "raw_pose": _pose_dict(raw_pose),
                "nearest_keyframe": None,
                "camera": {
                    **intrinsics,
                    "height_m": self.camera_height_m,
                },
            }
            (self.image_dir / filename).write_bytes(sample.jpeg)
            self.records.append(record)
            self._last_pose = map_pose
            self._last_capture_monotonic = now
            self._write_manifest()
            return dict(record), encode_image_capture(record, sample.jpeg)

    def seal(self) -> None:
        """Stop new captures, waiting for any in-flight capture to finish."""
        with self._lock:
            self._accepting_captures = False

    def reproject(self, keyframes) -> list[int]:
        """Move captures into the optimized map using their nearest keyframe."""
        if self.read_only:
            return []
        keyframes = list(keyframes)
        if not keyframes:
            return []
        sessions = {}
        for keyframe in keyframes:
            sessions.setdefault(keyframe.session_id, []).append(keyframe)
        updated = []
        with self._lock:
            for index, record in enumerate(self.records):
                candidates = sessions.get(int(record.get(
                    "slam_session_id", self.slam_session_id)), [])
                if not candidates:
                    continue
                nearest = min(
                    candidates,
                    key=lambda keyframe: abs(
                        float(keyframe.time_s) - float(record["pose_time_s"])
                    ),
                )
                raw_pose = _dict_pose(record["raw_pose"])
                map_pose = compose(
                    nearest.pose, between(nearest.raw_pose, raw_pose)
                )
                record["pose"] = _pose_dict(map_pose)
                record["nearest_keyframe"] = int(nearest.index)
                updated.append(index)
            if self.records:
                self._last_pose = _dict_pose(self.records[-1]["pose"])
            self._write_manifest()
        return updated

    def packet(self, index: int) -> bytes:
        """Return one saved capture in its live topic wire format."""
        with self._lock:
            record = dict(self.records[index])
            jpeg = (self.image_dir / record["image"]).read_bytes()
        return encode_image_capture(record, jpeg)

    def _intrinsics(self, width: int, height: int) -> dict[str, float | int]:
        focal = 0.5 * float(width) / math.tan(
            math.radians(self.horizontal_fov_deg) * 0.5
        )
        return {
            "width": int(width),
            "height": int(height),
            "fx": focal,
            "fy": focal,
            "cx": (float(width) - 1.0) * 0.5,
            "cy": (float(height) - 1.0) * 0.5,
            "horizontal_fov_deg": self.horizontal_fov_deg,
        }

    def _write_manifest(self) -> None:
        manifest = {
            "format": self.FORMAT,
            "frame": "map",
            "session_id": self.session_id,
            "created_at": self.created_at,
            "resumed": self._resumed,
            "selection": {
                "translation_spacing_m": self.translation_spacing_m,
                "rotation_spacing_deg": math.degrees(self.rotation_spacing_rad),
                "min_interval_s": self.min_interval_s,
            },
            "camera": {
                "horizontal_fov_deg": self.horizontal_fov_deg,
                "height_m": self.camera_height_m,
                "orientation": "forward, level, upright",
            },
            "captures": self.records,
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)
