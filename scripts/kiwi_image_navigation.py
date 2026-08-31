#!/usr/bin/env python3
"""Browse a Kiwi image map and drive to a selected capture pose.

The local web UI reads one ``kiwi-image-map-v1`` manifest, shows its saved
camera frames, and launches ``kiwi_navigation.py`` for the selected image's
map-frame position and heading.  Keep the SLAM process that recorded the
image map running: map coordinates are local to that SLAM session.

Example:
  python3 scripts/kiwi_image_navigation.py
  python3 scripts/kiwi_image_navigation.py --namespace kiwi/sim
  python3 scripts/kiwi_image_navigation.py --manifest \
      maps/kiwi_map.images/20260823T180000Z/manifest.json
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import webbrowser

from kiwi_client import DEFAULT_ROBOT_YAW_DEG, KiwiClient
from kiwi_agent_gateway import (
    ClipPlaceIndex,
    KiwiAgentGateway,
    OpenClipEncoder,
)
from kiwi_image_map import decode_image_capture
from kiwi_map import decode_occupancy_map
from kiwi_navigation_core import (
    DEFAULT_MAX_FOLLOWING_SPEED_MPS,
    DEFAULT_RUNTIME_COLLISION_RADIUS_M,
)


IMAGE_MAP_FORMAT = "kiwi-image-map-v1"
DEFAULT_IMAGE_ROOT = "maps/kiwi_map.images"


class ImageMapError(ValueError):
    """Raised when an image-map manifest cannot be used safely."""


class NavigationBusy(RuntimeError):
    """Raised when a second navigation is requested while one is active."""


@dataclass(frozen=True)
class ImageDestination:
    capture_id: int
    image_name: str
    image_path: Path
    x: float
    y: float
    yaw: float
    time_s: float | None

    def as_json(self) -> dict:
        return {
            "id": self.capture_id,
            "image": self.image_name,
            "image_url": f"/image/{self.capture_id}",
            "pose": {"x": self.x, "y": self.y, "yaw": self.yaw},
            "yaw_deg": math.degrees(self.yaw),
            "time_s": self.time_s,
        }


@dataclass(frozen=True)
class ImageMapDataset:
    manifest_path: Path
    session_id: str
    created_at: str | None
    captures: tuple[ImageDestination, ...]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ImageMapDataset":
        path = Path(manifest_path).expanduser().resolve()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ImageMapError(f"cannot read image-map manifest: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ImageMapError(f"invalid image-map JSON: {exc}") from exc

        if not isinstance(document, dict):
            raise ImageMapError("image-map manifest must be a JSON object")
        if document.get("format") != IMAGE_MAP_FORMAT:
            raise ImageMapError(
                f"unsupported image-map format {document.get('format')!r}"
            )
        if document.get("frame") != "map":
            raise ImageMapError("image-map captures must use the map frame")
        session_id = document.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ImageMapError("image-map session_id is missing")
        records = document.get("captures")
        if not isinstance(records, list):
            raise ImageMapError("image-map captures must be an array")

        image_root = path.parent.resolve()
        captures = []
        capture_ids = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ImageMapError(f"capture {index} must be an object")
            capture_id = record.get("id")
            if isinstance(capture_id, bool) or not isinstance(capture_id, int):
                raise ImageMapError(f"capture {index} has an invalid id")
            if capture_id in capture_ids:
                raise ImageMapError(f"duplicate capture id {capture_id}")
            capture_ids.add(capture_id)

            image_name = record.get("image")
            if not isinstance(image_name, str) or not image_name:
                raise ImageMapError(f"capture {capture_id} has no image")
            image_path = (image_root / image_name).resolve()
            try:
                image_path.relative_to(image_root)
            except ValueError as exc:
                raise ImageMapError(
                    f"capture {capture_id} image escapes its session directory"
                ) from exc
            if not image_path.is_file():
                raise ImageMapError(
                    f"capture {capture_id} image is missing: {image_path}"
                )

            pose = record.get("pose")
            if not isinstance(pose, dict):
                raise ImageMapError(f"capture {capture_id} has no map pose")
            try:
                x, y, yaw = (float(pose[key]) for key in ("x", "y", "yaw"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ImageMapError(
                    f"capture {capture_id} has an invalid map pose"
                ) from exc
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ImageMapError(f"capture {capture_id} pose is not finite")

            raw_time = record.get("time_s")
            if raw_time is None:
                capture_time = None
            else:
                try:
                    capture_time = float(raw_time)
                except (TypeError, ValueError) as exc:
                    raise ImageMapError(
                        f"capture {capture_id} has an invalid timestamp"
                    ) from exc
                if not math.isfinite(capture_time):
                    raise ImageMapError(
                        f"capture {capture_id} timestamp is not finite"
                    )

            captures.append(ImageDestination(
                capture_id=capture_id,
                image_name=image_name,
                image_path=image_path,
                x=x,
                y=y,
                yaw=yaw,
                time_s=capture_time,
            ))

        return cls(
            manifest_path=path,
            session_id=session_id,
            created_at=(str(document["created_at"])
                        if document.get("created_at") is not None else None),
            captures=tuple(captures),
        )

    def capture(self, capture_id: int) -> ImageDestination:
        for capture in self.captures:
            if capture.capture_id == capture_id:
                return capture
        raise KeyError(capture_id)

    def as_json(self) -> dict:
        return {
            "manifest": str(self.manifest_path),
            "session_id": self.session_id,
            "created_at": self.created_at,
            "captures": [capture.as_json() for capture in self.captures],
        }


def discover_manifest(image_root: str | Path) -> Path:
    """Return the most recently modified image-map manifest below a root."""
    root = Path(image_root).expanduser().resolve()
    candidates = [path for path in root.glob("*/manifest.json") if path.is_file()]
    if not candidates:
        raise ImageMapError(f"no image-map manifests found below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


class DatasetStore:
    """Reload a manifest after atomic updates from the running SLAM process."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self._lock = threading.RLock()
        self._mtime_ns = None
        self._dataset = None

    def snapshot(self) -> ImageMapDataset:
        with self._lock:
            try:
                mtime_ns = self.manifest_path.stat().st_mtime_ns
            except OSError as exc:
                raise ImageMapError(f"cannot stat image-map manifest: {exc}") from exc
            if self._dataset is None or mtime_ns != self._mtime_ns:
                self._dataset = ImageMapDataset.load(self.manifest_path)
                self._mtime_ns = mtime_ns
            return self._dataset


@dataclass(frozen=True)
class NavigationSettings:
    connect: str
    namespace: str
    robot_yaw_deg: float
    inflation_radius: float
    allow_unknown: bool
    lookahead: float
    max_linear_speed: float
    max_angular_speed: float
    position_tolerance: float
    yaw_tolerance_deg: float
    replan_distance: float
    max_duration: float
    kp_yaw: float = 2.5
    goal_yaw_blend_distance: float = 0.30
    command_topic: str = "cmd_vel"
    runtime_collision_radius: float = DEFAULT_RUNTIME_COLLISION_RADIUS_M
    calibration: str | None = None

    def command(self, destination: ImageDestination, *,
                action_id: str | None = None,
                max_travel_distance_m: float | None = None) -> list[str]:
        navigator = Path(__file__).resolve().with_name("kiwi_navigation.py")
        command = [
            sys.executable,
            "-u",
            str(navigator),
            repr(destination.x),
            repr(destination.y),
            "--goal-yaw-deg", repr(math.degrees(destination.yaw)),
            "--connect", self.connect,
            "--namespace", self.namespace,
            "--command-topic", self.command_topic,
            "--robot-yaw-deg", repr(self.robot_yaw_deg),
            "--inflation-radius", repr(self.inflation_radius),
            "--runtime-collision-radius", repr(self.runtime_collision_radius),
            "--lookahead", repr(self.lookahead),
            "--goal-yaw-blend-distance",
            repr(self.goal_yaw_blend_distance),
            "--kp-yaw", repr(self.kp_yaw),
            "--max-linear-speed", repr(self.max_linear_speed),
            "--max-angular-speed", repr(self.max_angular_speed),
            "--position-tolerance", repr(self.position_tolerance),
            "--yaw-tolerance-deg", repr(self.yaw_tolerance_deg),
            "--replan-distance", repr(self.replan_distance),
            "--max-duration", repr(self.max_duration),
        ]
        if action_id is not None:
            command += ["--action-id", str(action_id)]
        if max_travel_distance_m is not None:
            command += [
                "--max-travel-distance", repr(max_travel_distance_m)]
        if self.calibration:
            command += ["--calibration", self.calibration]
        if self.allow_unknown:
            command.append("--allow-unknown")
        return command


class NavigationManager:
    """Own the single navigation child process controlled by the web UI."""

    def __init__(self, settings: NavigationSettings):
        self.settings = settings
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._stop_requested = False
        self._safety_stop_started = False
        self._last_pose = None
        self._logs = deque(maxlen=100)
        self._state = {
            "phase": "idle",
            "action_id": None,
            "capture_ref": None,
            "capture_id": None,
            "goal": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "stop_reason": None,
            "max_travel_distance_m": None,
            "distance_traveled_m": 0.0,
            "remaining_path_m": None,
            "distance_budget_remaining_m": None,
        }

    def start(self, destination: ImageDestination, *,
              action_id: str | None = None,
              capture_ref: str | None = None,
              max_travel_distance_m: float | None = None) -> str:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise NavigationBusy("navigation is already active; stop it first")
            action_id = action_id or secrets.token_urlsafe(12)
            if max_travel_distance_m is not None:
                max_travel_distance_m = float(max_travel_distance_m)
                if (not math.isfinite(max_travel_distance_m) or
                        max_travel_distance_m <= 0.0):
                    raise ValueError("maximum travel distance must be positive")
            command = self.settings.command(
                destination, action_id=action_id,
                max_travel_distance_m=max_travel_distance_m)
            self._logs.clear()
            self._logs.append(
                "Selected image "
                f"{destination.capture_id}: goal "
                f"({destination.x:+.3f}, {destination.y:+.3f}, "
                f"{math.degrees(destination.yaw):+.1f} deg)"
            )
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                self._state.update(
                    phase="failed", finished_at=time.time(), exit_code=None)
                self._logs.append(f"Could not start navigation: {exc}")
                raise RuntimeError(f"could not start navigation: {exc}") from exc
            self._process = process
            self._stop_requested = False
            self._safety_stop_started = False
            self._last_pose = None
            self._state = {
                "phase": "running",
                "action_id": action_id,
                "capture_ref": capture_ref,
                "capture_id": destination.capture_id,
                "goal": {
                    "x": destination.x,
                    "y": destination.y,
                    "yaw": destination.yaw,
                    "yaw_deg": math.degrees(destination.yaw),
                },
                "started_at": time.time(),
                "finished_at": None,
                "exit_code": None,
                "stop_reason": None,
                "max_travel_distance_m": max_travel_distance_m,
                "distance_traveled_m": 0.0,
                "remaining_path_m": None,
                "distance_budget_remaining_m": max_travel_distance_m,
                "navigator_status": None,
            }
            threading.Thread(
                target=self._read_output,
                args=(process,),
                name="kiwi-image-navigation-output",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._wait_for_exit,
                args=(process,),
                name="kiwi-image-navigation-wait",
                daemon=True,
            ).start()
            return action_id

    def _request_safety_stop(self, reason: str) -> None:
        with self._lock:
            if (self._safety_stop_started or self._process is None or
                    self._process.poll() is not None):
                return
            self._safety_stop_started = True
            self._logs.append(f"Safety stop: {reason}")
            action_id = self._state.get("action_id")
        threading.Thread(
            target=self.stop,
            kwargs={"action_id": action_id, "reason": reason},
            name="kiwi-agent-safety-stop",
            daemon=True,
        ).start()

    def observe_pose(self, pose: dict) -> None:
        try:
            current = tuple(float(pose[key]) for key in ("x", "y"))
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in current):
            return
        with self._lock:
            active = self._state.get("phase") in ("running", "stopping")
            if not active:
                self._last_pose = None
                return
            # The SLAM pose can legitimately jump when scan matching or a loop
            # closure corrects the map transform. Physical travel is reported
            # by the navigator from integrated encoder speed; never charge a
            # map correction against the action's distance authorization.
            self._last_pose = current

    def observe_navigation_state(self, report: dict) -> None:
        if not isinstance(report, dict):
            return
        stop_reason = None
        with self._lock:
            if self._state.get("phase") not in ("running", "stopping"):
                return
            report_action = report.get("action_id")
            if (report_action is not None and
                    report_action != self._state.get("action_id")):
                return
            self._state["navigator_status"] = report.get("status")
            for source, target in (
                    ("remaining_m", "remaining_path_m"),
                    ("progress_m", "path_progress_m"),
                    ("cross_track_error_m", "cross_track_error_m")):
                value = report.get(source)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self._state[target] = float(value)
            if report.get("message"):
                self._state["navigator_message"] = str(report["message"])
            reported_distance = report.get("distance_traveled_m")
            if (isinstance(reported_distance, (int, float)) and
                    math.isfinite(reported_distance) and
                    reported_distance >= 0.0):
                self._state["distance_traveled_m"] = float(reported_distance)
            budget = self._state.get("max_travel_distance_m")
            remaining = self._state.get("remaining_path_m")
            traveled = self._state.get("distance_traveled_m", 0.0)
            if budget is not None:
                self._state["distance_budget_remaining_m"] = max(
                    0.0, budget - traveled)
                if (remaining is not None and
                        traveled + remaining > budget + 1e-6):
                    stop_reason = (
                        "replanned route exceeds the remaining travel budget")
        if stop_reason:
            self._request_safety_stop(stop_reason)

    def observe_mux_status(self, status: dict) -> None:
        if isinstance(status, dict) and status.get("source") == "teleop":
            self._request_safety_stop("teleop took priority")

    def _read_output(self, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            line = line.rstrip()
            if line:
                with self._lock:
                    self._logs.append(line)

    def _wait_for_exit(self, process: subprocess.Popen) -> None:
        exit_code = process.wait()
        with self._lock:
            if self._process is not process:
                return
            if self._stop_requested:
                phase = "stopped"
            else:
                phase = "succeeded" if exit_code == 0 else "failed"
            self._state.update(
                phase=phase,
                finished_at=time.time(),
                exit_code=exit_code,
            )
            self._process = None
            self._last_pose = None

    def stop(self, timeout_s: float = 5.0, *,
             action_id: str | None = None,
             reason: str | None = None) -> bool:
        with self._lock:
            current_action = self._state.get("action_id")
            if action_id is not None and action_id != current_action:
                raise ValueError(
                    f"action_id {action_id!r} is not the current or latest action")
            process = self._process
            if process is None or process.poll() is not None:
                return False
            self._stop_requested = True
            self._state["phase"] = "stopping"
            self._state["stop_reason"] = reason or "stop requested"
            self._logs.append(
                f"{self._state['stop_reason']}; sending a zero command and closing.")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with self._lock:
                self._logs.append("Navigation did not stop in time; terminating it.")
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._state, "logs": list(self._logs)}


class LiveSlamMonitor:
    """Confirm that the selected image map belongs to the live SLAM session."""

    def __init__(self, connect: str, namespace: str, robot_yaw_deg: float,
                 skip_session_check: bool = False,
                 require_mux: bool = False):
        self.skip_session_check = bool(skip_session_check)
        self.require_mux = bool(require_mux)
        self._lock = threading.RLock()
        self._client = None
        self._connection_error = None
        self._map_seen_at = None
        self._occupancy = None
        self._image_seen_at = None
        self._image_session_id = None
        self._scan_matched = False
        self._quality = None
        self._navigation_state = None
        self._navigation_seen_at = None
        self._trajectory = []
        self._mux_status = None
        self._mux_seen_at = None
        self._pose_callbacks = []
        self._navigation_callbacks = []
        self._mux_callbacks = []
        self._camera_callbacks = []
        try:
            client = KiwiClient(connect, namespace, robot_yaw_deg)
            client.add_slam_callback(self._on_slam)
            client.add_pose_callback(self._on_pose)
            client.subscribe("slam/map", self._on_map)
            client.subscribe("slam/image", self._on_image)
            # The diagnostic report samples this robot-visible stream only
            # while an MCP navigation action is active.
            client.subscribe("camera/jpeg", self._on_camera)
            client.subscribe("navigation/state", self._on_navigation_state)
            client.subscribe(
                "navigation/trajectory", self._on_navigation_trajectory)
            client.subscribe("cmd_vel/mux/status", self._on_mux_status)
            self._client = client
        except Exception as exc:  # Zenoh reports several transport exception types.
            self._connection_error = str(exc)

    def _on_map(self, payload: bytes) -> None:
        try:
            occupancy = decode_occupancy_map(payload)
        except ValueError:
            return
        with self._lock:
            self._occupancy = occupancy
            self._map_seen_at = time.monotonic()

    def _on_slam(self, report: dict) -> None:
        quality = report.get("quality")
        matched = (
            isinstance(quality, dict) and
            quality.get("scan_matched") is True and
            quality.get("relocalizing") is not True
        )
        with self._lock:
            self._scan_matched = matched
            self._quality = dict(quality) if isinstance(quality, dict) else None

    def _on_pose(self, pose: dict) -> None:
        with self._lock:
            callbacks = list(self._pose_callbacks)
        for callback in callbacks:
            callback(dict(pose))

    def _on_image(self, payload: bytes) -> None:
        try:
            metadata, _jpeg = decode_image_capture(payload)
            session_id = str(metadata["session_id"])
        except (KeyError, TypeError, ValueError):
            return
        with self._lock:
            self._image_session_id = session_id
            self._image_seen_at = time.monotonic()

    def _on_camera(self, payload: bytes) -> None:
        with self._lock:
            callbacks = list(self._camera_callbacks)
        for callback in callbacks:
            callback(bytes(payload))

    @staticmethod
    def _json_report(payload: bytes) -> dict | None:
        try:
            report = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return report if isinstance(report, dict) else None

    def _on_navigation_state(self, payload: bytes) -> None:
        report = self._json_report(payload)
        if report is None or report.get("frame") != "map":
            return
        with self._lock:
            self._navigation_state = report
            self._navigation_seen_at = time.monotonic()
            callbacks = list(self._navigation_callbacks)
        for callback in callbacks:
            callback(dict(report))

    def _on_navigation_trajectory(self, payload: bytes) -> None:
        report = self._json_report(payload)
        if report is None or report.get("frame") != "map":
            return
        try:
            points = [
                (float(point["x"]), float(point["y"]))
                for point in report["points"]
            ]
        except (KeyError, TypeError, ValueError):
            return
        if not points or not all(math.isfinite(value)
                                 for point in points for value in point):
            return
        with self._lock:
            self._trajectory = points

    def _on_mux_status(self, payload: bytes) -> None:
        report = self._json_report(payload)
        if report is None or report.get("source") not in (
                "idle", "teleop", "navigation"):
            return
        with self._lock:
            self._mux_status = report
            self._mux_seen_at = time.monotonic()
            callbacks = list(self._mux_callbacks)
        for callback in callbacks:
            callback(dict(report))

    def add_pose_callback(self, callback):
        with self._lock:
            self._pose_callbacks.append(callback)
        return callback

    def add_navigation_callback(self, callback):
        with self._lock:
            self._navigation_callbacks.append(callback)
        return callback

    def add_mux_callback(self, callback):
        with self._lock:
            self._mux_callbacks.append(callback)
        return callback

    def add_camera_callback(self, callback):
        with self._lock:
            self._camera_callbacks.append(callback)
        return callback

    def status(self, expected_session_id: str) -> dict:
        now = time.monotonic()
        with self._lock:
            client = self._client
            live_session = self._image_session_id
            image_seen_at = self._image_seen_at
            map_seen_at = self._map_seen_at
            scan_matched = self._scan_matched
            connection_error = self._connection_error
            quality = getattr(self, "_quality", None)
            mux_status = getattr(self, "_mux_status", None)
            mux_seen_at = getattr(self, "_mux_seen_at", None)

        pose_seen_at = (None if client is None else
                        getattr(client, "pose_received_at", None))
        client_pose = None if client is None else getattr(client, "pose", None)
        pose = None if client_pose is None else dict(client_pose)
        pose_age = None if pose_seen_at is None else max(0.0, now - pose_seen_at)
        map_age = None if map_seen_at is None else max(0.0, now - map_seen_at)
        image_age = None if image_seen_at is None else max(
            0.0, now - image_seen_at)
        mux_age = None if mux_seen_at is None else max(0.0, now - mux_seen_at)

        ready = False
        status_code = "not_ready"
        recovery = None
        if client is None:
            reason = "Zenoh connection failed"
            status_code = "connection_failed"
            if connection_error:
                reason += f": {connection_error}"
        elif pose_seen_at is None or pose_age > 1.0:
            reason = "waiting for a fresh SLAM pose"
            status_code = "pose_not_ready"
        elif not scan_matched:
            reason = "waiting for verified SLAM relocalization"
            status_code = "relocalizing"
        elif map_seen_at is None or now - map_seen_at > 8.0:
            reason = "waiting for the live SLAM occupancy map"
            status_code = "map_not_ready"
        elif not self.skip_session_check and (
                image_seen_at is None or now - image_seen_at > 4.0):
            reason = "waiting for the live image-map session"
            status_code = "image_session_not_ready"
        elif not self.skip_session_check and live_session != expected_session_id:
            reason = (
                f"manifest session {expected_session_id} does not match live "
                f"session {live_session}"
            )
            status_code = "session_mismatch"
            recovery = {
                "code": "session_mismatch",
                "expected_session_id": expected_session_id,
                "live_session_id": live_session,
                "action": (
                    "restart SLAM and image navigation with one explicit manifest"
                ),
            }
        elif (getattr(self, "require_mux", False) and
              (mux_age is None or mux_age > 1.0)):
            reason = "waiting for command mux status"
            status_code = "mux_not_ready"
        elif (getattr(self, "require_mux", False) and
              mux_status.get("source") == "teleop"):
            reason = "teleop is currently in control"
            status_code = "teleop_active"
        else:
            ready = True
            status_code = "ready"
            reason = "live navigation inputs ready; preview required before driving"
        return {
            "ready": ready,
            "reason": reason,
            "status_code": status_code,
            "recovery": recovery,
            "expected_session_id": expected_session_id,
            "live_session_id": live_session,
            "session_check_skipped": self.skip_session_check,
            "pose": pose,
            "pose_age_s": pose_age,
            "map_age_s": map_age,
            "image_age_s": image_age,
            "scan_matched": scan_matched,
            "quality": quality,
            "mux_source": (None if mux_status is None else
                           mux_status.get("source")),
            "mux_age_s": mux_age,
        }

    def snapshot(self, expected_session_id: str) -> dict:
        now = time.monotonic()
        status = self.status(expected_session_id)
        with self._lock:
            occupancy = getattr(self, "_occupancy", None)
            map_seen_at = self._map_seen_at
            navigation_state = getattr(self, "_navigation_state", None)
            trajectory = list(getattr(self, "_trajectory", []))
            quality = getattr(self, "_quality", None)
        return {
            "status": status,
            "pose": status.get("pose"),
            "pose_age_s": status.get("pose_age_s"),
            "occupancy": occupancy,
            "map_age_s": (None if map_seen_at is None else
                          max(0.0, now - map_seen_at)),
            "quality": quality,
            "navigation_state": (None if navigation_state is None else
                                 dict(navigation_state)),
            "trajectory": trajectory,
            "mux_source": status.get("mux_source"),
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class ImageNavigationApp:
    def __init__(self, dataset_store: DatasetStore,
                 navigation: NavigationManager, live: LiveSlamMonitor,
                 gateway: KiwiAgentGateway | None = None):
        self.dataset_store = dataset_store
        self.navigation = navigation
        self.live = live
        self.gateway = gateway
        self.csrf_token = secrets.token_urlsafe(24)

    def state(self) -> dict:
        dataset = self.dataset_store.snapshot()
        return {
            "dataset": dataset.as_json(),
            "live": self.live.status(dataset.session_id),
            "navigation": self.navigation.snapshot(),
        }


def _page(csrf_token: str) -> bytes:
    token = json.dumps(csrf_token)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Kiwi image navigation</title>
  <style>
    :root {{ color-scheme: dark; --bg:#111714; --panel:#1a211d; --line:#344039;
      --text:#eef6f0; --muted:#9aaca1; --kiwi:#b7df49; --blue:#63b7ff;
      --danger:#ff716c; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; min-height:100vh; background:radial-gradient(circle at 20% 0,
      #233426 0, var(--bg) 42%); color:var(--text); font:15px/1.45 system-ui,sans-serif }}
    header {{ position:sticky; top:0; z-index:4; display:flex; gap:18px;
      align-items:center; justify-content:space-between; padding:14px 22px;
      background:#111714ee; border-bottom:1px solid var(--line); backdrop-filter:blur(12px) }}
    h1 {{ margin:0; font-size:20px; letter-spacing:.02em }}
    .brand {{ color:var(--kiwi) }} .status {{ color:var(--muted); text-align:right }}
    .status.ready {{ color:var(--kiwi) }} .status.bad {{ color:#ffc46b }}
    main {{ display:grid; grid-template-columns:minmax(340px, 1fr) minmax(360px, 1.25fr);
      gap:18px; padding:18px; max-width:1500px; margin:auto }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
      overflow:hidden; box-shadow:0 18px 50px #0004 }}
    .viewer {{ position:sticky; top:82px; align-self:start }}
    .hero {{ aspect-ratio:4/3; display:grid; place-items:center; background:#090c0a }}
    .hero img {{ width:100%; height:100%; object-fit:contain }}
    .empty {{ color:var(--muted) }}
    .details {{ padding:16px }} .pose {{ font:600 17px ui-monospace,monospace; margin:6px 0 14px }}
    .controls {{ display:flex; flex-wrap:wrap; gap:9px }}
    button,input {{ font:inherit }} button {{ border:1px solid #526059; border-radius:9px;
      padding:9px 13px; background:#28312c; color:var(--text); cursor:pointer }}
    button:hover:not(:disabled) {{ border-color:var(--kiwi) }} button:disabled {{ opacity:.4; cursor:not-allowed }}
    .drive {{ background:var(--kiwi); color:#172007; border-color:var(--kiwi); font-weight:750 }}
    .stop {{ color:#ffd4d1; border-color:#854944 }}
    .log {{ margin-top:14px; padding:11px; height:154px; overflow:auto; white-space:pre-wrap;
      background:#0c100e; border:1px solid #2b332f; border-radius:9px;
      color:#bed0c5; font:12px/1.45 ui-monospace,monospace }}
    .gallery-head {{ display:flex; align-items:center; gap:12px; padding:13px;
      border-bottom:1px solid var(--line) }}
    .gallery-head input {{ min-width:0; width:100%; padding:9px 11px; border-radius:9px;
      border:1px solid #455149; color:var(--text); background:#101512 }}
    .count {{ color:var(--muted); white-space:nowrap }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
      gap:10px; padding:12px; max-height:calc(100vh - 145px); overflow:auto }}
    .card {{ padding:0; overflow:hidden; text-align:left; background:#111613 }}
    .card.selected {{ outline:3px solid var(--kiwi); border-color:var(--kiwi) }}
    .card img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#080a09 }}
    .card span {{ display:block; padding:7px 9px; color:var(--muted); font:12px ui-monospace,monospace }}
    @media (max-width:850px) {{ main {{ grid-template-columns:1fr }} .viewer {{ position:static }}
      .grid {{ max-height:none }} }}
  </style>
</head>
<body>
  <header><h1><span class="brand">Kiwi</span> image navigation</h1>
    <div id="live" class="status">Connecting to SLAM…</div></header>
  <main>
    <section class="panel viewer">
      <div class="hero"><div id="empty" class="empty">Select an image</div><img id="hero" hidden alt="Selected capture"></div>
      <div class="details">
        <div id="selection">No destination selected</div><div id="pose" class="pose">—</div>
        <div class="controls">
          <button id="previous" aria-label="Previous image">← Previous</button>
          <button id="next" aria-label="Next image">Next →</button>
          <button id="drive" class="drive">Drive to this pose</button>
          <button id="stop" class="stop">Stop robot</button>
        </div>
        <pre id="log" class="log">Navigation is idle.</pre>
      </div>
    </section>
    <section class="panel">
      <div class="gallery-head"><input id="filter" placeholder="Filter by image ID or pose" aria-label="Filter images">
        <span id="count" class="count"></span></div>
      <div id="grid" class="grid"></div>
    </section>
  </main>
<script>
const csrf = {token};
let state = null, captures = [], selectedId = null, captureKey = "";
const el = id => document.getElementById(id);
function fmt(c) {{ return `x ${{c.pose.x.toFixed(2)}}  y ${{c.pose.y.toFixed(2)}}  yaw ${{c.yaw_deg.toFixed(1)}}°`; }}
function selected() {{ return captures.find(c => c.id === selectedId) || null; }}
function choose(id) {{
  selectedId = id; const c = selected();
  document.querySelectorAll('.card').forEach(x => x.classList.toggle('selected', Number(x.dataset.id) === id));
  if (!c) return;
  el('empty').hidden = true; el('hero').hidden = false; el('hero').src = c.image_url;
  el('selection').textContent = `Image ${{c.id}} · session ${{state.dataset.session_id}}`;
  el('pose').textContent = fmt(c); updateButtons();
}}
function filtered() {{
  const q = el('filter').value.trim().toLowerCase();
  return captures.filter(c => !q || String(c.id).includes(q) || fmt(c).toLowerCase().includes(q));
}}
function renderGallery() {{
  const shown = filtered(); el('count').textContent = `${{shown.length}} / ${{captures.length}}`;
  el('grid').replaceChildren(...shown.map(c => {{
    const b = document.createElement('button'); b.className = 'card'; b.dataset.id = c.id;
    if (c.id === selectedId) b.classList.add('selected');
    const img = document.createElement('img'); img.src = c.image_url; img.loading = 'lazy'; img.alt = `Capture ${{c.id}}`;
    const label = document.createElement('span'); label.textContent = `#${{c.id}} · ${{fmt(c)}}`;
    b.append(img,label); b.onclick = () => choose(c.id); return b;
  }}));
}}
function updateButtons() {{
  const nav = state?.navigation || {{phase:'idle'}};
  const active = nav.phase === 'running' || nav.phase === 'stopping';
  el('drive').disabled = !selected() || !state?.live.ready || active;
  el('stop').disabled = !active;
  el('previous').disabled = captures.length < 2; el('next').disabled = captures.length < 2;
}}
async function post(path, body={{}}) {{
  const response = await fetch(path, {{method:'POST', headers:{{'Content-Type':'application/json','X-Kiwi-Token':csrf}}, body:JSON.stringify(body)}});
  const value = await response.json(); if (!response.ok) throw new Error(value.error || response.statusText); return value;
}}
async function refresh() {{
  try {{
    const response = await fetch('/api/state', {{cache:'no-store'}}); state = await response.json();
    if (!response.ok) throw new Error(state.error || response.statusText);
    captures = state.dataset.captures;
    const key = JSON.stringify(captures.map(c => [c.id,c.pose.x,c.pose.y,c.pose.yaw]));
    if (key !== captureKey) {{ captureKey = key; if (selectedId === null && captures.length) selectedId = captures[0].id; renderGallery(); choose(selectedId); }}
    el('live').textContent = `${{state.live.reason}} · ${{state.dataset.session_id}}`;
    el('live').className = `status ${{state.live.ready ? 'ready' : 'bad'}}`;
    const nav = state.navigation; const heading = `Navigation: ${{nav.phase}}${{nav.capture_id === null ? '' : ` · image ${{nav.capture_id}}`}}`;
    el('log').textContent = [heading, ...nav.logs].join('\\n'); el('log').scrollTop = el('log').scrollHeight;
    updateButtons();
  }} catch (error) {{ el('live').textContent = error.message; el('live').className = 'status bad'; }}
}}
function step(delta) {{ if (!captures.length) return; const i = Math.max(0,captures.findIndex(c => c.id === selectedId)); choose(captures[(i+delta+captures.length)%captures.length].id); }}
el('previous').onclick = () => step(-1); el('next').onclick = () => step(1);
el('filter').oninput = renderGallery;
el('drive').onclick = async () => {{
  const c = selected(); if (!c || !confirm(`Drive Kiwi to image ${{c.id}}?\n${{fmt(c)}}`)) return;
  try {{ await post('/api/navigate', {{id:c.id}}); await refresh(); }} catch (error) {{ alert(error.message); }}
}};
el('stop').onclick = async () => {{ try {{ await post('/api/stop'); await refresh(); }} catch (error) {{ alert(error.message); }} }};
document.addEventListener('keydown', event => {{ if (event.target.tagName === 'INPUT') return; if (event.key === 'ArrowLeft') step(-1); if (event.key === 'ArrowRight') step(1); }});
refresh(); setInterval(refresh, 1000);
</script>
</body></html>""".encode("utf-8")


def make_handler(app: ImageNavigationApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "KiwiImageNavigation/1"

        def _headers(self, status: int, content_type: str,
                     length: int | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; frame-ancestors 'none'",
            )
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def _json(self, status: int, value: dict) -> None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": str(message)})

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    body = _page(app.csrf_token)
                    self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
                    self.wfile.write(body)
                    return
                if path == "/api/state":
                    self._json(HTTPStatus.OK, app.state())
                    return
                if path.startswith("/image/"):
                    capture_id = int(unquote(path.removeprefix("/image/")))
                    capture = app.dataset_store.snapshot().capture(capture_id)
                    body = capture.image_path.read_bytes()
                    self._headers(HTTPStatus.OK, "image/jpeg", len(body))
                    self.wfile.write(body)
                    return
            except (ImageMapError, OSError, ValueError) as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "image not found")
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid request length") from exc
            if length < 0 or length > 4096:
                raise ValueError("request body is too large")
            try:
                value = json.loads(self.rfile.read(length) or b"{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body must be JSON") from exc
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def do_POST(self):
            if not secrets.compare_digest(
                    self.headers.get("X-Kiwi-Token", ""), app.csrf_token):
                self._error(HTTPStatus.FORBIDDEN, "invalid control token")
                return
            path = urlsplit(self.path).path
            try:
                body = self._body()
                if path == "/api/navigate":
                    capture_id = body.get("id")
                    if isinstance(capture_id, bool) or not isinstance(capture_id, int):
                        raise ValueError("id must be an integer")
                    dataset = app.dataset_store.snapshot()
                    live = app.live.status(dataset.session_id)
                    if not live["ready"]:
                        self._error(HTTPStatus.CONFLICT, live["reason"])
                        return
                    app.navigation.start(dataset.capture(capture_id))
                    self._json(HTTPStatus.ACCEPTED, app.navigation.snapshot())
                    return
                if path == "/api/stop":
                    stopped = app.navigation.stop()
                    self._json(HTTPStatus.OK, {
                        "stopped": stopped,
                        "navigation": app.navigation.snapshot(),
                    })
                    return
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "image not found")
                return
            except NavigationBusy as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            except (ImageMapError, RuntimeError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, format, *args):
            print(f"web: {format % args}")

    return Handler


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def _positive_float(value: str) -> float:
    number = _finite_float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_float(value: str) -> float:
    number = _finite_float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", help="image-map manifest to browse")
    source.add_argument(
        "--image-root", default=DEFAULT_IMAGE_ROOT,
        help=("choose the newest */manifest.json under this directory "
              f"(default {DEFAULT_IMAGE_ROOT})"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the gallery in the default browser")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument(
        "--calibration", help="yaw/LiDAR calibration YAML/JSON file")
    parser.add_argument(
        "--command-topic", default="cmd_vel",
        help=("command topic passed to kiwi_navigation.py; launch.py uses "
              "cmd_vel/navigation behind the command mux"))
    parser.add_argument("--robot-yaw-deg", type=_finite_float,
                        default=DEFAULT_ROBOT_YAW_DEG)
    parser.add_argument("--inflation-radius", type=_nonnegative_float,
                        default=0.25)
    parser.add_argument(
        "--runtime-collision-radius", type=_nonnegative_float,
        default=DEFAULT_RUNTIME_COLLISION_RADIUS_M,
        help=("hard live-following collision radius; planning keeps the larger "
              "--inflation-radius buffer"),
    )
    parser.add_argument("--allow-unknown", action="store_true")
    parser.add_argument("--lookahead", type=_positive_float, default=0.30)
    parser.add_argument(
        "--goal-yaw-blend-distance", type=_positive_float, default=0.30,
        help="smooth final-heading blend distance (default 0.30 m)",
    )
    parser.add_argument("--kp-yaw", type=_nonnegative_float, default=2.5)
    parser.add_argument(
        "--max-linear-speed",
        type=_positive_float,
        default=DEFAULT_MAX_FOLLOWING_SPEED_MPS,
        help=("maximum trajectory-following speed "
              f"(default {DEFAULT_MAX_FOLLOWING_SPEED_MPS:g} m/s)"),
    )
    parser.add_argument("--max-angular-speed", type=_positive_float, default=1.0)
    parser.add_argument("--position-tolerance", type=_positive_float, default=0.04)
    parser.add_argument("--yaw-tolerance-deg", type=_positive_float, default=3.0)
    parser.add_argument("--replan-distance", type=_positive_float, default=0.35)
    parser.add_argument("--max-duration", type=_positive_float, default=120.0)
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=8766)
    parser.add_argument("--no-mcp", action="store_true",
                        help="run only the gallery, without the agent MCP server")
    parser.add_argument(
        "--mcp-token-env", default="KIWI_MCP_TOKEN",
        help=("optional environment variable containing a static MCP bearer "
              "token (default KIWI_MCP_TOKEN)"))
    parser.add_argument("--agent-max-travel-distance", type=_positive_float,
                        default=5.0)
    parser.add_argument("--preview-ttl", type=_positive_float, default=30.0)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument(
        "--skip-session-check", action="store_true",
        help=("allow driving without proving that the manifest belongs to the "
              "live SLAM session; unsafe unless map-frame alignment is known"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("--host must be loopback; robot controls are intentionally local")
    if not 1 <= args.mcp_port <= 65535:
        parser.error("--mcp-port must be in [1, 65535]")
    if args.mcp_host not in ("127.0.0.1", "localhost", "::1"):
        parser.error(
            "--mcp-host must be loopback; robot controls are intentionally local")
    if (not args.no_mcp and args.mcp_host == args.host and
            args.mcp_port == args.port):
        parser.error("--mcp-port and gallery --port must differ")
    if args.runtime_collision_radius > args.inflation_radius + 1e-12:
        parser.error(
            "--runtime-collision-radius must not exceed --inflation-radius")

    try:
        manifest = (Path(args.manifest).expanduser().resolve()
                    if args.manifest else discover_manifest(args.image_root))
        store = DatasetStore(manifest)
        dataset = store.snapshot()
    except ImageMapError as exc:
        parser.error(str(exc))

    settings = NavigationSettings(
        connect=args.connect,
        namespace=args.namespace,
        robot_yaw_deg=args.robot_yaw_deg,
        inflation_radius=args.inflation_radius,
        allow_unknown=args.allow_unknown,
        lookahead=args.lookahead,
        kp_yaw=args.kp_yaw,
        goal_yaw_blend_distance=args.goal_yaw_blend_distance,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        position_tolerance=args.position_tolerance,
        yaw_tolerance_deg=args.yaw_tolerance_deg,
        replan_distance=args.replan_distance,
        max_duration=args.max_duration,
        command_topic=args.command_topic,
        runtime_collision_radius=args.runtime_collision_radius,
        calibration=args.calibration,
    )
    navigation = NavigationManager(settings)
    live = LiveSlamMonitor(
        args.connect, args.namespace, args.robot_yaw_deg,
        skip_session_check=args.skip_session_check,
        require_mux=args.command_topic.strip("/") == "cmd_vel/navigation",
    )
    gateway = KiwiAgentGateway(
        store, navigation, live,
        clip_index=ClipPlaceIndex(OpenClipEncoder(
            model_name=args.clip_model, pretrained=args.clip_pretrained)),
        preview_ttl_s=args.preview_ttl,
        max_action_distance_m=args.agent_max_travel_distance,
        watchdog_interval_s=0.2,
    )
    app = ImageNavigationApp(store, navigation, live, gateway)
    mcp_handle = None
    if not args.no_mcp:
        try:
            from kiwi_agent_mcp import McpServerHandle
            token = os.environ.get(args.mcp_token_env) if args.mcp_token_env else None
            mcp_handle = McpServerHandle(
                gateway, args.mcp_host, args.mcp_port, token)
        except RuntimeError as exc:
            live.close()
            parser.error(str(exc))
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    except OSError as exc:
        live.close()
        parser.error(f"cannot listen on {args.host}:{args.port}: {exc}")

    url = f"http://{args.host}:{args.port}/"
    print(f"image map: {dataset.session_id} ({len(dataset.captures)} captures)")
    print(f"manifest:  {dataset.manifest_path}")
    print(f"gallery:   {url}")
    if mcp_handle is not None:
        try:
            mcp_handle.start()
        except RuntimeError as exc:
            mcp_handle.stop()
            server.server_close()
            gateway.close()
            live.close()
            parser.error(str(exc))
        auth = (f"Bearer token from {args.mcp_token_env}"
                if os.environ.get(args.mcp_token_env) else "loopback only")
        print(f"MCP:       http://{args.mcp_host}:{args.mcp_port}/mcp ({auth})")
    print("Keep the matching kiwi_slam.py process running; Ctrl-C stops navigation.")
    if not args.no_open:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nstopping image navigation")
    finally:
        navigation.stop()
        server.server_close()
        if mcp_handle is not None:
            mcp_handle.stop()
        gateway.close()
        live.close()


if __name__ == "__main__":
    main()
