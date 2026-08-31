"""Kiwi SLAM calibration files, robust rotation fitting, and raw log I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct

import numpy as np
from scipy import optimize

from kiwi_yaw_estimator import YawEstimatorConfig


CALIBRATION_FORMAT = "kiwi-slam-calibration-v1"
LOG_FORMAT = "kiwi-calibration-log-v1"
LIDAR_RECORD_HEADER = struct.Struct("<QI")


def _parse_scalar(value):
    value = value.strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered in ("null", "~"):
            return None
        if lowered in ("true", "false"):
            return lowered == "true"
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value.strip("'\"")


def _load_simple_yaml(text):
    """Parse the mapping/scalar subset used by Kiwi calibration files."""
    root = {}
    stack = [(-1, root)]
    for line_number, source in enumerate(text.splitlines(), start=1):
        content = source.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"invalid calibration YAML line {line_number}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty calibration key on line {line_number}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        decoded = _parse_scalar(value)
        parent[key] = decoded
        if decoded == {} and not value.strip():
            stack.append((indent, decoded))
    return root


@dataclass
class LidarCalibration:
    time_offset_ms: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0

    @classmethod
    def from_mapping(cls, values):
        if values is None:
            return cls()
        if not isinstance(values, dict):
            raise ValueError("lidar calibration must be a mapping")
        return cls(**{key: values[key] for key in (
            "time_offset_ms", "x_m", "y_m", "yaw_deg") if key in values})

    def validate(self):
        values = asdict(self)
        try:
            values = {key: float(value) for key, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("LiDAR calibration values must be numeric") from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("LiDAR calibration values must be finite")
        if abs(values["time_offset_ms"]) > 250.0:
            raise ValueError("LiDAR time offset exceeds the 250 ms safety bound")
        if math.hypot(values["x_m"], values["y_m"]) > 0.5:
            raise ValueError("LiDAR translation exceeds the 0.5 m safety bound")
        if abs(values["yaw_deg"]) > 180.0:
            raise ValueError("LiDAR yaw must be between -180 and 180 degrees")
        for key, value in values.items():
            setattr(self, key, value)
        return self


@dataclass
class Calibration:
    format: str = CALIBRATION_FORMAT
    created_at: str = field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat())
    source_run: str | None = None
    yaw_estimator: YawEstimatorConfig = field(
        default_factory=YawEstimatorConfig)
    lidar: LidarCalibration = field(default_factory=LidarCalibration)
    validation: dict = field(default_factory=dict)
    source_path: str | None = field(default=None, repr=False)

    @classmethod
    def from_mapping(cls, values, source_path=None):
        if not isinstance(values, dict):
            raise ValueError("calibration document must be a mapping")
        document_format = values.get("format")
        if document_format != CALIBRATION_FORMAT:
            raise ValueError(
                f"unsupported calibration format: {document_format!r}")
        calibration = cls(
            format=document_format,
            created_at=str(values.get("created_at", "")),
            source_run=values.get("source_run"),
            yaw_estimator=YawEstimatorConfig.from_mapping(
                values.get("yaw_estimator")),
            lidar=LidarCalibration.from_mapping(values.get("lidar")),
            validation=(dict(values.get("validation", {}))
                        if isinstance(values.get("validation", {}), dict)
                        else {}),
            source_path=None if source_path is None else str(source_path),
        )
        calibration.yaw_estimator.validate()
        calibration.lidar.validate()
        return calibration

    def to_mapping(self):
        return {
            "format": self.format,
            "created_at": self.created_at,
            "source_run": self.source_run,
            "yaw_estimator": asdict(self.yaw_estimator),
            "lidar": asdict(self.lidar),
            "validation": self.validation,
        }


def load_calibration(path):
    path = Path(path).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read calibration {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = _load_simple_yaml(text)
    return Calibration.from_mapping(document, source_path=path.resolve())


def save_calibration(calibration, path):
    """Save JSON syntax, which is also a strict YAML 1.2 document."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration.to_mapping(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


class CalibrationLogWriter:
    """Append exact odometry JSON and LD19 payloads without Rerun."""

    def __init__(self, directory, metadata=None):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=False)
        run = {
            "format": LOG_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        (self.directory / "run.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8")
        self._odom = (self.directory / "odom.jsonl").open(
            "a", encoding="utf-8")
        self._lidar = (self.directory / "lidar.bin").open("ab")
        self._events = (self.directory / "events.jsonl").open(
            "a", encoding="utf-8")

    def write_odometry(self, payload, arrival_ns):
        if isinstance(payload, (bytes, bytearray, memoryview)):
            report = json.loads(bytes(payload).decode("utf-8"))
        elif isinstance(payload, dict):
            report = payload
        else:
            raise TypeError("odometry payload must be bytes or a mapping")
        self._odom.write(json.dumps({
            "arrival_ns": int(arrival_ns),
            "report": report,
        }, separators=(",", ":")) + "\n")
        self._odom.flush()

    def write_lidar(self, payload, arrival_ns):
        payload = bytes(payload)
        self._lidar.write(LIDAR_RECORD_HEADER.pack(
            int(arrival_ns), len(payload)))
        self._lidar.write(payload)
        self._lidar.flush()

    def write_event(self, label, arrival_ns, **details):
        self._events.write(json.dumps({
            "arrival_ns": int(arrival_ns),
            "label": str(label),
            **details,
        }, separators=(",", ":")) + "\n")
        self._events.flush()

    def close(self):
        self._odom.close()
        self._lidar.close()
        self._events.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def read_lidar_records(path):
    with Path(path).open("rb") as stream:
        while True:
            header = stream.read(LIDAR_RECORD_HEADER.size)
            if not header:
                return
            if len(header) != LIDAR_RECORD_HEADER.size:
                raise ValueError("truncated LiDAR calibration record header")
            arrival_ns, length = LIDAR_RECORD_HEADER.unpack(header)
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError("truncated LiDAR calibration payload")
            yield arrival_ns, payload


@dataclass(frozen=True)
class RotationObservation:
    dt_s: float
    wheel_delta_rad: float
    imu_delta_rad: float
    scan_delta_rad: float
    weight: float = 1.0


@dataclass(frozen=True)
class RotationFit:
    wheel_yaw_scale: float
    imu_yaw_scale: float
    imu_rate_bias_rad_s: float
    wheel_rmse_rad: float
    imu_rmse_rad: float
    imu_weight: float
    observations: int


def aggregate_rotation_segments(observations, min_rotation_rad=None):
    """Combine consecutive same-direction scan increments into robust turns."""
    minimum = (math.radians(20.0) if min_rotation_rad is None
               else float(min_rotation_rad))
    segments = []
    current = []
    current_sign = 0

    def flush():
        nonlocal current
        if not current:
            return
        wheel = sum(item.wheel_delta_rad for item in current)
        imu = sum(item.imu_delta_rad for item in current)
        scan = sum(item.scan_delta_rad for item in current)
        if max(abs(wheel), abs(imu), abs(scan)) >= minimum:
            segments.append(RotationObservation(
                dt_s=sum(item.dt_s for item in current),
                wheel_delta_rad=wheel,
                imu_delta_rad=imu,
                scan_delta_rad=scan,
                weight=sum(item.weight for item in current) / len(current),
            ))
        current = []

    for item in observations:
        direction_value = (item.wheel_delta_rad
                           if abs(item.wheel_delta_rad) > math.radians(0.05)
                           else item.imu_delta_rad)
        sign = 1 if direction_value > 0.0 else -1
        if current and sign != current_sign:
            flush()
        current_sign = sign
        current.append(item)
    flush()
    return segments


def fit_rotation_calibration(observations):
    """Robustly fit wheel scale and IMU scale/rate bias to LiDAR yaw."""
    observations = list(observations)
    if len(observations) < 3:
        raise ValueError("at least three rotation observations are required")
    values = np.asarray([(
        item.dt_s, item.wheel_delta_rad, item.imu_delta_rad,
        item.scan_delta_rad, item.weight,
    ) for item in observations], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0.0) or \
            np.any(values[:, 4] <= 0.0):
        raise ValueError("rotation observations must be finite with positive weights")
    dt, wheel, imu, scan, weights = values.T
    roots = np.sqrt(weights)

    wheel_fit = optimize.least_squares(
        lambda x: roots * (x[0] * wheel - scan),
        np.array((1.0,)), bounds=(0.5, 1.5), loss="huber",
        f_scale=math.radians(0.25),
    )
    imu_fit = optimize.least_squares(
        lambda x: roots * (x[0] * imu - x[1] * dt - scan),
        np.array((1.0, 0.0)),
        bounds=(np.array((0.5, math.radians(-1.0))),
                np.array((1.5, math.radians(1.0)))),
        loss="huber", f_scale=math.radians(0.25),
    )
    wheel_residual = wheel_fit.x[0] * wheel - scan
    imu_residual = imu_fit.x[0] * imu - imu_fit.x[1] * dt - scan
    wheel_rmse = float(np.sqrt(np.mean(np.square(wheel_residual))))
    imu_rmse = float(np.sqrt(np.mean(np.square(imu_residual))))
    # The runtime estimator expresses fusion as
    #   fused = wheel + imu_weight * (imu - wheel).
    # For two independent measurements, inverse-variance weighting gives the
    # IMU coefficient below.  The old fixed 0.85 default could make the noisier
    # sensor dominate even after the calibration run had measured otherwise.
    # Keep a small contribution from either sensor so one transient fault does
    # not become the entire heading estimate before the runtime innovation gate
    # can react.
    variance_floor = math.radians(0.25) ** 2
    wheel_variance = max(wheel_rmse ** 2, variance_floor)
    imu_variance = max(imu_rmse ** 2, variance_floor)
    imu_weight = min(
        0.95, max(0.05, wheel_variance / (wheel_variance + imu_variance)))
    return RotationFit(
        wheel_yaw_scale=float(wheel_fit.x[0]),
        imu_yaw_scale=float(imu_fit.x[0]),
        imu_rate_bias_rad_s=float(imu_fit.x[1]),
        wheel_rmse_rad=wheel_rmse,
        imu_rmse_rad=imu_rmse,
        imu_weight=imu_weight,
        observations=len(observations),
    )


@dataclass(frozen=True)
class PlanarCalibrationFit:
    time_offset_s: float
    x_m: float
    y_m: float
    objective: float


def fit_planar_calibration(objective, *, time_bound_s=0.05,
                           time_step_s=0.001, translation_bound_m=0.15):
    """Bounded time grid followed by robust planar translation refinement."""
    offsets = np.arange(-time_bound_s, time_bound_s + 0.5 * time_step_s,
                        time_step_s)
    scores = np.asarray([objective(float(offset), 0.0, 0.0)
                         for offset in offsets], dtype=np.float64)
    if not np.any(np.isfinite(scores)):
        raise ValueError("planar calibration objective has no finite solution")
    best_time = float(offsets[int(np.nanargmin(scores))])
    translation = optimize.minimize(
        lambda values: objective(best_time, values[0], values[1]),
        np.zeros(2), method="Powell",
        bounds=((-translation_bound_m, translation_bound_m),) * 2,
        options={"maxiter": 24, "xtol": 1.0e-4, "ftol": 1.0e-5},
    )
    joint_bounds = (
        (best_time - 2.0 * time_step_s,
         best_time + 2.0 * time_step_s),
        (-translation_bound_m, translation_bound_m),
        (-translation_bound_m, translation_bound_m),
    )
    result = optimize.minimize(
        lambda values: objective(*values),
        np.array((best_time, translation.x[0], translation.x[1])),
        method="Powell", bounds=joint_bounds,
        options={"maxiter": 24, "xtol": 1.0e-5, "ftol": 1.0e-6},
    )
    return PlanarCalibrationFit(
        time_offset_s=float(result.x[0]),
        x_m=float(result.x[1]),
        y_m=float(result.x[2]),
        objective=float(result.fun),
    )
