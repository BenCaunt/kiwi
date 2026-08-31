#!/usr/bin/env python3
"""Pure-Python 2D world and sensor model for the Kiwi Zenoh simulator.

The simulator keeps its world pose in the camera/LiDAR-aligned frame while
commands, wheel telemetry, and odometry use the same raw drivetrain frame as
the firmware.  Keeping this module transport-free makes the geometry and wire
payload generators straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import random
import struct
from typing import Iterable


LD19_FRAME_LEN = 47
LD19_POINTS_PER_FRAME = 12
LD19_FRAMES_PER_REVOLUTION = 40
LD19_ROTATION_HZ = 10.0
CAMERA_HEADER_LEN = 32
WHEEL_ANGLES_RAD = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return c * x - s * y, s * x + c * y


def aligned_to_robot(
    vx: float, vy: float, robot_yaw_deg: float
) -> tuple[float, float]:
    return rotate(vx, vy, -math.radians(robot_yaw_deg))


def robot_to_aligned(
    vx: float, vy: float, robot_yaw_deg: float
) -> tuple[float, float]:
    return rotate(vx, vy, math.radians(robot_yaw_deg))


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    color: tuple[int, int, int] = (95, 155, 210)


@dataclass
class Environment:
    name: str
    walls: list[Segment]
    start: tuple[float, float, float]
    description: str = ""

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [coordinate for wall in self.walls for coordinate in (wall.x1, wall.x2)]
        ys = [coordinate for wall in self.walls for coordinate in (wall.y1, wall.y2)]
        if not xs:
            return -1.0, -1.0, 1.0, 1.0
        return min(xs), min(ys), max(xs), max(ys)

    def raycast(
        self, x: float, y: float, angle_rad: float, max_range_m: float = 12.0
    ) -> tuple[float, Segment | None]:
        """Return distance and hit wall for a ray originating at ``x, y``."""
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        closest = max_range_m
        closest_wall = None
        for wall in self.walls:
            sx, sy = wall.x2 - wall.x1, wall.y2 - wall.y1
            denominator = dx * sy - dy * sx
            if abs(denominator) < 1e-12:
                continue
            qx, qy = wall.x1 - x, wall.y1 - y
            distance = (qx * sy - qy * sx) / denominator
            along_wall = (qx * dy - qy * dx) / denominator
            if 0.0 <= along_wall <= 1.0 and 0.0 <= distance < closest:
                closest = distance
                closest_wall = wall
        return closest, closest_wall

    def collides(self, x: float, y: float, radius_m: float) -> bool:
        radius_sq = radius_m * radius_m
        for wall in self.walls:
            sx, sy = wall.x2 - wall.x1, wall.y2 - wall.y1
            length_sq = sx * sx + sy * sy
            if length_sq <= 1e-12:
                closest_x, closest_y = wall.x1, wall.y1
            else:
                projection = ((x - wall.x1) * sx + (y - wall.y1) * sy) / length_sq
                projection = min(max(projection, 0.0), 1.0)
                closest_x = wall.x1 + projection * sx
                closest_y = wall.y1 + projection * sy
            if (x - closest_x) ** 2 + (y - closest_y) ** 2 < radius_sq:
                return True
        return False


def rectangle_walls(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int] = (95, 155, 210),
) -> list[Segment]:
    return [
        Segment(x1, y1, x2, y1, color),
        Segment(x2, y1, x2, y2, color),
        Segment(x2, y2, x1, y2, color),
        Segment(x1, y2, x1, y1, color),
    ]


def builtin_environments() -> dict[str, Environment]:
    boundary = (80, 135, 185)
    obstacle = (205, 145, 70)

    room_walls = rectangle_walls(-3.0, -2.2, 3.0, 2.2, boundary)
    room_walls += rectangle_walls(0.7, -0.55, 1.4, 0.55, obstacle)
    room_walls += rectangle_walls(-1.0, 1.25, 0.4, 1.75, (115, 185, 105))
    room = Environment(
        "room",
        room_walls,
        (-1.8, 0.0, 0.0),
        "A furnished 6 m x 4.4 m room with good scan-matching geometry.",
    )

    warehouse_walls = rectangle_walls(-5.0, -3.5, 5.0, 3.5, boundary)
    for x1, x2 in ((-2.7, -1.8), (-0.45, 0.45), (1.8, 2.7)):
        warehouse_walls += rectangle_walls(x1, -2.45, x2, -0.55, obstacle)
        warehouse_walls += rectangle_walls(x1, 0.55, x2, 2.45, obstacle)
    warehouse = Environment(
        "warehouse",
        warehouse_walls,
        (-4.25, 0.0, 0.0),
        "Long aisles and shelving for odometry, deskew, and loop-closure tests.",
    )

    maze_walls = rectangle_walls(-4.0, -3.0, 4.0, 3.0, boundary)
    maze_walls += [
        Segment(-2.6, -3.0, -2.6, 1.8, obstacle),
        Segment(-2.6, 1.8, -0.8, 1.8, obstacle),
        Segment(-0.8, 1.8, -0.8, -1.8, obstacle),
        Segment(-0.8, -1.8, 1.0, -1.8, obstacle),
        Segment(1.0, -1.8, 1.0, 1.8, obstacle),
        Segment(1.0, 1.8, 2.6, 1.8, obstacle),
        Segment(2.6, 1.8, 2.6, -1.7, obstacle),
    ]
    maze = Environment(
        "maze",
        maze_walls,
        (-3.35, -2.35, 0.0),
        "A tight orthogonal course that exercises strafing and collision handling.",
    )

    return {environment.name: environment for environment in (room, warehouse, maze)}


def _color(value: object, default=(95, 155, 210)) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    return tuple(min(max(int(channel), 0), 255) for channel in value)


def load_environment(path: str | Path) -> Environment:
    """Load a JSON environment with ``walls`` and optional box ``obstacles``."""
    source = Path(path)
    data = json.loads(source.read_text())
    walls: list[Segment] = []
    for item in data.get("walls", []):
        if isinstance(item, dict):
            start, end = item["from"], item["to"]
            color = _color(item.get("color"))
        else:
            start, end = item[:2], item[2:4]
            color = _color(item[4] if len(item) > 4 else None)
        walls.append(Segment(float(start[0]), float(start[1]),
                             float(end[0]), float(end[1]), color))
    for item in data.get("obstacles", []):
        lower, upper = item["min"], item["max"]
        walls.extend(rectangle_walls(
            float(lower[0]), float(lower[1]),
            float(upper[0]), float(upper[1]),
            _color(item.get("color"), (205, 145, 70)),
        ))
    start = data.get("start", [0.0, 0.0, 0.0])
    if not walls:
        raise ValueError(f"{source} contains no walls or obstacles")
    return Environment(
        str(data.get("name", source.stem)),
        walls,
        (float(start[0]), float(start[1]), math.radians(float(start[2]))),
        str(data.get("description", f"Loaded from {source}")),
    )


@dataclass
class Twist:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


@dataclass
class SimulatorConfig:
    robot_yaw_deg: float = 60.0
    robot_radius_m: float = 0.13
    drive_base_radius_m: float = 0.09
    wheel_radius_m: float = 0.025
    encoder_counts_per_revolution: int = 4096
    max_wheel_speed_mps: float = 3.23
    max_omega_radps: float = 6.0
    response_time_s: float = 0.12
    command_timeout_s: float = 0.25


@dataclass(frozen=True)
class HardwareSensorProfile:
    """Observable sensor envelope retained by the physical-robot map files."""

    name: str = "retained-robot-maps-v1"
    camera_hz: float = 9.69
    lidar_max_range_m: float = 8.0
    lidar_range_noise_std_m: float = 0.003
    # Residual loss after the simulated home's own doorway/no-hit geometry.
    lidar_random_dropout_probability: float = 0.06
    lidar_dropout_variation_std: float = 0.02
    lidar_blind_sector_min_deg: float = 5.0
    lidar_blind_sector_max_deg: float = 12.0
    odometry_linear_scale: float = 1.04
    odometry_axis_skew_deg: float = 0.35
    odometry_angular_scale: float = 1.005
    odometry_velocity_noise_std_mps: float = 0.004
    imu_yaw_scale: float = 1.003
    imu_yaw_drift_deg_per_second: float = 0.03
    imu_yaw_random_walk_deg_per_sqrt_second: float = 0.015
    imu_yaw_noise_std_deg: float = 0.12


RETAINED_ROBOT_PROFILE = HardwareSensorProfile()
IDEAL_SENSOR_PROFILE = HardwareSensorProfile(
    name="ideal",
    camera_hz=10.0,
    lidar_max_range_m=12.0,
    lidar_range_noise_std_m=0.0,
    lidar_random_dropout_probability=0.0,
    lidar_dropout_variation_std=0.0,
    lidar_blind_sector_min_deg=0.0,
    lidar_blind_sector_max_deg=0.0,
    odometry_linear_scale=1.0,
    odometry_axis_skew_deg=0.0,
    odometry_angular_scale=1.0,
    odometry_velocity_noise_std_mps=0.0,
    imu_yaw_scale=1.0,
    imu_yaw_drift_deg_per_second=0.0,
    imu_yaw_random_walk_deg_per_sqrt_second=0.0,
    imu_yaw_noise_std_deg=0.0,
)


@dataclass
class SimState:
    x: float
    y: float
    yaw: float
    command_raw: Twist = field(default_factory=Twist)
    measured_raw: Twist = field(default_factory=Twist)
    wheel_speed_mps: list[float] = field(default_factory=lambda: [0.0] * 3)
    wheel_angle_rad: list[float] = field(default_factory=lambda: [0.0] * 3)
    encoder_count: list[int] = field(default_factory=lambda: [0] * 3)
    imu_accel_mps2: list[float] = field(default_factory=lambda: [0.0, 0.0, 9.81])


def wheel_speeds_from_twist(
    twist: Twist, drive_base_radius_m: float
) -> list[float]:
    return [
        -math.sin(theta) * twist.vx
        + math.cos(theta) * twist.vy
        + drive_base_radius_m * twist.omega
        for theta in WHEEL_ANGLES_RAD
    ]


def twist_from_wheel_speeds(
    wheels: Iterable[float], drive_base_radius_m: float
) -> Twist:
    wheels = list(wheels)
    vx = (2.0 / 3.0) * sum(
        -math.sin(theta) * speed
        for theta, speed in zip(WHEEL_ANGLES_RAD, wheels)
    )
    vy = (2.0 / 3.0) * sum(
        math.cos(theta) * speed
        for theta, speed in zip(WHEEL_ANGLES_RAD, wheels)
    )
    omega = sum(wheels) / (3.0 * drive_base_radius_m)
    return Twist(vx, vy, omega)


class KiwiRobotModel:
    """Kinematic kiwi-drive robot with command timeout and wall collisions."""

    def __init__(
        self,
        environment: Environment,
        config: SimulatorConfig | None = None,
        start: tuple[float, float, float] | None = None,
        sensor_profile: HardwareSensorProfile = RETAINED_ROBOT_PROFILE,
        seed: int = 1,
    ):
        self.environment = environment
        self.config = config or SimulatorConfig()
        start = start or environment.start
        self.state = SimState(*start)
        self.sensor_profile = sensor_profile
        self.random = random.Random(seed)
        self._drive_velocity_raw = Twist()
        self._last_reported_raw = Twist()
        self._last_command_at = -math.inf
        self._command_timeout_s = self.config.command_timeout_s
        self._imu_yaw = start[2]
        self.command_active = False
        self.report_seq = 0

    def set_command_raw(
        self,
        vx: float,
        vy: float,
        omega: float,
        now: float,
        timeout_s: float | None = None,
    ) -> None:
        values = (float(vx), float(vy), float(omega))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("velocity command values must be finite")
        self.state.command_raw = Twist(*values)
        self._last_command_at = now
        self._command_timeout_s = (
            self.config.command_timeout_s if timeout_s is None
            else max(float(timeout_s), 0.001)
        )
        self.command_active = True

    def set_command_aligned(
        self, vx: float, vy: float, omega: float, now: float
    ) -> None:
        raw_vx, raw_vy = aligned_to_robot(vx, vy, self.config.robot_yaw_deg)
        self.set_command_raw(raw_vx, raw_vy, omega, now)

    def _limited_target(self) -> Twist:
        target = self.state.command_raw if self.command_active else Twist()
        omega = min(max(target.omega, -self.config.max_omega_radps),
                    self.config.max_omega_radps)
        wheels = wheel_speeds_from_twist(
            Twist(target.vx, target.vy, omega),
            self.config.drive_base_radius_m,
        )
        maximum = max((abs(speed) for speed in wheels), default=0.0)
        if maximum > self.config.max_wheel_speed_mps:
            scale = self.config.max_wheel_speed_mps / maximum
            wheels = [speed * scale for speed in wheels]
        return twist_from_wheel_speeds(wheels, self.config.drive_base_radius_m)

    def _move_with_collisions(self, dx: float, dy: float) -> tuple[float, float]:
        distance = math.hypot(dx, dy)
        step_limit = self.config.robot_radius_m * 0.35
        steps = max(1, math.ceil(distance / step_limit))
        step_x, step_y = dx / steps, dy / steps
        moved_x = moved_y = 0.0
        for _ in range(steps):
            x, y = self.state.x, self.state.y
            radius = self.config.robot_radius_m
            if not self.environment.collides(x + step_x, y + step_y, radius):
                self.state.x += step_x
                self.state.y += step_y
                moved_x += step_x
                moved_y += step_y
                continue
            if not self.environment.collides(x + step_x, y, radius):
                self.state.x += step_x
                moved_x += step_x
            if not self.environment.collides(self.state.x, y + step_y, radius):
                self.state.y = y + step_y
                moved_y += step_y
        return moved_x, moved_y

    def step(self, dt: float, now: float) -> None:
        dt = min(max(float(dt), 0.0), 0.1)
        if dt <= 0.0:
            return
        if now - self._last_command_at > self._command_timeout_s:
            self.command_active = False

        target = self._limited_target()
        tau = max(self.config.response_time_s, 1e-6)
        alpha = 1.0 - math.exp(-dt / tau)
        for field_name in ("vx", "vy", "omega"):
            current = getattr(self._drive_velocity_raw, field_name)
            desired = getattr(target, field_name)
            setattr(self._drive_velocity_raw, field_name,
                    current + alpha * (desired - current))

        aligned_vx, aligned_vy = robot_to_aligned(
            self._drive_velocity_raw.vx,
            self._drive_velocity_raw.vy,
            self.config.robot_yaw_deg,
        )
        world_vx, world_vy = rotate(aligned_vx, aligned_vy, self.state.yaw)
        moved_x, moved_y = self._move_with_collisions(world_vx * dt, world_vy * dt)
        old_yaw = self.state.yaw
        self.state.yaw = wrap_angle(
            self.state.yaw + self._drive_velocity_raw.omega * dt
        )

        actual_world_vx, actual_world_vy = moved_x / dt, moved_y / dt
        actual_aligned_vx, actual_aligned_vy = rotate(
            actual_world_vx, actual_world_vy, -old_yaw
        )
        true_raw_vx, true_raw_vy = aligned_to_robot(
            actual_aligned_vx, actual_aligned_vy, self.config.robot_yaw_deg
        )
        profile = self.sensor_profile
        skew = math.radians(profile.odometry_axis_skew_deg)
        skewed_vx, skewed_vy = rotate(true_raw_vx, true_raw_vy, skew)
        moving = math.hypot(true_raw_vx, true_raw_vy) > 0.002
        noise_std = (
            profile.odometry_velocity_noise_std_mps if moving else 0.0
        )
        self.state.measured_raw = Twist(
            profile.odometry_linear_scale * skewed_vx
            + self.random.gauss(0.0, noise_std),
            profile.odometry_linear_scale * skewed_vy
            + self.random.gauss(0.0, noise_std),
            profile.odometry_angular_scale * self._drive_velocity_raw.omega,
        )
        true_yaw_delta = wrap_angle(self.state.yaw - old_yaw)
        self._imu_yaw = wrap_angle(
            self._imu_yaw
            + profile.imu_yaw_scale * true_yaw_delta
            + math.radians(profile.imu_yaw_drift_deg_per_second) * dt
            + math.radians(profile.imu_yaw_random_walk_deg_per_sqrt_second)
            * math.sqrt(dt)
            * self.random.gauss(0.0, 1.0)
        )
        self.state.wheel_speed_mps = wheel_speeds_from_twist(
            self.state.measured_raw, self.config.drive_base_radius_m
        )
        for index, speed in enumerate(self.state.wheel_speed_mps):
            delta_angle = speed / self.config.wheel_radius_m * dt
            self.state.wheel_angle_rad[index] = (
                self.state.wheel_angle_rad[index] + delta_angle
            ) % (2.0 * math.pi)
            revolutions = delta_angle / (2.0 * math.pi)
            self.state.encoder_count[index] += round(
                revolutions * self.config.encoder_counts_per_revolution
            )

        self.state.imu_accel_mps2 = [
            (self.state.measured_raw.vx - self._last_reported_raw.vx) / dt,
            (self.state.measured_raw.vy - self._last_reported_raw.vy) / dt,
            9.81,
        ]
        self._last_reported_raw = Twist(
            self.state.measured_raw.vx,
            self.state.measured_raw.vy,
            self.state.measured_raw.omega,
        )

    def odometry_report(self, follower_time_us: int) -> dict:
        state = self.state
        imu_yaw = wrap_angle(
            self._imu_yaw
            + math.radians(self.sensor_profile.imu_yaw_noise_std_deg)
            * self.random.gauss(0.0, 1.0)
        )
        command = state.command_raw if self.command_active else Twist()
        report = {
            "follower_time_us": int(follower_time_us),
            "seq": self.report_seq,
            "measured": {
                "vx": state.measured_raw.vx,
                "vy": state.measured_raw.vy,
                "omega": state.measured_raw.omega,
            },
            "command": {
                "vx": command.vx,
                "vy": command.vy,
                "omega": command.omega,
            },
            "wheel_speed_mps": list(state.wheel_speed_mps),
            "wheel_angle_rad": list(state.wheel_angle_rad),
            "encoder_count": list(state.encoder_count),
            "imu_ready": True,
            "encoder_ready_mask": 7,
            "status_flags": 0 if self.command_active else 1,
            "imu_quat_ijkr": [
                0.0,
                0.0,
                math.sin(imu_yaw / 2.0),
                math.cos(imu_yaw / 2.0),
            ],
            "imu_accel_mps2": list(state.imu_accel_mps2),
        }
        self.report_seq += 1
        return report


def parse_velocity_payload(payload: bytes) -> tuple[Twist, float | None]:
    """Parse the three command forms accepted by the master firmware."""
    if len(payload) == 24:
        _time_us, vx, vy, omega, timeout_ms, mode, _reserved = struct.unpack(
            "<QfffHBB", payload
        )
        twist = Twist() if mode == 1 else Twist(vx, vy, omega)
        if not all(math.isfinite(value) for value in
                   (twist.vx, twist.vy, twist.omega)):
            raise ValueError("binary velocity command contains non-finite values")
        return twist, max(timeout_ms, 1) / 1000.0

    text = payload.decode("utf-8").strip()
    if text.startswith("{"):
        data = json.loads(text)
        twist = Twist(float(data["vx"]), float(data["vy"]), float(data["omega"]))
    else:
        values = text.replace(",", " ").split()
        if len(values) < 3:
            raise ValueError("text velocity command requires vx vy omega")
        twist = Twist(*(float(value) for value in values[:3]))
    if not all(math.isfinite(value) for value in
               (twist.vx, twist.vy, twist.omega)):
        raise ValueError("velocity command values must be finite")
    return twist, None


_CRC_TABLE = []
for _table_index in range(256):
    _crc = _table_index
    for _ in range(8):
        _crc = ((_crc << 1) ^ 0x4D if _crc & 0x80 else _crc << 1) & 0xFF
    _CRC_TABLE.append(_crc)


def crc8(data: bytes | bytearray) -> int:
    crc = 0
    for byte in data:
        crc = _CRC_TABLE[crc ^ byte]
    return crc


class LD19Simulator:
    """Produce CRC-valid raw LD19 frames at the real robot's framing rate."""

    def __init__(
        self,
        environment: Environment,
        max_range_m: float | None = None,
        range_noise_std_m: float | None = None,
        sensor_profile: HardwareSensorProfile = RETAINED_ROBOT_PROFILE,
        seed: int = 1,
    ):
        self.environment = environment
        self.sensor_profile = sensor_profile
        self.max_range_m = (
            sensor_profile.lidar_max_range_m
            if max_range_m is None else float(max_range_m)
        )
        self.range_noise_std_m = max(
            sensor_profile.lidar_range_noise_std_m
            if range_noise_std_m is None else float(range_noise_std_m),
            0.0,
        )
        self.frame_index = 0
        self.timestamp_ms = 0.0
        self.random = random.Random(seed)
        self.blind_sector_center_rad = 0.0
        self.blind_sector_width_rad = 0.0
        self.dropout_probability = sensor_profile.lidar_random_dropout_probability

    def _frame(self, x: float, y: float, yaw: float) -> bytes:
        frame_span_deg = 360.0 / LD19_FRAMES_PER_REVOLUTION
        start_deg = self.frame_index * frame_span_deg
        point_step_deg = frame_span_deg / LD19_POINTS_PER_FRAME
        end_deg = (
            start_deg + point_step_deg * (LD19_POINTS_PER_FRAME - 1)
        ) % 360.0
        if self.frame_index == 0:
            profile = self.sensor_profile
            self.blind_sector_center_rad = self.random.random() * 2.0 * math.pi
            self.blind_sector_width_rad = math.radians(self.random.uniform(
                profile.lidar_blind_sector_min_deg,
                profile.lidar_blind_sector_max_deg,
            ))
            self.dropout_probability = min(max(
                profile.lidar_random_dropout_probability
                + profile.lidar_dropout_variation_std
                * self.random.gauss(0.0, 1.0),
                0.0,
            ), 0.8)
        raw = bytearray(LD19_FRAME_LEN)
        raw[0] = 0x54
        raw[1] = 0x2C
        struct.pack_into("<HH", raw, 2,
                         round(360.0 * LD19_ROTATION_HZ),
                         round(start_deg * 100.0) % 36000)
        for point_index in range(LD19_POINTS_PER_FRAME):
            lidar_angle_deg = start_deg + point_step_deg * point_index
            world_angle = yaw - math.radians(lidar_angle_deg)
            distance, wall = self.environment.raycast(
                x, y, world_angle, self.max_range_m
            )
            local_angle = -math.radians(lidar_angle_deg)
            in_blind_sector = abs(wrap_angle(
                local_angle - self.blind_sector_center_rad
            )) <= self.blind_sector_width_rad / 2.0
            dropped = (
                in_blind_sector
                or self.random.random()
                < self.dropout_probability
            )
            if wall is None or dropped:
                distance_mm, intensity = 0, 0
            else:
                noisy = distance + self.random.gauss(0.0, self.range_noise_std_m)
                distance_mm = min(
                    max(round(noisy * 1000.0), 1),
                    round(self.max_range_m * 1000.0),
                )
                intensity = min(max(round(230.0 / (1.0 + 0.12 * distance)), 20), 255)
            struct.pack_into("<HB", raw, 6 + 3 * point_index,
                             distance_mm, intensity)
        struct.pack_into("<HH", raw, 42,
                         round(end_deg * 100.0) % 36000,
                         round(self.timestamp_ms) % 30000)
        raw[46] = crc8(raw[:46])
        self.frame_index = (self.frame_index + 1) % LD19_FRAMES_PER_REVOLUTION
        frame_period_ms = (
            1000.0 / (LD19_ROTATION_HZ * LD19_FRAMES_PER_REVOLUTION)
        )
        self.timestamp_ms = (self.timestamp_ms + frame_period_ms) % 30000.0
        return bytes(raw)

    def batch(
        self,
        state: SimState,
        frame_count: int = 20,
        start_pose: tuple[float, float, float] | None = None,
    ) -> bytes:
        """Render a batch, interpolating rolling-scan motion when available."""
        end_pose = (state.x, state.y, state.yaw)
        start_pose = end_pose if start_pose is None else start_pose
        yaw_delta = wrap_angle(end_pose[2] - start_pose[2])
        frames = []
        for frame_number in range(frame_count):
            fraction = (frame_number + 0.5) / frame_count
            x = start_pose[0] + fraction * (end_pose[0] - start_pose[0])
            y = start_pose[1] + fraction * (end_pose[1] - start_pose[1])
            yaw = wrap_angle(start_pose[2] + fraction * yaw_delta)
            frames.append(self._frame(x, y, yaw))
        return b"".join(frames)


def render_camera_jpeg(
    environment: Environment,
    state: SimState,
    width: int = 320,
    height: int = 240,
    fov_deg: float = 72.0,
) -> bytes:
    """Render a compact ray-cast camera view and return an upside-down JPEG.

    The physical camera is mounted inverted and the existing dashboard rotates
    it 180 degrees, so the simulator emits the same raw orientation.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (90, 150, 205))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height // 2, width, height), fill=(67, 72, 74))
    draw.line((0, height // 2, width, height // 2), fill=(210, 220, 225), width=1)
    fov_rad = math.radians(fov_deg)
    for column in range(width):
        fraction = (column + 0.5) / width - 0.5
        camera_angle = fraction * fov_rad
        distance, wall = environment.raycast(
            state.x, state.y, state.yaw - camera_angle, 12.0
        )
        if wall is None:
            continue
        corrected = max(distance * math.cos(camera_angle), 0.04)
        half_height = min(int(height * 0.46 / corrected), height // 2)
        shade = min(max(1.15 - corrected / 10.0, 0.25), 1.0)
        color = tuple(round(channel * shade) for channel in wall.color)
        draw.line(
            (column, height // 2 - half_height,
             column, height // 2 + half_height),
            fill=color,
        )
    draw.line((width // 2 - 8, height // 2, width // 2 + 8, height // 2),
              fill=(245, 245, 245))
    draw.line((width // 2, height // 2 - 8, width // 2, height // 2 + 8),
              fill=(245, 245, 245))
    image = image.transpose(Image.Transpose.ROTATE_180)
    output = io.BytesIO()
    image.save(output, "JPEG", quality=72, optimize=False)
    return output.getvalue()


def camera_payload(
    environment: Environment,
    state: SimState,
    sequence: int,
    timestamp_us: int,
    width: int = 320,
    height: int = 240,
) -> bytes:
    jpeg = render_camera_jpeg(environment, state, width, height)
    header = bytearray(CAMERA_HEADER_LEN)
    header[:4] = b"KVC1"
    header[4] = 1
    header[5] = 4  # esp_camera PIXFORMAT_JPEG
    struct.pack_into("<HHH", header, 6, width, height, CAMERA_HEADER_LEN)
    struct.pack_into("<I", header, 12, int(sequence))
    struct.pack_into("<Q", header, 16, int(timestamp_us))
    struct.pack_into("<I", header, 24, len(jpeg))
    return bytes(header) + jpeg
