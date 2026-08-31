"""Laptop-side time alignment and motion compensation for Kiwi LiDAR scans."""

from collections import deque
from dataclasses import dataclass
import bisect
import math


def unwrap_angle(angle, reference):
    """Return ``angle`` shifted by full turns to be nearest ``reference``."""
    return reference + math.atan2(math.sin(angle - reference),
                                  math.cos(angle - reference))


class SensorClock:
    """Map a sensor clock to the laptop clock using minimum observed latency."""

    def __init__(self, wrap_seconds=None, offset_window=200):
        self.wrap_seconds = wrap_seconds
        self._last_raw = None
        self._epoch = 0.0
        self._offsets = deque(maxlen=offset_window)

    def unwrap(self, raw_seconds):
        raw_seconds = float(raw_seconds)
        if self._last_raw is not None and self.wrap_seconds is not None:
            half_wrap = self.wrap_seconds / 2.0
            if raw_seconds < self._last_raw - half_wrap:
                self._epoch += self.wrap_seconds
        self._last_raw = raw_seconds
        return self._epoch + raw_seconds

    def observe(self, sensor_seconds, laptop_seconds):
        self._offsets.append(float(laptop_seconds) - float(sensor_seconds))

    @property
    def ready(self):
        return bool(self._offsets)

    @property
    def offset(self):
        if not self._offsets:
            raise RuntimeError("sensor clock has no laptop-time observation")
        # Transport delay is non-negative but variable. The lowest value in a
        # rolling window is the best available estimate of the clock offset.
        return min(self._offsets)

    def to_laptop(self, sensor_seconds):
        return float(sensor_seconds) + self.offset

    def from_laptop(self, laptop_seconds):
        return float(laptop_seconds) - self.offset


@dataclass(frozen=True)
class Pose:
    time_s: float
    x: float
    y: float
    yaw: float


class PoseHistory:
    """Timestamped planar poses with wrap-safe yaw interpolation."""

    def __init__(self, maxlen=500):
        self._poses = deque(maxlen=maxlen)
        self._snapshot_cache = None

    def __len__(self):
        return len(self._poses)

    @property
    def first_time(self):
        return self._poses[0].time_s if self._poses else None

    @property
    def last_time(self):
        return self._poses[-1].time_s if self._poses else None

    def append(self, time_s, x, y, yaw):
        time_s = float(time_s)
        yaw = float(yaw)
        if self._poses:
            if time_s <= self._poses[-1].time_s:
                return
            yaw = unwrap_angle(yaw, self._poses[-1].yaw)
        self._poses.append(Pose(time_s, float(x), float(y), yaw))
        self._snapshot_cache = None

    def interpolate(self, time_s):
        if not self._poses or time_s < self._poses[0].time_s \
                or time_s > self._poses[-1].time_s:
            return None
        if self._snapshot_cache is None:
            poses = tuple(self._poses)
            self._snapshot_cache = (
                poses, tuple(pose.time_s for pose in poses))
        poses, times = self._snapshot_cache
        right = bisect.bisect_left(times, time_s)
        if right == 0:
            return poses[0]
        if right == len(poses):
            return poses[-1]
        if poses[right].time_s == time_s:
            return poses[right]
        left_pose, right_pose = poses[right - 1], poses[right]
        fraction = ((time_s - left_pose.time_s) /
                    (right_pose.time_s - left_pose.time_s))
        return Pose(
            float(time_s),
            left_pose.x + fraction * (right_pose.x - left_pose.x),
            left_pose.y + fraction * (right_pose.y - left_pose.y),
            left_pose.yaw + fraction * (right_pose.yaw - left_pose.yaw),
        )


@dataclass(frozen=True)
class TimedFrame:
    frame: object
    time_s: float


class TimedScanAssembler:
    """Group timestamped LD19 frames into complete revolutions."""

    def __init__(self):
        self.frames = []
        self.last_angle = None

    def add(self, frame, time_s):
        done = None
        if (self.last_angle is not None and
                frame.start_angle_deg < self.last_angle - 180.0):
            done = self.frames
            self.frames = []
        self.last_angle = frame.start_angle_deg
        self.frames.append(TimedFrame(frame, float(time_s)))
        return done


@dataclass(frozen=True)
class DeskewedScan:
    """Motion-compensated points in the final LiDAR frame of a revolution."""

    time_s: float
    pose: Pose
    points: list  # [(x, y), ...] in the reference robot frame


@dataclass(frozen=True)
class LidarExtrinsics:
    """Planar body-from-LiDAR transform used by live and replay deskew."""

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    def transform_point(self, x, y):
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (
            self.x_m + c * x - s * y,
            self.y_m + s * x + c * y,
        )


def _frame_points(timed_frame):
    """Yield (time, x, y) points using the LD19 packet timestamp."""
    frame = timed_frame.frame
    for angle_deg, distance_m, _intensity in frame.points:
        if not 0.02 < distance_m < 12.0:
            continue
        angle = -math.radians(angle_deg)  # LD19 angles increase clockwise.
        yield (timed_frame.time_s,
               distance_m * math.cos(angle),
               distance_m * math.sin(angle))


def deskew_scan(timed_frames, poses, lidar_clock, pose_clock,
                lidar_time_offset_s=0.0, lidar_extrinsics=None):
    """Transform a rolling scan into world coordinates using interpolated pose.

    Returns ``None`` until pose history brackets every point in the scan.
    ``lidar_time_offset_s`` is a manual residual alignment adjustment; positive
    values treat LiDAR measurements as having happened later.
    """
    if not timed_frames or not lidar_clock.ready or not pose_clock.ready:
        return None
    extrinsics = lidar_extrinsics or LidarExtrinsics()
    points = []
    for timed_frame in timed_frames:
        laptop_time = (lidar_clock.to_laptop(timed_frame.time_s) +
                       lidar_time_offset_s)
        pose_time = pose_clock.from_laptop(laptop_time)
        pose = poses.interpolate(pose_time)
        if pose is None:
            return None
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for _lidar_time, lidar_x, lidar_y in _frame_points(timed_frame):
            body_x, body_y = extrinsics.transform_point(lidar_x, lidar_y)
            points.append((pose.x + c * body_x - s * body_y,
                           pose.y + s * body_x + c * body_y,
                           0.05))
    return points


def deskew_scan_local(timed_frames, poses, lidar_clock, pose_clock,
                      lidar_time_offset_s=0.0, lidar_extrinsics=None):
    """Deskew a revolution into its final robot frame for scan matching.

    The existing :func:`deskew_scan` output is in the wheel/IMU odometry world
    frame, which is ideal for visualization. SLAM needs the same compensated
    geometry without baking odometry drift into the scan. This function uses
    the final packet pose as a reference and expresses every compensated point
    relative to it.
    """
    world_points = deskew_scan(
        timed_frames, poses, lidar_clock, pose_clock, lidar_time_offset_s,
        lidar_extrinsics)
    if world_points is None:
        return None
    reference_lidar_time = max(frame.time_s for frame in timed_frames)
    reference_laptop_time = (
        lidar_clock.to_laptop(reference_lidar_time) + lidar_time_offset_s)
    reference_pose_time = pose_clock.from_laptop(reference_laptop_time)
    reference_pose = poses.interpolate(reference_pose_time)
    if reference_pose is None:
        return None
    c, s = math.cos(reference_pose.yaw), math.sin(reference_pose.yaw)
    local_points = []
    for world_x, world_y, _world_z in world_points:
        dx, dy = world_x - reference_pose.x, world_y - reference_pose.y
        local_points.append((c * dx + s * dy,
                             -s * dx + c * dy))
    return DeskewedScan(reference_pose_time, reference_pose, local_points)
