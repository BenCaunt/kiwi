#!/usr/bin/env python3
"""Laptop-side Zenoh client with the Kiwi robot frame correction.

The drivetrain's reported +X axis is mounted ``robot_yaw_deg`` counter-
clockwise from the lidar/camera forward direction. Public commands and
odometry use the aligned lidar/camera frame; conversion to and from the raw
drivetrain frame happens at this client boundary.
"""

import copy
import json
import math
import time


DEFAULT_ROBOT_YAW_DEG = 60.0


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quat(quat):
    i, j, k, real = quat
    return math.atan2(2.0 * (real * k + i * j),
                      1.0 - 2.0 * (j * j + k * k))


def se2_zoh_correct_twist(vx, vy, omega, dt_s):
    """Invert the SE(2) exponential for one desired Euler pose increment.

    The caller's finite-step intent is ``(vx*dt, vy*dt, omega*dt)``. A robot
    holding an unmodified body twist instead follows ``Exp(twist*dt)``, whose
    translation arcs whenever omega is nonzero. Return the constant body twist
    whose exponential lands on the intended finite-step pose.
    """
    vx, vy, omega, dt_s = map(float, (vx, vy, omega, dt_s))
    if not all(math.isfinite(value) for value in (vx, vy, omega, dt_s)):
        raise ValueError("twist and hold duration must be finite")
    if dt_s <= 0.0:
        raise ValueError("hold duration must be positive")
    theta = omega * dt_s
    if abs(theta) >= math.pi:
        raise ValueError("rotation per hold must be less than pi radians")
    half_theta = 0.5 * theta
    if abs(theta) < 1.0e-4:
        theta_squared = theta * theta
        diagonal = 1.0 - theta_squared / 12.0 - \
            theta_squared * theta_squared / 720.0
    else:
        diagonal = half_theta / math.tan(half_theta)
    return (
        diagonal * vx + half_theta * vy,
        -half_theta * vx + diagonal * vy,
        omega,
    )


class ImuYawContinuityFilter:
    """Remove nonphysical BNO08x yaw-origin jumps from odometry reports.

    Game Rotation Vector avoids magnetic-heading corrections, but its heading
    has no absolute reference. A sensor reinitialization can therefore choose
    a new yaw origin. Normal Game Rotation Vector deltas pass through exactly;
    only a delta that strongly disagrees with encoder angular velocity is
    rebased onto the continuous heading.
    """

    _CLOCK_WRAP_US = 1 << 32

    def __init__(self, base_jump_deg=25.0, expected_scale=0.0,
                 max_interval_s=0.5):
        self.base_jump_rad = math.radians(float(base_jump_deg))
        self.expected_scale = float(expected_scale)
        self.max_interval_s = float(max_interval_s)
        self.rejections = 0
        self._last_raw_yaw = None
        self._continuous_yaw = None
        self._last_time_s = None
        self._last_clock_us = None
        self._clock_epoch_us = 0

    def _report_time(self, report, arrival_time):
        raw_us = report.get("follower_time_us")
        if isinstance(raw_us, (int, float)) and math.isfinite(raw_us):
            raw_us = int(raw_us) % self._CLOCK_WRAP_US
            if (self._last_clock_us is not None and
                    raw_us < self._last_clock_us - self._CLOCK_WRAP_US // 2):
                self._clock_epoch_us += self._CLOCK_WRAP_US
            self._last_clock_us = raw_us
            return (self._clock_epoch_us + raw_us) / 1_000_000.0
        return float(arrival_time)

    @staticmethod
    def _apply_yaw_correction(quat, correction):
        """Left-multiply ``quat`` by a map-Z yaw correction."""
        i, j, k, real = quat
        half = 0.5 * correction
        sine, cosine = math.sin(half), math.cos(half)
        corrected = (
            cosine * i - sine * j,
            cosine * j + sine * i,
            cosine * k + sine * real,
            cosine * real - sine * k,
        )
        norm = math.sqrt(sum(value * value for value in corrected))
        if not math.isfinite(norm) or norm < 1.0e-6:
            return None
        return [value / norm for value in corrected]

    def filter_report(self, report, arrival_time=None):
        """Correct ``report`` in place and return whether a jump was rebased."""
        quat = report.get("imu_quat_ijkr")
        if (not report.get("imu_ready") or not isinstance(quat, (list, tuple))
                or len(quat) != 4):
            return False
        try:
            quat = tuple(float(value) for value in quat)
        except (TypeError, ValueError):
            return False
        norm = math.sqrt(sum(value * value for value in quat))
        if (not math.isfinite(norm) or norm < 0.5 or norm > 1.5):
            return False
        quat = tuple(value / norm for value in quat)
        raw_yaw = _yaw_from_quat(quat)
        now = time.monotonic() if arrival_time is None else arrival_time
        report_time = self._report_time(report, now)

        rejected = False
        if self._last_raw_yaw is None:
            self._continuous_yaw = raw_yaw
        else:
            raw_delta = _wrap_angle(raw_yaw - self._last_raw_yaw)
            dt = report_time - self._last_time_s
            corrected_delta = raw_delta
            if 0.0 < dt <= self.max_interval_s:
                measured = report.get("measured", {})
                try:
                    expected_delta = float(measured.get("omega", 0.0)) * dt
                except (AttributeError, TypeError, ValueError):
                    expected_delta = 0.0
                innovation = _wrap_angle(raw_delta - expected_delta)
                allowed = (self.base_jump_rad +
                           self.expected_scale * abs(expected_delta))
                if abs(innovation) > allowed:
                    corrected_delta = expected_delta
                    self.rejections += 1
                    rejected = True
            self._continuous_yaw += corrected_delta

        correction = self._continuous_yaw - raw_yaw
        corrected_quat = self._apply_yaw_correction(quat, correction)
        if corrected_quat is not None:
            report["imu_quat_ijkr"] = corrected_quat
        report["imu_yaw_rejections"] = self.rejections
        report["imu_yaw_discontinuity"] = rejected
        self._last_raw_yaw = raw_yaw
        self._last_time_s = report_time
        return rejected


class FrameTransform:
    """Convert planar twists between aligned and raw drivetrain frames."""

    def __init__(self, robot_yaw_deg=DEFAULT_ROBOT_YAW_DEG):
        self.robot_yaw_deg = float(robot_yaw_deg)

    @staticmethod
    def _rotate(vx, vy, angle_deg):
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return c * vx - s * vy, s * vx + c * vy

    def aligned_to_robot(self, vx, vy, omega=0.0):
        """Rotate an aligned command into the raw drivetrain frame."""
        vx, vy = self._rotate(vx, vy, -self.robot_yaw_deg)
        return vx, vy, omega

    def robot_to_aligned(self, vx, vy, omega=0.0):
        """Rotate raw drivetrain odometry into the aligned sensor frame."""
        vx, vy = self._rotate(vx, vy, self.robot_yaw_deg)
        return vx, vy, omega

    def odometry_to_aligned(self, report):
        """Return a copy of an odometry report with its twists aligned."""
        aligned = copy.deepcopy(report)
        for field in ("measured", "command"):
            twist = aligned.get(field)
            if not isinstance(twist, dict):
                continue
            vx, vy, omega = self.robot_to_aligned(
                twist.get("vx", 0.0),
                twist.get("vy", 0.0),
                twist.get("omega", 0.0),
            )
            twist.update(vx=vx, vy=vy, omega=omega)
        return aligned


class KiwiClient:
    """Zenoh transport whose public twist interface uses the aligned frame."""

    def __init__(self, connect, namespace, robot_yaw_deg=DEFAULT_ROBOT_YAW_DEG,
                 commanding=False, on_odometry=None,
                 command_suffix="cmd_vel"):
        import zenoh

        self.namespace = namespace.rstrip("/")
        self.frames = FrameTransform(robot_yaw_deg)
        self._imu_yaw_filter = ImuYawContinuityFilter()
        self.odometry = None
        self.pose = None
        self.slam_report = None
        self.pose_received_at = None
        self._odometry_callbacks = []
        self._pose_callbacks = []
        self._slam_callbacks = []
        self._subscribers = []
        self.command_suffix = str(command_suffix).strip("/")
        if commanding and not self.command_suffix:
            raise ValueError("command_suffix cannot be empty")

        conf = zenoh.Config()
        conf.insert_json5("mode", '"client"')
        conf.insert_json5("connect/endpoints", json.dumps([connect]))
        self.session = zenoh.open(conf)
        self._command_publisher = None
        if commanding:
            self._command_publisher = self.session.declare_publisher(
                f"{self.namespace}/{self.command_suffix}")
        if on_odometry is not None:
            self._odometry_callbacks.append(on_odometry)
        self._subscribers.append(self.session.declare_subscriber(
            f"{self.namespace}/odom/twist", self._on_odometry))
        self._subscribers.append(self.session.declare_subscriber(
            f"{self.namespace}/slam/pose", self._on_pose))

    @property
    def robot_yaw_deg(self):
        return self.frames.robot_yaw_deg

    @property
    def imu_yaw_rejections(self):
        return self._imu_yaw_filter.rejections

    def _on_odometry(self, sample):
        try:
            report = json.loads(bytes(sample.payload).decode())
        except (ValueError, UnicodeDecodeError):
            return
        self.odometry = self.frames.odometry_to_aligned(report)
        self._imu_yaw_filter.filter_report(self.odometry)
        for callback in self._odometry_callbacks:
            callback(self.odometry)

    def _on_pose(self, sample):
        try:
            report = json.loads(bytes(sample.payload).decode())
        except (ValueError, UnicodeDecodeError):
            return
        pose = report.get("pose")
        if not isinstance(pose, dict):
            return
        try:
            self.pose = {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": float(pose["yaw"]),
            }
        except (KeyError, TypeError, ValueError):
            return
        self.pose_received_at = time.monotonic()
        self.slam_report = report
        for callback in self._pose_callbacks:
            callback(self.pose)
        for callback in self._slam_callbacks:
            callback(self.slam_report)

    def subscribe(self, suffix, callback):
        """Subscribe to a namespaced topic and pass its payload as bytes."""
        def listener(sample):
            callback(bytes(sample.payload))

        subscriber = self.session.declare_subscriber(
            f"{self.namespace}/{suffix.lstrip('/')}", listener)
        self._subscribers.append(subscriber)
        return subscriber

    def add_odometry_callback(self, callback):
        """Register an additional callback for aligned odometry reports."""
        self._odometry_callbacks.append(callback)
        return callback

    def add_pose_callback(self, callback):
        """Register a callback for SLAM poses in the global map frame."""
        self._pose_callbacks.append(callback)
        return callback

    def add_slam_callback(self, callback):
        """Register a callback for the complete SLAM pose/quality report."""
        self._slam_callbacks.append(callback)
        return callback

    def send_twist(self, vx, vy, omega, *, active=None, hold_s=None):
        if self._command_publisher is None:
            raise RuntimeError("this KiwiClient was opened without commanding=True")
        if hold_s is not None:
            vx, vy, omega = se2_zoh_correct_twist(
                vx, vy, omega, hold_s)
        vx, vy, omega = self.frames.aligned_to_robot(vx, vy, omega)
        command = {
            "vx": vx,
            "vy": vy,
            "omega": omega,
        }
        if active is not None:
            command["active"] = bool(active)
        self._command_publisher.put(json.dumps(command))

    def close(self):
        if self._command_publisher is not None:
            self.send_twist(0.0, 0.0, 0.0)
            time.sleep(0.1)
        self.session.close()
