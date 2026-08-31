#!/usr/bin/env python3
"""Simple planar pose-stabilization feedback for the Kiwi robot."""

from dataclasses import dataclass
import math


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    yaw: float

    @classmethod
    def from_mapping(cls, value):
        return cls(float(value["x"]), float(value["y"]), float(value["yaw"]))


@dataclass(frozen=True)
class Twist2:
    vx: float
    vy: float
    omega: float


def compose_relative_pose(origin, relative):
    """Express ``relative`` in ``origin`` and return the resulting map pose."""
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose2(
        origin.x + c * relative.x - s * relative.y,
        origin.y + s * relative.x + c * relative.y,
        wrap_angle(origin.yaw + relative.yaw),
    )


class PoseStabilizingController:
    """Independent map-frame P control, converted to an aligned body twist."""

    def __init__(
        self,
        kp_x=0.0,
        kp_y=0.0,
        kp_yaw=3.0,
        max_linear_speed=0.25,
        max_angular_speed=1.0,
        position_tolerance=0.04,
        yaw_tolerance=math.radians(2.0),
    ):
        self.kp_x = float(kp_x)
        self.kp_y = float(kp_y)
        self.kp_yaw = float(kp_yaw)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.position_tolerance = float(position_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        if min(self.kp_x, self.kp_y, self.kp_yaw) < 0.0:
            raise ValueError("proportional gains must be non-negative")
        if min(self.max_linear_speed, self.max_angular_speed) <= 0.0:
            raise ValueError("speed limits must be positive")
        if min(self.position_tolerance, self.yaw_tolerance) < 0.0:
            raise ValueError("tolerances must be non-negative")

    @staticmethod
    def error(current, target):
        return Pose2(
            target.x - current.x,
            target.y - current.y,
            wrap_angle(target.yaw - current.yaw),
        )

    def at_target(self, current, target):
        error = self.error(current, target)
        return (
            self.within_position_tolerance(current, target)
            and abs(wrap_angle(error.yaw)) <= self.yaw_tolerance
        )

    def within_position_tolerance(self, current, target):
        return math.hypot(
            current.x - target.x,
            current.y - target.y,
        ) <= self.position_tolerance

    def command(self, current, target):
        """Return an aligned body-frame twist for a map-frame pose target."""
        error = self.error(current, target)

        vx_map = self.kp_x * error.x
        vy_map = self.kp_y * error.y

        # R(yaw)^T converts the desired map displacement into the aligned body
        # frame. The finite-hold SE(2) correction is applied at send_twist(),
        # where the actual command period is known.
        c, s = math.cos(current.yaw), math.sin(current.yaw)
        vx_body = c * vx_map + s * vy_map
        vy_body = -s * vx_map + c * vy_map

        linear_speed = math.hypot(vx_body, vy_body)
        if linear_speed > self.max_linear_speed:
            scale = self.max_linear_speed / linear_speed
            vx_body *= scale
            vy_body *= scale

        # Once position is settled, hold it while the heading finishes.
        if self.within_position_tolerance(current, target):
            vx_body = 0.0
            vy_body = 0.0

        omega = max(
            -self.max_angular_speed,
            min(self.kp_yaw * error.yaw, self.max_angular_speed),
        )
        return Twist2(vx_body, vy_body, omega)
