from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ActionMode = Literal[
    "relative_pose_v1", "relative_trajectory_v1", "twist_aligned_v1"
]
ObservationSchema = Literal["vision_goal_v1", "vision_v1"]
SensorProfile = Literal["ideal", "retained-robot-maps-v1"]
TaskId = Literal["image_goal_navigation_v1"]


@dataclass(frozen=True)
class VisionConfig:
    width: int = 320
    height: int = 240
    vertical_fov_deg: float = 72.0
    context_length: int = 6
    context_stride: int = 1

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Vision dimensions must be positive")
        if not 0 < self.vertical_fov_deg < 180:
            raise ValueError("vertical_fov_deg must be in (0, 180)")
        if self.context_length <= 0 or self.context_stride <= 0:
            raise ValueError("Vision context values must be positive")


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 1.0
    success: float = 5.0
    collision: float = 0.25
    time: float = 0.01
    controller: float = 0.001
    smoothness: float = 0.01


@dataclass(frozen=True)
class TaskConfig:
    success_radius_m: float = 0.25
    require_goal_heading: bool = False
    success_yaw_tolerance_rad: float = 0.25
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.success_radius_m <= 0:
            raise ValueError("success_radius_m must be positive")
        if self.success_yaw_tolerance_rad <= 0:
            raise ValueError("success_yaw_tolerance_rad must be positive")


@dataclass(frozen=True)
class ControllerConfig:
    kp_x: float = 0.8
    kp_y: float = 0.8
    kp_yaw: float = 1.5
    max_linear_speed: float = 0.25
    max_angular_speed: float = 1.0
    position_tolerance: float = 0.04
    yaw_tolerance: float = 0.03490658503988659

    def __post_init__(self) -> None:
        if min(self.kp_x, self.kp_y, self.kp_yaw) < 0:
            raise ValueError("Controller gains must be non-negative")
        if min(self.max_linear_speed, self.max_angular_speed) <= 0:
            raise ValueError("Controller speed limits must be positive")
        if min(self.position_tolerance, self.yaw_tolerance) < 0:
            raise ValueError("Controller tolerances must be non-negative")

    def protocol_document(self) -> dict[str, float]:
        return {
            "kpX": self.kp_x,
            "kpY": self.kp_y,
            "kpYaw": self.kp_yaw,
            "maxLinearSpeed": self.max_linear_speed,
            "maxAngularSpeed": self.max_angular_speed,
            "positionTolerance": self.position_tolerance,
            "yawTolerance": self.yaw_tolerance,
        }


@dataclass(frozen=True)
class EnvConfig:
    world_id: str = "home"
    observation_schema: ObservationSchema = "vision_goal_v1"
    action_mode: ActionMode = "relative_pose_v1"
    policy_hz: float | None = None
    controller_hz: float = 20.0
    trajectory_lookahead_index: int = 0
    max_episode_steps: int = 400
    max_relative_translation_m: float = 2.0
    max_relative_yaw_rad: float = 3.141592653589793
    max_trajectory_waypoints: int = 32
    sensor_profile: SensorProfile = "ideal"
    task: TaskId | None = "image_goal_navigation_v1"
    task_config: TaskConfig = field(default_factory=TaskConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    privileged_debug: bool = False
    simulator_web_dir: Path | None = None
    chromium_executable: Path | None = None
    build_if_missing: bool = True

    def __post_init__(self) -> None:
        if self.policy_hz is not None and self.policy_hz <= 0:
            raise ValueError("policy_hz must be positive")
        if self.controller_hz <= 0:
            raise ValueError("controller_hz must be positive")
        if self.trajectory_lookahead_index < 0:
            raise ValueError("trajectory_lookahead_index must be non-negative")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.max_relative_translation_m <= 0 or self.max_relative_yaw_rad <= 0:
            raise ValueError("Relative action bounds must be positive")
        if self.max_trajectory_waypoints <= 0:
            raise ValueError("max_trajectory_waypoints must be positive")
        if self.observation_schema == "vision_goal_v1" and self.task is None:
            raise ValueError("vision_goal_v1 requires image_goal_navigation_v1")
        if self.observation_schema == "vision_v1" and self.task is not None:
            raise ValueError("vision_v1 is task-free in protocol v1; set task=None")

    @property
    def resolved_policy_hz(self) -> float:
        if self.policy_hz is not None:
            return self.policy_hz
        return 20.0 if self.action_mode == "twist_aligned_v1" else 4.0

    def protocol_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "world_id": self.world_id,
            "observation_schema": self.observation_schema,
            "action_mode": self.action_mode,
            "policy_hz": self.resolved_policy_hz,
            "controller_hz": self.controller_hz,
            "trajectory_lookahead_index": self.trajectory_lookahead_index,
            "max_episode_steps": self.max_episode_steps,
            "max_relative_translation_m": self.max_relative_translation_m,
            "max_relative_yaw_rad": self.max_relative_yaw_rad,
            "max_trajectory_waypoints": self.max_trajectory_waypoints,
            "sensor_profile": self.sensor_profile,
            "task": self.task,
            "privileged_debug": self.privileged_debug,
            "success_radius_m": self.task_config.success_radius_m,
            "require_goal_heading": self.task_config.require_goal_heading,
            "success_yaw_tolerance_rad": self.task_config.success_yaw_tolerance_rad,
            "reward": asdict(self.task_config.reward),
            "controller": self.controller.protocol_document(),
            "vision_width": self.vision.width,
            "vision_height": self.vision.height,
            "vertical_fov_deg": self.vision.vertical_fov_deg,
            "context_length": self.vision.context_length,
            "context_stride": self.vision.context_stride,
        }
        return document

    def manifest_document(self) -> dict[str, object]:
        document = asdict(self)
        for key in ("simulator_web_dir", "chromium_executable", "build_if_missing"):
            document.pop(key, None)
        return document
