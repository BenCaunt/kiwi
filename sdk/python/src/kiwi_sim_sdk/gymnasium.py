from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Gymnasium support is optional. Install kiwi-sim-sdk[gymnasium]."
    ) from error

from .config import EnvConfig
from .env import KiwiEnv, Observation


def _policy_observation(observation: Observation) -> dict[str, np.ndarray[Any, Any]]:
    result = {
        "rgb": observation["rgb"],
        "rgb_valid": observation["rgb_valid"],
        "rgb_time_s": observation["rgb_time_s"],
        "rgb_sequence": observation["rgb_sequence"],
        "goal_rgb_valid": np.asarray(observation["goal_rgb_valid"], dtype=np.uint8),
    }
    if "goal_rgb" in observation:
        result["goal_rgb"] = observation["goal_rgb"]
    return result


class KiwiGymEnv(gym.Env[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]):
    """Optional Gymnasium adapter; the core SDK does not depend on Gymnasium."""

    metadata = KiwiEnv.metadata

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        vision = self.config.vision
        rgb_shape = (vision.context_length, vision.height, vision.width, 3)
        observation_spaces: dict[str, spaces.Space[Any]] = {
            "rgb": spaces.Box(0, 255, shape=rgb_shape, dtype=np.uint8),
            "rgb_valid": spaces.Box(0, 1, shape=(vision.context_length,), dtype=np.uint8),
            "rgb_time_s": spaces.Box(
                -np.inf, np.inf, shape=(vision.context_length,), dtype=np.float64
            ),
            "rgb_sequence": spaces.Box(
                0, np.iinfo(np.uint32).max, shape=(vision.context_length,), dtype=np.uint32
            ),
            "goal_rgb_valid": spaces.Box(0, 1, shape=(), dtype=np.uint8),
        }
        if self.config.observation_schema == "vision_goal_v1":
            observation_spaces["goal_rgb"] = spaces.Box(
                0, 255, shape=(vision.height, vision.width, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(observation_spaces)
        if self.config.action_mode == "relative_trajectory_v1":
            horizon = self.config.trajectory_lookahead_index + 1
            self.action_space = spaces.Box(-np.inf, np.inf, shape=(horizon, 3), dtype=np.float32)
        else:
            self.action_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self._environment = KiwiEnv(self.config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self._environment.reset(seed=seed, options=options)
        return _policy_observation(observation), info

    def step(
        self, action: np.ndarray[Any, Any]
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._environment.step(action)
        return _policy_observation(observation), reward, terminated, truncated, info

    def close(self) -> None:
        self._environment.close()
