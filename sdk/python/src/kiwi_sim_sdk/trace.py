from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ControllerConfig, EnvConfig, RewardConfig, TaskConfig, VisionConfig
from .env import KiwiEnv, Observation, encode_action

TRACE_SCHEMA = "kiwi_episode_trace_v1"


def observation_signature(observation: Observation) -> dict[str, object]:
    arrays: dict[str, object] = {}
    for name in ("rgb", "rgb_valid", "rgb_time_s", "rgb_sequence", "goal_rgb"):
        value = observation.get(name)
        if not isinstance(value, np.ndarray):
            continue
        contiguous = np.ascontiguousarray(value)
        arrays[name] = {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        }
    return {
        "schema": observation["schema"],
        "arrays": arrays,
        "goal_rgb_valid": int(observation["goal_rgb_valid"]),
        "calibration": observation["calibration"],
    }


def _config_from_manifest(
    document: dict[str, Any],
    simulator_web_dir: Path | None,
    chromium_executable: Path | None,
) -> EnvConfig:
    values = dict(document)
    vision = VisionConfig(**values.pop("vision"))
    task_values = dict(values.pop("task_config"))
    task_values["reward"] = RewardConfig(**task_values["reward"])
    task = TaskConfig(**task_values)
    controller = ControllerConfig(**values.pop("controller"))
    return EnvConfig(
        **values,
        vision=vision,
        task_config=task,
        controller=controller,
        simulator_web_dir=simulator_web_dir,
        chromium_executable=chromium_executable,
    )


class EpisodeRecorder:
    """Records actions, named rewards, events, provenance, and pixel hashes."""

    def __init__(self, environment: KiwiEnv) -> None:
        self.environment = environment
        self.trace: dict[str, Any] | None = None

    def reset(
        self, *, seed: int = 0, options: dict[str, object] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        observation, info = self.environment.reset(seed=seed, options=options)
        self.trace = {
            "schema": TRACE_SCHEMA,
            "env_config": self.environment.config.manifest_document(),
            "seed": seed,
            "reset_info": info,
            "reset_observation": observation_signature(observation),
            "steps": [],
            "metrics": {
                "steps": 0,
                "total_reward": 0.0,
                "collision_tick_count": 0,
                "success": False,
                "terminated": False,
                "truncated": False,
                "termination_reason": None,
            },
        }
        return observation, info

    def step(
        self, action: object
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self.trace is None:
            raise RuntimeError("EpisodeRecorder must be reset before step")
        encoded_action = encode_action(self.environment.config, action)
        observation, reward, terminated, truncated, info = self.environment.step(
            encoded_action
        )
        self.trace["steps"].append(
            {
                "action": encoded_action,
                "reward": reward,
                "reward_terms": info.get("reward_terms", {}),
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info.get("termination_reason"),
                "events": info.get("events", []),
                "controller_commands": info.get("controller_commands", []),
                "observation": observation_signature(observation),
            }
        )
        metrics = self.trace["metrics"]
        metrics["steps"] += 1
        metrics["total_reward"] += reward
        metrics["collision_tick_count"] += int(info.get("collision_tick_count", 0))
        metrics["success"] = bool(info.get("success", False))
        metrics["terminated"] = terminated
        metrics["truncated"] = truncated
        metrics["termination_reason"] = info.get("termination_reason")
        metrics["final_geodesic_distance_m"] = info.get("geodesic_distance_m")
        return observation, reward, terminated, truncated, info

    def save(self, path: str | Path) -> Path:
        if self.trace is None:
            raise RuntimeError("There is no episode trace to save")
        target = Path(path)
        target.write_text(
            json.dumps(self.trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target


@dataclass(frozen=True)
class ReplayReport:
    matched: bool
    steps: int
    mismatches: tuple[str, ...]


def replay_trace(
    path: str | Path,
    *,
    simulator_web_dir: Path | None = None,
    chromium_executable: Path | None = None,
) -> ReplayReport:
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"Unsupported trace schema {trace.get('schema')}")
    config = _config_from_manifest(
        trace["env_config"], simulator_web_dir, chromium_executable
    )
    mismatches: list[str] = []
    with KiwiEnv(config) as environment:
        observation, _ = environment.reset(seed=int(trace["seed"]))
        if observation_signature(observation) != trace["reset_observation"]:
            mismatches.append("reset_observation")
        for index, expected in enumerate(trace["steps"]):
            observation, reward, terminated, truncated, info = environment.step(
                expected["action"]
            )
            if not math.isclose(reward, float(expected["reward"]), rel_tol=0, abs_tol=1e-12):
                mismatches.append(f"step[{index}].reward")
            if info.get("reward_terms", {}) != expected["reward_terms"]:
                mismatches.append(f"step[{index}].reward_terms")
            if terminated != expected["terminated"] or truncated != expected["truncated"]:
                mismatches.append(f"step[{index}].termination")
            if info.get("termination_reason") != expected.get("termination_reason"):
                mismatches.append(f"step[{index}].termination_reason")
            if observation_signature(observation) != expected["observation"]:
                mismatches.append(f"step[{index}].observation")
    return ReplayReport(not mismatches, len(trace["steps"]), tuple(mismatches))
