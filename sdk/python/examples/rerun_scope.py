from __future__ import annotations

import argparse

import numpy as np
import rerun as rr

from kiwi_sim_sdk import EnvConfig, KiwiEnv, VisionConfig


def log_observation(observation: dict[str, object]) -> None:
    capture_time = float(observation["rgb_time_s"][-1])
    rr.set_time_seconds("simulation_time", capture_time)
    rr.log("policy/current_rgb", rr.Image(observation["rgb"][-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Kiwi RL episode in Rerun")
    parser.add_argument("--world", default="room")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    rr.init("kiwi_rl_environment", spawn=True)
    config = EnvConfig(
        world_id=args.world,
        action_mode="relative_pose_v1",
        vision=VisionConfig(context_length=4),
        privileged_debug=True,
    )
    with KiwiEnv(config) as environment:
        observation, info = environment.reset(seed=args.seed)
        rr.log("policy/goal_rgb", rr.Image(observation["goal_rgb"]), static=True)
        rr.log("episode/task_pair", rr.TextDocument(info["task_pair_id"]), static=True)
        log_observation(observation)

        for _ in range(args.steps):
            observation, reward, terminated, truncated, info = environment.step(
                np.array([0.12, 0.0, 0.0], dtype=np.float32)
            )
            log_observation(observation)
            rr.log("reward/total", rr.Scalar(reward))
            for name, value in info["reward_terms"].items():
                rr.log(f"reward/terms/{name}", rr.Scalar(value))
            privileged = info.get("privileged")
            if privileged:
                pose = privileged["pose"]
                goal = privileged["goal"]
                rr.log("map/robot", rr.Points2D([[pose["x"], pose["y"]]], radii=0.06))
                rr.log("map/goal", rr.Points2D([[goal["x"], goal["y"]]], radii=0.08))
            if terminated or truncated:
                break


if __name__ == "__main__":
    main()
