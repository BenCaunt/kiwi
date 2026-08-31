from __future__ import annotations

import numpy as np

from kiwi_sim_sdk import EnvConfig, KiwiEnv, VisionConfig


def main() -> None:
    config = EnvConfig(
        world_id="room",
        observation_schema="vision_goal_v1",
        action_mode="relative_pose_v1",
        task="image_goal_navigation_v1",
        vision=VisionConfig(context_length=6),
    )
    with KiwiEnv(config) as environment:
        observation, info = environment.reset(seed=42)
        print("reset", observation["rgb"].shape, info["provenance"])
        for _ in range(config.max_episode_steps):
            action = np.array([0.15, 0.0, 0.0], dtype=np.float32)
            observation, reward, terminated, truncated, info = environment.step(action)
            print(reward, info["reward_terms"])
            if terminated or truncated:
                break


if __name__ == "__main__":
    main()
