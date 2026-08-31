from __future__ import annotations

import numpy as np

from kiwi_sim_sdk import EnvConfig, KiwiEnv


def main() -> None:
    config = EnvConfig(
        world_id="room",
        action_mode="relative_trajectory_v1",
        trajectory_lookahead_index=4,
    )
    # Five cumulative poses, all in the robot frame captured when step accepts
    # the action. A real policy can produce the same generic SI-unit tensor.
    waypoints = np.array(
        [
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.15, 0.01, 0.00],
            [0.20, 0.02, 0.00],
            [0.25, 0.03, 0.00],
        ],
        dtype=np.float32,
    )
    with KiwiEnv(config) as environment:
        observation, _ = environment.reset(seed=42)
        while True:
            observation, reward, terminated, truncated, info = environment.step(
                waypoints
            )
            print(reward, info["controller_commands"][-1])
            if terminated or truncated:
                break


if __name__ == "__main__":
    main()
