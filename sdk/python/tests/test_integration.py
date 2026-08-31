from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from kiwi_sim_sdk import EnvConfig, KiwiEnv, VisionConfig


def _chrome_available() -> bool:
    configured = os.environ.get("KIWI_CHROMIUM_EXECUTABLE")
    candidates = [
        configured,
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
    ]
    return any(candidate and Path(candidate).is_file() for candidate in candidates)


@pytest.mark.skipif(not _chrome_available(), reason="Chrome/Chromium is not installed")
def test_seed_and_actions_reproduce_rewards_and_pixels() -> None:
    config = EnvConfig(
        world_id="room",
        vision=VisionConfig(width=64, height=48, context_length=2),
    )
    actions = [
        np.array([0.1, 0, 0], dtype=np.float32),
        np.array([0.1, 0.05, 0], dtype=np.float32),
    ]
    with KiwiEnv(config) as environment:
        traces = []
        for _ in range(2):
            observation, reset_info = environment.reset(seed=73)
            assert reset_info["provenance"]["controller_config"]["kpX"] == 0.8
            episode = [hashlib.sha256(observation["rgb"]).hexdigest()]
            for action in actions:
                observation, reward, terminated, truncated, info = environment.step(action)
                episode.append(
                    (
                        hashlib.sha256(observation["rgb"]).hexdigest(),
                        reward,
                        info["reward_terms"],
                        terminated,
                        truncated,
                    )
                )
            traces.append(episode)
    assert traces[0] == traces[1]
