from __future__ import annotations

import numpy as np
import pytest

from kiwi_sim_sdk import ControllerConfig, EnvConfig, VisionConfig, encode_action


def test_action_shapes_are_model_agnostic_and_fixed_by_mode() -> None:
    pose = encode_action(
        EnvConfig(action_mode="relative_pose_v1"),
        np.array([0.2, -0.1, 0.3], dtype=np.float32),
    )
    assert pose["kind"] == "relative_pose"
    assert pose["dx"] == pytest.approx(0.2)

    trajectory = encode_action(
        EnvConfig(action_mode="relative_trajectory_v1"),
        np.array([[0.1, 0, 0], [0.2, 0.1, 0.2]], dtype=np.float32),
    )
    assert trajectory["kind"] == "relative_trajectory"
    assert len(trajectory["waypoints"]) == 2


def test_invalid_schema_task_combinations_are_rejected() -> None:
    with pytest.raises(ValueError, match="task-free"):
        EnvConfig(observation_schema="vision_v1")
    with pytest.raises(ValueError, match="positive"):
        VisionConfig(context_length=0)


def test_controller_tuning_uses_the_canonical_protocol_names() -> None:
    document = EnvConfig(controller=ControllerConfig(kp_x=1.25)).protocol_document()
    assert document["controller"]["kpX"] == 1.25
    assert document["max_relative_translation_m"] == 2.0
