import json
import math
import pathlib
import sys
import types
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_pose_controller import (  # noqa: E402
    Pose2,
    PoseStabilizingController,
    compose_relative_pose,
    wrap_angle,
)
from kiwi_pose_test import wait_for_fresh_pose  # noqa: E402


class PoseStabilizingControllerTests(unittest.TestCase):
    def test_shared_typescript_controller_fixtures(self):
        fixture_path = (
            pathlib.Path(__file__).parents[1]
            / "simulator-web"
            / "src"
            / "control"
            / "pose-controller-fixtures.json"
        )
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
        for fixture in fixtures:
            config = fixture["config"]
            controller = PoseStabilizingController(
                kp_x=config["kpX"],
                kp_y=config["kpY"],
                kp_yaw=config["kpYaw"],
                max_linear_speed=config["maxLinearSpeed"],
                max_angular_speed=config["maxAngularSpeed"],
                position_tolerance=config["positionTolerance"],
                yaw_tolerance=config["yawTolerance"],
            )
            command = controller.command(
                Pose2.from_mapping(fixture["current"]),
                Pose2.from_mapping(fixture["target"]),
            )
            with self.subTest(fixture=fixture["name"]):
                self.assertAlmostEqual(command.vx, fixture["expected"]["vx"], places=12)
                self.assertAlmostEqual(command.vy, fixture["expected"]["vy"], places=12)
                self.assertAlmostEqual(
                    command.omega, fixture["expected"]["omega"], places=12
                )

    def test_map_command_is_rotated_into_body_frame(self):
        controller = PoseStabilizingController(
            kp_x=1.0, kp_y=1.0, max_linear_speed=10.0
        )
        current = Pose2(0.0, 0.0, math.pi / 2.0)
        command = controller.command(current, Pose2(1.0, 0.0, math.pi / 2.0))

        self.assertAlmostEqual(command.vx, 0.0, places=7)
        self.assertAlmostEqual(command.vy, -1.0)
        self.assertAlmostEqual(command.omega, 0.0)

    def test_independent_axis_gains_and_linear_speed_limit(self):
        controller = PoseStabilizingController(
            kp_x=2.0, kp_y=1.0, max_linear_speed=1.0
        )
        command = controller.command(Pose2(0.0, 0.0, 0.0),
                                     Pose2(1.0, 1.0, 0.0))

        self.assertAlmostEqual(math.hypot(command.vx, command.vy), 1.0)
        self.assertAlmostEqual(command.vx / command.vy, 2.0)

    def test_yaw_uses_shortest_wrapped_error_and_is_limited(self):
        controller = PoseStabilizingController(
            kp_yaw=2.0, max_angular_speed=0.5
        )
        command = controller.command(
            Pose2(0.0, 0.0, math.radians(179.0)),
            Pose2(0.0, 0.0, math.radians(-179.0)),
        )

        self.assertAlmostEqual(command.omega, 2.0 * math.radians(2.0))
        self.assertAlmostEqual(
            wrap_angle(math.radians(-179.0) - math.radians(179.0)),
            math.radians(2.0),
        )

    def test_relative_pose_is_composed_in_origin_frame(self):
        target = compose_relative_pose(
            Pose2(1.0, 2.0, math.pi / 2.0),
            Pose2(0.5, 0.5, 0.0),
        )

        self.assertAlmostEqual(target.x, 0.5)
        self.assertAlmostEqual(target.y, 2.5)
        self.assertAlmostEqual(target.yaw, math.pi / 2.0)

    def test_at_target_checks_translation_and_yaw(self):
        controller = PoseStabilizingController(
            position_tolerance=0.02,
            yaw_tolerance=math.radians(2.0),
        )
        self.assertTrue(controller.at_target(
            Pose2(0.0, 0.0, 0.0),
            Pose2(0.01, 0.01, math.radians(1.0)),
        ))
        self.assertFalse(controller.at_target(
            Pose2(0.0, 0.0, 0.0),
            Pose2(0.03, 0.0, 0.0),
        ))


class PoseRecoveryTests(unittest.TestCase):
    def test_wait_returns_when_pose_updates_resume(self):
        clock = types.SimpleNamespace(now=10.0)
        client = types.SimpleNamespace(pose_received_at=9.0)

        def sleep(duration):
            clock.now += duration
            client.pose_received_at = clock.now

        with mock.patch("kiwi_pose_test.time.monotonic",
                        side_effect=lambda: clock.now), \
                mock.patch("kiwi_pose_test.time.sleep", side_effect=sleep):
            wait_for_fresh_pose(client, max_age_s=0.5,
                                recovery_timeout_s=1.0)

    def test_wait_times_out_when_pose_does_not_recover(self):
        clock = types.SimpleNamespace(now=10.0)
        client = types.SimpleNamespace(pose_received_at=9.0)

        def sleep(duration):
            clock.now += duration

        with mock.patch("kiwi_pose_test.time.monotonic",
                        side_effect=lambda: clock.now), \
                mock.patch("kiwi_pose_test.time.sleep", side_effect=sleep):
            with self.assertRaises(TimeoutError):
                wait_for_fresh_pose(client, max_age_s=0.5,
                                    recovery_timeout_s=0.1)


if __name__ == "__main__":
    unittest.main()
