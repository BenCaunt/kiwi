import json
import math
import pathlib
import sys
import types
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_client import (  # noqa: E402
    FrameTransform,
    ImuYawContinuityFilter,
    KiwiClient,
    _yaw_from_quat,
    se2_zoh_correct_twist,
)


def _quat_from_yaw(degrees):
    half = 0.5 * math.radians(degrees)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


class FrameTransformTests(unittest.TestCase):
    def setUp(self):
        self.frames = FrameTransform(60.0)

    def assertTwistAlmostEqual(self, actual, expected):
        for value, wanted in zip(actual, expected):
            self.assertAlmostEqual(value, wanted)

    def test_aligned_forward_commands_robot_minus_sixty_degrees(self):
        self.assertTwistAlmostEqual(
            self.frames.aligned_to_robot(1.0, 0.0, 0.4),
            (0.5, -math.sqrt(3.0) / 2.0, 0.4),
        )

    def test_command_and_odometry_transforms_are_inverses(self):
        command = (0.31, -0.27, 1.2)
        raw = self.frames.aligned_to_robot(*command)
        self.assertTwistAlmostEqual(self.frames.robot_to_aligned(*raw), command)

    def test_odometry_report_is_aligned_without_mutating_input(self):
        report = {
            "measured": {"vx": 0.5, "vy": -math.sqrt(3.0) / 2.0, "omega": 0.2},
            "command": {"vx": 0.0, "vy": 1.0, "omega": -0.3},
            "wheel_speed_mps": [1.0, 2.0, 3.0],
        }
        aligned = self.frames.odometry_to_aligned(report)

        self.assertTwistAlmostEqual(
            tuple(aligned["measured"][axis] for axis in ("vx", "vy", "omega")),
            (1.0, 0.0, 0.2),
        )
        self.assertEqual(report["measured"]["vx"], 0.5)
        self.assertEqual(aligned["wheel_speed_mps"], [1.0, 2.0, 3.0])


class Se2CommandCorrectionTests(unittest.TestCase):
    def test_exponential_of_corrected_twist_matches_euler_increment(self):
        vx, vy, omega, dt = 0.31, -0.17, 1.2, 0.05
        corrected_x, corrected_y, corrected_omega = \
            se2_zoh_correct_twist(vx, vy, omega, dt)
        theta = corrected_omega * dt
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta
        reached_x = dt * (a * corrected_x - b * corrected_y)
        reached_y = dt * (b * corrected_x + a * corrected_y)

        self.assertAlmostEqual(reached_x, vx * dt)
        self.assertAlmostEqual(reached_y, vy * dt)
        self.assertAlmostEqual(theta, omega * dt)

    def test_forward_and_ccw_command_gets_compensating_right_velocity(self):
        vx, vy, omega = se2_zoh_correct_twist(0.2, 0.0, 1.0, 0.1)

        self.assertLess(vy, 0.0)
        self.assertAlmostEqual(vy, -0.01)
        self.assertAlmostEqual(omega, 1.0)

    def test_zero_rotation_is_unchanged(self):
        self.assertEqual(
            se2_zoh_correct_twist(0.2, -0.1, 0.0, 0.05),
            (0.2, -0.1, 0.0),
        )


class ImuYawContinuityFilterTests(unittest.TestCase):
    @staticmethod
    def report(yaw_deg, time_us, omega=0.0):
        return {
            "follower_time_us": time_us,
            "imu_ready": True,
            "imu_quat_ijkr": _quat_from_yaw(yaw_deg),
            "measured": {"omega": omega},
        }

    def test_normal_game_vector_motion_passes_through(self):
        tracker = ImuYawContinuityFilter()
        first = self.report(10.0, 1_000_000)
        second = self.report(20.0, 1_050_000,
                             omega=math.radians(200.0))
        tracker.filter_report(first)
        rejected = tracker.filter_report(second)

        self.assertFalse(rejected)
        self.assertAlmostEqual(
            math.degrees(_yaw_from_quat(second["imu_quat_ijkr"])),
            20.0,
        )
        self.assertEqual(tracker.rejections, 0)

    def test_origin_jump_is_rebased_and_following_motion_continues(self):
        tracker = ImuYawContinuityFilter()
        first = self.report(5.0, 1_000_000)
        jumped = self.report(-70.0, 1_050_000, omega=0.0)
        following = self.report(-60.0, 1_100_000,
                                omega=math.radians(200.0))
        tracker.filter_report(first)
        self.assertTrue(tracker.filter_report(jumped))
        self.assertFalse(tracker.filter_report(following))

        self.assertAlmostEqual(
            math.degrees(_yaw_from_quat(jumped["imu_quat_ijkr"])),
            5.0,
        )
        self.assertAlmostEqual(
            math.degrees(_yaw_from_quat(following["imu_quat_ijkr"])),
            15.0,
        )
        self.assertEqual(tracker.rejections, 1)

    def test_large_delta_matching_encoder_rotation_is_not_rejected(self):
        tracker = ImuYawContinuityFilter()
        first = self.report(0.0, 1_000_000)
        turning = self.report(40.0, 1_100_000,
                              omega=math.radians(400.0))
        tracker.filter_report(first)

        self.assertFalse(tracker.filter_report(turning))
        self.assertAlmostEqual(
            math.degrees(_yaw_from_quat(turning["imu_quat_ijkr"])),
            40.0,
        )


class _FakeConfig:
    def insert_json5(self, *_args):
        pass


class _FakePublisher:
    def __init__(self):
        self.payloads = []

    def put(self, payload):
        self.payloads.append(json.loads(payload))


class _FakeSession:
    def __init__(self):
        self.publisher = _FakePublisher()
        self.publisher_keys = []
        self.subscribers = {}
        self.closed = False

    def declare_publisher(self, key):
        self.publisher_keys.append(key)
        return self.publisher

    def declare_subscriber(self, key, callback):
        self.subscribers[key] = callback
        return callback

    def close(self):
        self.closed = True


class KiwiClientTests(unittest.TestCase):
    def test_transport_publishes_raw_frame_and_exposes_aligned_odometry(self):
        session = _FakeSession()
        fake_zenoh = types.SimpleNamespace(Config=_FakeConfig, open=lambda _conf: session)
        received = []
        with mock.patch.dict(sys.modules, {"zenoh": fake_zenoh}):
            client = KiwiClient("tcp/test:7447", "kiwi/test", commanding=True,
                                on_odometry=received.append)

        client.send_twist(1.0, 0.0, 0.4)
        sent = session.publisher.payloads[-1]
        self.assertAlmostEqual(sent["vx"], 0.5)
        self.assertAlmostEqual(sent["vy"], -math.sqrt(3.0) / 2.0)
        self.assertEqual(sent["omega"], 0.4)

        report = {"measured": sent, "command": sent}
        sample = types.SimpleNamespace(payload=json.dumps(report).encode())
        session.subscribers["kiwi/test/odom/twist"](sample)
        self.assertAlmostEqual(received[-1]["measured"]["vx"], 1.0)
        self.assertAlmostEqual(received[-1]["measured"]["vy"], 0.0)

        pose_sample = types.SimpleNamespace(payload=json.dumps({
            "pose": {"x": 1, "y": 2, "yaw": 0.3},
            "map_to_odom": {"x": 0.1, "y": -0.2, "yaw": 0.05},
        }).encode())
        session.subscribers["kiwi/test/slam/pose"](pose_sample)
        self.assertEqual(client.pose, {"x": 1.0, "y": 2.0, "yaw": 0.3})
        self.assertEqual(client.slam_report["map_to_odom"]["x"], 0.1)
        self.assertIsNotNone(client.pose_received_at)

        with mock.patch("kiwi_client.time.sleep"):
            client.close()
        self.assertEqual(session.publisher.payloads[-1],
                         {"vx": 0.0, "vy": 0.0, "omega": 0.0})
        self.assertTrue(session.closed)

    def test_hold_duration_applies_zoh_correction_before_frame_rotation(self):
        session = _FakeSession()
        fake_zenoh = types.SimpleNamespace(
            Config=_FakeConfig, open=lambda _conf: session)
        with mock.patch.dict(sys.modules, {"zenoh": fake_zenoh}):
            client = KiwiClient(
                "tcp/test:7447", "kiwi/test", commanding=True)

        client.send_twist(0.2, 0.0, 1.0, hold_s=0.1)

        sent = session.publisher.payloads[-1]
        aligned = client.frames.robot_to_aligned(
            sent["vx"], sent["vy"], sent["omega"])
        expected = se2_zoh_correct_twist(0.2, 0.0, 1.0, 0.1)
        for actual, wanted in zip(aligned, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_custom_command_topic_and_active_lease(self):
        session = _FakeSession()
        fake_zenoh = types.SimpleNamespace(Config=_FakeConfig, open=lambda _conf: session)
        with mock.patch.dict(sys.modules, {"zenoh": fake_zenoh}):
            client = KiwiClient(
                "tcp/test:7447", "kiwi/test", commanding=True,
                command_suffix="cmd_vel/teleop")

        client.send_twist(0.0, 0.0, 0.0, active=False)

        self.assertEqual(session.publisher_keys, ["kiwi/test/cmd_vel/teleop"])
        self.assertFalse(session.publisher.payloads[-1]["active"])


if __name__ == "__main__":
    unittest.main()
