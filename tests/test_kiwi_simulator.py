import json
import math
from pathlib import Path
import struct
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kiwi_lidar import parse_frames  # noqa: E402
from kiwi_sim_mapping_harness import summarize_slam_quality  # noqa: E402
from kiwi_sim_core import (  # noqa: E402
    Environment,
    IDEAL_SENSOR_PROFILE,
    KiwiRobotModel,
    LD19Simulator,
    RETAINED_ROBOT_PROFILE,
    Segment,
    SimulatorConfig,
    aligned_to_robot,
    camera_payload,
    parse_velocity_payload,
    rectangle_walls,
    robot_to_aligned,
)


def square_environment(size=4.0):
    half = size / 2.0
    return Environment(
        "test-square",
        rectangle_walls(-half, -half, half, half),
        (0.0, 0.0, 0.0),
    )


class MappingHarnessMetricsTests(unittest.TestCase):
    def test_summarizes_scan_quality_for_estimator_ab_comparison(self):
        summary = summarize_slam_quality([
            {
                "scan_matched": True,
                "heading_disagreement_deg": -1.0,
                "scan_feedback": "warming",
                "learned_rate_bias_deg_s": 0.01,
            },
            {
                "scan_matched": False,
                "heading_disagreement_deg": 3.0,
                "scan_feedback": "accepted",
                "learned_rate_bias_deg_s": 0.02,
            },
        ])

        self.assertEqual(summary["scan_samples"], 2)
        self.assertEqual(summary["scan_match_rate"], 0.5)
        self.assertEqual(summary["heading_disagreement_abs_p50_deg"], 2.0)
        self.assertAlmostEqual(
            summary["heading_disagreement_abs_p95_deg"], 2.9)
        self.assertEqual(summary["scan_feedback_accepted_samples"], 1)


class CommandContractTests(unittest.TestCase):
    def test_parses_same_json_and_text_commands_as_firmware(self):
        json_twist, json_timeout = parse_velocity_payload(
            json.dumps({"vx": 0.2, "vy": -0.1, "omega": 0.7}).encode()
        )
        text_twist, text_timeout = parse_velocity_payload(b"0.2 -0.1 0.7")

        self.assertEqual(json_twist, text_twist)
        self.assertIsNone(json_timeout)
        self.assertIsNone(text_timeout)

    def test_parses_binary_velocity_command_and_stop_mode(self):
        drive = struct.pack("<QfffHBB", 99, 0.2, -0.1, 0.7, 425, 0, 0)
        stop = struct.pack("<QfffHBB", 99, 0.2, -0.1, 0.7, 250, 1, 0)

        twist, timeout = parse_velocity_payload(drive)
        stopped, _ = parse_velocity_payload(stop)

        self.assertAlmostEqual(twist.vx, 0.2)
        self.assertAlmostEqual(twist.vy, -0.1)
        self.assertAlmostEqual(twist.omega, 0.7)
        self.assertEqual(timeout, 0.425)
        self.assertEqual((stopped.vx, stopped.vy, stopped.omega), (0.0, 0.0, 0.0))

    def test_raw_and_aligned_frame_mapping_are_inverses(self):
        raw = aligned_to_robot(0.4, -0.2, 60.0)
        aligned = robot_to_aligned(*raw, 60.0)

        self.assertAlmostEqual(aligned[0], 0.4)
        self.assertAlmostEqual(aligned[1], -0.2)


class WorldModelTests(unittest.TestCase):
    def test_aligned_forward_command_moves_along_lidar_forward(self):
        model = KiwiRobotModel(
            square_environment(10.0),
            SimulatorConfig(response_time_s=0.01),
        )
        now = 0.0
        for _ in range(100):
            model.set_command_aligned(0.5, 0.0, 0.0, now)
            now += 0.01
            model.step(0.01, now)

        self.assertGreater(model.state.x, 0.45)
        self.assertAlmostEqual(model.state.y, 0.0, places=6)

    def test_robot_cannot_cross_wall(self):
        environment = Environment(
            "wall",
            [Segment(0.5, -2.0, 0.5, 2.0)],
            (0.0, 0.0, 0.0),
        )
        model = KiwiRobotModel(
            environment,
            SimulatorConfig(response_time_s=0.01, robot_radius_m=0.13),
        )
        now = 0.0
        for _ in range(300):
            model.set_command_aligned(1.0, 0.0, 0.0, now)
            now += 0.01
            model.step(0.01, now)

        self.assertLessEqual(model.state.x, 0.5 - model.config.robot_radius_m + 1e-6)
        self.assertAlmostEqual(model.state.measured_raw.vx, 0.0, places=6)
        self.assertAlmostEqual(model.state.measured_raw.vy, 0.0, places=6)

    def test_command_watchdog_matches_robot_timeout_status(self):
        model = KiwiRobotModel(square_environment())
        model.set_command_raw(0.1, 0.0, 0.0, now=0.0)
        model.step(0.01, now=0.01)
        active = model.odometry_report(10_000)
        model.step(0.3, now=0.31)
        timed_out = model.odometry_report(310_000)

        self.assertEqual(active["status_flags"] & 1, 0)
        self.assertEqual(timed_out["status_flags"] & 1, 1)
        self.assertEqual(timed_out["command"], {"vx": 0.0, "vy": 0.0, "omega": 0.0})

    def test_odometry_has_firmware_fields_and_raw_frame_values(self):
        model = KiwiRobotModel(square_environment())
        model.set_command_raw(0.2, -0.1, 0.3, now=0.0)
        model.step(0.02, now=0.02)

        report = model.odometry_report(20_000)

        self.assertEqual(
            set(report),
            {
                "follower_time_us", "seq", "measured", "command",
                "wheel_speed_mps", "wheel_angle_rad", "encoder_count",
                "imu_ready", "encoder_ready_mask", "status_flags",
                "imu_quat_ijkr", "imu_accel_mps2",
            },
        )
        self.assertEqual(report["command"], {"vx": 0.2, "vy": -0.1, "omega": 0.3})
        self.assertEqual(len(report["wheel_speed_mps"]), 3)
        self.assertEqual(report["encoder_ready_mask"], 7)


class SensorContractTests(unittest.TestCase):
    def test_ld19_batch_is_twenty_crc_valid_firmware_frames(self):
        environment = square_environment()
        model = KiwiRobotModel(environment)
        payload = LD19Simulator(
            environment,
            range_noise_std_m=0.0,
            sensor_profile=IDEAL_SENSOR_PROFILE,
        ).batch(model.state, 20)

        frames = parse_frames(payload)

        self.assertEqual(len(payload), 20 * 47)
        self.assertEqual(len(frames), 20)
        self.assertTrue(all(frame is not None for frame in frames))
        self.assertTrue(all(
            0.0 < distance <= 12.0
            for frame in frames
            for _angle, distance, _intensity in frame.points
        ))

    def test_ld19_full_revolution_models_rolling_scan_motion(self):
        environment = square_environment()
        model = KiwiRobotModel(environment)
        model.state.x = 1.0
        payload = LD19Simulator(
            environment,
            range_noise_std_m=0.0,
            sensor_profile=IDEAL_SENSOR_PROFILE,
        ).batch(model.state, 40, start_pose=(0.0, 0.0, 0.0))

        frames = parse_frames(payload)
        first_forward_range = frames[0].points[0][1]
        last_forward_range = frames[-1].points[-1][1]

        self.assertGreater(first_forward_range, last_forward_range + 0.8)

    def test_retained_profile_adds_residual_missing_returns(self):
        environment = square_environment()
        model = KiwiRobotModel(environment)
        payload = LD19Simulator(
            environment,
            sensor_profile=RETAINED_ROBOT_PROFILE,
            seed=7,
        ).batch(model.state, 40)

        frames = parse_frames(payload)
        ranges = [
            distance
            for frame in frames
            for _angle, distance, _intensity in frame.points
            if distance > 0.0
        ]

        self.assertGreaterEqual(len(ranges), 420)
        self.assertLessEqual(len(ranges), 470)
        self.assertLessEqual(max(ranges), 8.0)

    def test_camera_packet_uses_kvc1_header_and_embeds_jpeg(self):
        from PIL import Image
        import io

        environment = square_environment()
        model = KiwiRobotModel(environment)
        payload = camera_payload(environment, model.state, 7, 123_456, 160, 120)

        self.assertEqual(payload[:4], b"KVC1")
        self.assertEqual(struct.unpack_from("<HHH", payload, 6), (160, 120, 32))
        self.assertEqual(struct.unpack_from("<I", payload, 12)[0], 7)
        self.assertEqual(struct.unpack_from("<Q", payload, 16)[0], 123_456)
        jpeg_length = struct.unpack_from("<I", payload, 24)[0]
        self.assertEqual(jpeg_length, len(payload) - 32)
        with Image.open(io.BytesIO(payload[32:])) as image:
            self.assertEqual(image.size, (160, 120))


if __name__ == "__main__":
    unittest.main()
