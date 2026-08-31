import math
import pathlib
import sys
import types
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_lidar_deskew import (  # noqa: E402
    LidarExtrinsics,
    PoseHistory,
    SensorClock,
    TimedFrame,
    deskew_scan,
    deskew_scan_local,
)


class SensorClockTests(unittest.TestCase):
    def test_unwraps_ld19_thirty_second_rollover(self):
        clock = SensorClock(wrap_seconds=30.0)

        self.assertAlmostEqual(clock.unwrap(29.990), 29.990)
        self.assertAlmostEqual(clock.unwrap(0.010), 30.010)

    def test_uses_minimum_observed_transport_latency(self):
        clock = SensorClock()
        clock.observe(10.0, 100.04)
        clock.observe(11.0, 101.01)

        self.assertAlmostEqual(clock.to_laptop(12.0), 102.01)
        self.assertAlmostEqual(clock.from_laptop(102.01), 12.0)


class PoseHistoryTests(unittest.TestCase):
    def test_interpolates_yaw_across_wrap(self):
        poses = PoseHistory()
        poses.append(0.0, 0.0, 0.0, math.radians(179.0))
        poses.append(1.0, 2.0, 4.0, math.radians(-179.0))

        pose = poses.interpolate(0.5)

        self.assertAlmostEqual(pose.x, 1.0)
        self.assertAlmostEqual(pose.y, 2.0)
        self.assertAlmostEqual(pose.yaw, math.pi)


class DeskewTests(unittest.TestCase):
    @staticmethod
    def frame(timestamp, angle):
        return TimedFrame(types.SimpleNamespace(
            start_angle_deg=angle,
            end_angle_deg=angle,
            speed_dps=3600.0,
            points=[(angle, 1.0, 100)],
        ), timestamp)

    def test_rotating_body_observations_land_on_same_world_point(self):
        lidar_clock = SensorClock()
        pose_clock = SensorClock()
        lidar_clock.observe(0.0, 100.0)
        pose_clock.observe(0.0, 100.0)
        poses = PoseHistory()
        poses.append(0.0, 0.0, 0.0, 0.0)
        poses.append(1.0, 0.0, 0.0, math.pi / 2.0)

        points = deskew_scan([
            self.frame(0.0, 0.0),
            self.frame(1.0, 90.0),
        ], poses, lidar_clock, pose_clock)

        self.assertEqual(len(points), 2)
        for x, y, _z in points:
            self.assertAlmostEqual(x, 1.0)
            self.assertAlmostEqual(y, 0.0, places=7)

    def test_waits_until_pose_history_brackets_scan(self):
        lidar_clock = SensorClock()
        pose_clock = SensorClock()
        lidar_clock.observe(0.0, 100.0)
        pose_clock.observe(0.0, 100.0)
        poses = PoseHistory()
        poses.append(0.0, 0.0, 0.0, 0.0)

        self.assertIsNone(deskew_scan(
            [self.frame(1.0, 0.0)], poses, lidar_clock, pose_clock))

    def test_local_deskew_expresses_points_in_final_robot_frame(self):
        lidar_clock = SensorClock()
        pose_clock = SensorClock()
        lidar_clock.observe(0.0, 100.0)
        pose_clock.observe(0.0, 100.0)
        poses = PoseHistory()
        poses.append(0.0, 0.0, 0.0, 0.0)
        poses.append(1.0, 0.0, 0.0, math.pi / 2.0)

        scan = deskew_scan_local([
            self.frame(0.0, 0.0),
            self.frame(1.0, 90.0),
        ], poses, lidar_clock, pose_clock)

        self.assertAlmostEqual(scan.pose.yaw, math.pi / 2.0)
        for x, y in scan.points:
            self.assertAlmostEqual(x, 0.0, places=7)
            self.assertAlmostEqual(y, -1.0, places=7)

    def test_applies_planar_lidar_extrinsics_before_body_motion(self):
        lidar_clock = SensorClock()
        pose_clock = SensorClock()
        lidar_clock.observe(0.0, 100.0)
        pose_clock.observe(0.0, 100.0)
        poses = PoseHistory()
        poses.append(0.0, 0.0, 0.0, 0.0)

        points = deskew_scan(
            [self.frame(0.0, 0.0)], poses, lidar_clock, pose_clock,
            lidar_extrinsics=LidarExtrinsics(
                x_m=0.2, y_m=0.1, yaw_rad=math.pi / 2.0))

        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0][0], 0.2, places=7)
        self.assertAlmostEqual(points[0][1], 1.1, places=7)


if __name__ == "__main__":
    unittest.main()
