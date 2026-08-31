import json
import math
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_dashboard import Dashboard, camera_to_map_rotation  # noqa: E402
from kiwi_image_map import encode_image_capture  # noqa: E402
from kiwi_map import encode_occupancy_map  # noqa: E402
from kiwi_slam_core import OccupancyMap  # noqa: E402


class DashboardSlamTests(unittest.TestCase):
    def test_slam_pose_corrects_dead_reckoned_pose(self):
        dashboard = Dashboard()
        dashboard.pose = [1.0, 2.0, 0.25]
        corrected = {"x": -0.4, "y": 3.2, "yaw": -0.5}

        with mock.patch.object(dashboard, "_log_robot_pose"):
            dashboard.on_slam_pose(corrected)

        np.testing.assert_allclose(
            dashboard._map_pose(dashboard.pose),
            [corrected["x"], corrected["y"], corrected["yaw"]],
        )

        moved = dashboard._map_pose([1.5, 2.0, 0.25])
        correction_yaw = corrected["yaw"] - 0.25
        self.assertAlmostEqual(
            math.hypot(moved[0] - corrected["x"],
                       moved[1] - corrected["y"]),
            0.5,
        )
        self.assertAlmostEqual(moved[2], corrected["yaw"])
        self.assertAlmostEqual(
            math.atan2(moved[1] - corrected["y"],
                       moved[0] - corrected["x"]),
            correction_yaw,
        )

    def test_live_map_logs_occupied_cells_in_map_coordinates(self):
        dashboard = Dashboard()
        occupancy = OccupancyMap(
            data=np.array(((-1, 100), (65, 0)), dtype=np.int16),
            resolution_m=0.2,
            origin_x=-1.0,
            origin_y=2.0,
        )
        payload = encode_occupancy_map(occupancy, 3)

        boxes_value = mock.Mock()
        boxes_value.centers = np.array(((-0.8, 2.0, 0.0), (-1.0, 2.2, 0.0)))
        with (mock.patch("kiwi_dashboard.rr.log") as log,
              mock.patch("kiwi_dashboard.rr.Boxes3D",
                         return_value=boxes_value) as boxes):
            dashboard.on_map(payload)

        np.testing.assert_allclose(
            np.asarray(boxes.call_args.kwargs["centers"]),
            [[-0.8, 2.0, 0.0], [-1.0, 2.2, 0.0]],
        )
        np.testing.assert_allclose(
            np.asarray(boxes.call_args.kwargs["half_sizes"]),
            [[0.1, 0.1, 0.005], [0.1, 0.1, 0.005]],
        )
        self.assertEqual(dashboard.map_keyframes, 3)

    def test_correlated_image_logs_a_map_frame_pinhole_camera(self):
        dashboard = Dashboard()
        metadata = {
            "id": 7,
            "session_id": "test-session",
            "pose": {"x": 1.2, "y": -0.4, "yaw": math.pi / 2.0},
            "camera": {
                "width": 320,
                "height": 240,
                "fx": 220.0,
                "fy": 221.0,
                "cx": 159.5,
                "cy": 119.5,
                "height_m": 0.11,
            },
        }
        payload = encode_image_capture(metadata, b"jpeg bytes")

        with (
            mock.patch("kiwi_dashboard.rr.log"),
            mock.patch("kiwi_dashboard.rr.Transform3D") as transform,
            mock.patch("kiwi_dashboard.rr.Clear"),
            mock.patch("kiwi_dashboard.rr.Pinhole") as pinhole,
            mock.patch("kiwi_dashboard.rr.EncodedImage"),
            mock.patch("kiwi_dashboard.rr.Points3D"),
            mock.patch("kiwi_dashboard.rr.Scalar"),
        ):
            dashboard.on_image_capture(payload)

        np.testing.assert_allclose(
            transform.call_args.kwargs["translation"], [1.2, -0.4, 0.11]
        )
        np.testing.assert_allclose(
            transform.call_args.kwargs["mat3x3"],
            camera_to_map_rotation(math.pi / 2.0),
        )
        np.testing.assert_allclose(
            pinhole.call_args.kwargs["image_from_camera"],
            [[220.0, 0.0, 159.5], [0.0, 221.0, 119.5], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(dashboard.image_capture_ids, {7})

    def test_slam_report_reanchors_the_dashboard_odometry_frame(self):
        dashboard = Dashboard()
        dashboard.pose = [1.0, 0.0, 0.2]
        report = {
            "pose": {"x": -0.4, "y": 3.2, "yaw": -0.5},
            # This transform belongs to SLAM's independently initialized
            # odometry frame and must not be applied to dashboard.pose.
            "map_to_odom": {"x": 2.0, "y": -1.0, "yaw": 0.3},
        }

        with mock.patch.object(dashboard, "_log_robot_pose"):
            dashboard.on_slam_report(report)

        np.testing.assert_allclose(
            dashboard._map_pose(dashboard.pose),
            [report["pose"][axis] for axis in ("x", "y", "yaw")],
        )
        self.assertNotEqual(dashboard.map_to_odom, (2.0, -1.0, 0.3))

    def test_navigation_trajectory_is_plotted_in_map_frame(self):
        dashboard = Dashboard()
        payload = json.dumps({
            "frame": "map",
            "planner": "astar",
            "inflation_radius_m": 0.15,
            "points": [
                {"x": -0.2, "y": 0.3},
                {"x": 0.4, "y": 0.7},
            ],
        }).encode()

        with (
            mock.patch("kiwi_dashboard.rr.log"),
            mock.patch("kiwi_dashboard.rr.Points3D"),
            mock.patch("kiwi_dashboard.rr.LineStrips3D") as lines,
            mock.patch("kiwi_dashboard.rr.Scalar"),
        ):
            dashboard.on_navigation_trajectory(payload)

        np.testing.assert_allclose(
            np.asarray(lines.call_args.args[0])[0],
            [[-0.2, 0.3, 0.025], [0.4, 0.7, 0.025]],
        )
        self.assertEqual(
            dashboard.navigation_trajectory,
            [[-0.2, 0.3], [0.4, 0.7]],
        )

    def test_navigation_state_plots_pose_goal_and_following_point(self):
        dashboard = Dashboard()
        payload = json.dumps({
            "frame": "map",
            "status": "following",
            "pose": {"x": 0.1, "y": 0.2, "yaw": math.pi / 2.0},
            "goal": {"x": 1.0, "y": 2.0, "yaw": 0.0},
            "following_point": {"x": 0.4, "y": 0.5, "yaw": 0.2},
            "progress_m": 0.3,
            "remaining_m": 1.2,
            "cross_track_error_m": 0.04,
            "heading_error_rad": -0.1,
            "command": {"vx": 0.1, "vy": 0.0, "omega": -0.25},
        }).encode()

        with (
            mock.patch("kiwi_dashboard.rr.log") as log,
            mock.patch("kiwi_dashboard.rr.Points3D"),
            mock.patch("kiwi_dashboard.rr.LineStrips3D"),
            mock.patch("kiwi_dashboard.rr.Scalar"),
        ):
            dashboard.on_navigation_state(payload)

        paths = [call.args[0] for call in log.call_args_list]
        self.assertIn("/map/navigation/current_pose", paths)
        self.assertIn("/map/navigation/goal", paths)
        self.assertIn("/map/navigation/following_point", paths)
        self.assertIn("/map/navigation/desired_heading", paths)
        self.assertIn("/system/navigation_heading_error_deg", paths)
        self.assertIn("/system/navigation_command_omega_rad_s", paths)
        self.assertEqual(dashboard.navigation_status, "following")


if __name__ == "__main__":
    unittest.main()
