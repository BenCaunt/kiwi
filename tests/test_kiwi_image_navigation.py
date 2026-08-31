import json
import math
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_image_navigation import (  # noqa: E402
    ImageMapDataset,
    ImageMapError,
    LiveSlamMonitor,
    NavigationManager,
    NavigationSettings,
    build_parser,
    discover_manifest,
)
from kiwi_navigation_core import DEFAULT_MAX_FOLLOWING_SPEED_MPS  # noqa: E402


class ImageMapDatasetTests(unittest.TestCase):
    @staticmethod
    def write_dataset(directory, session_id="test-session", x=1.25):
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "000007.jpg").write_bytes(b"jpeg")
        manifest = {
            "format": "kiwi-image-map-v1",
            "frame": "map",
            "session_id": session_id,
            "created_at": "2026-08-23T12:00:00Z",
            "captures": [{
                "id": 7,
                "image": "000007.jpg",
                "time_s": 123.0,
                "pose": {"x": x, "y": -0.5, "yaw": math.pi / 2.0},
            }],
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_loads_images_and_map_frame_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_dataset(directory)

            dataset = ImageMapDataset.load(manifest)

            capture = dataset.capture(7)
            self.assertEqual(dataset.session_id, "test-session")
            self.assertEqual(
                capture.image_path,
                (pathlib.Path(directory) / "000007.jpg").resolve(),
            )
            self.assertAlmostEqual(capture.x, 1.25)
            self.assertAlmostEqual(capture.yaw, math.pi / 2.0)

    def test_rejects_image_paths_outside_the_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            session = root / "session"
            manifest = self.write_dataset(session)
            (root / "outside.jpg").write_bytes(b"jpeg")
            document = json.loads(manifest.read_text())
            document["captures"][0]["image"] = "../outside.jpg"
            manifest.write_text(json.dumps(document))

            with self.assertRaisesRegex(ImageMapError, "escapes"):
                ImageMapDataset.load(manifest)

    def test_discovers_most_recently_updated_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            older = self.write_dataset(root / "older", session_id="older")
            time.sleep(0.002)
            newer = self.write_dataset(root / "newer", session_id="newer")

            self.assertEqual(discover_manifest(root), newer.resolve())
            self.assertNotEqual(discover_manifest(root), older.resolve())


class NavigationCommandTests(unittest.TestCase):
    def test_uses_the_shared_trajectory_following_speed_cap(self):
        args = build_parser().parse_args([])

        self.assertEqual(
            args.max_linear_speed, DEFAULT_MAX_FOLLOWING_SPEED_MPS)

    def test_selected_capture_pose_becomes_navigation_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = ImageMapDatasetTests.write_dataset(directory)
            capture = ImageMapDataset.load(manifest).capture(7)
            settings = NavigationSettings(
                connect="tcp/robot:7447",
                namespace="kiwi/sim",
                robot_yaw_deg=60.0,
                inflation_radius=0.25,
                allow_unknown=True,
                lookahead=0.30,
                max_linear_speed=0.20,
                max_angular_speed=0.80,
                position_tolerance=0.04,
                yaw_tolerance_deg=3.0,
                replan_distance=0.35,
                max_duration=60.0,
            )

            command = settings.command(capture)

            self.assertEqual(command[command.index("--goal-yaw-deg") + 1], "90.0")
            self.assertEqual(command[command.index("--namespace") + 1], "kiwi/sim")
            self.assertEqual(command[3:5], ["1.25", "-0.5"])
            self.assertIn("--allow-unknown", command)
            self.assertEqual(
                command[command.index("--command-topic") + 1], "cmd_vel")
            self.assertEqual(
                float(command[
                    command.index("--runtime-collision-radius") + 1]),
                settings.runtime_collision_radius,
            )
            self.assertEqual(
                float(command[command.index("--kp-yaw") + 1]),
                settings.kp_yaw,
            )
            self.assertEqual(
                float(command[
                    command.index("--goal-yaw-blend-distance") + 1]),
                settings.goal_yaw_blend_distance,
            )

    def test_agent_action_and_distance_budget_reach_the_navigator(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = ImageMapDatasetTests.write_dataset(directory)
            capture = ImageMapDataset.load(manifest).capture(7)
            settings = NavigationSettings(
                connect="tcp/robot:7447", namespace="kiwi/sim",
                robot_yaw_deg=60.0, inflation_radius=0.25,
                allow_unknown=False, lookahead=0.30,
                max_linear_speed=0.20, max_angular_speed=0.80,
                position_tolerance=0.04, yaw_tolerance_deg=3.0,
                replan_distance=0.35, max_duration=60.0,
            )

            command = settings.command(
                capture, action_id="action-1", max_travel_distance_m=2.5)

            self.assertEqual(
                command[command.index("--action-id") + 1], "action-1")
            self.assertEqual(
                command[command.index("--max-travel-distance") + 1], "2.5")

    def test_replanned_route_that_exceeds_remaining_budget_requests_stop(self):
        settings = mock.Mock()
        manager = NavigationManager(settings)
        manager._state.update(
            phase="running", action_id="action-1",
            max_travel_distance_m=1.0,
            distance_traveled_m=0.45,
        )

        with mock.patch.object(manager, "_request_safety_stop") as stop:
            manager.observe_navigation_state({
                "action_id": "action-1",
                "status": "following",
                "remaining_m": 0.60,
            })

        stop.assert_called_once_with(
            "replanned route exceeds the remaining travel budget")

    def test_slam_pose_correction_is_not_counted_as_physical_travel(self):
        manager = NavigationManager(mock.Mock())
        manager._state.update(
            phase="running", action_id="action-1",
            max_travel_distance_m=2.0,
            distance_traveled_m=0.25,
        )

        manager.observe_pose({"x": 0.0, "y": 0.0})
        manager.observe_pose({"x": 1.6, "y": 0.3})
        manager.observe_navigation_state({
            "action_id": "action-1",
            "status": "following",
            "distance_traveled_m": 0.31,
            "remaining_m": 1.0,
        })

        self.assertAlmostEqual(manager.snapshot()["distance_traveled_m"], 0.31)

    def test_teleop_preemption_requests_navigation_stop(self):
        manager = NavigationManager(mock.Mock())
        with mock.patch.object(manager, "_request_safety_stop") as stop:
            manager.observe_mux_status({"source": "teleop"})
        stop.assert_called_once_with("teleop took priority")


class LiveSlamMonitorTests(unittest.TestCase):
    def test_waits_for_verified_relocalization_before_driving(self):
        monitor = LiveSlamMonitor.__new__(LiveSlamMonitor)
        monitor.skip_session_check = False
        monitor._lock = threading.RLock()
        monitor._client = type("Client", (), {
            "pose_received_at": time.monotonic(),
        })()
        monitor._connection_error = None
        monitor._map_seen_at = time.monotonic()
        monitor._image_seen_at = time.monotonic()
        monitor._image_session_id = "test-session"
        monitor._scan_matched = False

        waiting = monitor.status("test-session")
        monitor._on_slam({
            "quality": {"scan_matched": True, "relocalizing": False},
        })
        ready = monitor.status("test-session")

        self.assertFalse(waiting["ready"])
        self.assertEqual(waiting["status_code"], "relocalizing")
        self.assertEqual(
            waiting["reason"], "waiting for verified SLAM relocalization")
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["status_code"], "ready")
        self.assertIn("preview required", ready["reason"])

    def test_session_mismatch_has_structured_recovery(self):
        monitor = LiveSlamMonitor.__new__(LiveSlamMonitor)
        monitor.skip_session_check = False
        monitor.require_mux = False
        monitor._lock = threading.RLock()
        monitor._client = type("Client", (), {
            "pose_received_at": time.monotonic(),
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        })()
        monitor._connection_error = None
        monitor._map_seen_at = time.monotonic()
        monitor._image_seen_at = time.monotonic()
        monitor._image_session_id = "live-session"
        monitor._scan_matched = True
        monitor._quality = {"scan_matched": True, "relocalizing": False}
        monitor._mux_status = None
        monitor._mux_seen_at = None

        status = monitor.status("saved-session")

        self.assertFalse(status["ready"])
        self.assertEqual(status["status_code"], "session_mismatch")
        self.assertEqual(status["recovery"]["code"], "session_mismatch")
        self.assertEqual(
            status["recovery"]["expected_session_id"], "saved-session")


if __name__ == "__main__":
    unittest.main()
