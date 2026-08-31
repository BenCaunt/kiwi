import json
import io
import math
import pathlib
import struct
import sys
import tempfile
import time
import types
import unittest

import numpy as np
from PIL import Image


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_agent_gateway import (  # noqa: E402
    AgentGatewayError,
    ClipPlaceIndex,
    KiwiAgentGateway,
    capture_ref,
    parse_capture_ref,
    render_pose_map,
)
from kiwi_map import LiveOccupancyMap  # noqa: E402


class Destination:
    def __init__(self, capture_id, image_path, x, y, yaw=0.0):
        self.capture_id = capture_id
        self.image_path = pathlib.Path(image_path)
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)


class Dataset:
    def __init__(self, root, captures, session_id="session-a"):
        self.manifest_path = pathlib.Path(root) / "manifest.json"
        self.manifest_path.write_text("{}", encoding="utf-8")
        self.captures = tuple(captures)
        self.session_id = session_id

    def capture(self, capture_id):
        for capture in self.captures:
            if capture.capture_id == capture_id:
                return capture
        raise KeyError(capture_id)


class FakeEncoder:
    model_name = "fake-clip"
    pretrained = "tests"
    preprocessing_version = "fake-v1"

    def __init__(self):
        self.image_calls = 0

    def encode_images(self, paths):
        self.image_calls += len(paths)
        values = []
        for path in paths:
            values.append([1.0, 0.0] if path.stem == "one" else [0.0, 1.0])
        return np.asarray(values, dtype=np.float32)

    def encode_text(self, text):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeLive:
    def __init__(self, occupancy, pose, *, ready=True, reason="ready to drive"):
        self.occupancy = occupancy
        self.pose = dict(pose)
        self.ready = ready
        self.reason = reason
        self.pose_callbacks = []
        self.navigation_callbacks = []
        self.mux_callbacks = []
        self.camera_callbacks = []
        self.live_session_id = None

    def status(self, session_id):
        return {
            "ready": self.ready,
            "reason": self.reason,
            "expected_session_id": session_id,
            "live_session_id": self.live_session_id or session_id,
            "pose": dict(self.pose),
            "pose_age_s": 0.02,
            "map_age_s": 0.03,
            "quality": {"scan_matched": True, "relocalizing": False},
            "mux_source": "idle",
        }

    def snapshot(self, session_id):
        return {
            "status": self.status(session_id),
            "occupancy": self.occupancy,
            "pose": dict(self.pose),
            "pose_age_s": 0.02,
            "map_age_s": 0.03,
            "quality": {"scan_matched": True},
            "trajectory": [],
            "navigation_state": None,
        }

    def add_pose_callback(self, callback):
        self.pose_callbacks.append(callback)

    def add_navigation_callback(self, callback):
        self.navigation_callbacks.append(callback)

    def add_mux_callback(self, callback):
        self.mux_callbacks.append(callback)

    def add_camera_callback(self, callback):
        self.camera_callbacks.append(callback)


class FakeNavigation:
    def __init__(self):
        self.settings = types.SimpleNamespace(
            inflation_radius=0.0,
            runtime_collision_radius=0.0,
            max_linear_speed=0.2,
        )
        self.state = {"phase": "idle", "action_id": None}
        self.started = None

    def snapshot(self):
        return dict(self.state)

    def start(self, destination, **kwargs):
        self.started = destination, kwargs
        self.state = {"phase": "running", "action_id": kwargs["action_id"]}

    def stop(self, **kwargs):
        active = self.state["phase"] == "running"
        self.state["phase"] = "stopped"
        return active

    def observe_pose(self, pose):
        pass

    def observe_navigation_state(self, state):
        pass

    def observe_mux_status(self, state):
        pass


class CaptureReferenceTests(unittest.TestCase):
    def test_round_trips_session_scoped_reference(self):
        value = capture_ref("20260823T195947Z", 152)
        self.assertEqual(value, "20260823T195947Z:152")
        self.assertEqual(parse_capture_ref(value), ("20260823T195947Z", 152))

    def test_rejects_unscoped_capture_id(self):
        with self.assertRaisesRegex(AgentGatewayError, "session_id"):
            parse_capture_ref("152")


class ClipPlaceIndexTests(unittest.TestCase):
    def test_persists_and_reuses_checksum_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            one = root / "one.jpg"
            two = root / "two.jpg"
            one.write_bytes(b"one")
            two.write_bytes(b"two")
            dataset = Dataset(root, [
                Destination(1, one, 0.0, 0.0),
                Destination(2, two, 1.0, 0.0),
            ])
            encoder = FakeEncoder()
            index = ClipPlaceIndex(encoder)

            results = index.search(dataset, "first place", 2)

            self.assertEqual(results[0][0].capture_id, 1)
            self.assertEqual(encoder.image_calls, 2)
            self.assertTrue((root / "clip-index-v1.json").is_file())
            metadata = json.loads((root / "clip-index-v1.json").read_text())
            self.assertEqual(metadata["session_id"], "session-a")

            second_encoder = FakeEncoder()
            second = ClipPlaceIndex(second_encoder)
            renumbered = Dataset(root, [
                Destination(11, one, 0.0, 0.0),
                Destination(12, two, 1.0, 0.0),
            ])
            second.search(renumbered, "first place", 1)
            self.assertEqual(second_encoder.image_calls, 0)


class MapRenderingTests(unittest.TestCase):
    def test_renders_long_heading_arrow_and_scale(self):
        occupancy = LiveOccupancyMap(
            data=np.zeros((40, 60), dtype=np.int8),
            resolution_m=0.1,
            origin_x=-3.0,
            origin_y=-2.0,
            keyframes=8,
        )

        png, metadata = render_pose_map(
            occupancy, {"x": 0.0, "y": 0.0, "yaw": math.pi / 3.0})

        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreaterEqual(image.width, 60)
        self.assertGreaterEqual(metadata["arrow_length_m"], 0.75)
        self.assertEqual(metadata["keyframes"], 8)


class GatewayPlanningTests(unittest.TestCase):
    @staticmethod
    def camera_payload(sequence=1):
        encoded = io.BytesIO()
        Image.new("RGB", (64, 48), (50 + sequence, 80, 120)).save(
            encoded, format="JPEG")
        jpeg = encoded.getvalue()
        header = bytearray(32)
        header[:4] = b"KVC1"
        struct.pack_into("<HHH", header, 6, 64, 48, len(header))
        struct.pack_into("<I", header, 12, sequence)
        struct.pack_into("<Q", header, 16, sequence * 1000)
        struct.pack_into("<I", header, 24, len(jpeg))
        return bytes(header) + jpeg

    def make_gateway(self, directory, *, occupancy_data=None, ready=True):
        root = pathlib.Path(directory)
        image = root / "goal.jpg"
        image.write_bytes(b"jpeg")
        destination = Destination(7, image, 0.8, 0.0)
        dataset = Dataset(root, [destination])
        store = types.SimpleNamespace(snapshot=lambda: dataset)
        occupancy = LiveOccupancyMap(
            data=(np.zeros((60, 60), dtype=np.int8)
                  if occupancy_data is None else occupancy_data),
            resolution_m=0.1,
            origin_x=-3.0,
            origin_y=-3.0,
            keyframes=12,
        )
        live = FakeLive(
            occupancy, {"x": 0.0, "y": 0.0, "yaw": 0.0}, ready=ready,
            reason=("ready to drive" if ready else "pose is stale"))
        navigation = FakeNavigation()
        gateway = KiwiAgentGateway(
            store, navigation, live,
            clip_index=ClipPlaceIndex(FakeEncoder()),
            preview_ttl_s=30.0,
            max_action_distance_m=2.0,
            trace_frame_interval_s=0.0,
        )
        return gateway, navigation

    def test_preview_then_execute_replans_and_starts_budgeted_action(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, navigation = self.make_gateway(directory)

            preview = gateway.preview_image_goal("session-a:7", 1.5)
            started = gateway.navigate_to_image(
                preview.structured["preview_id"])

            self.assertTrue(preview.structured["safe_to_start"])
            self.assertIn("expires_at", preview.structured)
            self.assertGreater(preview.structured["planned_path_distance_m"], 0.0)
            self.assertEqual(len(preview.images), 2)
            self.assertEqual(started.structured["phase"], "running")
            self.assertEqual(
                navigation.started[1]["max_travel_distance_m"], 1.5)
            self.assertEqual(
                navigation.started[1]["capture_ref"], "session-a:7")

    def test_preview_reports_budget_blocker_without_moving(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, navigation = self.make_gateway(directory)

            preview = gateway.preview_image_goal("session-a:7", 0.2)

            self.assertFalse(preview.structured["safe_to_start"])
            self.assertTrue(any("exceeds" in item
                                for item in preview.structured["blockers"]))
            with self.assertRaisesRegex(AgentGatewayError, "revalidation failed"):
                gateway.navigate_to_image(preview.structured["preview_id"])
            self.assertIsNone(navigation.started)

    def test_unknown_cells_remain_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            unknown = np.full((60, 60), -1, dtype=np.int8)
            gateway, _navigation = self.make_gateway(
                directory, occupancy_data=unknown)

            preview = gateway.preview_image_goal("session-a:7", 1.5)

            self.assertFalse(preview.structured["safe_to_start"])
            self.assertTrue(any("start" in item or "goal" in item
                                for item in preview.structured["blockers"]))

    def test_preview_allows_hard_safe_egress_from_soft_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            data = np.zeros((60, 60), dtype=np.int8)
            # Pose is at cell (30, 30); this obstacle is 0.224 m away.
            data[29, 32] = 100
            gateway, navigation = self.make_gateway(
                directory, occupancy_data=data)
            navigation.settings.inflation_radius = 0.25
            navigation.settings.runtime_collision_radius = 0.18

            preview = gateway.preview_image_goal("session-a:7", 1.5)

            self.assertTrue(preview.structured["safe_to_start"])
            self.assertTrue(preview.structured["soft_start_recovery"])
            self.assertEqual(
                preview.structured["planning_inflation_radius_m"], 0.25)
            self.assertEqual(
                preview.structured["runtime_collision_radius_m"], 0.18)
            self.assertGreater(
                preview.structured["planned_path_distance_m"], 0.8)

    def test_rejects_capture_from_an_inactive_session(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, _navigation = self.make_gateway(directory)
            with self.assertRaisesRegex(AgentGatewayError, "active session"):
                gateway.preview_image_goal("old-session:7", 1.5)

    def test_status_returns_machine_readable_session_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, _navigation = self.make_gateway(directory, ready=False)
            gateway.live.live_session_id = "different-session"
            gateway.live.reason = (
                "manifest session session-a does not match live session "
                "different-session")

            status = gateway.get_robot_status().structured

            self.assertFalse(status["ready"])
            self.assertEqual(status["live"]["status_code"], "session_mismatch")
            self.assertEqual(status["recovery"]["code"], "session_mismatch")
            self.assertEqual(
                status["recovery"]["manifest_path"],
                str(gateway.dataset_store.snapshot().manifest_path))

    def test_navigation_report_uses_only_camera_and_live_slam_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, navigation = self.make_gateway(directory)
            preview = gateway.preview_image_goal("session-a:7", 1.5)
            started = gateway.navigate_to_image(preview.structured["preview_id"])
            action_id = started.structured["action_id"]

            for index in range(5):
                pose = {
                    "x": 0.2 * index,
                    "y": 0.02 * index,
                    "yaw": 0.0,
                }
                for callback in gateway.live.pose_callbacks:
                    callback(pose)
                for callback in gateway.live.camera_callbacks:
                    callback(self.camera_payload(index + 1))
            navigation.state = {
                "phase": "succeeded",
                "action_id": action_id,
                "distance_traveled_m": 0.81,
                "remaining_path_m": 0.01,
                "cross_track_error_m": 0.02,
            }

            report = gateway.get_navigation_report(
                action_id, frame_count=4, brightness_gain=1.2)

            self.assertEqual(report.structured["phase"], "succeeded")
            self.assertEqual(report.structured["pose_count"], 6)
            self.assertEqual(report.structured["camera_frame_count"], 5)
            self.assertEqual(report.structured["selected_frame_count"], 4)
            self.assertFalse(report.structured["simulator_ground_truth_used"])
            self.assertEqual(
                report.structured["evidence_sources"],
                ["camera/jpeg", "slam/pose", "slam/map"])
            self.assertIn("navigator_message", report.structured)
            self.assertIn("logs", report.structured)
            self.assertEqual(len(report.images), 2)
            for attachment in report.images:
                with Image.open(io.BytesIO(attachment.data)) as rendered:
                    self.assertEqual(rendered.format, "PNG")

    def test_readiness_watchdog_terminates_an_active_action(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, navigation = self.make_gateway(directory)
            gateway.close()
            navigation.state = {"phase": "running", "action_id": "action-1"}
            gateway.live.ready = False
            gateway.live.reason = "manifest session mismatch"
            watched = KiwiAgentGateway(
                gateway.dataset_store, navigation, gateway.live,
                clip_index=ClipPlaceIndex(FakeEncoder()),
                watchdog_interval_s=0.01,
            )
            try:
                deadline = time.monotonic() + 1.0
                while (navigation.state["phase"] == "running" and
                       time.monotonic() < deadline):
                    time.sleep(0.01)
                self.assertEqual(navigation.state["phase"], "stopped")
            finally:
                watched.close()


if __name__ == "__main__":
    unittest.main()
