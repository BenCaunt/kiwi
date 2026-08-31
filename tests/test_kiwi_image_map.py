import json
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np


sys_path = str(pathlib.Path(__file__).parents[1] / "scripts")
sys.path.insert(0, sys_path)

from kiwi_image_map import (  # noqa: E402
    CameraSample,
    ImageMapRecorder,
    decode_image_capture,
    discover_compatible_image_manifest,
    encode_image_capture,
)
from kiwi_slam_core import Keyframe, Pose2  # noqa: E402


class ImageCaptureWireTests(unittest.TestCase):
    def test_round_trips_metadata_and_jpeg(self):
        metadata = {"id": 4, "pose": {"x": 1.0, "y": 2.0, "yaw": 0.3}}
        encoded = encode_image_capture(metadata, b"jpeg")

        decoded, jpeg = decode_image_capture(encoded)

        self.assertEqual(decoded, metadata)
        self.assertEqual(jpeg, b"jpeg")

    def test_rejects_truncated_capture(self):
        with self.assertRaises(ValueError):
            decode_image_capture(encode_image_capture({"id": 1}, b"jpeg")[:-1])


class ImageMapRecorderTests(unittest.TestCase):
    @staticmethod
    def sample(sequence=1):
        return CameraSample(
            width=320,
            height=240,
            sequence=sequence,
            sensor_time_us=123_000,
            pixels=np.zeros((240, 320, 3), dtype=np.uint8),
            jpeg=b"jpeg bytes",
        )

    def test_records_only_after_translation_or_rotation_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ImageMapRecorder(
                pathlib.Path(directory) / "map",
                translation_spacing_m=0.5,
                rotation_spacing_rad=math.radians(30.0),
                min_interval_s=0.25,
                session_id="test",
            )
            first = recorder.capture(
                self.sample(), Pose2(), Pose2(), 1.0,
                wall_time_s=10.0, monotonic_s=1.0,
            )
            too_close = recorder.capture(
                self.sample(2), Pose2(0.49, 0.0, 0.0),
                Pose2(0.49, 0.0, 0.0), 2.0,
                wall_time_s=11.0, monotonic_s=2.0,
            )
            rotated = recorder.capture(
                self.sample(3), Pose2(0.0, 0.0, math.radians(31.0)),
                Pose2(0.0, 0.0, math.radians(31.0)), 3.0,
                wall_time_s=12.0, monotonic_s=3.0,
            )

            self.assertIsNotNone(first)
            self.assertIsNone(too_close)
            self.assertIsNotNone(rotated)
            self.assertEqual(len(recorder), 2)
            manifest = json.loads(recorder.manifest_path.read_text())
            self.assertEqual(manifest["format"], "kiwi-image-map-v1")
            self.assertEqual(len(manifest["captures"]), 2)
            self.assertTrue((recorder.image_dir / "000000.jpg").exists())

    def test_read_only_resume_republishes_without_mutating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            recorder = ImageMapRecorder(prefix, session_id="saved-session")
            recorder.capture(
                self.sample(), Pose2(), Pose2(), 1.0, monotonic_s=1.0)
            manifest_before = recorder.manifest_path.read_bytes()
            files_before = sorted(path.name for path in recorder.image_dir.iterdir())

            resumed = ImageMapRecorder(
                prefix,
                resume_manifest=recorder.manifest_path,
                read_only=True,
            )
            resumed.activate()

            self.assertTrue(resumed.active)
            self.assertFalse(resumed.should_capture(
                Pose2(2.0, 0.0, 0.0), monotonic_s=2.0))
            self.assertIsNone(resumed.capture(
                self.sample(2), Pose2(2.0, 0.0, 0.0),
                Pose2(2.0, 0.0, 0.0), 2.0, monotonic_s=2.0))
            self.assertEqual(resumed.reproject([]), [])
            self.assertEqual(recorder.manifest_path.read_bytes(), manifest_before)
            self.assertEqual(
                sorted(path.name for path in recorder.image_dir.iterdir()),
                files_before,
            )

    def test_read_only_image_map_requires_resumed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "resumed manifest"):
                ImageMapRecorder(
                    pathlib.Path(directory) / "map", read_only=True)

    def test_reprojects_capture_through_nearest_optimized_keyframe(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ImageMapRecorder(
                pathlib.Path(directory) / "map", session_id="test"
            )
            recorder.capture(
                self.sample(),
                raw_pose=Pose2(1.5, 0.0, 0.0),
                map_pose=Pose2(1.5, 0.0, 0.0),
                pose_time_s=2.0,
                monotonic_s=1.0,
            )
            keyframe = Keyframe(
                index=3,
                time_s=1.9,
                raw_pose=Pose2(1.0, 0.0, 0.0),
                pose=Pose2(4.0, 2.0, math.pi / 2.0),
                points=np.empty((0, 2)),
                descriptor=np.empty(0),
                match_score=1.0,
            )

            recorder.reproject([keyframe])

            pose = recorder.records[0]["pose"]
            self.assertAlmostEqual(pose["x"], 4.0)
            self.assertAlmostEqual(pose["y"], 2.5)
            self.assertAlmostEqual(pose["yaw"], math.pi / 2.0)
            self.assertEqual(recorder.records[0]["nearest_keyframe"], 3)

    def test_seal_prevents_captures_after_shutdown_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ImageMapRecorder(
                pathlib.Path(directory) / "map", session_id="test"
            )
            recorder.capture(
                self.sample(), Pose2(), Pose2(), 1.0, monotonic_s=1.0)

            recorder.seal()

            self.assertFalse(recorder.should_capture(
                Pose2(1.0, 0.0, 0.0), monotonic_s=2.0))
            self.assertIsNone(recorder.capture(
                self.sample(2), Pose2(1.0, 0.0, 0.0),
                Pose2(1.0, 0.0, 0.0), 2.0, monotonic_s=2.0,
            ))
            self.assertEqual(len(recorder), 1)

    def test_resumes_old_captures_and_tracks_slam_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            old = ImageMapRecorder(prefix, session_id="old-captures")
            old.capture(
                self.sample(),
                raw_pose=Pose2(1.5, 0.0, 0.0),
                map_pose=Pose2(1.5, 0.0, 0.0),
                pose_time_s=2.0,
                monotonic_s=1.0,
            )
            old_keyframe = Keyframe(
                index=3,
                time_s=1.9,
                raw_pose=Pose2(1.0, 0.0, 0.0),
                pose=Pose2(4.0, 2.0, math.pi / 2.0),
                points=np.empty((0, 2)),
                descriptor=np.empty(0),
                match_score=1.0,
                session_id=0,
            )
            old.reproject([old_keyframe])
            document = json.loads(old.manifest_path.read_text())
            document["captures"][0].pop("slam_session_id")
            old.manifest_path.write_text(json.dumps(document))

            resumed = ImageMapRecorder(
                prefix,
                resume_manifest=old.manifest_path,
                slam_session_id=1,
                keyframe_sessions={3: 0},
            )

            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed.records[0]["slam_session_id"], 0)
            self.assertEqual(
                json.loads(old.manifest_path.read_text())["session_id"],
                "old-captures",
            )
            resumed.activate()
            migrated = json.loads(old.manifest_path.read_text())
            self.assertEqual(migrated["session_id"], "old-captures")

            captured = resumed.capture(
                self.sample(2),
                raw_pose=Pose2(0.5, 0.0, 0.0),
                map_pose=Pose2(10.5, 0.0, 0.0),
                pose_time_s=10.5,
                monotonic_s=2.0,
            )
            self.assertIsNotNone(captured)
            self.assertEqual(captured[0]["id"], 1)
            self.assertEqual(captured[0]["slam_session_id"], 1)
            new_keyframe = Keyframe(
                index=4,
                time_s=10.4,
                raw_pose=Pose2(0.0, 0.0, 0.0),
                pose=Pose2(10.0, 0.0, 0.0),
                points=np.empty((0, 2)),
                descriptor=np.empty(0),
                match_score=1.0,
                session_id=1,
            )
            resumed.reproject([old_keyframe, new_keyframe])

            self.assertEqual(resumed.records[0]["nearest_keyframe"], 3)
            self.assertEqual(resumed.records[1]["nearest_keyframe"], 4)
            metadata, _jpeg = decode_image_capture(resumed.packet(0))
            self.assertEqual(metadata["session_id"], "old-captures")

    def test_discovers_only_manifest_compatible_with_saved_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            keyframe = Keyframe(
                index=0,
                time_s=1.0,
                raw_pose=Pose2(),
                pose=Pose2(),
                points=np.empty((0, 2)),
                descriptor=np.empty(0),
                match_score=1.0,
                session_id=0,
            )
            compatible = ImageMapRecorder(prefix, session_id="compatible")
            compatible.capture(
                self.sample(), Pose2(), Pose2(), 1.0, monotonic_s=1.0)
            compatible.reproject([keyframe])
            incompatible = ImageMapRecorder(prefix, session_id="newer-bad")
            incompatible.capture(
                self.sample(), Pose2(), Pose2(4.0, 0.0, 0.0), 20.0,
                monotonic_s=2.0,
            )

            discovered = discover_compatible_image_manifest(
                prefix, [keyframe])

            self.assertEqual(discovered, compatible.manifest_path.resolve())

    def test_discovers_manifest_with_trailing_unanchored_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            keyframe = Keyframe(
                index=0,
                time_s=1.0,
                raw_pose=Pose2(),
                pose=Pose2(),
                points=np.empty((0, 2)),
                descriptor=np.empty(0),
                match_score=1.0,
                session_id=0,
            )
            recorder = ImageMapRecorder(prefix, session_id="saved")
            recorder.capture(
                self.sample(), Pose2(), Pose2(), 1.0, monotonic_s=1.0)
            recorder.reproject([keyframe])
            recorder.capture(
                self.sample(2), Pose2(1.0, 0.0, 0.0),
                Pose2(1.0, 0.0, 0.0), 20.0, monotonic_s=2.0,
            )

            discovered = discover_compatible_image_manifest(
                prefix, [keyframe])

            self.assertEqual(discovered, recorder.manifest_path.resolve())
            resumed = ImageMapRecorder(
                prefix,
                resume_manifest=discovered,
                slam_session_id=1,
                keyframe_sessions={0: 0},
            )
            self.assertEqual(len(resumed), 2)
            self.assertIsNone(resumed.records[1]["nearest_keyframe"])

            # Once shutdown reprojection anchors that delayed capture, the
            # explicit graph-session ID remains sufficient proof that it
            # belongs to this map even though its nearest keyframe is old.
            recorder.reproject([keyframe])
            self.assertEqual(
                discover_compatible_image_manifest(prefix, [keyframe]),
                recorder.manifest_path.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
