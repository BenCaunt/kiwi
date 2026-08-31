import json
import io
import math
import pathlib
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from PIL import Image


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_slam_core import (  # noqa: E402
    Constraint,
    Keyframe,
    MatchResult,
    Pose2,
    PoseGraphSlam,
    SlamConfig,
    between,
    compose,
    inverse,
    transform_points,
)
from kiwi_slam import decode_camera_frame  # noqa: E402
from kiwi_slam import SlamRunner  # noqa: E402
from kiwi_yaw_estimator import YawEstimator  # noqa: E402


def room_points():
    x = np.arange(-3.0, 3.001, 0.04)
    y = np.arange(-2.0, 2.001, 0.04)
    return np.vstack((
        np.column_stack((x, np.full_like(x, -2.0))),
        np.column_stack((x, np.full_like(x, 2.0))),
        np.column_stack((np.full_like(y, -3.0), y)),
        np.column_stack((np.full_like(y, 3.0), y)),
    ))


class CameraFrameTests(unittest.TestCase):
    def test_decodes_and_rotates_camera_pov_upright(self):
        source = np.array((
            ((255, 0, 0), (0, 255, 0)),
            ((0, 0, 255), (255, 255, 0)),
        ), dtype=np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(source, "RGB").save(encoded, format="PNG")
        header = bytearray(32)
        header[:4] = b"KVC1"
        struct.pack_into("<H", header, 10, len(header))

        pixels = decode_camera_frame(bytes(header) + encoded.getvalue())

        np.testing.assert_array_equal(pixels, np.rot90(source, 2))

    def test_rejects_bad_camera_header_and_image(self):
        self.assertIsNone(decode_camera_frame(b"not a camera frame"))
        header = bytearray(32)
        header[:4] = b"KVC1"
        struct.pack_into("<H", header, 10, len(header))
        self.assertIsNone(decode_camera_frame(bytes(header) + b"not jpeg"))


class PoseMathTests(unittest.TestCase):
    def test_compose_between_and_inverse_are_consistent(self):
        a = Pose2(1.2, -0.4, math.radians(35.0))
        delta = Pose2(0.7, 0.2, math.radians(-18.0))
        b = compose(a, delta)

        recovered = between(a, b)
        identity = compose(a, inverse(a))

        self.assertAlmostEqual(recovered.x, delta.x)
        self.assertAlmostEqual(recovered.y, delta.y)
        self.assertAlmostEqual(recovered.yaw, delta.yaw)
        self.assertAlmostEqual(identity.x, 0.0)
        self.assertAlmostEqual(identity.y, 0.0)
        self.assertAlmostEqual(identity.yaw, 0.0)


class ScanMatchingTests(unittest.TestCase):
    def test_ransac_wall_weighting_keeps_clutter_but_prefers_long_lines(self):
        config = SlamConfig(
            wall_point_weight=3.0,
            wall_line_min_points=8,
            wall_line_min_length_m=0.45,
            wall_line_distance_m=0.03,
        )
        slam = PoseGraphSlam(config)
        wall = np.column_stack((
            np.linspace(-2.0, 2.0, 80),
            np.full(80, 1.0),
        ))
        clutter = np.array((
            (2.8, -1.4), (1.7, -0.2), (0.9, -1.5),
            (-0.4, 0.1), (-1.5, -0.8), (-2.7, 0.2),
        ))
        points = np.vstack((wall, clutter))

        weights = slam.matcher._structural_weights(points)

        self.assertGreater(np.mean(weights[:len(wall)] > 1.0), 0.9)
        np.testing.assert_array_equal(weights[-len(clutter):], 1.0)
        diagnostics = slam.matcher.last_structural_diagnostics
        self.assertGreater(diagnostics.support_ratio, 0.85)
        self.assertGreater(diagnostics.supported_line_length_m, 3.5)
        self.assertTrue(diagnostics.orientations_rad)

    def test_scan_matching_reduces_drifting_odometry_error(self):
        world = room_points()
        local_scan = lambda pose: transform_points(inverse(pose), world)
        config = SlamConfig(
            keyframe_translation_m=0.05,
            min_match_score=0.25,
            min_hit_ratio=0.2,
        )
        slam = PoseGraphSlam(config)
        slam.process_scan(local_scan(Pose2()), Pose2(), 0.0)
        true_pose = Pose2(0.35, 0.20, 0.08)
        drifting_odom = Pose2(0.50, 0.05, 0.18)

        result = slam.process_scan(
            local_scan(true_pose), drifting_odom, 1.0)

        raw_error = between(true_pose, drifting_odom)
        slam_error = between(true_pose, result.pose)
        self.assertTrue(result.scan_matched)
        self.assertGreater(result.match_score, 0.7)
        self.assertLess(math.hypot(slam_error.x, slam_error.y),
                        math.hypot(raw_error.x, raw_error.y) * 0.6)
        self.assertLess(abs(slam_error.yaw), abs(raw_error.yaw) * 0.3)

    def test_scan_descriptor_recovers_heading_change(self):
        slam = PoseGraphSlam()
        world = room_points()
        reference = slam._descriptor(world)
        rotated_scan = transform_points(
            inverse(Pose2(yaw=math.radians(30.0))), world)
        query = slam._descriptor(rotated_scan)

        score, shift = slam._descriptor_similarity(reference, query)

        recovered_yaw = shift * 2.0 * math.pi / slam.config.descriptor_bins
        self.assertGreater(score, 0.95)
        self.assertAlmostEqual(recovered_yaw, math.radians(30.0), places=6)

    def test_absolute_imu_prediction_prevents_frontend_yaw_accumulation(self):
        slam = PoseGraphSlam(SlamConfig(
            min_scan_points=4,
            keyframe_translation_m=0.05,
            absolute_imu_heading_prediction=True,
        ))
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        slam.process_scan(points, Pose2(), 0.0)

        def biased_match(_points, initial_pose, _field, **_kwargs):
            return MatchResult(
                Pose2(initial_pose.x, initial_pose.y,
                      initial_pose.yaw + math.radians(2.0)),
                0.9, 0.9, 0.02, True)

        with mock.patch.object(slam.matcher, "match",
                               side_effect=biased_match):
            for index in range(1, 6):
                result = slam.process_scan(
                    points, Pose2(0.2 * index, 0.0, 0.0), float(index))

        self.assertAlmostEqual(
            math.degrees(result.heading_disagreement_rad), 2.0, places=6)

    def test_rejected_scan_does_not_poison_keyframes_or_constraints(self):
        slam = PoseGraphSlam(SlamConfig(
            min_scan_points=4,
            keyframe_translation_m=0.05,
        ))
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        slam.process_scan(points, Pose2(), 0.0)
        rejected = MatchResult(
            Pose2(0.2, 0.0, 0.0), 0.1, 0.1, 0.5, False)

        with mock.patch.object(slam.matcher, "match",
                               return_value=rejected):
            result = slam.process_scan(
                points, Pose2(0.2, 0.0, 0.0), 1.0)

        self.assertFalse(result.scan_matched)
        self.assertFalse(result.keyframe_added)
        self.assertEqual(result.keyframes, 1)
        self.assertEqual(slam.constraints, [])


class PoseGraphTests(unittest.TestCase):
    @staticmethod
    def keyframe(index, x, travel_m=0.0):
        points = np.array(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        return Keyframe(index, float(index), Pose2(x, 0.0, 0.0),
                        Pose2(x, 0.0, 0.0), points,
                        np.full(72, np.nan), 1.0, travel_m)

    def test_loop_edge_distributes_accumulated_graph_drift(self):
        slam = PoseGraphSlam()
        slam.keyframes = [self.keyframe(i, 1.1 * i) for i in range(4)]
        for i in range(3):
            slam.constraints.append(Constraint(
                i, i + 1, Pose2(1.1, 0.0, 0.0),
                0.10, 0.1, "odometry"))
        slam.constraints.append(Constraint(
            0, 3, Pose2(3.0, 0.0, 0.0),
            0.02, 0.02, "loop_closure"))

        slam.optimize()

        self.assertLess(abs(slam.keyframes[-1].pose.x - 3.0), 0.03)
        self.assertGreater(slam.keyframes[1].pose.x, 1.0)
        self.assertLess(slam.keyframes[1].pose.x, 1.1)

    def test_keyframe_travel_ignores_stationary_timed_keyframes(self):
        slam = PoseGraphSlam()
        points = np.array(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)

        slam._add_keyframe(points, Pose2(), Pose2(), 0.0, 1.0)
        slam._add_keyframe(
            points, Pose2(0.3, 0.4, 0.0), Pose2(0.3, 0.4, 0.0),
            1.0, 1.0)
        slam._add_keyframe(
            points, Pose2(0.3, 0.4, 0.5), Pose2(0.3, 0.4, 0.5),
            2.0, 1.0)

        self.assertAlmostEqual(slam.keyframes[1].travel_m, 0.5)
        self.assertAlmostEqual(slam.keyframes[2].travel_m, 0.5)

    def test_absolute_imu_heading_factors_stop_scan_yaw_accumulation(self):
        slam = PoseGraphSlam(SlamConfig(
            heading_prior_rotation_sigma_rad=math.radians(2.5),
            heading_prior_max_disagreement_rad=math.radians(25.0)))
        points = np.array(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        for index in range(6):
            slam.keyframes.append(Keyframe(
                index, float(index), Pose2(float(index), 0.0, 0.0),
                Pose2(float(index), 0.0, math.radians(4.0 * index)),
                points, np.full(72, np.nan), 0.8))
            if index:
                slam.constraints.append(Constraint(
                    index - 1, index, Pose2(1.0, 0.0, 0.0),
                    0.08, math.radians(3.0), "odometry"))
                slam.constraints.append(Constraint(
                    index - 1, index,
                    Pose2(1.0, 0.0, math.radians(4.0)),
                    0.04, math.radians(2.5), "scan_match", 0.8))

        slam.optimize()

        self.assertLess(abs(math.degrees(slam.keyframes[-1].pose.yaw)), 3.0)

    def test_heading_factor_does_not_distort_lidar_graph_after_sensor_fault(self):
        slam = PoseGraphSlam(SlamConfig(
            heading_prior_rotation_sigma_rad=math.radians(1.0),
            heading_prior_max_disagreement_rad=math.radians(12.0),
        ))
        points = np.array(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        slam.keyframes = [
            Keyframe(0, 0.0, Pose2(yaw=0.0), Pose2(yaw=0.0),
                     points, np.full(72, np.nan), 0.95),
            Keyframe(1, 1.0, Pose2(x=1.0, yaw=math.radians(35.0)),
                     Pose2(x=1.0, yaw=0.0),
                     points, np.full(72, np.nan), 0.95),
        ]
        slam.constraints = [Constraint(
            0, 1, Pose2(x=1.0, yaw=0.0),
            0.03, math.radians(1.0), "scan_match", 0.95)]

        slam.optimize()

        self.assertAlmostEqual(slam.keyframes[1].pose.yaw, 0.0, places=6)

    def test_loop_verifier_rejects_symmetric_half_turn(self):
        config = SlamConfig(loop_min_separation_keyframes=1,
                            loop_check_every_keyframes=1,
                            loop_min_path_separation_m=0.0,
                            loop_min_recent_motion_m=0.0,
                            loop_confirmation_count=1)
        slam = PoseGraphSlam(config)
        first = self.keyframe(0, 0.0)
        current = self.keyframe(1, 0.0)
        first.descriptor[:] = 1.0
        current.descriptor[:] = 1.0
        slam.keyframes = [first, current]
        false_symmetric_match = MatchResult(
            Pose2(0.0, 0.0, math.pi), 0.99, 0.99, 0.01, True)

        with mock.patch.object(slam.matcher, "match",
                               return_value=false_symmetric_match):
            accepted = slam._try_loop_closure()

        self.assertFalse(accepted)
        self.assertEqual(slam.loop_closure_count, 0)
        self.assertFalse(any(edge.kind == "loop_closure"
                             for edge in slam.constraints))

    def test_stationary_keyframes_do_not_propose_loop_closures(self):
        config = SlamConfig(
            loop_min_separation_keyframes=1,
            loop_check_every_keyframes=1,
            loop_min_path_separation_m=0.0,
            loop_min_recent_motion_m=0.1,
            loop_confirmation_count=1,
        )
        slam = PoseGraphSlam(config)
        first = self.keyframe(0, 0.0, travel_m=0.0)
        current = self.keyframe(1, 0.0, travel_m=0.0)
        first.descriptor[:] = 1.0
        current.descriptor[:] = 1.0
        slam.keyframes = [first, current]

        with mock.patch.object(slam.matcher, "match") as match:
            accepted = slam._try_loop_closure()

        self.assertFalse(accepted)
        match.assert_not_called()

    def test_stationary_timed_scans_do_not_add_keyframes(self):
        slam = PoseGraphSlam(SlamConfig(
            min_scan_points=4,
            keyframe_max_interval_s=0.5,
        ))
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        slam.process_scan(points, Pose2(), 0.0)

        result = slam.process_scan(points, Pose2(), 2.0)

        self.assertFalse(result.keyframe_added)
        self.assertEqual(result.keyframes, 1)

    def test_loop_verifier_tries_fused_heading_after_descriptor_alias(self):
        config = SlamConfig(
            loop_min_separation_keyframes=1,
            loop_check_every_keyframes=1,
            loop_min_path_separation_m=0.0,
            loop_min_recent_motion_m=0.0,
            loop_confirmation_count=1,
        )
        slam = PoseGraphSlam(config)
        first = self.keyframe(0, 0.0)
        current = self.keyframe(1, 0.0)
        slam.keyframes = [first, current]

        def match(_points, initial, _field, **_kwargs):
            if abs(initial.yaw) > math.radians(90.0):
                return MatchResult(
                    Pose2(yaw=math.pi), 0.99, 0.99, 0.01, True)
            return MatchResult(Pose2(), 0.90, 0.90, 0.02, True)

        with mock.patch.object(
                slam, "_loop_candidates",
                return_value=[(0.9, 0.0, 36, first)]), \
                mock.patch.object(slam.matcher, "match",
                                  side_effect=match) as matcher:
            accepted = slam._try_loop_closure()

        self.assertTrue(accepted)
        self.assertEqual(matcher.call_count, 2)
        self.assertEqual(slam.last_loop_status, "accepted:0->1")

    def test_loop_candidates_require_actual_path_separation(self):
        config = SlamConfig(
            loop_min_separation_keyframes=1,
            loop_check_every_keyframes=1,
            loop_min_path_separation_m=2.0,
            loop_min_recent_motion_m=0.0,
        )
        slam = PoseGraphSlam(config)
        first = self.keyframe(0, 0.0, travel_m=0.0)
        current = self.keyframe(1, 0.1, travel_m=1.5)
        first.descriptor[:] = 1.0
        current.descriptor[:] = 1.0
        slam.keyframes = [first, current]

        self.assertEqual(slam._loop_candidates(current), [])

    def test_loop_closure_requires_consistent_moving_scan_confirmation(self):
        config = SlamConfig(
            loop_min_separation_keyframes=1,
            loop_check_every_keyframes=1,
            loop_min_path_separation_m=1.0,
            loop_min_recent_motion_m=0.1,
            loop_confirmation_count=2,
            loop_confirmation_min_travel_m=0.3,
        )
        slam = PoseGraphSlam(config)
        first = self.keyframe(0, 0.0, travel_m=0.0)
        current = self.keyframe(1, 0.1, travel_m=2.0)
        first.descriptor[:] = 1.0
        current.descriptor[:] = 1.0
        slam.keyframes = [first, current]

        with mock.patch.object(
                slam.matcher, "match",
                side_effect=(
                    MatchResult(Pose2(0.0, 0.0, 0.0),
                                0.9, 0.8, 0.03, True),
                    MatchResult(Pose2(0.3, 0.0, 0.0),
                                0.9, 0.8, 0.03, True),
                )):
            self.assertFalse(slam._try_loop_closure())
            self.assertEqual(slam.loop_closure_count, 0)

            confirmed = self.keyframe(2, 0.4, travel_m=2.4)
            confirmed.descriptor[:] = 1.0
            slam.keyframes.append(confirmed)
            self.assertTrue(slam._try_loop_closure())

        self.assertEqual(slam.loop_closure_count, 1)
        edge = slam.constraints[-1]
        self.assertEqual((edge.i, edge.j, edge.kind),
                         (first.index, confirmed.index, "loop_closure"))

    def test_inconsistent_second_scan_restarts_loop_confirmation(self):
        config = SlamConfig(
            loop_min_separation_keyframes=1,
            loop_check_every_keyframes=1,
            loop_min_path_separation_m=1.0,
            loop_min_recent_motion_m=0.1,
            loop_confirmation_count=2,
            loop_confirmation_min_travel_m=0.3,
            loop_confirmation_translation_tolerance_m=0.15,
        )
        slam = PoseGraphSlam(config)
        first = self.keyframe(0, 0.0, travel_m=0.0)
        current = self.keyframe(1, 0.1, travel_m=2.0)
        first.descriptor[:] = 1.0
        current.descriptor[:] = 1.0
        slam.keyframes = [first, current]

        with mock.patch.object(
                slam.matcher, "match",
                side_effect=(
                    MatchResult(Pose2(0.0, 0.0, 0.0),
                                0.9, 0.8, 0.03, True),
                    MatchResult(Pose2(0.0, 0.0, 0.0),
                                0.9, 0.8, 0.03, True),
                )):
            self.assertFalse(slam._try_loop_closure())
            inconsistent = self.keyframe(2, 0.4, travel_m=2.4)
            inconsistent.descriptor[:] = 1.0
            slam.keyframes.append(inconsistent)
            self.assertFalse(slam._try_loop_closure())

        self.assertEqual(slam.loop_closure_count, 0)
        self.assertEqual(slam._pending_loop.confirmations, 1)

    def test_map_export_writes_ros_map_and_replay_state(self):
        slam = PoseGraphSlam(SlamConfig(min_scan_points=4))
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        slam.process_scan(points, Pose2(), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            paths = slam.save(pathlib.Path(directory) / "map")

            for path in paths.values():
                self.assertTrue(path.exists(), path)
            self.assertTrue(paths["map"].read_bytes().startswith(b"P5\n"))
            graph = json.loads(paths["graph"].read_text())
            self.assertEqual(graph["format"], "kiwi-pose-graph-v2")
            self.assertEqual(len(graph["nodes"]), 1)
            self.assertEqual(graph["nodes"][0]["travel_m"], 0.0)
            self.assertEqual(graph["nodes"][0]["session_id"], 0)
            with np.load(paths["state"]) as state:
                self.assertEqual(state["points"].shape, (4, 2))
                np.testing.assert_array_equal(state["travel_m"], [0.0])
                np.testing.assert_array_equal(state["session_ids"], [0])

    def test_loads_v1_state_and_reconstructs_descriptors(self):
        config = SlamConfig(min_scan_points=4)
        slam = PoseGraphSlam(config)
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        slam.process_scan(points, Pose2(), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            paths = slam.save(prefix)
            graph = json.loads(paths["graph"].read_text())
            graph["format"] = "kiwi-pose-graph-v1"
            for node in graph["nodes"]:
                node.pop("session_id")
            paths["graph"].write_text(json.dumps(graph))
            with np.load(paths["state"]) as state:
                old_state = {
                    name: np.array(state[name])
                    for name in state.files if name != "session_ids"
                }
            np.savez_compressed(paths["state"], **old_state)

            loaded = PoseGraphSlam.load(prefix)

            self.assertTrue(loaded.relocalization_required)
            self.assertEqual(loaded.current_session_id, 1)
            self.assertEqual(len(loaded.keyframes), 1)
            self.assertEqual(loaded.keyframes[0].session_id, 0)
            self.assertEqual(loaded.keyframes[0].descriptor.shape, (72,))
            np.testing.assert_allclose(loaded.keyframes[0].points, points)

    def test_resume_relocalizes_before_bridging_new_odometry_session(self):
        config = SlamConfig(
            min_scan_points=4,
            keyframe_translation_m=0.05,
        )
        saved = PoseGraphSlam(config)
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        saved._add_keyframe(points, Pose2(), Pose2(), 0.0, 1.0)
        saved._add_keyframe(
            points, Pose2(1.0, 0.0, 0.0), Pose2(1.0, 0.0, 0.0),
            1.0, 0.9)

        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            saved.save(prefix)
            slam = PoseGraphSlam.load(prefix)
            live_raw = Pose2(0.0, 0.0, 0.0)
            localized = MatchResult(
                Pose2(1.02, 0.01, 0.02), 0.85, 0.75, 0.03, True)
            with mock.patch.object(slam.matcher, "match",
                                   return_value=localized):
                result = slam.process_scan(points, live_raw, 0.1)

            self.assertTrue(result.scan_matched)
            self.assertTrue(result.keyframe_added)
            self.assertFalse(slam.relocalization_required)
            self.assertEqual(slam.keyframes[-1].session_id, 1)
            bridge = slam.constraints[-1]
            self.assertEqual(bridge.kind, "relocalization")
            self.assertEqual(bridge.j, 2)
            self.assertFalse(any(
                edge.i == 1 and edge.j == 2 and edge.kind == "odometry"
                for edge in slam.constraints
            ))
            self.assertAlmostEqual(result.map_to_odom.x, localized.pose.x)

            continued = MatchResult(
                Pose2(1.22, 0.01, 0.02), 0.82, 0.70, 0.03, True)
            with mock.patch.object(slam.matcher, "match",
                                   return_value=continued):
                result = slam.process_scan(
                    points, Pose2(0.2, 0.0, 0.0), 0.3)

            self.assertTrue(result.keyframe_added)
            self.assertEqual(slam.keyframes[-1].session_id, 1)
            self.assertTrue(any(
                edge.i == 2 and edge.j == 3 and edge.kind == "odometry"
                for edge in slam.constraints
            ))
            slam.save(prefix)
            reloaded = PoseGraphSlam.load(prefix)
            self.assertEqual(
                [keyframe.session_id for keyframe in reloaded.keyframes],
                [0, 0, 1, 1],
            )
            self.assertTrue(any(
                edge.kind == "relocalization"
                for edge in reloaded.constraints
            ))

    def test_heading_prior_is_relative_to_each_resumed_session(self):
        config = SlamConfig(
            heading_prior_rotation_sigma_rad=math.radians(2.0),
            heading_prior_max_disagreement_rad=math.radians(25.0))
        slam = PoseGraphSlam(config)
        points = np.array(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        frames = (
            Keyframe(0, 0.0, Pose2(yaw=0.0), Pose2(yaw=0.0),
                     points, np.full(72, np.nan), 1.0, session_id=0),
            Keyframe(1, 1.0, Pose2(yaw=0.1), Pose2(yaw=0.1),
                     points, np.full(72, np.nan), 1.0, session_id=0),
            Keyframe(2, 0.0, Pose2(yaw=0.0), Pose2(yaw=1.0),
                     points, np.full(72, np.nan), 1.0, session_id=1),
            Keyframe(3, 1.0, Pose2(yaw=0.1), Pose2(yaw=1.4),
                     points, np.full(72, np.nan), 1.0, session_id=1),
        )
        slam.keyframes = list(frames)
        slam.constraints = [
            Constraint(0, 1, Pose2(yaw=0.1), 0.1, 0.05, "odometry"),
            Constraint(1, 2, Pose2(yaw=0.9), 0.03, 0.02,
                       "relocalization"),
            Constraint(2, 3, Pose2(yaw=0.3), 0.1, 0.08, "scan_match"),
        ]

        slam.optimize()

        self.assertGreater(slam.keyframes[2].pose.yaw, 0.8)
        resumed_delta = between(
            slam.keyframes[2].pose, slam.keyframes[3].pose)
        self.assertLess(abs(resumed_delta.yaw - 0.1), 0.08)

    def test_resume_keeps_waiting_when_relocalization_fails(self):
        config = SlamConfig(min_scan_points=4)
        saved = PoseGraphSlam(config)
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        saved.process_scan(points, Pose2(), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            saved.save(prefix)
            slam = PoseGraphSlam.load(prefix)
            rejected = MatchResult(Pose2(), 0.2, 0.1, 0.4, False)
            with mock.patch.object(slam.matcher, "match",
                                   return_value=rejected):
                result = slam.process_scan(points, Pose2(), 0.1)

            self.assertFalse(result.scan_matched)
            self.assertFalse(result.keyframe_added)
            self.assertTrue(slam.relocalization_required)
            self.assertEqual(len(slam.keyframes), 1)
            self.assertEqual(result.loop_status, "relocalizing:no_match")

    def test_global_resume_refuses_two_equally_good_distinct_places(self):
        config = SlamConfig(
            min_scan_points=4,
            relocalization_min_score_margin=0.035,
        )
        slam = PoseGraphSlam(config)
        points = np.array(((1.0, 0.0), (0.0, 1.0),
                           (-1.0, 0.0), (0.0, -1.0)))
        slam._add_keyframe(points, Pose2(), Pose2(), 0.0, 1.0)
        slam._add_keyframe(
            points, Pose2(2.0, 0.0, 0.0), Pose2(2.0, 0.0, 0.0),
            1.0, 1.0)
        slam.current_session_id = 1
        slam.relocalization_required = True
        slam.relocalization_global = True
        slam.relocalization_hint = Pose2()
        first, second = slam.keyframes
        matches = (
            MatchResult(Pose2(0.0, 0.0, 0.0), 0.81, 0.7, 0.03, True),
            MatchResult(Pose2(2.0, 0.0, 0.0), 0.80, 0.7, 0.03, True),
        )
        with mock.patch.object(
                slam, "_relocalization_hypotheses",
                return_value=[(Pose2(), first),
                              (Pose2(2.0, 0.0, 0.0), second)]), \
                mock.patch.object(slam.matcher, "match",
                                  side_effect=matches):
            result = slam.process_scan(points, Pose2(), 0.1)

        self.assertFalse(result.scan_matched)
        self.assertTrue(slam.relocalization_required)
        self.assertEqual(result.loop_status, "relocalizing:ambiguous")
        self.assertEqual(len(slam.keyframes), 2)


class _Publisher:
    def __init__(self):
        self.payloads = []

    def put(self, payload):
        self.payloads.append(json.loads(payload))


class SlamRunnerPublishingTests(unittest.TestCase):
    def test_odometry_publishes_latest_map_corrected_pose(self):
        runner = SlamRunner.__new__(SlamRunner)
        runner.pose = [1.0, 2.0, 0.2]
        runner.yaw_offset = None
        runner.last_twist_t = 1.0
        runner.pose_history = mock.Mock()
        runner.pose_clock = mock.Mock()
        runner.pose_clock.observe = mock.Mock()
        runner.latest_result = mock.Mock(
            map_to_odom=Pose2(0.5, -0.25, 0.1),
            scan_matched=True,
            match_score=0.8,
            hit_ratio=0.7,
            rmse_m=0.03,
            heading_disagreement_rad=0.01,
            keyframes=4,
            loop_closures=0,
            processing_ms=600.0,
        )
        runner.latest_map_payload = None
        runner.last_map_publish_t = None
        runner.map_republish_s = 5.0
        runner._condition = threading.Condition(threading.RLock())
        runner._publisher = _Publisher()
        runner._map_publisher = _Publisher()

        runner.on_odometry({
            "follower_time_us": 1_100_000,
            "measured": {"vx": 0.0, "vy": 0.0, "omega": 1.0},
            "imu_ready": False,
        })

        self.assertEqual(len(runner._publisher.payloads), 1)
        payload = runner._publisher.payloads[0]
        expected = compose(
            runner.latest_result.map_to_odom,
            Pose2(*runner.pose),
        )
        self.assertEqual(payload["source"], "odometry_prediction")
        self.assertAlmostEqual(payload["pose"]["x"], expected.x)
        self.assertAlmostEqual(payload["pose"]["y"], expected.y)
        self.assertAlmostEqual(payload["pose"]["yaw"], expected.yaw)

    def test_fused_runner_uses_incremental_wheel_and_imu_yaw(self):
        runner = SlamRunner.__new__(SlamRunner)
        runner.pose = [0.0, 0.0, 0.0]
        runner.yaw_offset = None
        runner.last_twist_t = None
        runner.pose_history = mock.Mock()
        runner.pose_clock = mock.Mock()
        runner.latest_result = None
        runner.latest_map_payload = None
        runner.last_map_publish_t = None
        runner.map_republish_s = 5.0
        runner._condition = threading.Condition(threading.RLock())
        runner._publisher = _Publisher()
        runner._map_publisher = _Publisher()
        runner.yaw_estimator_mode = "fused"
        runner.yaw_estimator = YawEstimator()
        runner.yaw_estimate = runner.yaw_estimator.snapshot()

        def quat(yaw):
            return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

        runner.on_odometry({
            "follower_time_us": 1_000_000,
            "measured": {"vx": 0.0, "vy": 0.0, "omega": 0.0},
            "imu_ready": True,
            "imu_quat_ijkr": quat(0.0),
            "encoder_ready_mask": 7,
        })
        runner.on_odometry({
            "follower_time_us": 1_100_000,
            "measured": {"vx": 0.0, "vy": 0.0, "omega": 1.0},
            "imu_ready": True,
            "imu_quat_ijkr": quat(0.1),
            "encoder_ready_mask": 7,
        })

        self.assertAlmostEqual(runner.pose[2], 0.1, places=7)
        self.assertEqual(runner.yaw_estimate.source, "wheel+imu")


if __name__ == "__main__":
    unittest.main()
