import json
import math
import pathlib
import sys
import types
import unittest

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_navigation import (  # noqa: E402
    build_parser,
    follower_state_payload,
    trajectory_payload,
)
from kiwi_navigation_core import (  # noqa: E402
    AStarPlanner,
    DEFAULT_LIDAR_COLLISION_HORIZON_S,
    DEFAULT_MAX_FOLLOWING_SPEED_MPS,
    DEFAULT_RUNTIME_COLLISION_RADIUS_M,
    PathNotFound,
    PurePursuitFollower,
    SweptCircleCollisionGuard,
    WheelDistanceTracker,
    inflate_obstacles,
    stamp_lidar_obstacles,
)
from kiwi_pose_controller import Pose2, PoseStabilizingController  # noqa: E402


def occupancy(data, resolution=0.1, origin=(0.0, 0.0), keyframes=3):
    return types.SimpleNamespace(
        data=np.asarray(data, dtype=np.int8),
        resolution_m=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        keyframes=keyframes,
    )


class InflationTests(unittest.TestCase):
    def test_inflation_is_a_metric_disk(self):
        obstacles = np.zeros((11, 11), dtype=bool)
        obstacles[5, 5] = True

        inflated = inflate_obstacles(obstacles, 0.25, 0.1)

        self.assertTrue(inflated[5, 7])       # 0.20 m
        self.assertTrue(inflated[6, 7])       # sqrt(0.05) m
        self.assertFalse(inflated[7, 7])      # sqrt(0.08) m
        self.assertFalse(inflated[5, 8])      # 0.30 m

    def test_unknown_space_is_blocked_by_default(self):
        data = np.full((5, 5), -1, dtype=np.int8)
        data[2, 1:4] = 0

        safe = AStarPlanner(occupancy(data), inflation_radius_m=0.0)
        exploratory = AStarPlanner(
            occupancy(data), inflation_radius_m=0.0, allow_unknown=True)

        self.assertTrue(safe.blocked[0, 0])
        self.assertFalse(exploratory.blocked[0, 0])


class LiveCollisionGuardTests(unittest.TestCase):
    def test_blocks_only_translation_that_sweeps_into_wall(self):
        guard = SweptCircleCollisionGuard(
            radius_m=0.18,
            horizon_s=DEFAULT_LIDAR_COLLISION_HORIZON_S,
        )
        points = np.array(((0.30, 0.0), (0.0, 0.17)))

        self.assertTrue(guard.blocks(points, 0.20, 0.0))
        self.assertFalse(guard.blocks(points, -0.20, 0.0))
        self.assertFalse(guard.blocks(
            np.array(((0.0, 0.17),)), 0.20, 0.0))

    def test_stamps_live_scan_endpoints_into_a_map_copy(self):
        source = occupancy(np.zeros((9, 9), dtype=np.int8), resolution=0.1)
        pose = Pose2(0.2, 0.2, math.pi / 2.0)

        stamped = stamp_lidar_obstacles(
            source, pose, np.array(((0.2, 0.0),)))

        self.assertEqual(source.data[4, 2], 0)
        self.assertEqual(stamped.data[4, 2], 100)


class WheelDistanceTrackerTests(unittest.TestCase):
    def test_integrates_encoder_speed_and_ignores_duplicate_samples(self):
        tracker = WheelDistanceTracker()
        tracker.update({
            "follower_time_us": 1_000_000,
            "measured": {"vx": 0.3, "vy": 0.4},
        })
        tracker.update({
            "follower_time_us": 1_200_000,
            "measured": {"vx": 0.3, "vy": 0.4},
        })
        tracker.update({
            "follower_time_us": 1_200_000,
            "measured": {"vx": 4.0, "vy": 0.0},
        })

        self.assertAlmostEqual(tracker.distance_m, 0.1)


class AStarPlannerTests(unittest.TestCase):
    def test_path_routes_around_inflated_wall(self):
        data = np.zeros((15, 15), dtype=np.int8)
        data[2:13, 7] = 100
        planner = AStarPlanner(occupancy(data), inflation_radius_m=0.1)

        path = planner.plan((0.1, 0.7), (1.3, 0.7))

        np.testing.assert_allclose(path[0], (0.1, 0.7))
        np.testing.assert_allclose(path[-1], (1.3, 0.7))
        self.assertTrue(planner.path_is_free(path))
        self.assertTrue(np.any((path[:, 1] < 0.2) | (path[:, 1] > 1.2)))

    def test_diagonal_cannot_cut_an_obstacle_corner(self):
        data = np.zeros((2, 2), dtype=np.int8)
        data[0, 1] = 100
        data[1, 0] = 100
        planner = AStarPlanner(occupancy(data), inflation_radius_m=0.0)

        with self.assertRaises(PathNotFound):
            planner.plan((0.0, 0.0), (0.1, 0.1))

    def test_rejects_goal_inside_inflation_radius(self):
        data = np.zeros((9, 9), dtype=np.int8)
        data[4, 4] = 100
        planner = AStarPlanner(occupancy(data), inflation_radius_m=0.25)

        with self.assertRaisesRegex(PathNotFound, "goal"):
            planner.plan((0.0, 0.0), (0.6, 0.4))

    def test_collision_check_catches_a_thin_cell_corner_crossing(self):
        data = np.zeros((41, 41), dtype=np.int8)
        data[8:33, 20] = 100
        planner = AStarPlanner(occupancy(data, resolution=0.05),
                               inflation_radius_m=0.25)

        # This very short segment just enters an inflated cell. A sampled
        # checker can skip the sliver, but exact grid traversal must catch it.
        path = np.array(((0.774768837, 0.232364145),
                         (0.846925177, 0.150000000)))

        self.assertFalse(planner.path_is_free(path))

    def test_runtime_radius_can_recover_into_planning_buffer(self):
        data = np.zeros((21, 31), dtype=np.int8)
        data[10, 15] = 100
        map_value = occupancy(
            data, resolution=float(np.float32(0.05)))
        planning = AStarPlanner(map_value, inflation_radius_m=0.25)
        runtime = AStarPlanner(
            map_value,
            inflation_radius_m=DEFAULT_RUNTIME_COLLISION_RADIUS_M)
        route = np.array(((0.50, 0.25), (1.00, 0.25)))
        displaced_pose = Pose2(0.75, 0.30, 0.0)

        self.assertTrue(planning.path_is_free(route))
        self.assertFalse(planning.cell_is_free(
            planning.world_to_cell((displaced_pose.x, displaced_pose.y))))
        self.assertTrue(runtime.cell_is_free(
            runtime.world_to_cell((displaced_pose.x, displaced_pose.y))))

        controller = PoseStabilizingController(
            kp_x=1.0,
            kp_y=1.0,
            kp_yaw=1.0,
            max_linear_speed=1.0,
            max_angular_speed=1.0,
        )
        follower = PurePursuitFollower(
            route, controller, lookahead_m=0.15)
        output = follower.update(displaced_pose, runtime.path_is_free)

        self.assertTrue(output.pursuit_path_clear)
        self.assertLess(output.command.vy, 0.0)

    def test_plans_hard_safe_egress_from_preferred_start_buffer(self):
        data = np.zeros((41, 41), dtype=np.int8)
        # The obstacle is 0.224 m from the start: outside the 0.18 m hard
        # envelope, but inside the preferred 0.25 m routing buffer.
        data[16, 22] = 100
        map_value = occupancy(data, resolution=0.05)
        preferred = AStarPlanner(map_value, inflation_radius_m=0.25)
        runtime = AStarPlanner(map_value, inflation_radius_m=0.18)
        start = (1.0, 1.0)
        goal = (1.5, 1.0)

        path = preferred.plan_with_start_recovery(start, goal, runtime)

        self.assertFalse(preferred.cell_is_free(
            preferred.world_to_cell(start)))
        self.assertTrue(runtime.cell_is_free(runtime.world_to_cell(start)))
        self.assertTrue(runtime.path_is_free(path))
        self.assertTrue(preferred.path_is_free_with_start_recovery(
            path, runtime))
        self.assertFalse(preferred.path_is_free(path))
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], goal)

    def test_start_recovery_never_weakens_hard_collision_envelope(self):
        data = np.zeros((21, 21), dtype=np.int8)
        data[10, 11] = 100
        map_value = occupancy(data, resolution=0.1)
        preferred = AStarPlanner(map_value, inflation_radius_m=0.25)
        runtime = AStarPlanner(map_value, inflation_radius_m=0.18)

        with self.assertRaisesRegex(PathNotFound, "hard collision envelope"):
            preferred.plan_with_start_recovery(
                (1.0, 1.0), (1.8, 1.0), runtime)


class PurePursuitFollowerTests(unittest.TestCase):
    def setUp(self):
        self.controller = PoseStabilizingController(
            kp_x=1.0,
            kp_y=1.0,
            kp_yaw=1.0,
            max_linear_speed=10.0,
            max_angular_speed=10.0,
            position_tolerance=0.04,
            yaw_tolerance=math.radians(2.0),
        )

    def test_selects_interpolated_lookahead_on_trajectory(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            self.controller,
            lookahead_m=0.3,
        )

        output = follower.update(Pose2(0.2, 0.1, 0.0))

        self.assertAlmostEqual(output.progress_m, 0.2)
        self.assertAlmostEqual(output.cross_track_error_m, 0.1)
        self.assertAlmostEqual(output.following_pose.x, 0.5)
        self.assertAlmostEqual(output.following_pose.y, 0.0)
        self.assertAlmostEqual(output.command.vx, 0.3)
        self.assertAlmostEqual(output.command.vy, -0.1)

    def test_progress_does_not_move_backward(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (2.0, 0.0)], self.controller, lookahead_m=0.25)

        follower.update(Pose2(1.0, 0.0, 0.0))
        follower.update(Pose2(0.2, 0.0, 0.0))

        self.assertAlmostEqual(follower.progress_m, 1.0)

    def test_pose_stabilizer_rotates_map_command_into_robot_body(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (1.0, 0.0)],
            self.controller,
            lookahead_m=0.3,
            goal_yaw=0.0,
        )

        output = follower.update(Pose2(0.0, 0.0, math.pi / 2.0))

        self.assertAlmostEqual(output.command.vx, 0.0, places=7)
        self.assertAlmostEqual(output.command.vy, -0.3)
        self.assertAlmostEqual(output.command.omega, -math.pi / 2.0)

    def test_heading_blends_continuously_across_path_corner(self):
        route = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        before = PurePursuitFollower(
            route, self.controller, lookahead_m=0.3,
            goal_yaw_blend_distance_m=0.3,
        ).update(Pose2(0.69, 0.0, 0.0))
        after = PurePursuitFollower(
            route, self.controller, lookahead_m=0.3,
            goal_yaw_blend_distance_m=0.3,
        ).update(Pose2(0.71, 0.0, 0.0))

        self.assertAlmostEqual(before.following_pose.yaw, 0.0)
        self.assertGreater(after.following_pose.yaw, 0.0)
        self.assertLess(after.following_pose.yaw, math.radians(10.0))

    def test_final_heading_blends_over_approach_distance(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (1.0, 0.0)],
            self.controller,
            lookahead_m=0.3,
            goal_yaw=math.pi / 2.0,
            goal_yaw_blend_distance_m=0.4,
        )

        before_blend = follower.update(Pose2(0.60, 0.0, 0.0))
        halfway = follower.update(Pose2(0.80, 0.0, 0.0))

        self.assertAlmostEqual(before_blend.following_pose.yaw, 0.0)
        self.assertAlmostEqual(halfway.following_pose.yaw, math.pi / 4.0)

    def test_final_position_is_held_while_goal_yaw_settles(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (1.0, 0.0)],
            self.controller,
            lookahead_m=0.3,
            goal_yaw=math.pi / 2.0,
        )

        turning = follower.update(Pose2(0.99, 0.0, 0.0))
        reached = follower.update(Pose2(1.0, 0.0, math.pi / 2.0))

        self.assertEqual((turning.command.vx, turning.command.vy), (0.0, 0.0))
        self.assertGreater(turning.command.omega, 0.0)
        self.assertFalse(turning.complete)
        self.assertTrue(reached.complete)
        self.assertEqual(
            (reached.command.vx, reached.command.vy, reached.command.omega),
            (0.0, 0.0, 0.0),
        )

    def test_lookahead_shortens_instead_of_cutting_blocked_corner(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (1.0, 0.0)],
            self.controller,
            lookahead_m=0.8,
        )

        output = follower.update(
            Pose2(0.0, 0.0, 0.0),
            path_is_free=lambda segment: segment[-1, 0] <= 0.45,
        )

        self.assertTrue(output.pursuit_path_clear)
        self.assertGreater(output.following_pose.x, 0.0)
        self.assertLessEqual(output.following_pose.x, 0.45)
        self.assertAlmostEqual(output.command.vx, output.following_pose.x)

    def test_stops_when_no_forward_pursuit_segment_is_clear(self):
        follower = PurePursuitFollower(
            [(0.0, 0.0), (1.0, 0.0)],
            self.controller,
            lookahead_m=0.3,
        )

        output = follower.update(
            Pose2(0.0, 0.0, 0.0),
            path_is_free=lambda _segment: False,
        )

        self.assertFalse(output.pursuit_path_clear)
        self.assertEqual(
            (output.command.vx, output.command.vy, output.command.omega),
            (0.0, 0.0, 0.0),
        )


class NavigationDefaultsTests(unittest.TestCase):
    def test_trajectory_following_has_a_conservative_default_speed_cap(self):
        args = build_parser().parse_args(["1.0", "0.0"])

        self.assertEqual(
            args.max_linear_speed, DEFAULT_MAX_FOLLOWING_SPEED_MPS)
        self.assertEqual(DEFAULT_MAX_FOLLOWING_SPEED_MPS, 0.12)
        self.assertEqual(args.kp_yaw, 2.5)
        self.assertEqual(args.goal_yaw_blend_distance, 0.30)

    def test_accepts_coordinator_action_and_hard_distance_envelope(self):
        args = build_parser().parse_args([
            "1.0", "0.0", "--action-id", "action-1",
            "--max-travel-distance", "2.5",
        ])

        self.assertEqual(args.action_id, "action-1")
        self.assertEqual(args.max_travel_distance, 2.5)


class NavigationPayloadTests(unittest.TestCase):
    def test_payloads_are_json_serializable_and_map_framed(self):
        map_value = occupancy(np.zeros((3, 3), dtype=np.int8))
        planner = AStarPlanner(map_value, inflation_radius_m=0.25)
        points = np.array(((0.0, 0.0), (0.1, 0.1)))
        trajectory = trajectory_payload(points, planner, map_value)
        follower = PurePursuitFollower(points, self.controller(), lookahead_m=0.1)
        current = Pose2(0.0, 0.0, 0.0)
        output = follower.update(current)
        state = follower_state_payload(
            "following", current, Pose2(0.1, 0.1, follower.goal_yaw), output,
            action_id="action-1", distance_traveled_m=0.25,
            max_travel_distance_m=1.0)

        json.dumps(trajectory)
        json.dumps(state)
        self.assertEqual(trajectory["frame"], "map")
        self.assertEqual(trajectory["inflation_radius_m"], 0.25)
        self.assertEqual(state["following_point"]["x"], output.following_pose.x)
        self.assertEqual(
            state["heading_setpoint_rad"], output.following_pose.yaw)
        self.assertAlmostEqual(
            state["heading_error_rad"], output.following_pose.yaw-current.yaw)
        self.assertEqual(state["action_id"], "action-1")
        self.assertEqual(state["distance_budget_remaining_m"], 0.75)

    def test_trajectory_reports_planning_and_runtime_clearance(self):
        map_value = occupancy(np.zeros((3, 3), dtype=np.int8))
        planner = AStarPlanner(map_value, inflation_radius_m=0.25)

        payload = trajectory_payload(
            np.array(((0.0, 0.0), (0.1, 0.0))),
            planner,
            map_value,
            collision_radius_m=0.18,
        )

        self.assertEqual(payload["inflation_radius_m"], 0.25)
        self.assertEqual(payload["runtime_collision_radius_m"], 0.18)
        self.assertAlmostEqual(payload["tracking_buffer_m"], 0.07)

    @staticmethod
    def controller():
        return PoseStabilizingController(
            kp_x=1.0, kp_y=1.0, kp_yaw=1.0,
            max_linear_speed=1.0, max_angular_speed=1.0)


if __name__ == "__main__":
    unittest.main()
