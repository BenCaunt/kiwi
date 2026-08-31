#!/usr/bin/env python3
"""Plan on Kiwi's live SLAM map and follow the path with pure pursuit.

Run ``kiwi_slam.py`` first, then provide a map-frame goal in meters:

  python3 scripts/kiwi_navigation.py 1.5 -0.4 --goal-yaw-deg 90

The process publishes the active trajectory and follower state for
``kiwi_dashboard.py`` while commanding the robot on ``cmd_vel``.
"""

import argparse
import json
import math
import threading
import time

import numpy as np

from kiwi_calibration import load_calibration
from kiwi_client import DEFAULT_ROBOT_YAW_DEG, KiwiClient
from kiwi_lidar import ScanAssembler, parse_frames
from kiwi_lidar_deskew import LidarExtrinsics
from kiwi_map import decode_occupancy_map
from kiwi_navigation_core import (
    AStarPlanner,
    DEFAULT_LIDAR_COLLISION_HORIZON_S,
    DEFAULT_MAX_FOLLOWING_SPEED_MPS,
    DEFAULT_RUNTIME_COLLISION_RADIUS_M,
    PathNotFound,
    PurePursuitFollower,
    SweptCircleCollisionGuard,
    WheelDistanceTracker,
    stamp_lidar_obstacles,
)
from kiwi_pose_controller import Pose2, PoseStabilizingController, wrap_angle


class LiveMapBuffer:
    """Thread-safe holder for the most recently decoded SLAM map."""

    def __init__(self):
        self._lock = threading.Lock()
        self._occupancy = None
        self._generation = 0

    def update(self, payload):
        try:
            occupancy = decode_occupancy_map(payload)
        except ValueError:
            return
        with self._lock:
            self._occupancy = occupancy
            self._generation += 1

    def snapshot(self):
        with self._lock:
            return self._occupancy, self._generation


class LiveLidarBuffer:
    """Latest complete raw LiDAR revolution in the aligned body frame."""

    def __init__(self, extrinsics=None):
        self._lock = threading.Lock()
        self._assembler = ScanAssembler()
        self._extrinsics = extrinsics or LidarExtrinsics()
        self._points = None
        self._received_at = None

    def update(self, payload):
        for frame in parse_frames(bytes(payload)):
            if frame is None:
                continue
            revolution = self._assembler.add(frame)
            if not revolution:
                continue
            points = []
            for angle_deg, distance_m, _intensity in revolution:
                if not 0.02 < distance_m < 12.0:
                    continue
                angle = -math.radians(angle_deg)
                points.append(self._extrinsics.transform_point(
                    distance_m * math.cos(angle),
                    distance_m * math.sin(angle),
                ))
            if points:
                with self._lock:
                    self._points = np.asarray(points, dtype=float)
                    self._received_at = time.monotonic()

    def snapshot(self):
        with self._lock:
            return self._points, self._received_at


def wait_for_inputs(client, maps, timeout_s, max_pose_age_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        occupancy, generation = maps.snapshot()
        pose_is_fresh = (
            client.pose is not None
            and client.pose_received_at is not None
            and time.monotonic() - client.pose_received_at <= max_pose_age_s
        )
        if occupancy is not None and pose_is_fresh:
            return occupancy, generation, Pose2.from_mapping(client.pose)
        time.sleep(0.02)
    missing = []
    if client.pose is None:
        missing.append(f"{client.namespace}/slam/pose")
    if maps.snapshot()[0] is None:
        missing.append(f"{client.namespace}/slam/map")
    raise TimeoutError(
        "no fresh navigation input received" +
        (": " + ", ".join(missing) if missing else "") +
        "; start scripts/kiwi_slam.py first"
    )


def wait_for_fresh_pose(client, max_age_s, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (
            client.pose_received_at is not None
            and time.monotonic() - client.pose_received_at <= max_age_s
        ):
            return
        time.sleep(0.02)
    raise TimeoutError(f"no fresh SLAM pose for {timeout_s:g} s; robot stopped")


def wait_for_fresh_lidar(lidar, max_age_s, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        points, received_at = lidar.snapshot()
        if (points is not None and received_at is not None and
                time.monotonic() - received_at <= max_age_s):
            return points
        time.sleep(0.02)
    raise TimeoutError(f"no fresh LiDAR scan for {timeout_s:g} s; robot stopped")


def trajectory_payload(points, planner, occupancy, *, collision_radius_m=None):
    collision_radius = (
        planner.inflation_radius_m
        if collision_radius_m is None else float(collision_radius_m))
    return {
        "frame": "map",
        "planner": "astar",
        "inflation_radius_m": planner.inflation_radius_m,
        "runtime_collision_radius_m": collision_radius,
        "tracking_buffer_m": max(
            0.0, planner.inflation_radius_m - collision_radius),
        "map_keyframes": occupancy.keyframes,
        "points": [
            {"x": float(point[0]), "y": float(point[1])}
            for point in points
        ],
    }


def follower_state_payload(status, current, goal, output=None, message=None,
                           *, action_id=None, distance_traveled_m=None,
                           max_travel_distance_m=None):
    state = {
        "frame": "map",
        "status": status,
        "pose": {"x": current.x, "y": current.y, "yaw": current.yaw},
        "goal": {"x": goal.x, "y": goal.y, "yaw": goal.yaw},
    }
    if output is not None:
        heading_setpoint = output.following_pose.yaw
        state.update({
            "following_point": {
                "x": output.following_pose.x,
                "y": output.following_pose.y,
                "yaw": output.following_pose.yaw,
            },
            "progress_m": output.progress_m,
            "remaining_m": output.remaining_m,
            "cross_track_error_m": output.cross_track_error_m,
            "heading_setpoint_rad": heading_setpoint,
            "heading_error_rad": wrap_angle(
                heading_setpoint - current.yaw),
            "pursuit_path_clear": output.pursuit_path_clear,
            "command": {
                "vx": output.command.vx,
                "vy": output.command.vy,
                "omega": output.command.omega,
            },
        })
    if message:
        state["message"] = str(message)
    if action_id is not None:
        state["action_id"] = str(action_id)
    if distance_traveled_m is not None:
        state["distance_traveled_m"] = float(distance_traveled_m)
    if max_travel_distance_m is not None:
        state["max_travel_distance_m"] = float(max_travel_distance_m)
        state["distance_budget_remaining_m"] = max(
            0.0, float(max_travel_distance_m) -
            float(distance_traveled_m or 0.0))
    return state


def _publish_json(publisher, value):
    publisher.put(json.dumps(value, separators=(",", ":")))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal_x", type=float, help="map-frame goal X in meters")
    parser.add_argument("goal_y", type=float, help="map-frame goal Y in meters")
    parser.add_argument(
        "--goal-yaw-deg", type=float,
        help="final map-frame heading; defaults to the final path direction")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument(
        "--command-topic", default="cmd_vel",
        help=("namespaced command topic suffix; launch.py uses "
              "cmd_vel/navigation behind the command mux"))
    parser.add_argument("--robot-yaw-deg", type=float,
                        default=DEFAULT_ROBOT_YAW_DEG)
    parser.add_argument("--inflation-radius", type=float, default=0.25,
                        help="metric A* obstacle inflation radius (default 0.25 m)")
    parser.add_argument(
        "--runtime-collision-radius", type=float,
        default=DEFAULT_RUNTIME_COLLISION_RADIUS_M,
        help=("hard live-following collision radius; must not exceed the A* "
              "inflation radius (default 0.18 m)"),
    )
    parser.add_argument("--occupied-threshold", type=int, default=65)
    parser.add_argument(
        "--allow-unknown", action="store_true",
        help="allow A* through unobserved cells (off by default for safety)")
    parser.add_argument("--lookahead", type=float, default=0.30,
                        help="pure-pursuit lookahead distance (default 0.30 m)")
    parser.add_argument(
        "--goal-yaw-blend-distance", type=float, default=0.30,
        help=("distance over which path heading blends into the final saved "
              "heading (default 0.30 m)"),
    )
    parser.add_argument("--kp-x", type=float, default=0.8)
    parser.add_argument("--kp-y", type=float, default=0.8)
    parser.add_argument("--kp-yaw", type=float, default=2.5)
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=DEFAULT_MAX_FOLLOWING_SPEED_MPS,
        help=("maximum trajectory-following speed "
              f"(default {DEFAULT_MAX_FOLLOWING_SPEED_MPS:g} m/s)"),
    )
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument("--position-tolerance", type=float, default=0.04)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--replan-distance", type=float, default=0.35,
                        help="replan after this much cross-track error")
    parser.add_argument(
        "--localization-jump-distance", type=float, default=0.25,
        help="replan instead of following across a larger SLAM pose jump")
    parser.add_argument(
        "--localization-jump-angle-deg", type=float, default=15.0,
        help="replan instead of following across a larger SLAM yaw jump")
    parser.add_argument(
        "--lidar-collision-horizon", type=float,
        default=DEFAULT_LIDAR_COLLISION_HORIZON_S,
        help="seconds of commanded/measured translation checked in raw LiDAR")
    parser.add_argument("--max-lidar-age", type=float, default=0.5)
    parser.add_argument(
        "--calibration", help="yaw/LiDAR calibration YAML/JSON file")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--input-timeout", type=float, default=8.0)
    parser.add_argument("--max-pose-age", type=float, default=0.5)
    parser.add_argument("--pose-recovery-timeout", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument(
        "--action-id",
        help="opaque coordinator action ID copied into navigation state")
    parser.add_argument(
        "--max-travel-distance", type=float,
        help=("hard total translated-distance envelope; replanning must also "
              "fit inside its remaining budget"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    positive = {
        "--lookahead": args.lookahead,
        "--goal-yaw-blend-distance": args.goal_yaw_blend_distance,
        "--max-linear-speed": args.max_linear_speed,
        "--max-angular-speed": args.max_angular_speed,
        "--position-tolerance": args.position_tolerance,
        "--yaw-tolerance-deg": args.yaw_tolerance_deg,
        "--replan-distance": args.replan_distance,
        "--localization-jump-distance": args.localization_jump_distance,
        "--localization-jump-angle-deg": args.localization_jump_angle_deg,
        "--lidar-collision-horizon": args.lidar_collision_horizon,
        "--max-lidar-age": args.max_lidar_age,
        "--rate": args.rate,
        "--input-timeout": args.input_timeout,
        "--max-pose-age": args.max_pose_age,
        "--pose-recovery-timeout": args.pose_recovery_timeout,
        "--max-duration": args.max_duration,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be positive")
    if not math.isfinite(args.inflation_radius) or args.inflation_radius < 0.0:
        parser.error("--inflation-radius must be non-negative")
    if (not math.isfinite(args.runtime_collision_radius) or
            args.runtime_collision_radius < 0.0):
        parser.error("--runtime-collision-radius must be non-negative")
    if args.runtime_collision_radius > args.inflation_radius + 1e-12:
        parser.error(
            "--runtime-collision-radius must not exceed --inflation-radius")
    if (args.max_travel_distance is not None and
            (not math.isfinite(args.max_travel_distance) or
             args.max_travel_distance <= 0.0)):
        parser.error("--max-travel-distance must be positive")
    if not 0 <= args.occupied_threshold <= 100:
        parser.error("--occupied-threshold must be in [0, 100]")
    if not np.isfinite((args.goal_x, args.goal_y)).all():
        parser.error("goal coordinates must be finite")
    if (not math.isfinite(args.robot_yaw_deg) or
            (args.goal_yaw_deg is not None and
             not math.isfinite(args.goal_yaw_deg))):
        parser.error("frame and goal yaw values must be finite")

    calibration = None
    if args.calibration:
        try:
            calibration = load_calibration(args.calibration)
        except ValueError as exc:
            parser.error(str(exc))
    lidar_calibration = calibration.lidar if calibration is not None else None
    lidar_extrinsics = LidarExtrinsics(
        x_m=lidar_calibration.x_m if lidar_calibration is not None else 0.0,
        y_m=lidar_calibration.y_m if lidar_calibration is not None else 0.0,
        yaw_rad=math.radians(
            lidar_calibration.yaw_deg if lidar_calibration is not None else 0.0),
    )
    for name, value in {
        "--kp-x": args.kp_x,
        "--kp-y": args.kp_y,
        "--kp-yaw": args.kp_yaw,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{name} must be non-negative")

    controller = PoseStabilizingController(
        kp_x=args.kp_x,
        kp_y=args.kp_y,
        kp_yaw=args.kp_yaw,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        position_tolerance=args.position_tolerance,
        yaw_tolerance=math.radians(args.yaw_tolerance_deg),
    )
    maps = LiveMapBuffer()
    lidar = LiveLidarBuffer(lidar_extrinsics)
    wheel_distance = WheelDistanceTracker()
    client = KiwiClient(
        args.connect,
        args.namespace,
        args.robot_yaw_deg,
        commanding=True,
        on_odometry=wheel_distance.update,
        command_suffix=args.command_topic,
    )
    client.subscribe("slam/map", maps.update)
    client.subscribe("lidar/ld19/raw", lidar.update)
    trajectory_publisher = client.session.declare_publisher(
        f"{client.namespace}/navigation/trajectory")
    state_publisher = client.session.declare_publisher(
        f"{client.namespace}/navigation/state")

    follower = None
    planner_instance = None
    route_planner_instance = None
    goal_yaw = (None if args.goal_yaw_deg is None
                else math.radians(args.goal_yaw_deg))
    last_map_generation = -1
    last_trajectory_payload = None
    distance_traveled_m = 0.0
    last_control_pose = None
    live_overlay = None
    live_overlay_generation = -1
    last_lidar_replan_at = -math.inf
    collision_guard = SweptCircleCollisionGuard(
        args.runtime_collision_radius, args.lidar_collision_horizon)
    try:
        occupancy, last_map_generation, current = wait_for_inputs(
            client, maps, args.input_timeout, args.max_pose_age)
        wait_for_fresh_lidar(lidar, args.max_lidar_age, args.input_timeout)
        last_control_pose = current

        def replan(active_map, pose, *, recovery=False):
            nonlocal follower, planner_instance, route_planner_instance
            nonlocal last_trajectory_payload
            preferred_planner = AStarPlanner(
                active_map,
                inflation_radius_m=args.inflation_radius,
                occupied_threshold=args.occupied_threshold,
                allow_unknown=args.allow_unknown,
            )
            runtime_planner = AStarPlanner(
                active_map,
                inflation_radius_m=args.runtime_collision_radius,
                occupied_threshold=args.occupied_threshold,
                allow_unknown=args.allow_unknown,
            )
            start_xy = (pose.x, pose.y)
            start_cell = preferred_planner.world_to_cell(start_xy)
            soft_start_recovery = (
                not preferred_planner.cell_is_free(start_cell)
                and runtime_planner.cell_is_free(start_cell))
            try:
                points = preferred_planner.plan_with_start_recovery(
                    start_xy, (args.goal_x, args.goal_y), runtime_planner)
                route_planner_instance = preferred_planner
                if soft_start_recovery:
                    print(
                        f"start is inside the {args.inflation_radius:.2f} m "
                        f"preferred buffer but outside the "
                        f"{args.runtime_collision_radius:.2f} m hard envelope; "
                        "planning a hard-safe egress"
                    )
            except PathNotFound:
                if not recovery:
                    raise
                # Preserve the previous recovery behavior for a route whose
                # updated obstacles leave no preferred-clearance continuation.
                route_planner_instance = runtime_planner
                points = runtime_planner.plan(
                    start_xy, (args.goal_x, args.goal_y))
                print(
                    "preferred-clearance recovery path unavailable; using "
                    f"the {args.runtime_collision_radius:.2f} m hard envelope"
                )
            planner_instance = runtime_planner
            follower = PurePursuitFollower(
                points,
                controller,
                lookahead_m=args.lookahead,
                goal_yaw=goal_yaw,
                goal_yaw_blend_distance_m=args.goal_yaw_blend_distance,
            )
            if (args.max_travel_distance is not None and
                    distance_traveled_m + follower.length_m >
                    args.max_travel_distance + 1e-9):
                raise PathNotFound(
                    f"route requires {follower.length_m:.3f} m with "
                    f"{distance_traveled_m:.3f} m already traveled, exceeding "
                    f"the {args.max_travel_distance:.3f} m authorization")
            last_trajectory_payload = trajectory_payload(
                points, route_planner_instance, active_map,
                collision_radius_m=planner_instance.inflation_radius_m)
            _publish_json(trajectory_publisher, last_trajectory_payload)
            print(
                f"A* path: {len(points)} vertices, {follower.length_m:.2f} m, "
                f"planning inflation {route_planner_instance.inflation_radius_m:.2f} m, "
                f"runtime collision radius {planner_instance.inflation_radius_m:.2f} m, "
                f"map keyframes {active_map.keyframes}"
            )

        replan(occupancy, current)
        active_goal_yaw = follower.goal_yaw
        goal = Pose2(args.goal_x, args.goal_y, active_goal_yaw)
        print(
            f"following to ({goal.x:+.2f}, {goal.y:+.2f}, "
            f"{math.degrees(goal.yaw):+.1f} deg) at {args.rate:g} Hz"
        )

        def wait_on_transient_start_block(exc, pose, output, context):
            message = str(exc)
            if not message.startswith("start is outside"):
                return False
            client.send_twist(0.0, 0.0, 0.0)
            print(f"{context} waiting for map clearance: {message}")
            _publish_json(state_publisher, follower_state_payload(
                "replanning", pose, goal, output,
                f"{context}; waiting for map clearance: {message}",
                action_id=args.action_id,
                distance_traveled_m=distance_traveled_m,
                max_travel_distance_m=args.max_travel_distance))
            time.sleep(period_s)
            return True

        period_s = 1.0 / args.rate
        deadline = time.monotonic() + args.max_duration
        next_cycle = time.monotonic()
        next_status_publish = 0.0
        next_trajectory_publish = time.monotonic() + 1.0
        last_output = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            if client.pose_received_at is None or \
                    now - client.pose_received_at > args.max_pose_age:
                client.send_twist(0.0, 0.0, 0.0)
                if client.pose is not None:
                    stale_pose = Pose2.from_mapping(client.pose)
                    _publish_json(state_publisher, follower_state_payload(
                        "paused", stale_pose, goal, last_output,
                        "SLAM pose is stale", action_id=args.action_id,
                        distance_traveled_m=distance_traveled_m,
                        max_travel_distance_m=args.max_travel_distance))
                print("SLAM pose stale; stopped and waiting for recovery")
                paused_at = time.monotonic()
                wait_for_fresh_pose(
                    client, args.max_pose_age, args.pose_recovery_timeout)
                pause_duration = time.monotonic() - paused_at
                deadline += pause_duration
                next_cycle = time.monotonic()
                print("SLAM pose recovered; resuming")
                continue

            current = Pose2.from_mapping(client.pose)
            # A SLAM correction is not physical travel. Integrate measured
            # encoder speed so relocalization cannot consume the route budget.
            distance_traveled_m = wheel_distance.distance_m
            if (args.max_travel_distance is not None and
                    distance_traveled_m > args.max_travel_distance + 1e-9):
                client.send_twist(0.0, 0.0, 0.0)
                raise PathNotFound(
                    f"authorized travel distance "
                    f"{args.max_travel_distance:.3f} m exhausted")
            base_map, map_generation = maps.snapshot()
            map_changed = map_generation != last_map_generation
            if map_changed:
                last_map_generation = map_generation
                live_overlay = None
                live_overlay_generation = -1
            active_map = (
                live_overlay
                if live_overlay is not None and
                live_overlay_generation == map_generation
                else base_map
            )
            if last_control_pose is not None:
                pose_jump = math.hypot(
                    current.x - last_control_pose.x,
                    current.y - last_control_pose.y)
                yaw_jump = abs(math.atan2(
                    math.sin(current.yaw - last_control_pose.yaw),
                    math.cos(current.yaw - last_control_pose.yaw)))
                if (pose_jump > args.localization_jump_distance or
                        yaw_jump > math.radians(
                            args.localization_jump_angle_deg)):
                    client.send_twist(0.0, 0.0, 0.0)
                    print(
                        f"SLAM correction {pose_jump:.2f} m / "
                        f"{math.degrees(yaw_jump):.1f} deg; replanning "
                        "without charging it to wheel travel"
                    )
                    try:
                        replan(active_map, current, recovery=True)
                    except PathNotFound as exc:
                        if wait_on_transient_start_block(
                                exc, current, last_output,
                                "SLAM-correction replan"):
                            last_control_pose = current
                            continue
                        raise
                    goal = Pose2(
                        args.goal_x, args.goal_y, follower.goal_yaw)
            last_control_pose = current
            if map_changed:
                candidate_route_planner = AStarPlanner(
                    active_map,
                    inflation_radius_m=args.inflation_radius,
                    occupied_threshold=args.occupied_threshold,
                    allow_unknown=args.allow_unknown,
                )
                candidate_planner = AStarPlanner(
                    active_map,
                    inflation_radius_m=args.runtime_collision_radius,
                    occupied_threshold=args.occupied_threshold,
                    allow_unknown=args.allow_unknown,
                )
                current_cell = candidate_planner.world_to_cell(
                    (current.x, current.y))
                remaining = follower.remaining_trajectory()
                if candidate_route_planner.cell_is_free(current_cell):
                    remaining_path_is_free = \
                        candidate_route_planner.path_is_free(remaining)
                else:
                    remaining_path_is_free = \
                        candidate_route_planner.path_is_free_with_start_recovery(
                            remaining, candidate_planner)
                if (not candidate_planner.cell_is_free(current_cell) or
                        not remaining_path_is_free):
                    client.send_twist(0.0, 0.0, 0.0)
                    print("updated SLAM map blocks the trajectory; replanning")
                    try:
                        replan(active_map, current, recovery=True)
                    except PathNotFound as exc:
                        if wait_on_transient_start_block(
                                exc, current, last_output,
                                "updated-map replan"):
                            continue
                        raise
                    goal = Pose2(args.goal_x, args.goal_y, follower.goal_yaw)
                else:
                    planner_instance = candidate_planner
                    route_planner_instance = candidate_route_planner
                    last_trajectory_payload["map_keyframes"] = \
                        active_map.keyframes

            output = follower.update(current, planner_instance.path_is_free)
            last_output = output
            if not output.pursuit_path_clear:
                client.send_twist(0.0, 0.0, 0.0)
                print("no collision-free lookahead segment; replanning")
                try:
                    replan(active_map, current, recovery=True)
                except PathNotFound as exc:
                    if wait_on_transient_start_block(
                            exc, current, output, "lookahead replan"):
                        continue
                    raise
                goal = Pose2(args.goal_x, args.goal_y, follower.goal_yaw)
                output = follower.update(current, planner_instance.path_is_free)
                last_output = output
                if not output.pursuit_path_clear:
                    raise PathNotFound(
                        "no collision-free pursuit segment from the current pose")
            if output.cross_track_error_m > args.replan_distance:
                client.send_twist(0.0, 0.0, 0.0)
                print(
                    f"cross-track error {output.cross_track_error_m:.2f} m; "
                    "replanning"
                )
                try:
                    replan(active_map, current)
                except PathNotFound as exc:
                    if wait_on_transient_start_block(
                            exc, current, output, "cross-track replan"):
                        continue
                    raise
                goal = Pose2(args.goal_x, args.goal_y, follower.goal_yaw)
                output = follower.update(current, planner_instance.path_is_free)
                last_output = output

            if output.complete:
                client.send_twist(0.0, 0.0, 0.0)
                _publish_json(state_publisher, follower_state_payload(
                    "reached", current, goal, output,
                    action_id=args.action_id,
                    distance_traveled_m=distance_traveled_m,
                    max_travel_distance_m=args.max_travel_distance))
                print(
                    f"goal reached: ({current.x:+.3f}, {current.y:+.3f}, "
                    f"{math.degrees(current.yaw):+.1f} deg)"
                )
                return

            lidar_points, lidar_received_at = lidar.snapshot()
            if (lidar_points is None or lidar_received_at is None or
                    now - lidar_received_at > args.max_lidar_age):
                client.send_twist(0.0, 0.0, 0.0)
                _publish_json(state_publisher, follower_state_payload(
                    "paused", current, goal, output,
                    "raw LiDAR is stale", action_id=args.action_id,
                    distance_traveled_m=distance_traveled_m,
                    max_travel_distance_m=args.max_travel_distance))
                print("raw LiDAR stale; stopped and waiting for recovery")
                paused_at = time.monotonic()
                wait_for_fresh_lidar(
                    lidar, args.max_lidar_age, args.pose_recovery_timeout)
                deadline += time.monotonic() - paused_at
                next_cycle = time.monotonic()
                continue

            measured = ((client.odometry or {}).get("measured", {})
                        if isinstance(client.odometry, dict) else {})
            measured_vx = float(measured.get("vx", 0.0))
            measured_vy = float(measured.get("vy", 0.0))
            lidar_blocked = (
                collision_guard.blocks(
                    lidar_points, output.command.vx, output.command.vy)
                or collision_guard.blocks(
                    lidar_points, measured_vx, measured_vy)
            )
            if lidar_blocked:
                client.send_twist(0.0, 0.0, 0.0)
                if now - last_lidar_replan_at >= 0.5:
                    last_lidar_replan_at = now
                    live_overlay = stamp_lidar_obstacles(
                        base_map, current, lidar_points)
                    live_overlay_generation = map_generation
                    active_map = live_overlay
                    print(
                        "raw LiDAR blocks the swept footprint; "
                        "stamping live obstacles and replanning"
                    )
                    try:
                        replan(active_map, current, recovery=True)
                        goal = Pose2(
                            args.goal_x, args.goal_y, follower.goal_yaw)
                    except PathNotFound as exc:
                        print(f"live obstacle replan waiting: {exc}")
                _publish_json(state_publisher, follower_state_payload(
                    "replanning", current, goal, output,
                    "raw LiDAR blocks the swept footprint",
                    action_id=args.action_id,
                    distance_traveled_m=distance_traveled_m,
                    max_travel_distance_m=args.max_travel_distance))
                next_cycle += period_s
                time.sleep(max(0.0, next_cycle - time.monotonic()))
                continue

            client.send_twist(
                output.command.vx, output.command.vy, output.command.omega,
                hold_s=period_s)
            if now >= next_status_publish:
                _publish_json(state_publisher, follower_state_payload(
                    "following", current, goal, output,
                    action_id=args.action_id,
                    distance_traveled_m=distance_traveled_m,
                    max_travel_distance_m=args.max_travel_distance))
                next_status_publish = now + 0.1
            if now >= next_trajectory_publish:
                _publish_json(trajectory_publisher, last_trajectory_payload)
                next_trajectory_publish = now + 1.0

            next_cycle += period_s
            time.sleep(max(0.0, next_cycle - time.monotonic()))

        client.send_twist(0.0, 0.0, 0.0)
        current = Pose2.from_mapping(client.pose)
        _publish_json(state_publisher, follower_state_payload(
            "timed_out", current, goal, last_output,
            f"maximum duration {args.max_duration:g} s exceeded",
            action_id=args.action_id,
            distance_traveled_m=distance_traveled_m,
            max_travel_distance_m=args.max_travel_distance))
        raise TimeoutError(f"navigation timed out after {args.max_duration:g} s")
    except (PathNotFound, TimeoutError) as exc:
        client.send_twist(0.0, 0.0, 0.0)
        if client.pose is not None:
            current = Pose2.from_mapping(client.pose)
            fallback_yaw = current.yaw if goal_yaw is None else goal_yaw
            goal = Pose2(args.goal_x, args.goal_y, fallback_yaw)
            _publish_json(state_publisher, follower_state_payload(
                "failed", current, goal, message=exc,
                action_id=args.action_id,
                distance_traveled_m=distance_traveled_m,
                max_travel_distance_m=args.max_travel_distance))
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        print("\nnavigation stopped")
    finally:
        client.close()


if __name__ == "__main__":
    main()
