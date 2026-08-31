#!/usr/bin/env python3
"""Drive 0.5 m forward, 0.5 m left, and 0 rad relative to the current pose.

Run ``kiwi_slam.py`` first so ``<namespace>/slam/pose`` is available. This
script reads that pose, composes the relative reference in the robot's current
frame, and closes the pose loop entirely on the laptop in Python.
"""

import argparse
import math
import time
import numpy as np
from kiwi_client import DEFAULT_ROBOT_YAW_DEG, KiwiClient
from kiwi_pose_controller import (
    Pose2,
    PoseStabilizingController,
    compose_relative_pose,
)


def wait_for_pose(client, timeout_s):
    deadline = time.monotonic() + timeout_s
    while client.pose is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if client.pose is None:
        raise TimeoutError(
            f"no pose received on {client.namespace}/slam/pose; "
            "start scripts/kiwi_slam.py first"
        )
    return Pose2.from_mapping(client.pose)


def wait_for_fresh_pose(client, max_age_s, recovery_timeout_s):
    """Wait for SLAM to resume after commanding a safe stop."""
    deadline = time.monotonic() + recovery_timeout_s
    while time.monotonic() < deadline:
        received_at = client.pose_received_at
        if (
            received_at is not None
            and time.monotonic() - received_at <= max_age_s
        ):
            return
        time.sleep(0.02)
    raise TimeoutError(
        f"no fresh SLAM pose for {recovery_timeout_s:g} s; stopping robot"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--robot-yaw-deg", type=float,
                        default=DEFAULT_ROBOT_YAW_DEG)
    parser.add_argument("--kp-x", type=float, default=0.8)
    parser.add_argument("--kp-y", type=float, default=0.8)
    parser.add_argument("--kp-yaw", type=float, default=1.5)
    parser.add_argument("--max-linear-speed", type=float, default=0.25)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument("--position-tolerance", type=float, default=0.02)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--pose-timeout", type=float, default=5.0)
    parser.add_argument("--max-pose-age", type=float, default=0.5)
    parser.add_argument("--pose-recovery-timeout", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()

    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    if args.max_duration <= 0.0:
        parser.error("--max-duration must be positive")
    if args.max_pose_age <= 0.0:
        parser.error("--max-pose-age must be positive")
    if args.pose_recovery_timeout <= 0.0:
        parser.error("--pose-recovery-timeout must be positive")

    controller = PoseStabilizingController(
        kp_x=args.kp_x,
        kp_y=args.kp_y,
        kp_yaw=args.kp_yaw,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        position_tolerance=args.position_tolerance,
        yaw_tolerance=math.radians(args.yaw_tolerance_deg),
    )
    client = KiwiClient(
        args.connect,
        args.namespace,
        args.robot_yaw_deg,
        commanding=True,
    )
    try:
        start = wait_for_pose(client, args.pose_timeout)
        target = compose_relative_pose(start, Pose2(1.0, 0.0, np.pi))
        print(
            f"current: x={start.x:+.3f} y={start.y:+.3f} "
            f"yaw={math.degrees(start.yaw):+.1f} deg"
        )
        print(
            f"target:  x={target.x:+.3f} y={target.y:+.3f} "
            f"yaw={math.degrees(target.yaw):+.1f} deg"
        )

        period_s = 1.0 / args.rate
        deadline = time.monotonic() + args.max_duration
        next_print = 0.0
        reached = False
        while time.monotonic() < deadline:
            if (
                client.pose_received_at is None
                or time.monotonic() - client.pose_received_at > args.max_pose_age
            ):
                client.send_twist(0.0, 0.0, 0.0)
                print("\nSLAM pose is stale; stopped and waiting for recovery...")
                paused_at = time.monotonic()
                wait_for_fresh_pose(
                    client,
                    args.max_pose_age,
                    args.pose_recovery_timeout,
                )
                deadline += time.monotonic() - paused_at
                next_print = 0.0
                print("SLAM pose recovered; resuming.")
                continue
            current = Pose2.from_mapping(client.pose)
            if controller.at_target(current, target):
                reached = True
                break
            command = controller.command(current, target)
            client.send_twist(
                command.vx, command.vy, command.omega,
                hold_s=period_s)
            now = time.monotonic()
            if now >= next_print:
                error = controller.error(current, target)
                print(
                    f"\rerror x={error.x:+.3f} y={error.y:+.3f} "
                    f"yaw={math.degrees(error.yaw):+.1f} deg | "
                    f"cmd vx={command.vx:+.3f} vy={command.vy:+.3f} "
                    f"omega={command.omega:+.3f}",
                    end="",
                    flush=True,
                )
                next_print = now + 0.25
            time.sleep(period_s)

        client.send_twist(0.0, 0.0, 0.0)
        final_pose = Pose2.from_mapping(client.pose)
        final_error = controller.error(final_pose, target)
        print(
            f"\n{'target reached' if reached else 'timed out'}: "
            f"x error={final_error.x:+.3f} m, "
            f"y error={final_error.y:+.3f} m, "
            f"yaw error={math.degrees(final_error.yaw):+.1f} deg"
        )
        if not reached:
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        raise SystemExit(str(exc)) from None
