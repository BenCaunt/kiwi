#!/usr/bin/env python3
"""Live Rerun dashboard for Kiwi sensors and the live SLAM occupancy map.

Subscribes to kiwi/xiao/** over Zenoh and streams the camera, SLAM map and
corrected pose, lidar, twists, wheels, accelerometer, and system stats into a
Rerun viewer.

Run:  python3 scripts/kiwi_dashboard.py            # spawns the rerun viewer
      python3 scripts/kiwi_dashboard.py --connect tcp/127.0.0.1:7447
"""

import argparse
from collections import deque
import json
import math
import sys
import threading
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, "scripts")
from kiwi_client import DEFAULT_ROBOT_YAW_DEG, KiwiClient  # noqa: E402
from kiwi_calibration import load_calibration  # noqa: E402
from kiwi_lidar import parse_frames, ScanAssembler  # noqa: E402
from kiwi_lidar_deskew import (  # noqa: E402
    LidarExtrinsics,
    PoseHistory,
    SensorClock,
    TimedScanAssembler,
    deskew_scan,
    unwrap_angle,
)
from kiwi_image_map import (  # noqa: E402
    decode_camera_sample,
    decode_image_capture,
)
from kiwi_map import decode_occupancy_map  # noqa: E402


def make_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="camera", origin="/camera"),
                rrb.Spatial3DView(name="image-correlated map", origin="/map"),
                rrb.Spatial3DView(name="robot + lidar", origin="/world"),
                row_shares=[2, 4, 3],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="twist vx", origin="/twist/vx"),
                rrb.TimeSeriesView(name="twist vy", origin="/twist/vy"),
                rrb.TimeSeriesView(name="twist omega", origin="/twist/omega"),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="wheel speeds", origin="/wheels"),
                rrb.TimeSeriesView(name="imu accel", origin="/imu/accel"),
                rrb.TimeSeriesView(name="system", origin="/system"),
            ),
            column_shares=[4, 3, 3],
        ),
        collapse_panels=True,
    )


def yaw_from_quat(i, j, k, r):
    return math.atan2(2.0 * (r * k + i * j), 1.0 - 2.0 * (j * j + k * k))


def camera_to_map_rotation(yaw):
    """Return the camera RDF-to-map FLU rotation for a planar robot yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    robot_from_camera = np.array((
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ))
    map_from_robot = np.array((
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    ))
    return map_from_robot @ robot_from_camera


class Dashboard:
    def __init__(self, lidar_deskew=True, lidar_time_offset_ms=0.0,
                 lidar_extrinsics=None):
        self.pose = [0.0, 0.0, 0.0]  # x, y, yaw (dead-reckoned, IMU heading)
        self.yaw_offset = None
        self.last_twist_t = None
        self.trajectory = []
        self.lidar_deskew = lidar_deskew
        self.lidar_time_offset_s = lidar_time_offset_ms / 1000.0
        self.lidar_extrinsics = lidar_extrinsics or LidarExtrinsics()
        self.assembler = ScanAssembler()
        self.timed_assembler = TimedScanAssembler()
        self.lidar_clock = SensorClock(wrap_seconds=30.0)
        self.pose_clock = SensorClock()
        self.pose_history = PoseHistory()
        self.pending_scans = deque(maxlen=20)
        self.deskewed_scan_count = 0
        self.map_to_odom = None
        self.map_keyframes = 0
        self.navigation_trajectory = []
        self.navigation_status = None
        self.image_capture_ids = set()
        self.image_capture_poses = {}
        self.image_session_id = None
        self.lock = threading.RLock()

    def on_camera(self, payload):
        sample = decode_camera_sample(bytes(payload))
        if sample is None:
            return
        rr.log("/camera", rr.Image(sample.pixels))

    def on_twist(self, m):
        with self.lock:
            meas, cmd = m.get("measured", {}), m.get("command", {})
            for axis in ("vx", "vy", "omega"):
                rr.log(f"/twist/{axis}/measured", rr.Scalar(meas.get(axis, 0.0)))
                rr.log(f"/twist/{axis}/commanded", rr.Scalar(cmd.get(axis, 0.0)))
            for i, w in enumerate(m.get("wheel_speed_mps", [])):
                rr.log(f"/wheels/w{i}", rr.Scalar(w))
            accel = m.get("imu_accel_mps2", [0, 0, 0])
            for name, val in zip("xyz", accel):
                rr.log(f"/imu/accel/{name}", rr.Scalar(val))

            # Use the follower timestamp for pose integration and history. The
            # arrival time is only used to align its clock with the LD19 clock.
            arrival_t = time.monotonic()
            follower_time_us = m.get("follower_time_us")
            if isinstance(follower_time_us, (int, float)):
                pose_t = follower_time_us / 1_000_000.0
            else:
                pose_t = arrival_t
            self.pose_clock.observe(pose_t, arrival_t)

            # Pose: IMU yaw for heading, integrate measured body twist for position.
            q = m.get("imu_quat_ijkr")
            if q and m.get("imu_ready"):
                yaw = yaw_from_quat(*q)
                if self.yaw_offset is None:
                    self.yaw_offset = yaw
                relative_yaw = yaw - self.yaw_offset
                self.pose[2] = unwrap_angle(relative_yaw, self.pose[2])
            if self.last_twist_t is not None:
                dt = min(max(pose_t - self.last_twist_t, 0.0), 0.2)
                vx, vy = meas.get("vx", 0.0), meas.get("vy", 0.0)
                c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
                self.pose[0] += (c * vx - s * vy) * dt
                self.pose[1] += (s * vx + c * vy) * dt
            self.last_twist_t = pose_t
            self.pose_history.append(pose_t, *self.pose)

            display_pose = self._map_pose(self.pose)
            self._log_robot_pose(display_pose)
            self.trajectory.append([display_pose[0], display_pose[1], 0.0])
            if len(self.trajectory) > 3000:
                del self.trajectory[:1000]
            rr.log("/world/trajectory", rr.LineStrips3D([self.trajectory]))
            rr.log("/map/trajectory", rr.LineStrips3D([self.trajectory]))
            self._flush_deskewed_scans(arrival_t)

    def _map_pose(self, odom_pose):
        if self.map_to_odom is None:
            return list(odom_pose)
        tx, ty, yaw = self.map_to_odom
        c, s = math.cos(yaw), math.sin(yaw)
        return [
            tx + c * odom_pose[0] - s * odom_pose[1],
            ty + s * odom_pose[0] + c * odom_pose[1],
            odom_pose[2] + yaw,
        ]

    def _log_robot_pose(self, pose):
        rr.log("/world/robot",
               rr.Transform3D(translation=[pose[0], pose[1], 0.0],
                              rotation=rr.Quaternion(
                                  xyzw=[0.0, 0.0, math.sin(pose[2] / 2),
                                        math.cos(pose[2] / 2)])))
        rr.log("/map/robot",
               rr.Transform3D(translation=[pose[0], pose[1], 0.0],
                              rotation=rr.Quaternion(
                                  xyzw=[0.0, 0.0, math.sin(pose[2] / 2),
                                        math.cos(pose[2] / 2)])))

    def on_slam_pose(self, pose):
        """Track map->odom so dashboard geometry follows SLAM corrections."""
        with self.lock:
            correction_yaw = float(pose["yaw"]) - self.pose[2]
            c, s = math.cos(correction_yaw), math.sin(correction_yaw)
            self.map_to_odom = (
                float(pose["x"]) - c * self.pose[0] + s * self.pose[1],
                float(pose["y"]) - s * self.pose[0] - c * self.pose[1],
                correction_yaw,
            )
            self._log_robot_pose(self._map_pose(self.pose))

    def on_slam_report(self, report):
        """Align this dashboard's odometry frame to SLAM's map frame.

        SLAM and the dashboard integrate odometry independently and may start
        at different times, so SLAM's explicit ``map_to_odom`` transform does
        not apply to the dashboard's local odometry origin.  Re-anchor from
        the published map-frame robot pose instead.
        """
        pose = report.get("pose")
        if isinstance(pose, dict):
            self.on_slam_pose(pose)

    def on_map(self, payload):
        """Render occupied cells from kiwi_slam's live occupancy grid."""
        try:
            occupancy = decode_occupancy_map(payload)
        except ValueError:
            return
        rows, columns = np.nonzero(occupancy.data >= 65)
        centers_2d = np.column_stack((
            occupancy.origin_x + columns * occupancy.resolution_m,
            occupancy.origin_y + rows * occupancy.resolution_m,
        ))
        centers = np.column_stack((
            centers_2d,
            np.zeros(len(centers_2d), dtype=np.float32),
        ))
        half_sizes = np.full(
            (len(centers), 3),
            occupancy.resolution_m * 0.5,
            dtype=np.float32,
        )
        half_sizes[:, 2] = 0.005
        rr.log("/map/occupancy", rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            colors=[[200, 205, 215]],
        ))
        self.map_keyframes = occupancy.keyframes
        rr.log("/system/slam_map_keyframes", rr.Scalar(self.map_keyframes))
        if self.map_keyframes == 1:
            print("SLAM map active on dashboard")

    @staticmethod
    def _navigation_pose(value):
        if not isinstance(value, dict):
            raise ValueError("navigation pose must be an object")
        pose = tuple(float(value[axis]) for axis in ("x", "y", "yaw"))
        if not np.isfinite(pose).all():
            raise ValueError("navigation pose must be finite")
        return pose

    def on_navigation_trajectory(self, payload):
        """Render the A* path currently being followed in the map frame."""
        try:
            report = json.loads(bytes(payload).decode())
            if report.get("frame") != "map":
                return
            points = np.array([
                (float(point["x"]), float(point["y"]), 0.025)
                for point in report["points"]
            ], dtype=float)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError):
            return
        if points.ndim != 2 or points.shape[1:] != (3,) or \
                len(points) == 0 or not np.isfinite(points).all():
            return
        self.navigation_trajectory = points[:, :2].tolist()
        rr.log(
            "/map/navigation/path_vertices",
            rr.Points3D(points, radii=0.018, colors=[[70, 145, 255]]),
        )
        if len(points) >= 2:
            rr.log(
                "/map/navigation/planned_trajectory",
                rr.LineStrips3D([points], radii=0.012,
                                colors=[[70, 145, 255]]),
            )
        else:
            rr.log("/map/navigation/planned_trajectory", rr.Clear(recursive=False))
        inflation = report.get("inflation_radius_m")
        if isinstance(inflation, (int, float)) and math.isfinite(inflation):
            rr.log("/system/navigation_inflation_radius_m", rr.Scalar(inflation))

    def on_navigation_state(self, payload):
        """Render controller pose, goal, and pure-pursuit following point."""
        try:
            report = json.loads(bytes(payload).decode())
            if report.get("frame") != "map":
                return
            pose = self._navigation_pose(report["pose"])
            goal = self._navigation_pose(report["goal"])
            following = (self._navigation_pose(report["following_point"])
                         if "following_point" in report else None)
            status = str(report["status"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError):
            return

        self.navigation_status = status
        pose_xyz = [pose[0], pose[1], 0.055]
        rr.log(
            "/map/navigation/current_pose",
            rr.Points3D([pose_xyz], radii=0.045, colors=[[70, 220, 130]],
                        labels=["controller pose"]),
        )
        heading_end = [
            pose[0] + 0.18 * math.cos(pose[2]),
            pose[1] + 0.18 * math.sin(pose[2]),
            pose_xyz[2],
        ]
        rr.log(
            "/map/navigation/current_heading",
            rr.LineStrips3D([[pose_xyz, heading_end]], radii=0.014,
                            colors=[[70, 220, 130]]),
        )
        rr.log(
            "/map/navigation/goal",
            rr.Points3D([[goal[0], goal[1], 0.045]], radii=0.06,
                        colors=[[235, 85, 175]], labels=["goal"]),
        )
        if following is not None:
            rr.log(
                "/map/navigation/following_point",
                rr.Points3D([[following[0], following[1], 0.065]],
                            radii=0.05, colors=[[255, 195, 55]],
                            labels=["pure pursuit"]),
            )
            desired_heading_end = [
                pose[0] + 0.18 * math.cos(following[2]),
                pose[1] + 0.18 * math.sin(following[2]),
                pose_xyz[2] + 0.005,
            ]
            rr.log(
                "/map/navigation/desired_heading",
                rr.LineStrips3D(
                    [[pose_xyz, desired_heading_end]], radii=0.010,
                    colors=[[255, 195, 55]]),
            )
        else:
            rr.log("/map/navigation/following_point", rr.Clear(recursive=False))
            rr.log("/map/navigation/desired_heading", rr.Clear(recursive=False))

        for key in ("progress_m", "remaining_m", "cross_track_error_m"):
            value = report.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                rr.log(f"/system/navigation_{key}", rr.Scalar(value))

        if following is not None:
            heading_error = report.get(
                "heading_error_rad",
                math.atan2(
                    math.sin(following[2] - pose[2]),
                    math.cos(following[2] - pose[2]),
                ),
            )
            rr.log("/system/navigation_heading_deg", rr.Scalar(
                math.degrees(pose[2])))
            rr.log("/system/navigation_heading_setpoint_deg", rr.Scalar(
                math.degrees(following[2])))
            if isinstance(heading_error, (int, float)) and math.isfinite(
                    heading_error):
                rr.log("/system/navigation_heading_error_deg", rr.Scalar(
                    math.degrees(heading_error)))
        command = report.get("command")
        if isinstance(command, dict):
            omega = command.get("omega")
            if isinstance(omega, (int, float)) and math.isfinite(omega):
                rr.log("/system/navigation_command_omega_rad_s", rr.Scalar(
                    omega))

    def on_image_capture(self, payload):
        """Render one saved image as a pinhole camera in the 3D map scene."""
        try:
            metadata, jpeg = decode_image_capture(payload)
            capture_id = int(metadata["id"])
            session_id = str(metadata.get("session_id", "legacy"))
            pose = metadata["pose"]
            camera = metadata["camera"]
            x = float(pose["x"])
            y = float(pose["y"])
            yaw = float(pose["yaw"])
            height_m = float(camera["height_m"])
            width = int(camera["width"])
            height = int(camera["height"])
            fx, fy = float(camera["fx"]), float(camera["fy"])
            cx, cy = float(camera["cx"]), float(camera["cy"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
            return

        if self.image_session_id != session_id:
            rr.log("/map/captures", rr.Clear(recursive=True), static=True)
            self.image_session_id = session_id
            self.image_capture_ids.clear()
            self.image_capture_poses.clear()

        capture_pose = (x, y, yaw, height_m)
        previous_pose = self.image_capture_poses.get(capture_id)
        if previous_pose == capture_pose:
            return

        camera_path = f"/map/captures/{capture_id:06d}/camera"
        rr.log(
            camera_path,
            rr.Transform3D(
                translation=[x, y, height_m],
                mat3x3=camera_to_map_rotation(yaw),
            ),
            static=True,
        )
        first_seen = previous_pose is None
        if first_seen:
            rr.log(
                camera_path,
                rr.Pinhole(
                    image_from_camera=np.array((
                        (fx, 0.0, cx),
                        (0.0, fy, cy),
                        (0.0, 0.0, 1.0),
                    )),
                    resolution=[width, height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.25,
                ),
                static=True,
            )
            rr.log(
                camera_path,
                rr.EncodedImage(contents=jpeg, media_type="image/jpeg"),
                static=True,
            )
            rr.log(
                f"{camera_path}/origin",
                rr.Points3D(
                    [[0.0, 0.0, 0.0]],
                    radii=0.025,
                    colors=[[245, 180, 55]],
                    labels=[f"image {capture_id}"],
                ),
                static=True,
            )
        self.image_capture_ids.add(capture_id)
        self.image_capture_poses[capture_id] = capture_pose
        rr.log(
            "/system/slam_image_captures",
            rr.Scalar(len(self.image_capture_ids)),
        )
        if first_seen:
            print(
                f"image map capture {capture_id}: "
                f"({x:+.2f}, {y:+.2f}, {math.degrees(yaw):+.1f} deg)"
            )

    def on_lidar(self, payload):
        with self.lock:
            frames = [frame for frame in parse_frames(bytes(payload))
                      if frame is not None]
            if not self.lidar_deskew:
                for frame in frames:
                    rev = self.assembler.add(frame)
                    if not rev:
                        continue
                    pts = [
                        (d * math.cos(-math.radians(a)),
                         d * math.sin(-math.radians(a)), 0.05)
                        for (a, d, _inten) in rev if 0.02 < d < 12.0
                    ]
                    if pts:
                        rr.log("/world/robot/lidar",
                               rr.Points3D(pts, radii=0.01))
                return

            arrival_t = time.monotonic()
            timed_frames = []
            for frame in frames:
                lidar_t = self.lidar_clock.unwrap(frame.timestamp_ms / 1000.0)
                timed_frames.append((frame, lidar_t))
            if timed_frames:
                # The newest frame in a batch has the least batching delay.
                self.lidar_clock.observe(timed_frames[-1][1], arrival_t)
            for frame, lidar_t in timed_frames:
                scan = self.timed_assembler.add(frame, lidar_t)
                if scan:
                    self.pending_scans.append((scan, arrival_t))
            self._flush_deskewed_scans(arrival_t)

    def _flush_deskewed_scans(self, now_t):
        if not self.lidar_deskew:
            return
        while self.pending_scans:
            scan, queued_t = self.pending_scans[0]
            points = deskew_scan(
                scan,
                self.pose_history,
                self.lidar_clock,
                self.pose_clock,
                self.lidar_time_offset_s,
                self.lidar_extrinsics,
            )
            if points is None:
                # Normally this means the scan is waiting for the next pose
                # report. Drop an unalignable startup scan rather than blocking
                # all newer revolutions behind it indefinitely.
                if now_t - queued_t > 1.0:
                    self.pending_scans.popleft()
                    continue
                break
            self.pending_scans.popleft()
            if points:
                # These are already world coordinates; do not inherit the
                # current robot transform a second time.
                if self.map_to_odom is not None:
                    tx, ty, yaw = self.map_to_odom
                    c, s = math.cos(yaw), math.sin(yaw)
                    points = [
                        (tx + c * x - s * y, ty + s * x + c * y, z)
                        for x, y, z in points
                    ]
                rr.log("/world/lidar", rr.Points3D(points, radii=0.01))
                self.deskewed_scan_count += 1
                rr.log("/system/lidar_deskew_scans",
                       rr.Scalar(self.deskewed_scan_count))
                if self.deskewed_scan_count == 1:
                    print(f"LiDAR deskew active: {len(points)} points in first scan")

    def on_status(self, payload):
        try:
            s = json.loads(payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        rr.log("/system/heap_kb", rr.Scalar(s.get("free_heap", 0) / 1024))
        rr.log("/system/rssi_dbm", rr.Scalar(s.get("rssi", 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument(
        "--robot-yaw-deg", type=float, default=DEFAULT_ROBOT_YAW_DEG,
        help=("raw drivetrain +X yaw counter-clockwise from lidar/camera "
              f"forward (default {DEFAULT_ROBOT_YAW_DEG:g} deg)"))
    parser.add_argument(
        "--no-lidar-deskew", dest="lidar_deskew", action="store_false",
        help="show each revolution rigidly in the robot frame (comparison mode)")
    parser.add_argument(
        "--calibration",
        help="kiwi-slam-calibration-v1 YAML/JSON file")
    parser.add_argument(
        "--lidar-time-offset-ms", type=float,
        help=("residual LiDAR-to-IMU time adjustment; positive means the LiDAR "
              "measurements happened later (default 0)"))
    parser.add_argument("--lidar-x-m", type=float)
    parser.add_argument("--lidar-y-m", type=float)
    parser.add_argument("--lidar-yaw-deg", type=float)
    args = parser.parse_args()

    calibration = None
    if args.calibration:
        try:
            calibration = load_calibration(args.calibration)
        except ValueError as exc:
            parser.error(str(exc))
    lidar = calibration.lidar if calibration is not None else None
    # The historic zero default remains an explicit CLI override only when no
    # calibration file is supplied.
    time_offset_ms = (
        args.lidar_time_offset_ms if args.lidar_time_offset_ms is not None
        else (lidar.time_offset_ms if lidar is not None else 0.0))
    lidar_extrinsics = LidarExtrinsics(
        x_m=(args.lidar_x_m if args.lidar_x_m is not None else
             (lidar.x_m if lidar is not None else 0.0)),
        y_m=(args.lidar_y_m if args.lidar_y_m is not None else
             (lidar.y_m if lidar is not None else 0.0)),
        yaw_rad=math.radians(
            args.lidar_yaw_deg if args.lidar_yaw_deg is not None else
            (lidar.yaw_deg if lidar is not None else 0.0)),
    )

    rr.init("kiwi_dashboard", spawn=True, default_blueprint=make_blueprint())
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("/map", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # static scene: robot body footprint
    rr.log("/world/robot/body",
           rr.Boxes3D(centers=[[0, 0, 0.04]], half_sizes=[[0.11, 0.11, 0.04]],
                      colors=[[80, 200, 120]]),
           static=True)
    rr.log("/map/robot/body",
           rr.Boxes3D(centers=[[0, 0, 0.04]], half_sizes=[[0.11, 0.11, 0.04]],
                      colors=[[80, 200, 120]]),
           static=True)

    def timed(handler):
        def listener(payload):
            rr.set_time_seconds("time", time.time())
            handler(payload)
        return listener

    dash = Dashboard(args.lidar_deskew, time_offset_ms, lidar_extrinsics)
    client = KiwiClient(args.connect, args.namespace, args.robot_yaw_deg,
                         on_odometry=timed(dash.on_twist))
    client.add_slam_callback(timed(dash.on_slam_report))
    for suffix, handler in {
        "camera/jpeg": dash.on_camera,
        "lidar/ld19/raw": dash.on_lidar,
        "status/master": dash.on_status,
        "slam/map": dash.on_map,
        "slam/image": dash.on_image_capture,
        "navigation/trajectory": dash.on_navigation_trajectory,
        "navigation/state": dash.on_navigation_state,
    }.items():
        client.subscribe(suffix, timed(handler))
    deskew = (f"deskew on ({time_offset_ms:+g} ms offset, "
              f"xy=({lidar_extrinsics.x_m:+g}, "
              f"{lidar_extrinsics.y_m:+g}) m, "
              f"yaw={math.degrees(lidar_extrinsics.yaw_rad):+g} deg)"
              if args.lidar_deskew else "deskew off")
    print(f"dashboard streaming; frame correction {client.robot_yaw_deg:+g} deg; "
          f"{deskew}; ctrl-c to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
