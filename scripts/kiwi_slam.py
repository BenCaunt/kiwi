#!/usr/bin/env python3
"""Run Kiwi's native deskewed 2D graph-SLAM pipeline.

The process consumes the existing aligned odometry and raw LD19 Zenoh topics,
publishes the corrected pose on ``<namespace>/slam/pose``, and saves a
ROS-compatible occupancy map plus the optimized pose graph on exit.

Examples:
  python3 scripts/kiwi_slam.py
  python3 scripts/kiwi_slam.py --viewer --output maps/downstairs
  python3 scripts/kiwi_slam.py --viewer --resume maps/downstairs
  python3 scripts/kiwi_slam.py --lidar-time-offset-ms 8
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kiwi_client import DEFAULT_ROBOT_YAW_DEG, KiwiClient  # noqa: E402
from kiwi_calibration import load_calibration  # noqa: E402
from kiwi_lidar import parse_frames  # noqa: E402
from kiwi_lidar_deskew import (  # noqa: E402
    LidarExtrinsics,
    PoseHistory,
    SensorClock,
    TimedScanAssembler,
    deskew_scan_local,
    unwrap_angle,
)
from kiwi_yaw_estimator import (  # noqa: E402
    YawEstimator,
    YawEstimatorConfig,
)
from kiwi_image_map import (  # noqa: E402
    ImageMapRecorder,
    decode_camera_sample,
    discover_compatible_image_manifest,
    parse_camera_header,
)
from kiwi_map import encode_occupancy_map  # noqa: E402
from kiwi_slam_core import (  # noqa: E402
    Pose2,
    SlamConfig,
    PoseGraphSlam,
    compose,
    transform_points,
)

def decode_camera_frame(payload):
    """Decode and orient one camera topic payload, or return ``None``."""
    sample = decode_camera_sample(payload)
    return None if sample is None else sample.pixels


def make_viewer_blueprint(rrb):
    """Put the camera POV beside the map and SLAM quality plots."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="camera POV", origin="/camera"),
            rrb.Vertical(
                rrb.Spatial2DView(name="SLAM map", origin="/map"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="match quality",
                                       origin="/slam/quality"),
                    rrb.TimeSeriesView(name="processing latency",
                                       origin="/slam/processing_ms"),
                ),
                row_shares=[4, 1],
            ),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def yaw_from_quat(i, j, k, r):
    return math.atan2(2.0 * (r * k + i * j),
                      1.0 - 2.0 * (j * j + k * k))


class SlamRunner:
    """Time-align sensor callbacks and run scan matching off callback threads."""

    def __init__(self, client, config=None, lidar_time_offset_ms=0.0,
                 output_prefix="maps/kiwi_map", viewer=False,
                 print_every=10, map_publish_every=5,
                 image_map=True, image_distance_m=0.50,
                 image_angle_deg=30.0, image_min_interval_s=0.50,
                 camera_horizontal_fov_deg=72.0,
                 camera_height_m=0.10, image_map_read_only=False, slam=None,
                 image_resume_manifest=None, yaw_estimator_mode="legacy",
                 yaw_estimator_config=None, lidar_extrinsics=None,
                 calibration_path=None):
        self.client = client
        self.slam = slam if slam is not None else PoseGraphSlam(config)
        self.output_prefix = output_prefix
        self.print_every = max(1, int(print_every))
        self.map_publish_every = max(1, int(map_publish_every))
        self.lidar_time_offset_s = lidar_time_offset_ms / 1000.0
        self.lidar_extrinsics = lidar_extrinsics or LidarExtrinsics()
        if yaw_estimator_mode not in ("legacy", "fused"):
            raise ValueError("yaw estimator must be 'legacy' or 'fused'")
        self.yaw_estimator_mode = yaw_estimator_mode
        self.yaw_estimator = (
            YawEstimator(yaw_estimator_config or YawEstimatorConfig())
            if yaw_estimator_mode == "fused" else None)
        self.yaw_estimate = (
            self.yaw_estimator.snapshot()
            if self.yaw_estimator is not None else None)
        self.pose = [0.0, 0.0, 0.0]
        self.yaw_offset = None
        self.last_twist_t = None
        self.pose_history = PoseHistory(maxlen=1200)
        self.pose_clock = SensorClock(offset_window=400)
        self.lidar_clock = SensorClock(wrap_seconds=30.0, offset_window=400)
        self.camera_clock = SensorClock(offset_window=400)
        self.assembler = TimedScanAssembler()
        self.slam.runtime_metadata = {
            "yaw_estimator": yaw_estimator_mode,
            "calibration": calibration_path,
            "lidar": {
                "time_offset_ms": float(lidar_time_offset_ms),
                "x_m": self.lidar_extrinsics.x_m,
                "y_m": self.lidar_extrinsics.y_m,
                "yaw_deg": math.degrees(self.lidar_extrinsics.yaw_rad),
            },
        }
        self.pending_scans = deque(maxlen=30)
        self.bad_lidar_frames = 0
        self.camera_frames = 0
        self.bad_camera_frames = 0
        self.completed_scans = 0
        self.dropped_scans = 0
        self.latest_result = None
        self.latest_map_payload = None
        self.last_map_publish_t = None
        self.map_republish_s = 5.0
        self.image_republish_s = 1.0
        self.last_image_republish_t = None
        self.image_republish_index = 0
        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._publisher = client.session.declare_publisher(
            f"{client.namespace}/slam/pose")
        self._map_publisher = client.session.declare_publisher(
            f"{client.namespace}/slam/map")
        self._image_publisher = client.session.declare_publisher(
            f"{client.namespace}/slam/image")
        if self.slam.keyframes:
            occupancy = self.slam.build_occupancy_map()
            self.latest_map_payload = encode_occupancy_map(
                occupancy, len(self.slam.keyframes))
        self.image_recorder = (
            ImageMapRecorder(
                output_prefix,
                translation_spacing_m=image_distance_m,
                rotation_spacing_rad=math.radians(image_angle_deg),
                min_interval_s=image_min_interval_s,
                horizontal_fov_deg=camera_horizontal_fov_deg,
                camera_height_m=camera_height_m,
                resume_manifest=image_resume_manifest,
                slam_session_id=self.slam.current_session_id,
                keyframe_sessions={
                    keyframe.index: keyframe.session_id
                    for keyframe in self.slam.keyframes
                },
                read_only=image_map_read_only,
            )
            if image_map
            else None
        )
        self._rr = None
        if viewer:
            import rerun as rr
            import rerun.blueprint as rrb
            rr.init("kiwi_slam", spawn=True,
                    default_blueprint=make_viewer_blueprint(rrb))
            self._rr = rr
        self._worker = threading.Thread(
            target=self._run, name="kiwi-slam", daemon=True)
        self._worker.start()
        if self.latest_map_payload is not None:
            self._map_publisher.put(self.latest_map_payload)
            self.last_map_publish_t = time.monotonic()

    def on_odometry(self, report):
        arrival_t = time.monotonic()
        follower_time_us = report.get("follower_time_us")
        pose_t = (follower_time_us / 1_000_000.0
                  if isinstance(follower_time_us, (int, float))
                  else arrival_t)
        measured = report.get("measured", {})
        predicted_pose = None
        map_payload = None
        image_payload = None
        latest_result = None
        with self._condition:
            self.pose_clock.observe(pose_t, arrival_t)
            dt = 0.0
            if self.last_twist_t is not None:
                dt = min(max(pose_t - self.last_twist_t, 0.0), 0.2)

            q = report.get("imu_quat_ijkr")
            imu_yaw = None
            if q and report.get("imu_ready"):
                imu_yaw = yaw_from_quat(*q)
                if getattr(self, "yaw_estimator_mode", "legacy") == "legacy":
                    if self.yaw_offset is None:
                        self.yaw_offset = imu_yaw
                    imu_yaw = unwrap_angle(imu_yaw - self.yaw_offset,
                                           self.pose[2])

            # Integrate body translation at the interval midpoint. The BNO08x
            # fused heading supplies yaw when available; wheel omega is the
            # fallback during IMU startup or a transient IMU fault.
            previous_yaw = self.pose[2]
            vx = float(measured.get("vx", 0.0))
            vy = float(measured.get("vy", 0.0))
            if getattr(self, "yaw_estimator", None) is not None:
                self.yaw_estimate = self.yaw_estimator.update_odometry(
                    time_s=pose_t,
                    wheel_omega_rad_s=measured.get("omega", 0.0),
                    imu_yaw_rad=imu_yaw,
                    imu_valid=imu_yaw is not None,
                    imu_discontinuity=bool(
                        report.get("imu_yaw_discontinuity", False)),
                    wheel_valid=bool(report.get("encoder_ready_mask", 1)),
                    linear_speed_m_s=math.hypot(vx, vy),
                )
                dt = self.yaw_estimate.dt_s
                next_yaw = self.yaw_estimate.yaw
            else:
                next_yaw = (imu_yaw if imu_yaw is not None else
                            previous_yaw + measured.get("omega", 0.0) * dt)
            midpoint_yaw = previous_yaw + 0.5 * (next_yaw - previous_yaw)
            c, s = math.cos(midpoint_yaw), math.sin(midpoint_yaw)
            self.pose[0] += (c * vx - s * vy) * dt
            self.pose[1] += (s * vx + c * vy) * dt
            self.pose[2] = next_yaw
            self.last_twist_t = pose_t
            self.pose_history.append(pose_t, *self.pose)
            latest_result = self.latest_result
            if latest_result is not None:
                # Scan matching estimates map->odom at the LiDAR scan time.
                # Apply that correction to the newest IMU/wheel odometry so
                # consumers never have to control from a delayed scan pose.
                predicted_pose = compose(
                    latest_result.map_to_odom,
                    Pose2(*self.pose),
                )
            if (self.latest_map_payload is not None and
                    (self.last_map_publish_t is None or
                     arrival_t - self.last_map_publish_t >=
                     self.map_republish_s)):
                map_payload = self.latest_map_payload
                self.last_map_publish_t = arrival_t
            image_recorder = getattr(self, "image_recorder", None)
            if (
                image_recorder is not None
                and image_recorder.active
                and not self.slam.relocalization_required
                and len(image_recorder)
                and (
                    self.last_image_republish_t is None
                    or arrival_t - self.last_image_republish_t
                    >= self.image_republish_s
                )
            ):
                image_payload = image_recorder.packet(
                    self.image_republish_index % len(image_recorder)
                )
                self.image_republish_index += 1
                self.last_image_republish_t = arrival_t
            self._condition.notify()
        if predicted_pose is not None:
            self._publish_pose(
                predicted_pose,
                latest_result,
                source="odometry_prediction",
            )
        if map_payload is not None:
            self._map_publisher.put(map_payload)
        if image_payload is not None:
            self._image_publisher.put(image_payload)

    def on_lidar(self, payload):
        arrival_t = time.monotonic()
        frames = parse_frames(bytes(payload))
        with self._condition:
            timed_frames = []
            for frame in frames:
                if frame is None:
                    self.bad_lidar_frames += 1
                    continue
                lidar_t = self.lidar_clock.unwrap(frame.timestamp_ms / 1000.0)
                timed_frames.append((frame, lidar_t))
            if timed_frames:
                # The last packet in a Zenoh batch has the least batching
                # latency and therefore gives the best clock-offset sample.
                self.lidar_clock.observe(timed_frames[-1][1], arrival_t)
            for frame, lidar_t in timed_frames:
                scan = self.assembler.add(frame, lidar_t)
                if scan:
                    if len(self.pending_scans) == self.pending_scans.maxlen:
                        self.dropped_scans += 1
                    self.pending_scans.append((scan, arrival_t))
            self._condition.notify()

    def on_camera(self, payload):
        """Record a pose-spaced upright image and optionally log the live POV."""
        payload = bytes(payload)
        header = parse_camera_header(payload)
        if header is None:
            self.bad_camera_frames += 1
            return
        arrival_t = time.monotonic()
        map_pose = None
        raw_pose = None
        pose_time_s = None
        with self._condition:
            if header.sensor_time_us > 0:
                camera_time_s = header.sensor_time_us / 1_000_000.0
                self.camera_clock.observe(camera_time_s, arrival_t)
                if self.pose_clock.ready:
                    laptop_time = self.camera_clock.to_laptop(camera_time_s)
                    candidate_time = self.pose_clock.from_laptop(laptop_time)
                    interpolated = self.pose_history.interpolate(candidate_time)
                    if interpolated is not None:
                        raw_pose = Pose2(
                            interpolated.x, interpolated.y, interpolated.yaw
                        )
                        pose_time_s = candidate_time
            if raw_pose is None and self.last_twist_t is not None:
                raw_pose = Pose2(*self.pose)
                pose_time_s = self.last_twist_t
            if (raw_pose is not None and self.latest_result is not None and
                    self.latest_result.scan_matched and
                    not self.slam.relocalization_required):
                map_pose = compose(self.latest_result.map_to_odom, raw_pose)

        wants_capture = (
            self.image_recorder is not None
            and map_pose is not None
            and self.image_recorder.should_capture(map_pose, arrival_t)
        )
        if self._rr is None and not wants_capture:
            return
        sample = decode_camera_sample(payload)
        if sample is None:
            self.bad_camera_frames += 1
            return
        self.camera_frames += 1
        wall_time_s = time.time()
        if self._rr is not None:
            self._rr.set_time_seconds("time", wall_time_s)
            self._rr.log("/camera", self._rr.Image(sample.pixels))
        if wants_capture:
            captured = self.image_recorder.capture(
                sample,
                raw_pose,
                map_pose,
                pose_time_s,
                wall_time_s=wall_time_s,
                monotonic_s=arrival_t,
            )
            if captured is not None:
                metadata, encoded = captured
                self._image_publisher.put(encoded)
                print(
                    f"image {metadata['id']:04d}  "
                    f"pose=({map_pose.x:+.2f}, {map_pose.y:+.2f}, "
                    f"{math.degrees(map_pose.yaw):+.1f} deg)"
                )

    def _take_ready_scan(self):
        with self._condition:
            while not self._stop.is_set():
                if not self.pending_scans:
                    self._condition.wait(timeout=0.2)
                    continue

                # A pose estimate used for feedback must be recent. If scan
                # matching falls behind, processing every queued revolution
                # only increases control latency. Prefer the newest complete
                # scan and discard older revolutions once it can be deskewed.
                scan, queued_t = self.pending_scans[-1]
                deskewed = deskew_scan_local(
                    scan,
                    self.pose_history,
                    self.lidar_clock,
                    self.pose_clock,
                    self.lidar_time_offset_s,
                    self.lidar_extrinsics,
                )
                if deskewed is not None:
                    superseded = len(self.pending_scans) - 1
                    self.pending_scans.clear()
                    self.dropped_scans += superseded
                    return deskewed

                oldest_queued_t = self.pending_scans[0][1]
                if time.monotonic() - oldest_queued_t > 1.5:
                    self.pending_scans.popleft()
                    self.dropped_scans += 1
                    continue
                self._condition.wait(timeout=0.025)
            return None

    def _run(self):
        while not self._stop.is_set():
            scan = self._take_ready_scan()
            if scan is None:
                continue
            points = np.asarray(scan.points, dtype=np.float64)
            raw_pose = Pose2(scan.pose.x, scan.pose.y, scan.pose.yaw)
            try:
                result = self.slam.process_scan(points, raw_pose, scan.time_s)
            except ValueError:
                self.dropped_scans += 1
                continue
            if self.yaw_estimator is not None:
                self.yaw_estimator.observe_scan(
                    time_s=scan.time_s,
                    heading_disagreement_rad=
                    result.heading_disagreement_rad,
                    scan_matched=result.scan_matched,
                    score=result.match_score,
                    hit_ratio=result.hit_ratio,
                    rmse_m=result.rmse_m,
                    wall_support_ratio=result.wall_support_ratio,
                    loop_closed=result.loop_closed,
                    relocalized=result.loop_status.startswith("relocalized:"),
                )
                self.yaw_estimate = self.yaw_estimator.snapshot()
            self.completed_scans += 1
            if (result.loop_status.startswith("relocalized:") and
                    self.image_recorder is not None):
                self.image_recorder.activate()
            with self._condition:
                self.latest_result = result
                current_raw_pose = Pose2(*self.pose)
            current_pose = compose(result.map_to_odom, current_raw_pose)
            self._publish_pose(
                current_pose,
                result,
                source="scan_correction",
            )
            if (result.keyframe_added and
                    (result.keyframes == 1 or result.loop_closed or
                     result.keyframes % self.map_publish_every == 0)):
                self._publish_map()
            if result.loop_closed and self.image_recorder is not None:
                updated = self.image_recorder.reproject(self.slam.keyframes)
                for index in updated:
                    self._image_publisher.put(
                        self.image_recorder.packet(index)
                    )
            if self._rr is not None:
                self._log_viewer(points, result)
            if (result.loop_closed or result.loop_status.startswith(
                    "relocalized:") or
                    self.completed_scans % self.print_every == 0):
                status = " LOOP CLOSED" if result.loop_closed else ""
                matched = "match" if result.scan_matched else "odom fallback"
                print(
                    f"scan {self.completed_scans:5d}  "
                    f"pose=({result.pose.x:+.2f}, {result.pose.y:+.2f}, "
                    f"{math.degrees(result.pose.yaw):+.1f} deg)  "
                    f"score={result.match_score:.3f} "
                    f"hits={result.hit_ratio:.0%}  {matched}  "
                    f"heading_delta="
                    f"{math.degrees(result.heading_disagreement_rad):+.1f} deg  "
                    f"{result.processing_ms:.1f} ms  "
                    f"kf={result.keyframes} loops={result.loop_closures} "
                    f"loop={result.loop_status}"
                    f"{status}")

    def _publish_pose(self, pose, result, source):
        loop_status = getattr(result, "loop_status", "idle")
        if not isinstance(loop_status, str):
            loop_status = "idle"
        wall_support_ratio = getattr(result, "wall_support_ratio", 0.0)
        supported_line_length_m = getattr(
            result, "supported_line_length_m", 0.0)
        if not isinstance(wall_support_ratio, (int, float)):
            wall_support_ratio = 0.0
        if not isinstance(supported_line_length_m, (int, float)):
            supported_line_length_m = 0.0
        payload = {
            "time_s": time.time(),
            "frame": "map",
            "child_frame": "robot",
            "source": source,
            "pose": {
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
            },
            "map_to_odom": {
                "x": result.map_to_odom.x,
                "y": result.map_to_odom.y,
                "yaw": result.map_to_odom.yaw,
            },
            "quality": {
                "scan_matched": result.scan_matched,
                "relocalizing": getattr(
                    getattr(self, "slam", None),
                    "relocalization_required", False),
                "score": result.match_score,
                "hit_ratio": result.hit_ratio,
                "rmse_m": result.rmse_m,
                "heading_disagreement_deg": math.degrees(
                    result.heading_disagreement_rad),
                "wall_support_ratio": wall_support_ratio,
                "supported_line_length_m": supported_line_length_m,
            },
            "keyframes": result.keyframes,
            "loop_closures": result.loop_closures,
            "loop_status": loop_status,
            "imu_yaw_rejections": getattr(
                getattr(self, "client", None), "imu_yaw_rejections", 0),
            "processing_ms": result.processing_ms,
        }
        estimate = getattr(self, "yaw_estimate", None)
        payload["yaw_estimator"] = {
            "mode": getattr(self, "yaw_estimator_mode", "legacy"),
            **(estimate.diagnostics() if estimate is not None else {}),
        }
        self._publisher.put(json.dumps(payload, separators=(",", ":")))

    def _publish_map(self):
        occupancy = self.slam.build_occupancy_map()
        payload = encode_occupancy_map(occupancy, len(self.slam.keyframes))
        with self._condition:
            self.latest_map_payload = payload
            self.last_map_publish_t = time.monotonic()
        self._map_publisher.put(payload)

    def _log_viewer(self, local_points, result):
        rr = self._rr
        world_points = transform_points(result.pose, local_points)
        rr.set_time_seconds("time", time.time())
        rr.log("/map/current_scan", rr.Points2D(world_points, radii=0.012))
        rr.log("/map/robot", rr.Points2D(
            [[result.pose.x, result.pose.y]], radii=0.09,
            colors=[[80, 220, 120]]))
        rr.log("/slam/quality/match_score", rr.Scalar(result.match_score))
        rr.log("/slam/quality/hit_ratio", rr.Scalar(result.hit_ratio))
        rr.log("/slam/quality/heading_disagreement_deg", rr.Scalar(
            math.degrees(result.heading_disagreement_rad)))
        rr.log("/slam/quality/wall_support_ratio", rr.Scalar(
            getattr(result, "wall_support_ratio", 0.0)))
        estimate = getattr(self, "yaw_estimate", None)
        if estimate is not None:
            rr.log("/slam/yaw/learned_rate_bias_deg_s", rr.Scalar(
                math.degrees(estimate.learned_rate_bias_rad_s)))
            rr.log("/slam/yaw/uncertainty_deg", rr.Scalar(
                math.degrees(estimate.uncertainty_rad)))
            rr.log("/slam/yaw/imu_weight", rr.Scalar(estimate.imu_weight))
        rr.log("/slam/processing_ms", rr.Scalar(result.processing_ms))
        if result.keyframe_added:
            trajectory = [[keyframe.pose.x, keyframe.pose.y]
                          for keyframe in self.slam.keyframes]
            rr.log("/map/trajectory", rr.LineStrips2D([trajectory]))
            if result.keyframes % 10 == 0 or result.loop_closed:
                mapped = np.concatenate([
                    transform_points(k.pose, k.points)
                    for k in self.slam.keyframes
                ])
                rr.log("/map/keyframe_hits",
                       rr.Points2D(mapped, radii=0.008))

    def close(self, save=True):
        self._stop.set()
        if self.image_recorder is not None:
            # Block until an in-flight camera capture completes, then prevent
            # callbacks from appending after the final reprojection below.
            self.image_recorder.seal()
        with self._condition:
            self._condition.notify_all()
        self._worker.join(timeout=5.0)
        if (save and self.slam.keyframes and
                self.slam.relocalization_required and
                hasattr(self.slam, "loaded_prefix")):
            print("relocalization never succeeded; saved map left unchanged")
            return {}
        if save and self.slam.keyframes:
            self.slam.optimize()
            if self.yaw_estimator is not None:
                self.slam.runtime_metadata[
                    "yaw_estimator_diagnostics"] = \
                    self.yaw_estimator.summary()
            if self.image_recorder is not None:
                self.image_recorder.reproject(self.slam.keyframes)
            paths = self.slam.save(self.output_prefix)
            if self.image_recorder is not None:
                paths["image_map"] = self.image_recorder.manifest_path
            print("saved " + ", ".join(str(path) for path in paths.values()))
            return paths
        return {}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--robot-yaw-deg", type=float,
                        default=DEFAULT_ROBOT_YAW_DEG)
    parser.add_argument(
        "--calibration",
        help="kiwi-slam-calibration-v1 YAML/JSON file")
    parser.add_argument(
        "--yaw-estimator", choices=("legacy", "fused"), default="legacy",
        help="continuous yaw implementation (default legacy until robot A/B)")
    parser.add_argument("--lidar-time-offset-ms", type=float)
    parser.add_argument("--lidar-x-m", type=float)
    parser.add_argument("--lidar-y-m", type=float)
    parser.add_argument("--lidar-yaw-deg", type=float)
    parser.add_argument(
        "--output",
        help=("output path prefix (default maps/kiwi_map, or the loaded prefix "
              "when --resume is used)"),
    )
    parser.add_argument(
        "--resume",
        help=("load PREFIX.graph.json and PREFIX.slam.npz, relocalize, and "
              "continue the saved map"),
    )
    parser.add_argument(
        "--resume-pose", nargs=3, type=float,
        metavar=("X", "Y", "YAW_DEG"),
        help=("initial map-frame relocalization hint; defaults to the saved "
              "final pose"),
    )
    parser.add_argument(
        "--resume-search-distance", type=float, metavar="METERS",
        help=("translation radius around the resume pose to search during "
              "relocalization (default: saved map setting, normally 1 m)"),
    )
    parser.add_argument(
        "--resume-global", action="store_true",
        help=("also search descriptor-selected locations across the saved map "
              "(slower; rejects ambiguous matches)"),
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--viewer", action="store_true",
                        help="spawn a live Rerun SLAM viewer")
    parser.add_argument("--map-resolution", type=float)
    parser.add_argument("--keyframe-distance", type=float)
    parser.add_argument("--keyframe-angle-deg", type=float)
    parser.add_argument(
        "--scan-search-angle-deg", type=float,
        help="scan matcher yaw-error search window around IMU prediction")
    parser.add_argument(
        "--scan-yaw-prior-sigma-deg", type=float,
        help="front-end BNO08x heading prior sigma (smaller is stronger)")
    parser.add_argument(
        "--heading-prior-sigma-deg", type=float,
        help="pose-graph absolute BNO08x heading sigma (smaller is stronger)")
    parser.add_argument(
        "--heading-prior-max-disagreement-deg", type=float,
        help=("discard a pose-graph heading prior after this much disagreement "
              "with LiDAR (default 12 degrees)"))
    parser.add_argument(
        "--absolute-imu-heading-prediction", action="store_true",
        help="snap each front-end yaw prediction to the startup-relative IMU yaw")
    parser.add_argument(
        "--no-heading-prior", action="store_true",
        help="disable absolute BNO heading anchoring in front end and graph")
    parser.add_argument("--no-loop-closure", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--map-publish-every", type=int, default=5,
        help="publish the live occupancy map every N keyframes (default 5)")
    parser.add_argument(
        "--no-image-map", action="store_true",
        help="disable pose-spaced camera image recording")
    parser.add_argument(
        "--resume-image-manifest",
        help=("image-map manifest path or session directory to load and append; "
              "--resume auto-discovers a compatible manifest by default"),
    )
    parser.add_argument(
        "--no-resume-images", action="store_true",
        help="start a new image manifest instead of loading compatible captures",
    )
    parser.add_argument(
        "--image-distance-m", type=float, default=0.50,
        help="capture after translating this far from the last image (default 0.5)")
    parser.add_argument(
        "--image-angle-deg", type=float, default=30.0,
        help="capture after rotating this far from the last image (default 30)")
    parser.add_argument(
        "--image-min-interval-s", type=float, default=0.50,
        help="minimum time between image captures (default 0.5)")
    parser.add_argument(
        "--camera-horizontal-fov-deg", type=float, default=72.0,
        help="horizontal camera field of view used for intrinsics (default 72)")
    parser.add_argument(
        "--camera-height-m", type=float, default=0.10,
        help="camera height above the occupancy floor (default 0.10)")
    args = parser.parse_args()

    if not args.resume and (args.resume_pose or args.resume_global or
                            args.resume_search_distance is not None):
        parser.error("resume localization options require --resume")
    if (args.resume_search_distance is not None and
            (not math.isfinite(args.resume_search_distance) or
             args.resume_search_distance <= 0.0)):
        parser.error("--resume-search-distance must be positive and finite")
    if not args.resume and args.resume_image_manifest:
        parser.error("--resume-image-manifest requires --resume")
    if args.no_image_map and args.resume_image_manifest:
        parser.error(
            "--resume-image-manifest cannot be used with --no-image-map")
    if args.no_resume_images and args.resume_image_manifest:
        parser.error(
            "--resume-image-manifest conflicts with --no-resume-images")

    calibration = None
    if args.calibration:
        try:
            calibration = load_calibration(args.calibration)
        except ValueError as exc:
            parser.error(str(exc))
    yaw_estimator_config = (
        calibration.yaw_estimator if calibration is not None
        else YawEstimatorConfig())
    lidar_calibration = calibration.lidar if calibration is not None else None
    lidar_time_offset_ms = (
        args.lidar_time_offset_ms if args.lidar_time_offset_ms is not None
        else (lidar_calibration.time_offset_ms
              if lidar_calibration is not None else 0.0))
    lidar_extrinsics = LidarExtrinsics(
        x_m=(args.lidar_x_m if args.lidar_x_m is not None else
             (lidar_calibration.x_m if lidar_calibration is not None else 0.0)),
        y_m=(args.lidar_y_m if args.lidar_y_m is not None else
             (lidar_calibration.y_m if lidar_calibration is not None else 0.0)),
        yaw_rad=math.radians(
            args.lidar_yaw_deg if args.lidar_yaw_deg is not None else
            (lidar_calibration.yaw_deg if lidar_calibration is not None
             else 0.0)),
    )

    config = SlamConfig(
        map_resolution_m=(0.05 if args.map_resolution is None
                          else args.map_resolution),
        keyframe_translation_m=(0.12 if args.keyframe_distance is None
                                else args.keyframe_distance),
        keyframe_rotation_rad=math.radians(
            8.0 if args.keyframe_angle_deg is None
            else args.keyframe_angle_deg),
        search_rotation_rad=math.radians(
            16.0 if args.scan_search_angle_deg is None
            else args.scan_search_angle_deg),
        odom_prior_rotation_sigma_rad=math.radians(
            8.0 if args.scan_yaw_prior_sigma_deg is None
            else args.scan_yaw_prior_sigma_deg),
        heading_prior_rotation_sigma_rad=(
            0.0 if args.no_heading_prior else
            math.radians(
                15.0 if args.heading_prior_sigma_deg is None
                else args.heading_prior_sigma_deg)),
        heading_prior_max_disagreement_rad=math.radians(
            12.0 if args.heading_prior_max_disagreement_deg is None
            else args.heading_prior_max_disagreement_deg),
        absolute_imu_heading_prediction=(
            args.absolute_imu_heading_prediction and
            not args.no_heading_prior),
    )
    if args.no_loop_closure:
        config.loop_check_every_keyframes = 1_000_000_000

    slam = None
    image_resume_manifest = None
    output_prefix = args.output or "maps/kiwi_map"
    if args.resume:
        resume_hint = (
            Pose2(args.resume_pose[0], args.resume_pose[1],
                  math.radians(args.resume_pose[2]))
            if args.resume_pose else None
        )
        try:
            slam = PoseGraphSlam.load(
                args.resume,
                relocalization_hint=resume_hint,
                global_relocalization=args.resume_global,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            parser.error(str(exc))
        config = slam.config
        if args.resume_search_distance is not None:
            config.relocalization_search_translation_m = \
                args.resume_search_distance
        # Saved tuning is the default for a continued graph. Explicit command
        # line tuning remains available for deliberate experiments.
        if args.map_resolution is not None:
            config.map_resolution_m = args.map_resolution
        if args.keyframe_distance is not None:
            config.keyframe_translation_m = args.keyframe_distance
        if args.keyframe_angle_deg is not None:
            config.keyframe_rotation_rad = math.radians(
                args.keyframe_angle_deg)
        if args.scan_search_angle_deg is not None:
            config.search_rotation_rad = math.radians(
                args.scan_search_angle_deg)
        if args.scan_yaw_prior_sigma_deg is not None:
            config.odom_prior_rotation_sigma_rad = math.radians(
                args.scan_yaw_prior_sigma_deg)
        if args.heading_prior_sigma_deg is not None:
            config.heading_prior_rotation_sigma_rad = math.radians(
                args.heading_prior_sigma_deg)
        if args.heading_prior_max_disagreement_deg is not None:
            config.heading_prior_max_disagreement_rad = math.radians(
                args.heading_prior_max_disagreement_deg)
        if args.no_heading_prior:
            config.heading_prior_rotation_sigma_rad = 0.0
        if args.absolute_imu_heading_prediction:
            config.absolute_imu_heading_prediction = not args.no_heading_prior
        if args.no_loop_closure:
            config.loop_check_every_keyframes = 1_000_000_000
        output_prefix = args.output or str(slam.loaded_prefix)
        if not args.no_image_map and not args.no_resume_images:
            if args.resume_image_manifest:
                candidate = Path(args.resume_image_manifest).expanduser()
                if candidate.is_dir():
                    candidate = candidate / "manifest.json"
                if not candidate.is_file():
                    by_session = Path(
                        f"{slam.loaded_prefix}.images"
                    ) / args.resume_image_manifest / "manifest.json"
                    candidate = by_session if by_session.is_file() else candidate
                if not candidate.is_file():
                    parser.error(
                        f"image-map manifest not found: {candidate}")
                image_resume_manifest = candidate.resolve()
            else:
                image_resume_manifest = discover_compatible_image_manifest(
                    slam.loaded_prefix, slam.keyframes)

    client = KiwiClient(
        args.connect,
        args.namespace,
        args.robot_yaw_deg,
    )
    runner = SlamRunner(
        client,
        config,
        lidar_time_offset_ms,
        output_prefix,
        args.viewer,
        args.print_every,
        args.map_publish_every,
        # A no-save replay may republish an existing image session, but it must
        # never create or append persistent image-map data without saving the
        # graph keyframes those captures reference.
        image_map=(
            not args.no_image_map
            and (not args.no_save or image_resume_manifest is not None)
        ),
        image_distance_m=args.image_distance_m,
        image_angle_deg=args.image_angle_deg,
        image_min_interval_s=args.image_min_interval_s,
        camera_horizontal_fov_deg=args.camera_horizontal_fov_deg,
        camera_height_m=args.camera_height_m,
        image_map_read_only=args.no_save,
        slam=slam,
        image_resume_manifest=image_resume_manifest,
        yaw_estimator_mode=args.yaw_estimator,
        yaw_estimator_config=yaw_estimator_config,
        lidar_extrinsics=lidar_extrinsics,
        calibration_path=(calibration.source_path
                          if calibration is not None else None),
    )
    client.add_odometry_callback(runner.on_odometry)
    client.subscribe("lidar/ld19/raw", runner.on_lidar)
    if args.viewer or not args.no_image_map:
        client.subscribe("camera/jpeg", runner.on_camera)
    stopping = threading.Event()

    def stop(_signum=None, _frame=None):
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"SLAM streaming from {args.namespace}; pose -> "
        f"{args.namespace}/slam/pose; map -> {args.namespace}/slam/map; "
        f"images -> {args.namespace}/slam/image; output {output_prefix}.*; "
        f"ctrl-c to stop")
    print(
        f"yaw estimator {args.yaw_estimator}; LiDAR calibration "
        f"time={lidar_time_offset_ms:+g} ms "
        f"xy=({lidar_extrinsics.x_m:+g}, {lidar_extrinsics.y_m:+g}) m "
        f"yaw={math.degrees(lidar_extrinsics.yaw_rad):+g} deg")
    if slam is not None:
        if args.resume_global:
            search = "saved final pose plus global candidates"
        elif args.resume_pose:
            search = (f"supplied pose {tuple(args.resume_pose)}")
        else:
            search = "saved final pose"
        print(
            f"loaded {len(slam.keyframes)} keyframes from {slam.loaded_prefix}; "
            f"relocalizing within "
            f"{config.relocalization_search_translation_m:g} m of {search}; "
            f"keep the robot stopped until a "
            f"'relocalized' scan is reported")
        if image_resume_manifest is not None:
            print(
                f"loaded {len(runner.image_recorder)} saved image captures from "
                f"{image_resume_manifest}; live image session "
                f"{runner.image_recorder.session_id}")
    try:
        while not stopping.wait(0.5):
            pass
    finally:
        runner.close(save=not args.no_save)
        client.close()


if __name__ == "__main__":
    main()
