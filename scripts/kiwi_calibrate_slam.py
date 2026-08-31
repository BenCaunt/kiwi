#!/usr/bin/env python3
"""Record, solve, and validate Kiwi yaw/LiDAR calibration datasets.

The recorder retains exact Zenoh odometry JSON and LD19 byte payloads.  The
solver replays those payloads through the production continuity, deskew, and
SLAM code; it never consumes simulator ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from kiwi_calibration import (  # noqa: E402
    Calibration,
    CalibrationLogWriter,
    LidarCalibration,
    LOG_FORMAT,
    RotationObservation,
    aggregate_rotation_segments,
    fit_planar_calibration,
    fit_rotation_calibration,
    load_calibration,
    read_lidar_records,
    save_calibration,
)
from kiwi_client import (  # noqa: E402
    DEFAULT_ROBOT_YAW_DEG,
    FrameTransform,
    ImuYawContinuityFilter,
    KiwiClient,
)
from kiwi_lidar import parse_frames  # noqa: E402
from kiwi_lidar_deskew import (  # noqa: E402
    LidarExtrinsics,
    PoseHistory,
    SensorClock,
    TimedScanAssembler,
    deskew_scan_local,
)
from kiwi_slam import yaw_from_quat  # noqa: E402
from kiwi_slam_core import (  # noqa: E402
    DistanceField,
    Pose2,
    PoseGraphSlam,
    SlamConfig,
    transform_points,
    wrap_angle,
)
from kiwi_yaw_estimator import YawEstimator, YawEstimatorConfig  # noqa: E402


@dataclass
class ReplayData:
    observations: list
    scans: list
    poses: PoseHistory
    lidar_clock: SensorClock
    pose_clock: SensorClock
    motion_times: np.ndarray
    linear_speeds: np.ndarray
    angular_speeds: np.ndarray
    scan_metrics: dict


def _read_run(directory):
    directory = Path(directory).expanduser()
    try:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibration run {directory}: {exc}") from exc
    if run.get("format") != LOG_FORMAT:
        raise ValueError(f"unsupported calibration log: {run.get('format')!r}")
    return directory, run


def _read_odometry(directory):
    records = []
    try:
        lines = (directory / "odom.jsonl").read_text(
            encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read odometry log: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
            records.append((int(item["arrival_ns"]), item["report"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid odometry record on line {line_number}") from exc
    if len(records) < 2:
        raise ValueError("calibration run has fewer than two odometry reports")
    return records


def _interpolate(times, values, query):
    if query < times[0] or query > times[-1]:
        return None
    return float(np.interp(query, times, values))


def replay_run(directory, calibration=None, max_scans=None):
    """Replay raw topics and derive LiDAR-referenced rotation observations."""
    directory, run = _read_run(directory)
    odometry = _read_odometry(directory)
    robot_yaw_deg = float(run.get("robot_yaw_deg", DEFAULT_ROBOT_YAW_DEG))
    frames = FrameTransform(robot_yaw_deg)
    continuity = ImuYawContinuityFilter()
    estimator_config = (calibration.yaw_estimator if calibration is not None
                        else YawEstimatorConfig())
    estimator = YawEstimator(estimator_config)
    poses = PoseHistory(maxlen=max(2000, len(odometry) + 10))
    pose_clock = SensorClock(offset_window=max(400, len(odometry)))
    pose_sensor_clock = SensorClock(
        wrap_seconds=(1 << 32) / 1_000_000.0,
        offset_window=max(400, len(odometry)),
    )

    pose = [0.0, 0.0, 0.0]
    last_pose_t = None
    wheel_integral = 0.0
    imu_integral = 0.0
    last_imu = None
    times = []
    wheel_integrals = []
    imu_integrals = []
    linear_speeds = []
    angular_speeds = []

    for arrival_ns, raw_report in odometry:
        report = frames.odometry_to_aligned(raw_report)
        arrival_s = arrival_ns / 1_000_000_000.0
        continuity.filter_report(report, arrival_time=arrival_s)
        follower_us = report.get("follower_time_us")
        if not isinstance(follower_us, (int, float)):
            continue
        pose_t = pose_sensor_clock.unwrap(float(follower_us) / 1_000_000.0)
        pose_clock.observe(pose_t, arrival_s)
        measured = report.get("measured", {})
        omega = float(measured.get("omega", 0.0))
        vx = float(measured.get("vx", 0.0))
        vy = float(measured.get("vy", 0.0))
        q = report.get("imu_quat_ijkr")
        imu_yaw = (yaw_from_quat(*q) if q and report.get("imu_ready") else None)

        estimate = estimator.update_odometry(
            time_s=pose_t,
            wheel_omega_rad_s=omega,
            imu_yaw_rad=imu_yaw,
            imu_valid=imu_yaw is not None,
            imu_discontinuity=bool(report.get("imu_yaw_discontinuity")),
            wheel_valid=bool(report.get("encoder_ready_mask", 1)),
        )
        dt = estimate.dt_s
        if last_pose_t is not None:
            wheel_integral += omega * dt
            if imu_yaw is not None and last_imu is not None and not \
                    report.get("imu_yaw_discontinuity"):
                imu_integral += wrap_angle(imu_yaw - last_imu)
            midpoint = pose[2] + 0.5 * (estimate.yaw - pose[2])
            c, s = math.cos(midpoint), math.sin(midpoint)
            pose[0] += (c * vx - s * vy) * dt
            pose[1] += (s * vx + c * vy) * dt
        pose[2] = estimate.yaw
        if imu_yaw is not None:
            last_imu = imu_yaw
        last_pose_t = pose_t
        poses.append(pose_t, *pose)
        times.append(pose_t)
        wheel_integrals.append(wheel_integral)
        imu_integrals.append(imu_integral)
        linear_speeds.append(math.hypot(vx, vy))
        angular_speeds.append(omega)

    if len(times) < 2:
        raise ValueError("calibration run has no usable timestamped odometry")
    times = np.asarray(times)
    wheel_integrals = np.asarray(wheel_integrals)
    imu_integrals = np.asarray(imu_integrals)
    linear_speeds = np.asarray(linear_speeds)
    angular_speeds = np.asarray(angular_speeds)

    lidar_clock = SensorClock(wrap_seconds=30.0, offset_window=400)
    assembler = TimedScanAssembler()
    scans = []
    for arrival_ns, payload in read_lidar_records(directory / "lidar.bin"):
        parsed = []
        for frame in parse_frames(payload):
            if frame is None:
                continue
            lidar_t = lidar_clock.unwrap(frame.timestamp_ms / 1000.0)
            parsed.append((frame, lidar_t))
        if parsed:
            lidar_clock.observe(parsed[-1][1], arrival_ns / 1_000_000_000.0)
        for frame, lidar_t in parsed:
            completed = assembler.add(frame, lidar_t)
            if completed:
                scans.append(completed)
                if max_scans is not None and len(scans) >= max_scans:
                    break
        if max_scans is not None and len(scans) >= max_scans:
            break
    if not scans:
        raise ValueError("calibration run contains no complete LiDAR revolutions")

    lidar = calibration.lidar if calibration is not None else LidarCalibration()
    extrinsics = LidarExtrinsics(
        lidar.x_m, lidar.y_m, math.radians(lidar.yaw_deg))
    slam = PoseGraphSlam(SlamConfig())
    previous = None
    observations = []
    usable_scans = []
    matched_scans = 0
    for timed_scan in scans:
        deskewed = deskew_scan_local(
            timed_scan, poses, lidar_clock, pose_clock,
            lidar.time_offset_ms / 1000.0, extrinsics)
        if deskewed is None:
            continue
        usable_scans.append(timed_scan)
        points = np.asarray(deskewed.points, dtype=np.float64)
        try:
            result = slam.process_scan(
                points, Pose2(deskewed.pose.x, deskewed.pose.y,
                              deskewed.pose.yaw), deskewed.time_s)
        except ValueError:
            continue
        if result.scan_matched:
            matched_scans += 1
        wheel_value = _interpolate(
            times, wheel_integrals, deskewed.time_s)
        imu_value = _interpolate(times, imu_integrals, deskewed.time_s)
        if previous is not None and result.scan_matched and \
                previous[0].scan_matched and wheel_value is not None and \
                imu_value is not None:
            previous_result, previous_time, previous_wheel, previous_imu = previous
            scan_delta = wrap_angle(result.pose.yaw - previous_result.pose.yaw)
            wheel_delta = wheel_value - previous_wheel
            imu_delta = imu_value - previous_imu
            dt = deskewed.time_s - previous_time
            if dt > 0.0 and max(abs(scan_delta), abs(wheel_delta),
                                abs(imu_delta)) >= math.radians(0.25):
                observations.append(RotationObservation(
                    dt, wheel_delta, imu_delta, scan_delta,
                    max(0.05, result.match_score * result.hit_ratio),
                ))
        previous = (result, deskewed.time_s, wheel_value, imu_value)

    metrics = {
        "complete_scans": len(scans),
        "usable_scans": len(usable_scans),
        "matched_scans": matched_scans,
        "match_rate": matched_scans / max(1, len(usable_scans)),
    }
    return ReplayData(
        observations, usable_scans, poses, lidar_clock, pose_clock,
        times, linear_speeds, angular_speeds, metrics)


def _planar_objective(replay, yaw_deg, held_out=False):
    indices = np.arange(len(replay.scans))
    reference = []
    moving = []
    for index, scan in enumerate(replay.scans):
        lidar_time = max(frame.time_s for frame in scan)
        laptop_time = replay.lidar_clock.to_laptop(lidar_time)
        pose_time = replay.pose_clock.from_laptop(laptop_time)
        speed = _interpolate(
            replay.motion_times, replay.linear_speeds, pose_time)
        omega = _interpolate(
            replay.motion_times, replay.angular_speeds, pose_time)
        if speed is None or omega is None:
            continue
        if speed < 0.03 and abs(omega) < math.radians(3.0):
            reference.append(index)
        else:
            moving.append(index)
    reference = reference[:8]
    moving = moving[1::5] if held_out else moving[::5]
    if len(reference) < 2 or len(moving) < 2:
        return None

    def objective(time_offset_s, x_m, y_m):
        extrinsics = LidarExtrinsics(x_m, y_m, math.radians(yaw_deg))
        references = []
        for index in reference:
            scan = deskew_scan_local(
                replay.scans[index], replay.poses, replay.lidar_clock,
                replay.pose_clock, time_offset_s, extrinsics)
            if scan is not None:
                references.append((
                    Pose2(scan.pose.x, scan.pose.y, scan.pose.yaw),
                    np.asarray(scan.points)[::3],
                ))
        field = DistanceField.from_point_sets(
            references, resolution_m=0.035, padding_m=0.5,
            max_distance_m=0.5)
        if field is None:
            return math.inf
        residuals = []
        for index in moving:
            scan = deskew_scan_local(
                replay.scans[index], replay.poses, replay.lidar_clock,
                replay.pose_clock, time_offset_s, extrinsics)
            if scan is None:
                continue
            world = transform_points(
                Pose2(scan.pose.x, scan.pose.y, scan.pose.yaw),
                np.asarray(scan.points)[::4])
            residuals.extend(field.sample(world))
        if not residuals:
            return math.inf
        residuals = np.asarray(residuals)
        return float(np.median(residuals) + 0.25 * np.percentile(residuals, 90))

    return objective


def record_command(args):
    metadata = {
        "namespace": args.namespace,
        "connect": args.connect,
        "robot_yaw_deg": args.robot_yaw_deg,
        "clock": "time.monotonic_ns",
    }
    writer = CalibrationLogWriter(args.output, metadata)
    stopping = threading.Event()
    client = KiwiClient(args.connect, args.namespace, args.robot_yaw_deg)

    def odometry(payload):
        try:
            writer.write_odometry(payload, time.monotonic_ns())
        except (ValueError, UnicodeDecodeError):
            pass

    client.subscribe("odom/twist", odometry)
    client.subscribe("lidar/ld19/raw", lambda payload: writer.write_lidar(
        payload, time.monotonic_ns()))
    writer.write_event("recording_started", time.monotonic_ns())

    def markers():
        for line in sys.stdin:
            if stopping.is_set():
                return
            label = line.strip()
            if label:
                writer.write_event(label, time.monotonic_ns())

    threading.Thread(target=markers, daemon=True).start()

    def stop(_signum=None, _frame=None):
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"recording raw calibration topics in {writer.directory}")
    print("type a phase label and press enter to add an event marker")
    started = time.monotonic()
    try:
        while not stopping.wait(0.2):
            if args.duration_s and time.monotonic() - started >= args.duration_s:
                break
    finally:
        stopping.set()
        writer.write_event("recording_stopped", time.monotonic_ns())
        client.close()
        writer.close()
    print(f"saved calibration run {writer.directory}")


def solve_command(args):
    replay = replay_run(args.run, max_scans=args.max_scans)
    rotation_segments = aggregate_rotation_segments(replay.observations)
    if len(rotation_segments) < 3:
        raise ValueError(
            "rotation solve needs at least three CW/CCW segments over 20 degrees")
    rotation = fit_rotation_calibration(rotation_segments)
    lidar = LidarCalibration(yaw_deg=args.lidar_yaw_deg)
    planar_metrics = {}
    if not args.rotation_only:
        objective = _planar_objective(replay, args.lidar_yaw_deg)
        if objective is None:
            raise ValueError(
                "planar solve needs at least two stationary and two moving scans")
        baseline = objective(0.0, 0.0, 0.0)
        planar = fit_planar_calibration(objective)
        lidar.time_offset_ms = 1000.0 * planar.time_offset_s
        lidar.x_m = planar.x_m
        lidar.y_m = planar.y_m
        planar_metrics = {
            "planar_baseline_objective_m": baseline,
            "planar_objective_m": planar.objective,
        }
    calibration = Calibration(
        source_run=str(Path(args.run).expanduser().resolve()),
        yaw_estimator=YawEstimatorConfig(
            wheel_yaw_scale=rotation.wheel_yaw_scale,
            imu_yaw_scale=rotation.imu_yaw_scale,
            initial_rate_bias_deg_s=math.degrees(rotation.imu_rate_bias_rad_s),
            imu_weight=rotation.imu_weight,
        ),
        lidar=lidar,
        validation={
            **replay.scan_metrics,
            "rotation_observations": rotation.observations,
            "rotation_scan_increments": len(replay.observations),
            "wheel_rotation_rmse_deg": math.degrees(rotation.wheel_rmse_rad),
            "imu_rotation_rmse_deg": math.degrees(rotation.imu_rmse_rad),
            **planar_metrics,
        },
    )
    output = save_calibration(calibration, args.output)
    print(json.dumps({
        "calibration": str(output),
        "rotation": asdict(rotation),
        "lidar": asdict(lidar),
        "replay": replay.scan_metrics,
    }, indent=2))


def validate_command(args):
    calibration = load_calibration(args.calibration)
    replay = replay_run(args.run, calibration, max_scans=args.max_scans)
    segments = aggregate_rotation_segments(replay.observations)
    if len(segments) < 3:
        raise ValueError("validation run has too few rotation segments")
    fit = fit_rotation_calibration(segments)
    result = {
        **replay.scan_metrics,
        "rotation_observations": len(segments),
        "rotation_scan_increments": len(replay.observations),
        "fitted_wheel_yaw_scale": fit.wheel_yaw_scale,
        "fitted_imu_yaw_scale": fit.imu_yaw_scale,
        "fitted_imu_bias_deg_s": math.degrees(fit.imu_rate_bias_rad_s),
        "configured_imu_weight": calibration.yaw_estimator.imu_weight,
        "recommended_imu_weight": fit.imu_weight,
        "residual_wheel_scale_ratio": (
            fit.wheel_yaw_scale /
            calibration.yaw_estimator.wheel_yaw_scale),
        "residual_imu_scale_ratio": (
            fit.imu_yaw_scale /
            calibration.yaw_estimator.imu_yaw_scale),
        "residual_imu_bias_deg_s": (
            math.degrees(fit.imu_rate_bias_rad_s) -
            calibration.yaw_estimator.initial_rate_bias_deg_s),
        "wheel_rotation_rmse_deg": math.degrees(fit.wheel_rmse_rad),
        "imu_rotation_rmse_deg": math.degrees(fit.imu_rmse_rad),
    }
    objective = _planar_objective(
        replay, calibration.lidar.yaw_deg, held_out=True)
    if objective is not None:
        result["held_out_planar_objective_m"] = objective(
            calibration.lidar.time_offset_ms / 1000.0,
            calibration.lidar.x_m, calibration.lidar.y_m)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="retain raw robot topics")
    record.add_argument("output")
    record.add_argument("--connect", default="tcp/127.0.0.1:7447")
    record.add_argument("--namespace", default="kiwi/xiao")
    record.add_argument("--robot-yaw-deg", type=float,
                        default=DEFAULT_ROBOT_YAW_DEG)
    record.add_argument("--duration-s", type=float, default=0.0)
    record.set_defaults(function=record_command)

    solve = subparsers.add_parser("solve", help="fit a calibration artifact")
    solve.add_argument("run")
    solve.add_argument("--output", required=True)
    solve.add_argument("--lidar-yaw-deg", type=float, default=0.0)
    solve.add_argument("--rotation-only", action="store_true")
    solve.add_argument("--max-scans", type=int)
    solve.set_defaults(function=solve_command)

    validate = subparsers.add_parser(
        "validate", help="evaluate a calibration on retained topics")
    validate.add_argument("run")
    validate.add_argument("--calibration", required=True)
    validate.add_argument("--max-scans", type=int)
    validate.set_defaults(function=validate_command)

    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
