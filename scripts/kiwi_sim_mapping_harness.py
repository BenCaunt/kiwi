#!/usr/bin/env python3
"""Privileged simulation harness for full-house SLAM and image-map runs.

Ground truth is consumed only from the browser bridge's loopback WebSocket.
The unprivileged SLAM process receives the same Zenoh odometry, LD19, and
camera topics as the physical robot and receives no simulator-only state.
Commands are sent through KiwiClient's ordinary aligned ``cmd_vel`` API.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kiwi_client import KiwiClient  # noqa: E402
from kiwi_pose_controller import (  # noqa: E402
    Pose2,
    PoseStabilizingController,
)


@dataclass(frozen=True)
class TrajectoryWaypoint:
    name: str
    pose: Pose2
    dwell_s: float = 0.20


def waypoint(name, x, y, yaw_deg, dwell_s=0.20):
    return TrajectoryWaypoint(
        str(name), Pose2(float(x), float(y), math.radians(float(yaw_deg))),
        float(dwell_s),
    )


def scan_station(name, x, y):
    return [
        waypoint(f"{name} north", x, y, 90, 0.60),
        waypoint(f"{name} west", x, y, 180, 0.60),
        waypoint(f"{name} south", x, y, -90, 0.60),
        waypoint(f"{name} east", x, y, 0, 0.60),
    ]


HOME_MAPPING_TRAJECTORY = [
    waypoint("entry", 0.00, -3.00, 180),
    waypoint("clear entry plant", 0.00, -2.75, 180),
    waypoint("living approach", -2.00, -2.75, 90),
    waypoint("living north", -2.00, -0.35, 180),
    waypoint("west bedroom door", -4.00, -0.15, 90),
    waypoint("west bedroom threshold", -4.00, 1.28, 90),
    *scan_station("west bedroom scan", -4.00, 1.28),
    waypoint("leave west bedroom", -4.00, -0.15, 0),
    waypoint("hall entry", 0.00, 0.00, 90),
    waypoint("hall", 0.00, 1.75, 90),
    waypoint("bathroom threshold", 0.00, 2.78, 90),
    *scan_station("bathroom scan", 0.00, 2.78),
    waypoint("leave bathroom", 0.00, 0.00, 0),
    waypoint("east bedroom door", 4.00, -0.15, 90),
    waypoint("east bedroom threshold", 4.00, 1.30, 90),
    *scan_station("east bedroom scan", 4.00, 1.30),
    waypoint("leave east bedroom", 4.00, -0.15, -90),
    waypoint("east kitchen lane", 4.55, -3.00, 180),
    waypoint("south kitchen lane", 2.80, -3.00, 180),
    waypoint("dining lane", 1.40, -3.00, 180),
    waypoint("close loop", 0.00, -3.00, 90, 1.0),
]

SMOKE_TRAJECTORY = [
    waypoint("smoke north", 0.00, -2.70, 90),
    waypoint("smoke west", -0.50, -2.70, 180),
    waypoint("smoke return", 0.00, -3.00, 90),
]

ROTATION_REGRESSION_TRAJECTORY = [
    waypoint("rotation start", 0.00, -3.00, 180, 0.50),
    *scan_station("rotation lap one", 0.00, -3.00),
    *scan_station("rotation lap two", 0.00, -3.00),
    waypoint("rotation close", 0.00, -3.00, 180, 0.75),
]

TRAJECTORIES = {
    "home-map": HOME_MAPPING_TRAJECTORY,
    "rotation-regression": ROTATION_REGRESSION_TRAJECTORY,
    "smoke": SMOKE_TRAJECTORY,
}


def default_output_prefix():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "maps" / f"web-home-{stamp}"


def ensure_simulation_namespace(namespace, allow_non_sim_namespace=False):
    normalized = "/" + str(namespace).strip("/")
    if (normalized.endswith("/sim") or "/sim/" in normalized or
            allow_non_sim_namespace):
        return
    raise ValueError(
        f"refusing privileged harness on non-simulation namespace {namespace!r}; "
        "use --allow-non-sim-namespace only after independently proving no "
        "physical robot can receive commands"
    )


def slam_command(args, output_prefix):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "kiwi_slam.py"),
        "--connect", args.connect,
        "--namespace", args.namespace,
        "--output", str(output_prefix),
        "--print-every", str(args.slam_print_every),
        "--image-distance-m", str(args.image_distance_m),
        "--image-angle-deg", str(args.image_angle_deg),
        "--image-min-interval-s", str(args.image_min_interval_s),
        # The Three.js sensor camera is mounted at 0.22 m. This is a normal
        # runtime calibration flag and does not alter physical-robot defaults.
        "--camera-height-m", "0.22",
        "--yaw-estimator", args.yaw_estimator,
    ]
    if args.calibration:
        command.extend(("--calibration", args.calibration))
    if args.viewer:
        command.append("--viewer")
    return command


def stop_slam(process, timeout=20.0):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)


def mapping_artifacts(output_prefix):
    prefix = Path(output_prefix)
    manifests = sorted(Path(f"{prefix}.images").glob("*/manifest.json"))
    return {
        "graph": Path(f"{prefix}.graph.json"),
        "state": Path(f"{prefix}.slam.npz"),
        "occupancy": Path(f"{prefix}.pgm"),
        "metadata": Path(f"{prefix}.yaml"),
        "image_manifest": manifests[-1] if manifests else None,
        "harness_report": Path(f"{prefix}.harness.json"),
    }


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_slam_quality(samples):
    headings = [abs(float(sample["heading_disagreement_deg"]))
                for sample in samples]
    matched = [bool(sample["scan_matched"]) for sample in samples]
    accepted = sum(
        sample.get("scan_feedback") == "accepted" for sample in samples)
    final = samples[-1] if samples else {}
    return {
        "scan_samples": len(samples),
        "scan_match_rate": (sum(matched) / len(matched) if matched else None),
        "heading_disagreement_abs_p50_deg": percentile(headings, 0.50),
        "heading_disagreement_abs_p95_deg": percentile(headings, 0.95),
        "heading_disagreement_abs_max_deg": max(headings) if headings else None,
        "final_heading_disagreement_deg": final.get(
            "heading_disagreement_deg"),
        "final_learned_rate_bias_deg_s": final.get(
            "learned_rate_bias_deg_s"),
        "scan_feedback_accepted_samples": accepted,
    }


class PrivilegedTrajectoryActor:
    def __init__(
        self,
        bridge_url,
        client,
        controller,
        expected_world="home",
        waypoint_timeout_s=45.0,
    ):
        self.bridge_url = bridge_url
        self.client = client
        self.controller = controller
        self.expected_world = expected_world
        self.waypoint_timeout_s = float(waypoint_timeout_s)
        self.last_ground_truth = None
        self.last_simulation_time = None
        self.ground_truth_distance_m = 0.0
        self.samples = 0

    @staticmethod
    def _decode_ground_truth(message):
        if not isinstance(message, str):
            return None
        try:
            document = json.loads(message)
        except json.JSONDecodeError:
            return None
        if document.get("type") != "ground-truth":
            return None
        pose = document.get("pose")
        if not isinstance(pose, dict):
            return None
        try:
            decoded = Pose2.from_mapping(pose)
            simulation_time = float(document["simulation_time_s"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (
                decoded.x, decoded.y, decoded.yaw, simulation_time)):
            return None
        return document.get("world"), decoded, simulation_time

    async def _next_ground_truth(self, websocket):
        while True:
            decoded = self._decode_ground_truth(await websocket.recv())
            if decoded is None:
                continue
            world, pose, simulation_time = decoded
            if world != self.expected_world:
                raise RuntimeError(
                    f"expected simulator world {self.expected_world!r}, got {world!r}"
                )
            if (self.last_ground_truth is not None and
                    self.last_simulation_time is not None and
                    simulation_time > self.last_simulation_time):
                self.ground_truth_distance_m += math.hypot(
                    pose.x - self.last_ground_truth.x,
                    pose.y - self.last_ground_truth.y,
                )
            self.last_ground_truth = pose
            self.last_simulation_time = simulation_time
            self.samples += 1
            return pose

    async def drive(self, trajectory):
        from websockets.asyncio.client import connect

        started = time.monotonic()
        async with connect(
            self.bridge_url,
            max_size=2 * 1024 * 1024,
            compression=None,
        ) as websocket:
            await websocket.send(json.dumps({
                "type": "hello",
                "client": "kiwi-mapping-harness",
                "role": "privileged-harness",
            }))
            await asyncio.wait_for(self._next_ground_truth(websocket), timeout=5.0)
            print(
                f"privileged actor connected; {len(trajectory)} waypoints in "
                f"{self.expected_world!r}", flush=True,
            )
            try:
                for index, target in enumerate(trajectory, start=1):
                    waypoint_started = time.monotonic()
                    last_report = 0.0
                    while True:
                        current = await asyncio.wait_for(
                            self._next_ground_truth(websocket), timeout=3.0
                        )
                        if self.controller.at_target(current, target.pose):
                            self.client.send_twist(0.0, 0.0, 0.0)
                            print(
                                f"[{index:02d}/{len(trajectory):02d}] reached "
                                f"{target.name}", flush=True,
                            )
                            if target.dwell_s:
                                await asyncio.sleep(target.dwell_s)
                            break
                        elapsed = time.monotonic() - waypoint_started
                        if elapsed > self.waypoint_timeout_s:
                            raise TimeoutError(
                                f"timed out reaching {target.name!r}; pose="
                                f"({current.x:.2f}, {current.y:.2f}, "
                                f"{math.degrees(current.yaw):.1f} deg)"
                            )
                        command = self.controller.command(current, target.pose)
                        self.client.send_twist(
                            command.vx, command.vy, command.omega
                        )
                        if elapsed - last_report >= 2.0:
                            remaining = math.hypot(
                                target.pose.x - current.x,
                                target.pose.y - current.y,
                            )
                            print(
                                f"[{index:02d}/{len(trajectory):02d}] "
                                f"{target.name}: {remaining:.2f} m remaining",
                                flush=True,
                            )
                            last_report = elapsed
            finally:
                self.client.send_twist(0.0, 0.0, 0.0)
        return {
            "world": self.expected_world,
            "waypoints": len(trajectory),
            "ground_truth_distance_m": self.ground_truth_distance_m,
            "ground_truth_samples": self.samples,
            "wall_elapsed_s": time.monotonic() - started,
            "final_pose": (
                None if self.last_ground_truth is None else {
                    "x": self.last_ground_truth.x,
                    "y": self.last_ground_truth.y,
                    "yaw": self.last_ground_truth.yaw,
                }
            ),
        }


async def async_main(args):
    ensure_simulation_namespace(args.namespace, args.allow_non_sim_namespace)
    output_prefix = Path(args.output).expanduser() if args.output else default_output_prefix()
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".graph.json", ".slam.npz", ".pgm", ".yaml"):
        if Path(f"{output_prefix}{suffix}").exists():
            raise FileExistsError(f"refusing to overwrite {output_prefix}{suffix}")

    client = KiwiClient(
        args.connect,
        args.namespace,
        commanding=True,
    )
    slam_process = None
    image_samples = [0]
    map_samples = [0]
    slam_quality_samples = []
    client.subscribe("slam/image", lambda _payload: image_samples.__setitem__(0, image_samples[0] + 1))
    client.subscribe("slam/map", lambda _payload: map_samples.__setitem__(0, map_samples[0] + 1))

    def retain_slam_quality(report):
        if report.get("source") != "scan_correction":
            return
        quality = report.get("quality", {})
        estimator = report.get("yaw_estimator", {})
        try:
            slam_quality_samples.append({
                "scan_matched": bool(quality["scan_matched"]),
                "heading_disagreement_deg": float(
                    quality["heading_disagreement_deg"]),
                "wall_support_ratio": float(
                    quality.get("wall_support_ratio", 0.0)),
                "learned_rate_bias_deg_s": (
                    float(estimator["learned_rate_bias_deg_s"])
                    if "learned_rate_bias_deg_s" in estimator else None),
                "scan_feedback": estimator.get("scan_feedback"),
            })
        except (KeyError, TypeError, ValueError):
            return

    client.add_slam_callback(retain_slam_quality)
    actor = PrivilegedTrajectoryActor(
        args.bridge,
        client,
        PoseStabilizingController(
            kp_x=args.kp_position,
            kp_y=args.kp_position,
            kp_yaw=args.kp_yaw,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            position_tolerance=args.position_tolerance,
            yaw_tolerance=math.radians(args.yaw_tolerance_deg),
        ),
        expected_world="home",
        waypoint_timeout_s=args.waypoint_timeout,
    )

    try:
        if not args.drive_only:
            # Move out of the tight spawn pocket before SLAM exists. The entry
            # pose has substantially better scan coverage, and keeping this
            # privileged setup move outside the SLAM process preserves the
            # same clean sensor-only boundary used on the physical robot.
            print("pre-positioning simulator at the home entry", flush=True)
            await actor.drive([HOME_MAPPING_TRAJECTORY[0]])
            await asyncio.sleep(0.75)
            command = slam_command(args, output_prefix)
            print("starting unprivileged SLAM: " + " ".join(command), flush=True)
            slam_process = subprocess.Popen(command, cwd=PROJECT_ROOT)
            deadline = time.monotonic() + args.slam_startup_timeout
            while client.pose is None and time.monotonic() < deadline:
                if slam_process.poll() is not None:
                    raise RuntimeError(
                        "unprivileged SLAM exited during startup with code "
                        f"{slam_process.returncode}"
                    )
                await asyncio.sleep(0.1)
            if client.pose is None:
                raise TimeoutError("unprivileged SLAM did not publish an initial pose")
            print("unprivileged SLAM pose stream is ready", flush=True)

        trajectory = TRAJECTORIES[args.trajectory]
        report = await actor.drive(trajectory)
        report.update({
            "trajectory": args.trajectory,
            "namespace": args.namespace,
            "output_prefix": str(output_prefix),
            "image_topic_samples": image_samples[0],
            "map_topic_samples": map_samples[0],
            "slam_quality": summarize_slam_quality(slam_quality_samples),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        try:
            client.send_twist(0.0, 0.0, 0.0)
        except Exception:
            pass
        stop_slam(slam_process)
        client.close()

    if args.drive_only:
        print("drive-only trajectory complete", flush=True)
        return

    report_path = Path(f"{output_prefix}.harness.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    artifacts = mapping_artifacts(output_prefix)
    missing = [name for name, path in artifacts.items() if path is None or not path.is_file()]
    if missing:
        raise RuntimeError(
            "mapping finished but required artifacts are missing: " + ", ".join(missing)
        )
    manifest = json.loads(artifacts["image_manifest"].read_text(encoding="utf-8"))
    captures = len(manifest.get("captures", []))
    print(
        f"mapping complete: {report['ground_truth_distance_m']:.2f} m ground truth; "
        f"{captures} saved images; {image_samples[0]} image-topic samples; "
        f"{map_samples[0]} map-topic samples",
        flush=True,
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/sim")
    parser.add_argument("--bridge", default="ws://127.0.0.1:8767")
    parser.add_argument("--output")
    parser.add_argument("--trajectory", choices=sorted(TRAJECTORIES), default="home-map")
    parser.add_argument("--drive-only", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--yaw-estimator", choices=("legacy", "fused"), default="fused")
    parser.add_argument("--calibration")
    parser.add_argument("--allow-non-sim-namespace", action="store_true")
    parser.add_argument("--slam-startup-timeout", type=float, default=20.0)
    parser.add_argument("--waypoint-timeout", type=float, default=45.0)
    parser.add_argument("--slam-print-every", type=int, default=10)
    parser.add_argument("--max-linear-speed", type=float, default=0.25)
    parser.add_argument("--max-angular-speed", type=float, default=1.2)
    parser.add_argument("--kp-position", type=float, default=1.2)
    parser.add_argument("--kp-yaw", type=float, default=5.0)
    parser.add_argument("--position-tolerance", type=float, default=0.07)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=4.0)
    parser.add_argument("--image-distance-m", type=float, default=0.45)
    parser.add_argument("--image-angle-deg", type=float, default=30.0)
    parser.add_argument("--image-min-interval-s", type=float, default=0.35)
    args = parser.parse_args()
    positive = (
        args.slam_startup_timeout, args.waypoint_timeout,
        args.max_linear_speed, args.max_angular_speed,
        args.kp_position, args.kp_yaw, args.position_tolerance,
        args.yaw_tolerance_deg, args.image_distance_m,
        args.image_angle_deg, args.image_min_interval_s,
    )
    if any(value <= 0.0 for value in positive):
        parser.error("timing, gain, tolerance, and spacing values must be positive")
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"mapping harness failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
