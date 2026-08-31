#!/usr/bin/env python3
"""Run a driveable Kiwi robot simulator on the hardware Zenoh keyspace.

The process subscribes to ``<namespace>/cmd_vel`` and publishes the same
odometry, raw LD19, camera, and status payloads as the real two-board robot.
Existing tools can connect without a simulator mode:

  python3 scripts/kiwi_simulator.py --environment room
  python3 scripts/kiwi_teleop.py --namespace kiwi/sim
  python3 scripts/kiwi_dashboard.py --namespace kiwi/sim

The optional viewer also accepts W/S forward/back, A/D strafe, Q/E rotate,
space to stop, R to reset, and Escape to quit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

from kiwi_sim_core import (
    Environment,
    KiwiRobotModel,
    LD19Simulator,
    RETAINED_ROBOT_PROFILE,
    SimulatorConfig,
    builtin_environments,
    camera_payload,
    load_environment,
    parse_velocity_payload,
)


class KiwiZenohSimulator:
    PHYSICS_HZ = 100.0
    ODOM_HZ = 20.0
    LIDAR_BATCH_HZ = 20.0
    CAMERA_HZ = RETAINED_ROBOT_PROFILE.camera_hz
    STATUS_HZ = 1.0

    def __init__(
        self,
        environment: Environment,
        connect: str,
        namespace: str,
        robot_yaw_deg: float,
        seed: int = 1,
        lidar_noise_std_m: float = 0.003,
    ):
        import zenoh

        self.environment = environment
        self.connect = connect
        self.namespace = namespace.rstrip("/")
        self.config = SimulatorConfig(robot_yaw_deg=robot_yaw_deg)
        self.model = KiwiRobotModel(environment, self.config)
        self.seed = seed
        self.lidar_noise_std_m = lidar_noise_std_m
        self.lidar = LD19Simulator(
            environment, range_noise_std_m=lidar_noise_std_m, seed=seed
        )
        self.last_lidar_pose = (
            self.model.state.x, self.model.state.y, self.model.state.yaw
        )
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.last_step_at = self.started_at
        self.next_odom_at = self.started_at
        self.next_lidar_at = self.started_at
        self.next_camera_at = self.started_at
        self.next_status_at = self.started_at
        self.camera_sequence = 0
        self.running = True
        self.stats = {
            "velocity_commands": 0,
            "invalid_commands": 0,
            "camera_published": 0,
            "lidar_published": 0,
            "lidar_batches": 0,
            "twist_published": 0,
        }

        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([connect]))
        self.session = zenoh.open(config)
        self.publishers = {
            suffix: self.session.declare_publisher(f"{self.namespace}/{suffix}")
            for suffix in (
                "odom/twist",
                "lidar/ld19/raw",
                "camera/jpeg",
                "status/master",
            )
        }
        self.command_subscriber = self.session.declare_subscriber(
            f"{self.namespace}/cmd_vel", self._on_command
        )

    def _elapsed_us(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        return int((now - self.started_at) * 1_000_000)

    def _on_command(self, sample) -> None:
        now = time.monotonic()
        try:
            twist, timeout_s = parse_velocity_payload(bytes(sample.payload))
            with self.lock:
                self.model.set_command_raw(
                    twist.vx, twist.vy, twist.omega, now, timeout_s
                )
                self.stats["velocity_commands"] += 1
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            self.stats["invalid_commands"] += 1
            print(f"ignored invalid cmd_vel: {exc}", file=sys.stderr)

    def set_local_command_aligned(
        self, vx: float, vy: float, omega: float, now: float
    ) -> None:
        with self.lock:
            self.model.set_command_aligned(vx, vy, omega, now)
            self.stats["velocity_commands"] += 1

    def reset(self) -> None:
        with self.lock:
            self.model = KiwiRobotModel(self.environment, self.config)
            self.lidar = LD19Simulator(
                self.environment,
                range_noise_std_m=self.lidar_noise_std_m,
                seed=self.seed,
            )
            self.last_lidar_pose = (
                self.model.state.x, self.model.state.y, self.model.state.yaw
            )

    def _publish_odom(self, now: float) -> None:
        with self.lock:
            report = self.model.odometry_report(self._elapsed_us(now))
        self.publishers["odom/twist"].put(json.dumps(
            report, separators=(",", ":"), allow_nan=False
        ))
        self.stats["twist_published"] += 1

    def _publish_lidar(self) -> None:
        with self.lock:
            payload = self.lidar.batch(
                self.model.state, 20, self.last_lidar_pose
            )
            self.last_lidar_pose = (
                self.model.state.x, self.model.state.y, self.model.state.yaw
            )
        self.publishers["lidar/ld19/raw"].put(payload)
        self.stats["lidar_published"] += 20
        self.stats["lidar_batches"] += 1

    def _publish_camera(self, now: float) -> None:
        with self.lock:
            payload = camera_payload(
                self.environment,
                self.model.state,
                self.camera_sequence,
                self._elapsed_us(now),
            )
        self.publishers["camera/jpeg"].put(payload)
        self.camera_sequence += 1
        self.stats["camera_published"] += 1

    def _publish_status(self, now: float) -> None:
        status = {
            "esp_ms": self._elapsed_us(now) // 1000,
            "sta_connected": True,
            "sta_ip": "127.0.0.1",
            "rssi": -25,
            "camera_ready": True,
            "zenoh_ready": True,
            "lidar_frames": self.stats["lidar_published"],
            "lidar_bad_frames": 0,
            "follower_reports": self.stats["twist_published"],
            "follower_bad_packets": self.stats["invalid_commands"],
            "velocity_commands": self.stats["velocity_commands"],
            "camera_published": self.stats["camera_published"],
            "camera_errors": 0,
            "lidar_published": self.stats["lidar_published"],
            "lidar_errors": 0,
            "lidar_batches": self.stats["lidar_batches"],
            "twist_published": self.stats["twist_published"],
            "twist_errors": 0,
            "loop_gap_max_us": 0,
            "publish_max_us": 0,
            "lidar_rx_high_water": 940,
            "follower_rx_high_water": 116,
            "free_heap": 2_000_000,
            "simulator": True,
            "environment": self.environment.name,
            "simulator_sensor_profile": RETAINED_ROBOT_PROFILE.name,
        }
        self.publishers["status/master"].put(json.dumps(
            status, separators=(",", ":")
        ))

    @staticmethod
    def _advance_deadline(deadline: float, period: float, now: float) -> float:
        while deadline <= now:
            deadline += period
        return deadline

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self.lock:
            self.model.step(now - self.last_step_at, now)
            self.last_step_at = now

        if now >= self.next_odom_at:
            self._publish_odom(now)
            self.next_odom_at = self._advance_deadline(
                self.next_odom_at, 1.0 / self.ODOM_HZ, now
            )
        if now >= self.next_lidar_at:
            self._publish_lidar()
            self.next_lidar_at = self._advance_deadline(
                self.next_lidar_at, 1.0 / self.LIDAR_BATCH_HZ, now
            )
        if now >= self.next_camera_at:
            self._publish_camera(now)
            self.next_camera_at = self._advance_deadline(
                self.next_camera_at, 1.0 / self.CAMERA_HZ, now
            )
        if now >= self.next_status_at:
            self._publish_status(now)
            self.next_status_at = self._advance_deadline(
                self.next_status_at, 1.0 / self.STATUS_HZ, now
            )

    def close(self) -> None:
        self.running = False
        self.session.close()


class PygameViewer:
    WIDTH = 1100
    HEIGHT = 760
    HUD_HEIGHT = 74

    def __init__(self, simulator: KiwiZenohSimulator, speed: float, omega: float):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for the viewer; install requirements-sim.txt "
                "or pass --headless"
            ) from exc
        self.pygame = pygame
        self.simulator = simulator
        self.speed = speed
        self.omega = omega
        pygame.init()
        pygame.display.set_caption(
            f"Kiwi simulator — {simulator.environment.name}"
        )
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont("Menlo", 16)
        self.clock = pygame.time.Clock()
        self.bounds = simulator.environment.bounds
        min_x, min_y, max_x, max_y = self.bounds
        available_width = self.WIDTH - 60
        available_height = self.HEIGHT - self.HUD_HEIGHT - 50
        self.scale = min(
            available_width / max(max_x - min_x, 0.1),
            available_height / max(max_y - min_y, 0.1),
        )
        self.origin_x = (self.WIDTH - (min_x + max_x) * self.scale) / 2.0
        world_height = (max_y - min_y) * self.scale
        self.origin_y = self.HUD_HEIGHT + 20 + world_height / 2.0
        self.origin_y += (min_y + max_y) * self.scale / 2.0
        self.local_command_active = False

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (
            round(self.origin_x + x * self.scale),
            round(self.origin_y - y * self.scale),
        )

    def _keyboard_command(self, now: float) -> None:
        pygame = self.pygame
        keys = pygame.key.get_pressed()
        vx = self.speed * (int(keys[pygame.K_w]) - int(keys[pygame.K_s]))
        vy = self.speed * (int(keys[pygame.K_a]) - int(keys[pygame.K_d]))
        omega = self.omega * (int(keys[pygame.K_q]) - int(keys[pygame.K_e]))
        if keys[pygame.K_SPACE]:
            vx = vy = omega = 0.0
        if vx or vy or omega or keys[pygame.K_SPACE]:
            self.simulator.set_local_command_aligned(vx, vy, omega, now)
            self.local_command_active = bool(vx or vy or omega)
        elif self.local_command_active:
            self.simulator.set_local_command_aligned(0.0, 0.0, 0.0, now)
            self.local_command_active = False

    def _draw_grid(self) -> None:
        pygame = self.pygame
        min_x, min_y, max_x, max_y = self.bounds
        for x in range(math.floor(min_x), math.ceil(max_x) + 1):
            pygame.draw.line(
                self.screen, (42, 49, 57),
                self.world_to_screen(x, min_y),
                self.world_to_screen(x, max_y), 1,
            )
        for y in range(math.floor(min_y), math.ceil(max_y) + 1):
            pygame.draw.line(
                self.screen, (42, 49, 57),
                self.world_to_screen(min_x, y),
                self.world_to_screen(max_x, y), 1,
            )

    def draw(self) -> None:
        pygame = self.pygame
        self.screen.fill((24, 29, 35))
        self._draw_grid()
        for wall in self.simulator.environment.walls:
            pygame.draw.line(
                self.screen, wall.color,
                self.world_to_screen(wall.x1, wall.y1),
                self.world_to_screen(wall.x2, wall.y2),
                max(3, round(self.scale * 0.035)),
            )

        with self.simulator.lock:
            state = self.simulator.model.state
            x, y, yaw = state.x, state.y, state.yaw
            measured = state.measured_raw
        center = self.world_to_screen(x, y)
        radius = max(7, round(self.simulator.config.robot_radius_m * self.scale))
        pygame.draw.circle(self.screen, (75, 215, 135), center, radius)
        forward = self.world_to_screen(
            x + math.cos(yaw) * self.simulator.config.robot_radius_m * 1.5,
            y + math.sin(yaw) * self.simulator.config.robot_radius_m * 1.5,
        )
        pygame.draw.line(self.screen, (245, 250, 250), center, forward, 3)

        title = (
            f"{self.simulator.environment.name}: "
            f"{self.simulator.environment.description}"
        )
        help_text = (
            "W/S forward  A/D strafe  Q/E rotate  SPACE stop  "
            "R reset  ESC quit   |   external kiwi_teleop.py also works"
        )
        telemetry = (
            f"pose x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.1f}°   "
            f"raw odom vx={measured.vx:+.2f} vy={measured.vy:+.2f} "
            f"ω={measured.omega:+.2f}"
        )
        for row, text in enumerate((title, help_text, telemetry)):
            surface = self.font.render(text, True, (226, 232, 237))
            self.screen.blit(surface, (18, 8 + row * 22))
        pygame.display.flip()

    def run(self) -> None:
        pygame = self.pygame
        try:
            while self.simulator.running:
                now = time.monotonic()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.simulator.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.simulator.running = False
                        elif event.key == pygame.K_r:
                            self.simulator.reset()
                self._keyboard_command(now)
                self.simulator.tick(now)
                self.draw()
                self.clock.tick(60)
        finally:
            pygame.quit()


def parse_start(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("start must be x,y,yaw_degrees")
    return parts[0], parts[1], math.radians(parts[2])


def main() -> None:
    environments = builtin_environments()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--environment", choices=sorted(environments), default="room",
        help="built-in world (default room)",
    )
    parser.add_argument(
        "--environment-file", type=Path,
        help="load a custom JSON world instead of a built-in environment",
    )
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument(
        "--namespace", default="kiwi/sim",
        help=("Zenoh namespace (default kiwi/sim, separate from physical "
              "robot namespace kiwi/xiao)"),
    )
    parser.add_argument("--robot-yaw-deg", type=float, default=60.0)
    parser.add_argument(
        "--start", type=parse_start,
        help="override spawn as x,y,yaw_degrees",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lidar-noise-std-m", type=float, default=0.003)
    parser.add_argument(
        "--headless", action="store_true",
        help="publish sensors without opening the interactive map",
    )
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--omega", type=float, default=1.2)
    args = parser.parse_args()

    environment = (
        load_environment(args.environment_file)
        if args.environment_file else environments[args.environment]
    )
    if args.start is not None:
        environment = Environment(
            environment.name,
            environment.walls,
            args.start,
            environment.description,
        )
    simulator = KiwiZenohSimulator(
        environment,
        args.connect,
        args.namespace,
        args.robot_yaw_deg,
        args.seed,
        args.lidar_noise_std_m,
    )
    print(
        f"Kiwi simulator ready: environment={environment.name} "
        f"namespace={args.namespace} connect={args.connect}"
    )
    print(
        "Zenoh mapping: cmd_vel -> odom/twist, camera/jpeg, "
        "lidar/ld19/raw, status/master"
    )
    try:
        if args.headless:
            while simulator.running:
                simulator.tick()
                time.sleep(1.0 / simulator.PHYSICS_HZ)
        else:
            PygameViewer(simulator, args.speed, args.omega).run()
    except KeyboardInterrupt:
        pass
    finally:
        simulator.close()
        print("simulator stopped")


if __name__ == "__main__":
    main()
