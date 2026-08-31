#!/usr/bin/env python3
"""Arbitrate teleop and navigation commands before Kiwi's real cmd_vel topic.

Both inputs contain raw drivetrain-frame JSON commands because the publishing
KiwiClient has already applied the robot frame correction. Teleop wins while
its ``active`` lease is true; otherwise fresh navigation wins. If neither
source is fresh, the mux continuously publishes a zero command so the robot's
firmware watchdog and this process agree on the safe state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import signal
import threading
import time


ZERO_COMMAND = {"vx": 0.0, "vy": 0.0, "omega": 0.0}


@dataclass(frozen=True)
class SourceCommand:
    command: dict[str, float]
    received_at: float
    active: bool = True


def decode_command(payload: bytes | str) -> tuple[dict[str, float], bool | None]:
    """Decode a finite JSON twist and its optional teleop lease flag."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("command must be a JSON object")
    try:
        command = {
            name: float(document[name]) for name in ("vx", "vy", "omega")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("command must contain numeric vx, vy, and omega") from exc
    if not all(math.isfinite(value) for value in command.values()):
        raise ValueError("command values must be finite")
    active = document.get("active")
    if active is not None and not isinstance(active, bool):
        raise ValueError("active must be a boolean")
    return command, active


class CommandMux:
    """Thread-safe priority and timeout policy, independent of Zenoh."""

    def __init__(self, teleop_timeout_s=0.35, navigation_timeout_s=0.35):
        self.teleop_timeout_s = float(teleop_timeout_s)
        self.navigation_timeout_s = float(navigation_timeout_s)
        self._lock = threading.Lock()
        self._teleop: SourceCommand | None = None
        self._navigation: SourceCommand | None = None

    def update(self, source: str, payload: bytes | str,
               now: float | None = None) -> None:
        command, active_flag = decode_command(payload)
        timestamp = time.monotonic() if now is None else float(now)
        if source == "teleop":
            # Old clients have no lease bit: nonzero means take control and a
            # zero command means release it.
            active = (any(abs(value) > 0.0 for value in command.values())
                      if active_flag is None else active_flag)
            value = SourceCommand(command, timestamp, active)
            with self._lock:
                self._teleop = value
        elif source == "navigation":
            value = SourceCommand(command, timestamp)
            with self._lock:
                self._navigation = value
        else:
            raise ValueError(f"unknown command source: {source}")

    def select(self, now: float | None = None) -> tuple[str, dict[str, float]]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            teleop = self._teleop
            navigation = self._navigation
        if (teleop is not None and teleop.active and
                timestamp - teleop.received_at <= self.teleop_timeout_s):
            return "teleop", dict(teleop.command)
        if (navigation is not None and
                timestamp - navigation.received_at <= self.navigation_timeout_s):
            return "navigation", dict(navigation.command)
        return "idle", dict(ZERO_COMMAND)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--teleop-topic", default="cmd_vel/teleop")
    parser.add_argument("--navigation-topic", default="cmd_vel/navigation")
    parser.add_argument("--output-topic", default="cmd_vel")
    parser.add_argument("--teleop-timeout", type=_positive_float, default=0.35)
    parser.add_argument("--navigation-timeout", type=_positive_float, default=0.35)
    parser.add_argument("--rate", type=_positive_float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import zenoh

    namespace = args.namespace.rstrip("/")
    conf = zenoh.Config()
    conf.insert_json5("mode", '"client"')
    conf.insert_json5("connect/endpoints", json.dumps([args.connect]))
    session = zenoh.open(conf)
    mux = CommandMux(args.teleop_timeout, args.navigation_timeout)
    output = session.declare_publisher(
        f"{namespace}/{args.output_topic.strip('/')}")
    status = session.declare_publisher(f"{namespace}/cmd_vel/mux/status")

    def receive(source):
        def listener(sample):
            try:
                mux.update(source, bytes(sample.payload))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                return
        return listener

    subscribers = [
        session.declare_subscriber(
            f"{namespace}/{args.teleop_topic.strip('/')}", receive("teleop")),
        session.declare_subscriber(
            f"{namespace}/{args.navigation_topic.strip('/')}",
            receive("navigation")),
    ]
    stopping = threading.Event()

    def stop(_signum=None, _frame=None):
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"command mux: {namespace}/{args.teleop_topic.strip('/')} (priority) + "
        f"{namespace}/{args.navigation_topic.strip('/')} -> "
        f"{namespace}/{args.output_topic.strip('/')} at {args.rate:g} Hz; "
        "Ctrl-C stops the robot"
    )
    period_s = 1.0 / args.rate
    next_cycle = time.monotonic()
    last_source = None
    next_status = 0.0
    try:
        while not stopping.is_set():
            now = time.monotonic()
            source, command = mux.select(now)
            output.put(json.dumps(command, separators=(",", ":")))
            if source != last_source:
                print(f"command mux source: {source}")
                last_source = source
            if now >= next_status:
                status.put(json.dumps({
                    "source": source,
                    "command": command,
                }, separators=(",", ":")))
                next_status = now + 0.2
            next_cycle += period_s
            stopping.wait(max(0.0, next_cycle - time.monotonic()))
    finally:
        zero = json.dumps(ZERO_COMMAND, separators=(",", ":"))
        for _ in range(3):
            output.put(zero)
            time.sleep(0.03)
        # Keep Zenoh handles alive until after the final stop packets flush.
        del subscribers
        session.close()


if __name__ == "__main__":
    main()
