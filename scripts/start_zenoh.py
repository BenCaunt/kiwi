#!/usr/bin/env python3
"""Start the local Zenoh router with Kiwi's required UDP and TCP listeners."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


LISTENERS = ("udp/0.0.0.0:7447", "tcp/0.0.0.0:7447")


def zenohd_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-x", "zenohd"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line) for line in result.stdout.splitlines() if line.isdigit()]


def process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "zenohd"


def has_required_listeners(command: str) -> bool:
    return all(listener in command for listener in LISTENERS)


def zenohd_path() -> str | None:
    configured = os.environ.get("KIWI_ZENOHD")
    if configured:
        return configured

    cargo_binary = Path.home() / ".cargo" / "bin" / "zenohd"
    if cargo_binary.is_file() and os.access(cargo_binary, os.X_OK):
        return str(cargo_binary)

    return shutil.which("zenohd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="stop any existing zenohd process before starting the correct listeners",
    )
    args = parser.parse_args()

    pids = zenohd_pids()
    commands = [(pid, process_command(pid)) for pid in pids]
    correctly_running = (
        len(commands) == 1 and has_required_listeners(commands[0][1])
    )

    if correctly_running and not args.restart:
        pid, command = commands[0]
        print(f"Zenoh is already running correctly (PID {pid}):")
        print(f"  {command}")
        return 0

    if commands:
        print("Existing zenohd process(es):")
        for pid, command in commands:
            print(f"  PID {pid}: {command}")

        if not args.restart:
            print(
                "Not starting a duplicate. Run ./scripts/start_zenoh.sh --restart "
                "to replace the existing router."
            )
            return 1

        stop_script = Path(__file__).with_name("stop_zenoh.py")
        result = subprocess.run([sys.executable, str(stop_script)], check=False)
        if result.returncode != 0:
            return result.returncode

    binary = zenohd_path()
    if binary is None:
        print(
            "zenohd was not found at ~/.cargo/bin/zenohd or on PATH. "
            "Set KIWI_ZENOHD to its full path.",
            file=sys.stderr,
        )
        return 1

    command = [binary]
    for listener in LISTENERS:
        command.extend(("--listen", listener))

    print("Starting Zenoh for Kiwi with UDP and TCP on port 7447:")
    print("  " + " ".join(command), flush=True)
    os.execv(binary, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
