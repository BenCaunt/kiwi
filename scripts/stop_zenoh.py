#!/usr/bin/env python3
"""Stop every local zenohd router process."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def zenohd_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-x", "zenohd"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line) for line in result.stdout.splitlines() if line.isdigit()]


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "zenohd"


def send_signal(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="show processes without stopping them"
    )
    args = parser.parse_args()

    pids = zenohd_pids()
    if not pids:
        print("No zenohd processes are running.")
        return 0

    print(f"Found {len(pids)} zenohd process(es):")
    for pid in pids:
        print(f"  PID {pid}: {process_command(pid)}")

    if args.dry_run:
        print("Dry run: nothing stopped.")
        return 0

    print("Stopping zenohd gracefully...")
    send_signal(pids, signal.SIGINT)
    deadline = time.monotonic() + 5
    remaining = pids
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        remaining = [pid for pid in remaining if is_running(pid)]

    if remaining:
        print("Graceful shutdown timed out; sending TERM to: " + " ".join(map(str, remaining)))
        send_signal(remaining, signal.SIGTERM)
        time.sleep(1)
        remaining = [pid for pid in remaining if is_running(pid)]

    if remaining:
        print("Could not stop zenohd PID(s): " + " ".join(map(str, remaining)))
        return 1

    print("All zenohd transports are stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
