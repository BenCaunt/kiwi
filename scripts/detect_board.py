#!/usr/bin/env python3
"""Match connected USB serial devices against hardware/boards.json."""

from __future__ import annotations

import json
from pathlib import Path

from serial.tools import list_ports


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "hardware" / "boards.json"


def normalize(value: str | None) -> str:
    return (value or "").replace(":", "").replace("-", "").upper()


def main() -> int:
    inventory = json.loads(INVENTORY.read_text())
    boards = inventory.get("boards", [])
    ports = list(list_ports.comports())

    if not ports:
      print("No serial devices found.")
      return 1

    found = False
    for port in ports:
        vid_pid = ""
        if port.vid is not None and port.pid is not None:
            vid_pid = f"{port.vid:04X}:{port.pid:04X}"
        serial = normalize(port.serial_number)

        matches = [
            board
            for board in boards
            if normalize(board.get("usb_serial")) == serial
            or normalize(board.get("mac")) == serial
        ]

        if matches:
            found = True
            for board in matches:
                print(
                    f"{port.device}: role={board.get('role')} "
                    f"label={board.get('label')} serial={port.serial_number} "
                    f"vid_pid={vid_pid}"
                )
        else:
            print(f"{port.device}: unknown serial={port.serial_number or ''} vid_pid={vid_pid}")

    return 0 if found else 2


if __name__ == "__main__":
    raise SystemExit(main())
