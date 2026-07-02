#!/usr/bin/env python
"""Quick serial probe: optionally send commands, then print output for a while.

Usage:
  serial_probe.py [--port PORT] [--listen SECONDS] [--send CMD] [--send CMD2] ...

Commands are sent one per second after an initial settle delay.
"""

import argparse
import sys
import time

import serial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbmodem1101")
    parser.add_argument("--listen", type=float, default=5.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--send", action="append", default=[])
    parser.add_argument("--send-gap", type=float, default=1.0)
    args = parser.parse_args()

    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = 115200
    ser.timeout = 0.1
    # ESP32-S3 native USB CDC: keep DTR asserted so TX from the board flows,
    # avoid RTS toggling that could wiggle the auto-download circuit.
    ser.dtr = True
    ser.rts = False
    ser.open()

    deadline = time.time() + args.settle
    pending = list(args.send)
    next_send = time.time() + args.settle
    end = time.time() + args.settle + args.listen

    buf = b""
    while time.time() < end:
        data = ser.read(256)
        if data:
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                print(line.decode(errors="replace").rstrip())
                sys.stdout.flush()
        if pending and time.time() >= next_send:
            cmd = pending.pop(0)
            print(f">>> sending: {cmd!r}")
            ser.write((cmd + "\n").encode())
            ser.flush()
            next_send = time.time() + args.send_gap
    if buf:
        print(buf.decode(errors="replace").rstrip())
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
