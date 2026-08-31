#!/usr/bin/env python3
"""Decode LD19 lidar frames from the robot's Zenoh raw topic.

Frame format (47 bytes, LDROBOT LD19 dev manual):
  [0]     0x54 header
  [1]     0x2C ver/len (12 points)
  [2:4]   rotation speed, deg/s, LE16
  [4:6]   start angle, 0.01 deg, LE16
  [6:42]  12 x (distance mm LE16, intensity u8)
  [42:44] end angle, 0.01 deg, LE16
  [44:46] timestamp ms LE16 (rolls over at 30000)
  [46]    CRC8 (poly 0x4D, MSB-first, init 0) over bytes [0:46]

Usage:
  python3 scripts/kiwi_lidar.py            # per-revolution stats
  python3 scripts/kiwi_lidar.py --check    # CRC validation report only
  python3 scripts/kiwi_lidar.py --plot     # live polar plot (needs matplotlib)

Importable:
  parse_frame(bytes) -> Frame | None (None = CRC failure)
  parse_frames(bytes) -> list[Frame | None] (one or more concatenated frames)
"""

import argparse
import json
import math
import struct
import time
from dataclasses import dataclass

FRAME_LEN = 47
_CRC_TABLE = []
for i in range(256):
    crc = i
    for _ in range(8):
        crc = ((crc << 1) ^ 0x4D if crc & 0x80 else crc << 1) & 0xFF
    _CRC_TABLE.append(crc)


def crc8(data):
    crc = 0
    for b in data:
        crc = _CRC_TABLE[crc ^ b]
    return crc


@dataclass
class Frame:
    speed_dps: float
    start_angle_deg: float
    end_angle_deg: float
    timestamp_ms: int
    points: list  # [(angle_deg, distance_m, intensity), ...] x12


def parse_frame(raw):
    """Decode one 47-byte LD19 frame. Returns None if the CRC fails."""
    if len(raw) != FRAME_LEN or raw[0] != 0x54:
        return None
    if crc8(raw[:46]) != raw[46]:
        return None
    speed, start = struct.unpack_from("<HH", raw, 2)
    end, ts = struct.unpack_from("<HH", raw, 42)
    start_deg = start / 100.0
    end_deg = end / 100.0
    sweep = (end_deg - start_deg) % 360.0
    points = []
    for i in range(12):
        dist, intensity = struct.unpack_from("<HB", raw, 6 + 3 * i)
        angle = (start_deg + sweep * i / 11.0) % 360.0
        points.append((angle, dist / 1000.0, intensity))
    return Frame(speed, start_deg, end_deg, ts, points)


def parse_frames(raw):
    """Decode a Zenoh sample containing one or more concatenated LD19 frames."""
    if not raw or len(raw) % FRAME_LEN:
        return []
    return [parse_frame(raw[offset:offset + FRAME_LEN])
            for offset in range(0, len(raw), FRAME_LEN)]


class ScanAssembler:
    """Groups frames into full revolutions (angle wraps past 0)."""

    def __init__(self):
        self.points = []
        self.last_angle = None

    def add(self, frame):
        """Returns a completed revolution's points or None."""
        done = None
        if self.last_angle is not None and frame.start_angle_deg < self.last_angle - 180:
            done = self.points
            self.points = []
        self.last_angle = frame.start_angle_deg
        self.points.extend(frame.points)
        return done


def open_session(connect):
    import zenoh
    conf = zenoh.Config()
    conf.insert_json5("mode", '"client"')
    conf.insert_json5("connect/endpoints", json.dumps([connect]))
    return zenoh.open(conf)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--check", action="store_true",
                        help="CRC validation report over ~5 s, then exit")
    parser.add_argument("--plot", action="store_true",
                        help="live polar plot (requires matplotlib)")
    args = parser.parse_args()

    stats = {"frames": 0, "crc_bad": 0}
    assembler = ScanAssembler()
    scans = []

    def listener(sample):
        raw = bytes(sample.payload)
        decoded = parse_frames(raw)
        if not decoded:
            stats["frames"] += 1
            stats["crc_bad"] += 1
            return
        for frame in decoded:
            stats["frames"] += 1
            if frame is None:
                stats["crc_bad"] += 1
                continue
            rev = assembler.add(frame)
            if rev:
                scans.append((time.time(), frame.speed_dps, rev))
                if len(scans) > 10:
                    del scans[:5]

    session = open_session(args.connect)
    sub = session.declare_subscriber(f"{args.namespace}/lidar/ld19/raw", listener)

    try:
        if args.check:
            time.sleep(5)
            n, bad = stats["frames"], stats["crc_bad"]
            print(f"frames: {n} in 5 s ({n / 5:.0f}/s), CRC failures: {bad} "
                  f"({100 * bad / max(n, 1):.2f}%)")
            return

        if args.plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig = plt.figure("kiwi LD19")
            ax = fig.add_subplot(projection="polar")
            ax.set_theta_direction(-1)  # LD19 angles increase clockwise
            ax.set_theta_zero_location("N")
            while plt.fignum_exists(fig.number):
                if scans:
                    _, _, rev = scans[-1]
                    good = [(a, d) for (a, d, i) in rev if 0.02 < d < 12.0]
                    ax.clear()
                    ax.set_theta_direction(-1)
                    ax.set_theta_zero_location("N")
                    ax.set_ylim(0, 4)
                    if good:
                        ax.scatter([math.radians(a) for a, _ in good],
                                   [d for _, d in good], s=2)
                plt.pause(0.15)
            return

        # default: per-revolution stats
        print("streaming; ctrl-c to stop")
        shown = 0
        while True:
            time.sleep(0.5)
            while shown < len(scans):
                t, speed, rev = scans[shown]
                shown += 1
                good = [(a, d, i) for (a, d, i) in rev if d > 0.02]
                if good:
                    nearest = min(good, key=lambda p: p[1])
                    print(f"rev: {len(rev):4d} pts ({len(good)} valid) "
                          f"{speed / 360.0:4.1f} Hz  "
                          f"nearest {nearest[1]:5.2f} m @ {nearest[0]:5.1f} deg  "
                          f"crc_bad {stats['crc_bad']}")
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


if __name__ == "__main__":
    main()
