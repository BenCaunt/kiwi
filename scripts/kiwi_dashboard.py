#!/usr/bin/env python3
"""Live rerun dashboard for the kiwi robot: camera, IMU, twists, wheels, lidar.

Subscribes to kiwi/xiao/** over Zenoh and streams everything into a rerun
viewer: camera feed, 3D robot pose (IMU orientation + dead-reckoned
position), lidar scan points, commanded-vs-measured twist plots, wheel
speeds, accelerometer, and link/system stats.

Run:  python3 scripts/kiwi_dashboard.py            # spawns the rerun viewer
      python3 scripts/kiwi_dashboard.py --connect tcp/127.0.0.1:7447
"""

import argparse
import json
import math
import struct
import sys
import time

import rerun as rr
import rerun.blueprint as rrb
import zenoh

sys.path.insert(0, "scripts")
from kiwi_lidar import parse_frame, ScanAssembler  # noqa: E402

CAMERA_MAGIC = b"KVC1"


def make_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="camera", origin="/camera"),
                rrb.Spatial3DView(name="robot + lidar", origin="/world"),
                row_shares=[2, 3],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="twist vx", origin="/twist/vx"),
                rrb.TimeSeriesView(name="twist vy", origin="/twist/vy"),
                rrb.TimeSeriesView(name="twist omega", origin="/twist/omega"),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="wheel speeds", origin="/wheels"),
                rrb.TimeSeriesView(name="imu accel", origin="/imu/accel"),
                rrb.TimeSeriesView(name="system", origin="/system"),
            ),
            column_shares=[4, 3, 3],
        ),
        collapse_panels=True,
    )


def yaw_from_quat(i, j, k, r):
    return math.atan2(2.0 * (r * k + i * j), 1.0 - 2.0 * (j * j + k * k))


class Dashboard:
    def __init__(self):
        self.pose = [0.0, 0.0, 0.0]  # x, y, yaw (dead-reckoned, IMU heading)
        self.yaw_offset = None
        self.last_twist_t = None
        self.trajectory = []
        self.assembler = ScanAssembler()

    def on_camera(self, payload):
        if len(payload) < 32 or payload[:4] != CAMERA_MAGIC:
            return
        header_len = struct.unpack_from("<H", payload, 10)[0]
        jpeg = bytes(payload[header_len:])
        rr.log("/camera", rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))

    def on_twist(self, payload):
        try:
            m = json.loads(payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        meas, cmd = m.get("measured", {}), m.get("command", {})
        for axis in ("vx", "vy", "omega"):
            rr.log(f"/twist/{axis}/measured", rr.Scalar(meas.get(axis, 0.0)))
            rr.log(f"/twist/{axis}/commanded", rr.Scalar(cmd.get(axis, 0.0)))
        for i, w in enumerate(m.get("wheel_speed_mps", [])):
            rr.log(f"/wheels/w{i}", rr.Scalar(w))
        accel = m.get("imu_accel_mps2", [0, 0, 0])
        for name, val in zip("xyz", accel):
            rr.log(f"/imu/accel/{name}", rr.Scalar(val))

        # Pose: IMU yaw for heading, integrate measured body twist for position.
        q = m.get("imu_quat_ijkr")
        now = time.time()
        if q and m.get("imu_ready"):
            yaw = yaw_from_quat(*q)
            if self.yaw_offset is None:
                self.yaw_offset = yaw
            self.pose[2] = yaw - self.yaw_offset
        if self.last_twist_t is not None:
            dt = min(now - self.last_twist_t, 0.2)
            vx, vy = meas.get("vx", 0.0), meas.get("vy", 0.0)
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            self.pose[0] += (c * vx - s * vy) * dt
            self.pose[1] += (s * vx + c * vy) * dt
        self.last_twist_t = now

        rr.log("/world/robot",
               rr.Transform3D(translation=[self.pose[0], self.pose[1], 0.0],
                              rotation=rr.Quaternion(
                                  xyzw=[0.0, 0.0, math.sin(self.pose[2] / 2),
                                        math.cos(self.pose[2] / 2)])))
        self.trajectory.append([self.pose[0], self.pose[1], 0.0])
        if len(self.trajectory) > 3000:
            del self.trajectory[:1000]
        rr.log("/world/trajectory", rr.LineStrips3D([self.trajectory]))

    def on_lidar(self, payload):
        frame = parse_frame(bytes(payload))
        if frame is None:
            return
        rev = self.assembler.add(frame)
        if not rev:
            return
        pts = [(d * math.cos(-math.radians(a)), d * math.sin(-math.radians(a)), 0.05)
               for (a, d, inten) in rev if 0.02 < d < 12.0]
        if pts:
            rr.log("/world/robot/lidar", rr.Points3D(pts, radii=0.01))

    def on_status(self, payload):
        try:
            s = json.loads(payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        rr.log("/system/heap_kb", rr.Scalar(s.get("free_heap", 0) / 1024))
        rr.log("/system/rssi_dbm", rr.Scalar(s.get("rssi", 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    args = parser.parse_args()

    rr.init("kiwi_dashboard", spawn=True, default_blueprint=make_blueprint())
    # static scene: robot body footprint
    rr.log("/world/robot/body",
           rr.Boxes3D(centers=[[0, 0, 0.04]], half_sizes=[[0.11, 0.11, 0.04]],
                      colors=[[80, 200, 120]]),
           static=True)

    dash = Dashboard()
    routes = {
        "camera/jpeg": dash.on_camera,
        "odom/twist": dash.on_twist,
        "lidar/ld19/raw": dash.on_lidar,
        "status/master": dash.on_status,
    }

    def listener(sample):
        key = str(sample.key_expr)
        rr.set_time_seconds("time", time.time())
        for suffix, handler in routes.items():
            if key.endswith(suffix):
                handler(bytes(sample.payload))
                return

    conf = zenoh.Config()
    conf.insert_json5("mode", '"client"')
    conf.insert_json5("connect/endpoints", json.dumps([args.connect]))
    session = zenoh.open(conf)
    sub = session.declare_subscriber(f"{args.namespace}/**", listener)
    print("dashboard streaming; ctrl-c to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


if __name__ == "__main__":
    main()
