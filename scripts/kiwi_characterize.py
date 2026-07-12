#!/usr/bin/env python3
"""Characterize the kiwi drivetrain and auto-tune the wheel PI controller.

ROBOT MUST BE WHEELS-UP on a test bed. Requires zenohd running locally and
both boards flashed with DriveParams v36 firmware (deadband + PI fields).

Phases:
  A. Breakaway: ramp each motor individually until its encoder moves
     -> motor_deadband_pct
  B. Speed sweep: measure speed vs command with deadbands applied
     -> true max wheel speed (feedforward gain / max_wheel_surface_speed_mps)
  C. Step response: measure the velocity time constant tau
     -> IMC PI gains (kp = 100/G with lambda=tau, ki = kp/tau)
  D. Apply closed loop and verify low-speed tracking.

Run:  python3 scripts/kiwi_characterize.py [--host 192.168.1.157] [--dry-run]
"""

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request

import zenoh

R_BASE = 0.09  # must match drive_base_radius_m for isolated-wheel math


def http_config(host, payload):
    req = urllib.request.Request(
        f"http://{host}/config", data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        out = json.loads(resp.read().decode())
    if not out.get("ok"):
        sys.exit(f"config rejected: {out}")
    time.sleep(2.5)  # let the follower ack the new version
    return out


def http_status(host):
    with urllib.request.urlopen(f"http://{host}/status", timeout=5) as resp:
        return json.loads(resp.read().decode())


class Rig:
    def __init__(self, connect, namespace):
        conf = zenoh.Config()
        conf.insert_json5("mode", '"client"')
        conf.insert_json5("connect/endpoints", json.dumps([connect]))
        self.session = zenoh.open(conf)
        self.pub = self.session.declare_publisher(f"{namespace}/cmd_vel")
        self.samples = []  # (t, [w0,w1,w2])
        self.sub = self.session.declare_subscriber(
            f"{namespace}/odom/twist", self._on_twist)

    def _on_twist(self, sample):
        try:
            m = json.loads(bytes(sample.payload).decode())
            self.samples.append((time.time(), m["wheel_speed_mps"]))
            if len(self.samples) > 4000:
                del self.samples[:2000]
        except Exception:
            pass

    def send_wheels(self, wheels):
        """Command a twist that produces exactly these wheel surface speeds."""
        vx = (2 / 3) * sum(-math.sin(i * 2 * math.pi / 3) * wheels[i] for i in range(3))
        vy = (2 / 3) * sum(math.cos(i * 2 * math.pi / 3) * wheels[i] for i in range(3))
        om = sum(wheels) / (3 * R_BASE)
        self.pub.put(json.dumps({"vx": vx, "vy": vy, "omega": om}))

    def hold(self, wheels, secs):
        end = time.time() + secs
        while time.time() < end:
            self.send_wheels(wheels)
            time.sleep(0.08)

    def stop(self, settle=1.0):
        self.send_wheels([0.0, 0.0, 0.0])
        time.sleep(settle)

    def window(self, since):
        return [(t, w) for (t, w) in self.samples if t >= since]

    def mean_speed(self, wheel, since):
        vals = [w[wheel] for (_, w) in self.window(since) if len(w) == 3]
        return statistics.mean(vals) if vals else 0.0

    def close(self):
        self.stop(0.2)
        self.session.close()


def phase_a_breakaway(rig, max_speed):
    """With deadband=0 and max_speed as FF scale, command pct = wheel/max*100."""
    print("\n=== Phase A: breakaway per motor ===")
    deadbands = []
    for wheel in range(3):
        found = {}
        for sign in (+1, -1):
            breakaway = None
            for pct in range(15, 85, 5):
                target = sign * (pct / 100.0) * max_speed
                cmd = [0.0, 0.0, 0.0]
                cmd[wheel] = target
                rig.hold(cmd, 1.1)
                t0 = time.time() - 0.4
                rig.hold(cmd, 0.4)
                speed = rig.mean_speed(wheel, t0)
                rig.stop(0.8)
                if abs(speed) > 0.05:
                    breakaway = pct
                    break
            found[sign] = breakaway if breakaway is not None else 85
            print(f"  motor{wheel} {'+' if sign > 0 else '-'}: breakaway ~{found[sign]}%")
        db = max(found[+1], found[-1])
        deadbands.append(max(db - 4, 0))  # small margin below breakaway
    print(f"  -> motor_deadband_pct = {deadbands}")
    return deadbands


def phase_b_speed_sweep(rig, max_speed):
    """Deadbands applied: measure actual speed vs commanded, fit gain G."""
    print("\n=== Phase B: speed sweep (feedforward gain) ===")
    gains = []
    for wheel in range(3):
        pts = []
        for pct in (30, 50, 70, 90):
            for sign in (+1, -1):
                target = sign * (pct / 100.0) * max_speed
                cmd = [0.0, 0.0, 0.0]
                cmd[wheel] = target
                rig.hold(cmd, 1.4)
                t0 = time.time() - 0.5
                rig.hold(cmd, 0.5)
                actual = rig.mean_speed(wheel, t0)
                rig.stop(0.7)
                if abs(actual) > 0.05:
                    pts.append((target, actual))
        if pts:
            g = statistics.median(a / t for (t, a) in pts)
            gains.append(g)
            print(f"  motor{wheel}: actual/commanded gain = {g:.2f}")
    if not gains:
        sys.exit("no wheel motion measured in sweep; aborting")
    g_med = statistics.median(gains)
    true_max = g_med * max_speed
    print(f"  -> true max wheel speed ~= {true_max:.2f} m/s "
          f"(setting max_wheel_surface_speed_mps)")
    return true_max


def phase_c_step_response(rig, true_max):
    """FF calibrated: step all wheels to 40% of max, estimate tau."""
    print("\n=== Phase C: step response (time constant) ===")
    taus = []
    target = 0.4 * true_max
    for trial in range(3):
        rig.stop(1.0)
        t_step = time.time()
        rig.hold([target] * 3, 1.5)
        rig.stop(0.8)
        for wheel in range(3):
            win = rig.window(t_step)
            final_vals = [w[wheel] for (t, w) in win if t > t_step + 0.8]
            if not final_vals:
                continue
            final = statistics.mean(final_vals)
            if abs(final) < 0.1:
                continue
            thresh = 0.632 * final
            crossed = [t for (t, w) in win if (w[wheel] >= thresh if final > 0
                                               else w[wheel] <= thresh)]
            if crossed:
                taus.append(max(crossed[0] - t_step, 0.02))
    if not taus:
        print("  could not estimate tau; defaulting to 0.20 s")
        tau = 0.20
    else:
        tau = statistics.median(taus)
    print(f"  -> tau ~= {tau:.3f} s  (n={len(taus)})")
    return tau


def compute_gains(true_max, tau):
    # IMC with lambda = tau: kp = tau / (Kplant * lambda) = 100 / (true_max)
    # where Kplant = true_max/100 m/s per percent. ki = kp / tau.
    kp = 100.0 / true_max
    ki = kp / tau
    # conservative clamps
    kp = min(kp, 200.0)
    ki = min(ki, 1500.0)
    return round(kp, 1), round(ki, 1)


def phase_d_verify(rig, true_max):
    print("\n=== Phase D: closed-loop low-speed tracking ===")
    ok = True
    for target in (0.15, 0.4):
        rig.hold([target] * 3, 2.0)
        t0 = time.time() - 0.6
        rig.hold([target] * 3, 0.6)
        meas = [rig.mean_speed(w, t0) for w in range(3)]
        rig.stop(0.8)
        errs = [m - target for m in meas]
        worst = max(abs(e) for e in errs)
        status = "OK" if worst < 0.08 else "CHECK"
        if status != "OK":
            ok = False
        print(f"  target {target:+.2f} m/s all wheels -> measured "
              f"{[round(m, 3) for m in meas]}  worst err {worst:.3f}  {status}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.1.157")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--dry-run", action="store_true",
                        help="characterize but do not persist the final config")
    args = parser.parse_args()

    status = http_status(args.host)
    drive = status["drive"]
    if not drive.get("acked_by_follower"):
        sys.exit("follower has not acked drive params -- flash both boards first")
    polarity = drive["motor_polarity"]
    print(f"robot ok, polarity {polarity}, starting drive version {drive['version']}")
    print("WHEELS UP? starting in 4 s, ctrl-c aborts.")
    time.sleep(4)

    baseline_max = 1.0
    print("baseline: deadband 0, open loop, max=1.0 (command == percent)")
    http_config(args.host, {"motor_deadband_pct": [0, 0, 0], "closed_loop": 0,
                            "pid_kp": 0, "pid_ki": 0,
                            "max_wheel_surface_speed_mps": baseline_max})

    rig = Rig(args.connect, args.namespace)
    try:
        time.sleep(1.5)
        deadbands = phase_a_breakaway(rig, baseline_max)
        http_config(args.host, {"motor_deadband_pct": deadbands})
        true_max = phase_b_speed_sweep(rig, baseline_max)
        http_config(args.host, {"max_wheel_surface_speed_mps": round(true_max, 2)})
        tau = phase_c_step_response(rig, true_max)
        kp, ki = compute_gains(true_max, tau)
        print(f"\n=== gains: kp={kp} %/(m/s), ki={ki} %/(m/s*s) ===")
        http_config(args.host, {"pid_kp": kp, "pid_ki": ki, "closed_loop": 1})
        verified = phase_d_verify(rig, true_max)

        final = http_status(args.host)["drive"]
        print("\nFinal drive config:")
        print(json.dumps(final, indent=2))
        if args.dry_run:
            print("\n--dry-run: reverting closed loop off "
                  "(params remain persisted on the robot until changed)")
            http_config(args.host, {"closed_loop": 0})
        print("\nDone." if verified else
              "\nDone, but tracking errors were large -- rerun or inspect.")
    finally:
        rig.close()


if __name__ == "__main__":
    main()
