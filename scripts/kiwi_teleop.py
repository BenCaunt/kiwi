#!/usr/bin/env python3
"""Drive the kiwi robot over Zenoh: keyboard teleop or automated axis test.

Prerequisites: zenohd running on this machine
(`zenohd --listen udp/0.0.0.0:7447 --listen tcp/0.0.0.0:7447`) and the robot
provisioned to connect to it (see scripts/kiwi_provision.py).

Teleop (default):
  python3 scripts/kiwi_teleop.py
  w/s: +/-vx (forward/back)   a/d: +/-vy (left/right)
  q/e: +/-omega (spin CCW/CW) space: stop   +/-: speed   ctrl-c: quit

Gamepad (proportional, needs pygame):
  python3 scripts/kiwi_teleop.py --gamepad
  left stick: translate   right stick X: rotate   ctrl-c: quit
  --speed/--omega set full-deflection rates; --ax-vx/--ax-vy/--ax-om remap axes.

Axis test -- put the robot WHEELS UP first:
  python3 scripts/kiwi_teleop.py --test
  Drives +vx, +vy, +omega in turn while printing the encoder-measured twist.
  A healthy axis reports the same sign as commanded. If a wheel spins the
  wrong way, flip its entry with:
  scripts/kiwi_provision.py --host <robot-ip> --motor-polarity 1,-1,1
"""

import argparse
import json
import select
import sys
import termios
import time
import tty

import zenoh

CMD_PERIOD_S = 0.1  # robot failsafe stops motors 250 ms after the last command


class Link:
    def __init__(self, connect, namespace):
        conf = zenoh.Config()
        conf.insert_json5("mode", '"client"')
        conf.insert_json5("connect/endpoints", json.dumps([connect]))
        self.session = zenoh.open(conf)
        self.pub = self.session.declare_publisher(f"{namespace}/cmd_vel")
        self.measured = None
        self.sub = self.session.declare_subscriber(
            f"{namespace}/odom/twist", self._on_twist)

    def _on_twist(self, sample):
        try:
            self.measured = json.loads(bytes(sample.payload).decode())
        except (ValueError, UnicodeDecodeError):
            pass

    def send(self, vx, vy, omega):
        self.pub.put(json.dumps({"vx": vx, "vy": vy, "omega": omega}))

    def close(self):
        self.send(0.0, 0.0, 0.0)
        time.sleep(0.1)
        self.session.close()


def axis_test(link, speed, omega):
    print("AXIS TEST -- robot should be wheels-up. Starting in 3 s, ctrl-c aborts.")
    time.sleep(3)
    axes = [
        ("+vx (forward)", (speed, 0.0, 0.0), "vx"),
        ("+vy (left)", (0.0, speed, 0.0), "vy"),
        ("+omega (CCW)", (0.0, 0.0, omega), "omega"),
    ]
    results = []
    for name, (vx, vy, om), field in axes:
        print(f"\n--- commanding {name} for 3 s ---")
        readings = []
        end = time.time() + 3.0
        while time.time() < end:
            link.send(vx, vy, om)
            time.sleep(CMD_PERIOD_S)
            if link.measured is not None:
                readings.append(link.measured)
        link.send(0.0, 0.0, 0.0)
        if readings:
            last = readings[-1]
            value = last.get("measured", {}).get(field, 0.0)
            commanded = vx or vy or om
            ok = value * commanded > 0
            results.append((name, commanded, value, ok))
            print(f"  commanded={commanded:+.2f}  measured {field}={value:+.3f}  "
                  f"{'SIGN OK' if ok else 'WRONG SIGN'}")
            print(f"  wheels m/s: {last.get('wheel_speed_mps', '?')}")
        else:
            results.append((name, 0, 0, False))
            print("  no odometry received -- check zenohd and robot status")
        print("  pausing 2 s...")
        time.sleep(2)

    print("\nSummary:")
    for name, commanded, value, ok in results:
        print(f"  {name:16s} {'OK' if ok else 'CHECK'}  "
              f"(commanded {commanded:+.2f}, measured {value:+.3f})")
    print("\nIf an axis shows WRONG SIGN or individual wheels fight each other,")
    print("adjust polarity at runtime, e.g.:")
    print("  python3 scripts/kiwi_provision.py --host <robot-ip> --motor-polarity 1,-1,1")


def teleop(link, speed, omega):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    vx = vy = om = 0.0
    print(__doc__.split("Teleop (default):")[1].split("Axis test")[0])
    print(f"speed={speed:.2f} m/s, omega={omega:.2f} rad/s. Driving...")
    try:
        tty.setcbreak(fd)
        last_print = 0.0
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "w":
                    vx, vy, om = speed, 0.0, 0.0
                elif key == "s":
                    vx, vy, om = -speed, 0.0, 0.0
                elif key == "a":
                    vx, vy, om = 0.0, speed, 0.0
                elif key == "d":
                    vx, vy, om = 0.0, -speed, 0.0
                elif key == "q":
                    vx, vy, om = 0.0, 0.0, omega
                elif key == "e":
                    vx, vy, om = 0.0, 0.0, -omega
                elif key == " ":
                    vx = vy = om = 0.0
                elif key == "+":
                    speed = min(speed + 0.05, 1.0)
                    omega = min(omega + 0.25, 6.0)
                elif key == "-":
                    speed = max(speed - 0.05, 0.05)
                    omega = max(omega - 0.25, 0.25)
            link.send(vx, vy, om)
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                m = (link.measured or {}).get("measured", {})
                print(f"\rcmd vx={vx:+.2f} vy={vy:+.2f} om={om:+.2f} | "
                      f"meas vx={m.get('vx', 0):+.2f} "
                      f"vy={m.get('vy', 0):+.2f} "
                      f"om={m.get('omega', 0):+.2f} "
                      f"(speed {speed:.2f})   ", end="", flush=True)
            time.sleep(CMD_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\nstopping")


GAMEPAD_DEADZONE = 0.12


def _deadzone(value):
    if abs(value) < GAMEPAD_DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * min((abs(value) - GAMEPAD_DEADZONE) / (1.0 - GAMEPAD_DEADZONE), 1.0)


def gamepad_teleop(link, speed, omega, ax_vx, ax_vy, ax_om):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("no gamepad detected (pair/plug it in, then rerun)")
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"gamepad: {pad.get_name()} ({pad.get_numaxes()} axes)")
    print(f"full deflection = {speed:.2f} m/s translate, {omega:.2f} rad/s rotate. ctrl-c to quit.")

    def axis(i):
        return _deadzone(float(pad.get_axis(i))) if i < pad.get_numaxes() else 0.0

    last_print = 0.0
    try:
        while True:
            pygame.event.pump()
            x = axis(ax_vy)   # left stick X: + is right
            y = axis(ax_vx)   # left stick Y: + is down
            r = axis(ax_om)   # right stick X: + is right
            mag = (x * x + y * y) ** 0.5
            if mag > 1.0:
                x /= mag
                y /= mag
            vx = -y * speed       # stick up = forward
            vy = -x * speed       # stick left = +vy (robot left)
            om = -r * omega       # stick right = clockwise = -omega
            link.send(vx, vy, om)
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                m = (link.measured or {}).get("measured", {})
                print(f"\rcmd vx={vx:+.2f} vy={vy:+.2f} om={om:+.2f} | "
                      f"meas vx={m.get('vx', 0):+.2f} vy={m.get('vy', 0):+.2f} "
                      f"om={m.get('omega', 0):+.2f}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447",
                        help="zenoh router endpoint (default local zenohd)")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--speed", type=float, default=0.15,
                        help="linear speed in m/s (default 0.15)")
    parser.add_argument("--omega", type=float, default=1.0,
                        help="angular speed in rad/s (default 1.0)")
    parser.add_argument("--test", action="store_true",
                        help="run the automated axis/polarity test instead of teleop")
    parser.add_argument("--gamepad", action="store_true",
                        help="drive with a gamepad (left stick translate, right stick X rotate)")
    parser.add_argument("--ax-vx", type=int, default=1, help="gamepad axis for forward/back")
    parser.add_argument("--ax-vy", type=int, default=0, help="gamepad axis for strafe")
    parser.add_argument("--ax-om", type=int, default=2, help="gamepad axis for rotation")
    args = parser.parse_args()

    link = Link(args.connect, args.namespace)
    try:
        if args.test:
            axis_test(link, args.speed, args.omega)
        elif args.gamepad:
            gamepad_teleop(link, args.speed, args.omega,
                           args.ax_vx, args.ax_vy, args.ax_om)
        else:
            teleop(link, args.speed, args.omega)
    finally:
        link.close()


if __name__ == "__main__":
    main()
