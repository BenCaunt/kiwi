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
  Start/Menu (button 6): toggle teleop armed / agent control
  --speed/--omega set full-deflection rates; --ax-vx/--ax-vy/--ax-om remap axes.

Axis test -- put the robot WHEELS UP first:
  python3 scripts/kiwi_teleop.py --test
  Drives +vx, +vy, +omega in turn while printing the encoder-measured twist.
  A healthy axis reports the same sign as commanded. If a wheel spins the
  wrong way, flip its entry with:
  scripts/kiwi_provision.py --host <robot-ip> --motor-polarity 1,-1,1
"""

import argparse
import select
import sys
import termios
import time
import tty

from kiwi_client import DEFAULT_ROBOT_YAW_DEG, KiwiClient

CMD_PERIOD_S = 0.1  # robot failsafe stops motors 250 ms after the last command


def send_command(link, vx, vy, omega, active=True):
    """Publish a direct command or a mux lease, depending on the topic."""
    muxed = link.command_suffix != "cmd_vel"
    link.send_twist(vx, vy, omega, active=active if muxed else None)


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
            send_command(link, vx, vy, om)
            time.sleep(CMD_PERIOD_S)
            if link.odometry is not None:
                readings.append(link.odometry)
        send_command(link, 0.0, 0.0, 0.0, active=False)
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
    active = False
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
                    active = True
                elif key == "s":
                    vx, vy, om = -speed, 0.0, 0.0
                    active = True
                elif key == "a":
                    vx, vy, om = 0.0, speed, 0.0
                    active = True
                elif key == "d":
                    vx, vy, om = 0.0, -speed, 0.0
                    active = True
                elif key == "q":
                    vx, vy, om = 0.0, 0.0, omega
                    active = True
                elif key == "e":
                    vx, vy, om = 0.0, 0.0, -omega
                    active = True
                elif key == " ":
                    vx = vy = om = 0.0
                    active = False
                elif key == "+":
                    speed = min(speed + 0.05, 1.0)
                    omega = min(omega + 0.25, 6.0)
                elif key == "-":
                    speed = max(speed - 0.05, 0.05)
                    omega = max(omega - 0.25, 0.25)
            send_command(link, vx, vy, om, active=active)
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                m = (link.odometry or {}).get("measured", {})
                print(f"\rcmd vx={vx:+.2f} vy={vy:+.2f} om={om:+.2f} | "
                      f"meas vx={m.get('vx', 0):+.2f} "
                      f"vy={m.get('vy', 0):+.2f} "
                      f"om={m.get('omega', 0):+.2f} "
                      f"(speed {speed:.2f})   ", end="", flush=True)
            time.sleep(CMD_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        send_command(link, 0.0, 0.0, 0.0, active=False)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\nstopping")


GAMEPAD_DEADZONE = 0.08
GAMEPAD_PERIOD_S = 0.05


def _deadzone(value, deadzone=GAMEPAD_DEADZONE):
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * min((abs(value) - deadzone) / (1.0 - deadzone), 1.0)


class GamepadHandoff:
    """Edge-triggered teleop/agent ownership toggle."""

    def __init__(self):
        self.teleop_enabled = True
        self._button_down = False

    def update(self, button_down):
        button_down = bool(button_down)
        toggled = button_down and not self._button_down
        if toggled:
            self.teleop_enabled = not self.teleop_enabled
        self._button_down = button_down
        return toggled

    def reclaim_for_stick_input(self, stick_active):
        """Guarantee that deliberate stick motion always restores teleop."""
        if stick_active and not self.teleop_enabled:
            self.teleop_enabled = True
            return True
        return False


def gamepad_teleop(
    link, speed, omega, ax_vx, ax_vy, ax_om,
    deadzone=GAMEPAD_DEADZONE, handoff_button=6,
):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.joystick.init()
    handoff = GamepadHandoff()
    pad = None
    pad_instance_id = None
    handoff_available = False

    def attach(device_index=0):
        nonlocal pad, pad_instance_id, handoff_available
        candidate = pygame.joystick.Joystick(device_index)
        candidate.init()
        pad = candidate
        pad_instance_id = pad.get_instance_id()
        handoff_available = 0 <= handoff_button < pad.get_numbuttons()
        print(
            f"\ngamepad connected: {pad.get_name()} "
            f"({pad.get_numaxes()} axes, instance {pad_instance_id})"
        )
        print(
            f"full deflection = {speed:.2f} m/s translate, "
            f"{omega:.2f} rad/s rotate; deadzone={deadzone:.0%}."
        )
        if handoff_available:
            print(
                f"Start/Menu button {handoff_button}: toggle TELEOP / AGENT "
                "control; centered sticks release the mux and moving a stick "
                "always reclaims TELEOP. ctrl-c quits."
            )
        else:
            print(
                "No handoff button is available; centered sticks release the "
                "mux. ctrl-c quits."
            )

    if pygame.joystick.get_count() == 0:
        sys.exit("no gamepad detected (pair/plug it in, then rerun)")
    attach()

    last_print = 0.0
    try:
        while True:
            for event in pygame.event.get():
                if (event.type == pygame.JOYDEVICEREMOVED and pad is not None and
                        getattr(event, "instance_id", None) ==
                        pad_instance_id):
                    send_command(link, 0.0, 0.0, 0.0, active=False)
                    print("\ngamepad disconnected; teleop released")
                    pad.quit()
                    pad = None
                    pad_instance_id = None
                    handoff_available = False
                elif event.type == pygame.JOYDEVICEADDED and pad is None:
                    attach(getattr(event, "device_index", 0))
            if pad is None:
                # Some SDL backends do not deliver a hot-plug event. Polling
                # provides a safe fallback while continuing to publish release.
                send_command(link, 0.0, 0.0, 0.0, active=False)
                if pygame.joystick.get_count() > 0:
                    attach()
                else:
                    time.sleep(0.10)
                    continue

            button_down = (
                bool(pad.get_button(handoff_button))
                if handoff_available else False
            )
            if handoff.update(button_down):
                mode = "TELEOP" if handoff.teleop_enabled else "AGENT"
                send_command(link, 0.0, 0.0, 0.0, active=False)
                print(f"\ncontrol handed to {mode}")
            def axis(index):
                return (_deadzone(float(pad.get_axis(index)), deadzone)
                        if index < pad.get_numaxes() else 0.0)

            x = axis(ax_vy)   # left stick X: + is right
            y = axis(ax_vx)   # left stick Y: + is down
            r = axis(ax_om)   # right stick X: + is right
            mag = (x * x + y * y) ** 0.5
            if mag > 1.0:
                x /= mag
                y /= mag
            stick_active = any(abs(value) > 0.0 for value in (x, y, r))
            if handoff.reclaim_for_stick_input(stick_active):
                print("\nstick input reclaimed TELEOP control")
            if handoff.teleop_enabled:
                vx = -y * speed       # stick up = forward
                vy = -x * speed       # stick left = +vy (robot left)
                om = -r * omega       # stick right = clockwise = -omega
                active = stick_active
            else:
                vx = vy = om = 0.0
                active = False
            link.send_twist(
                vx, vy, om,
                active=active if link.command_suffix != "cmd_vel" else None,
                hold_s=GAMEPAD_PERIOD_S,
            )
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                m = (link.odometry or {}).get("measured", {})
                mode = "TELEOP" if handoff.teleop_enabled else "AGENT "
                print(f"\r{mode} | cmd vx={vx:+.2f} vy={vy:+.2f} om={om:+.2f} | "
                      f"meas vx={m.get('vx', 0):+.2f} vy={m.get('vy', 0):+.2f} "
                      f"om={m.get('omega', 0):+.2f}   ", end="", flush=True)
            time.sleep(GAMEPAD_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        send_command(link, 0.0, 0.0, 0.0, active=False)
        if pad is not None:
            pad.quit()
        pygame.joystick.quit()
        pygame.quit()
        print("\nstopping")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447",
                        help="zenoh router endpoint (default local zenohd)")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument(
        "--command-topic", default="cmd_vel",
        help=("namespaced command topic suffix; launch.py uses cmd_vel/teleop "
              "behind the command mux"))
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
    parser.add_argument(
        "--gamepad-deadzone", type=float, default=GAMEPAD_DEADZONE,
        help=f"centered-stick deadzone fraction (default {GAMEPAD_DEADZONE:g})")
    parser.add_argument(
        "--handoff-button", type=int, default=6,
        help="gamepad button that toggles teleop/agent control; -1 disables")
    parser.add_argument(
        "--robot-yaw-deg", type=float, default=DEFAULT_ROBOT_YAW_DEG,
        help=("raw drivetrain +X yaw counter-clockwise from lidar/camera "
              f"forward (default {DEFAULT_ROBOT_YAW_DEG:g} deg)"))
    args = parser.parse_args()
    if not 0.0 <= args.gamepad_deadzone < 1.0:
        parser.error("--gamepad-deadzone must be in [0, 1)")
    if args.handoff_button < -1:
        parser.error("--handoff-button must be -1 or nonnegative")

    link = KiwiClient(args.connect, args.namespace, args.robot_yaw_deg,
                      commanding=True, command_suffix=args.command_topic)
    print(f"frame correction: {link.robot_yaw_deg:+g} deg robot yaw")
    try:
        if args.test:
            axis_test(link, args.speed, args.omega)
        elif args.gamepad:
            gamepad_teleop(link, args.speed, args.omega,
                           args.ax_vx, args.ax_vy, args.ax_om,
                           args.gamepad_deadzone, args.handoff_button)
        else:
            teleop(link, args.speed, args.omega)
    finally:
        link.close()


if __name__ == "__main__":
    main()
