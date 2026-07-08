#!/usr/bin/env python3
"""Provision the kiwi master over its soft-AP -- no reflash, no USB.

The master always serves HTTP on http://192.168.4.1/ from the KIWI-MASTER
AP (password: seeedstudio), and on its station IP once it has joined a
network. Settings are persisted to NVS on the robot; compiled defaults are
first-boot values only. Drive parameters are forwarded to the follower over
UART and persist there too.

Typical travel workflow (new network, new laptop IP):
  1. Note your laptop's IP on the target network (e.g. `ipconfig getifaddr en0`
     while connected to the hotel/home WiFi).
  2. Join the KIWI-MASTER WiFi network.
  3. python3 scripts/kiwi_provision.py --ssid MyNet --password hunter2 --pc-ip 192.168.8.42
  4. Rejoin your normal network; the script printed the robot's new IP.
  5. zenohd --listen tcp/0.0.0.0:7447

Calibration workflow (robot already on your network):
  python3 scripts/kiwi_provision.py --host <robot-sta-ip> --wheel-radius 0.024

Status check:
  python3 scripts/kiwi_provision.py --status
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "192.168.4.1"
AP_SUBNET_PREFIX = "192.168.4."


def http_json(method, url, body=None, timeout=5):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def detect_lan_ip():
    """Best-effort detection of this machine's LAN IP (macOS-first)."""
    try:
        route = subprocess.run(
            ["route", "-n", "get", "default"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        iface = next(
            (line.split(":", 1)[1].strip() for line in route.splitlines()
             if "interface:" in line),
            None,
        )
        candidates = [iface] if iface else []
    except (OSError, subprocess.SubprocessError):
        candidates = []
    candidates += ["en0", "en1"]

    for candidate in candidates:
        try:
            ip = subprocess.run(
                ["ipconfig", "getifaddr", candidate],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if ip:
            return ip
    return None


def build_config(args):
    config = {}
    if args.ssid is not None:
        config["wifi_ssid"] = args.ssid
    if args.password is not None:
        config["wifi_password"] = args.password
    if args.zenoh_connect is not None:
        config["zenoh_connect"] = args.zenoh_connect
    elif args.pc_ip is not None:
        config["zenoh_connect"] = f"tcp/{args.pc_ip}:{args.zenoh_port}"
    if args.zenoh_mode is not None:
        config["zenoh_mode"] = args.zenoh_mode
    if args.wheel_radius is not None:
        config["wheel_radius_m"] = args.wheel_radius
    if args.base_radius is not None:
        config["drive_base_radius_m"] = args.base_radius
    if args.max_speed is not None:
        config["max_wheel_surface_speed_mps"] = args.max_speed
    if args.cmd_timeout_ms is not None:
        config["velocity_command_timeout_ms"] = args.cmd_timeout_ms
    if args.motor_polarity is not None:
        polarity = [int(p) for p in args.motor_polarity.split(",")]
        if len(polarity) != 3 or any(p not in (-1, 1) for p in polarity):
            sys.exit("--motor-polarity must be three comma-separated values of 1 or -1")
        config["motor_polarity"] = polarity
    return config


def print_status(status):
    print(json.dumps(status, indent=2))
    if status.get("sta_connected"):
        sta_ip = status.get("sta_ip")
        print(f"\nRobot is on '{status.get('wifi_ssid')}' at {sta_ip}")
        print(f"Reprovision later without the AP: --host {sta_ip}")
    else:
        print("\nRobot STA is NOT connected; it is reachable only on the AP.")
    if status.get("zenoh_connect"):
        print(f"Run the router on your laptop: zenohd --listen tcp/0.0.0.0:"
              f"{status['zenoh_connect'].rsplit(':', 1)[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"robot address (default {DEFAULT_HOST} via the KIWI-MASTER AP)")
    parser.add_argument("--status", action="store_true",
                        help="print robot status and exit")
    parser.add_argument("--ssid", help="WiFi network the robot should join")
    parser.add_argument("--password", help="WiFi password")
    parser.add_argument("--pc-ip", help="this laptop's IP on the target network "
                        "(builds zenoh_connect; auto-detected only when not on the robot AP)")
    parser.add_argument("--zenoh-port", type=int, default=7447)
    parser.add_argument("--zenoh-connect",
                        help="full zenoh locator, overrides --pc-ip (e.g. tcp/192.168.8.42:7447)")
    parser.add_argument("--zenoh-mode", choices=["client", "peer"])
    parser.add_argument("--wheel-radius", type=float, help="wheel radius in meters")
    parser.add_argument("--base-radius", type=float, help="drive base radius in meters")
    parser.add_argument("--max-speed", type=float, help="max wheel surface speed in m/s")
    parser.add_argument("--cmd-timeout-ms", type=int, help="velocity command timeout")
    parser.add_argument("--motor-polarity", help="three comma-separated 1/-1, e.g. '1,-1,1'")
    args = parser.parse_args()

    base = f"http://{args.host}"

    if args.status:
        print_status(http_json("GET", f"{base}/status"))
        return

    if args.ssid is not None and args.password is None:
        sys.exit("--ssid requires --password (use --password '' for an open network)")

    # A new network almost always means a new laptop IP: if WiFi is being set
    # and no zenoh endpoint was given, derive it, but never from the robot's
    # own AP subnet -- that address is useless on the target network.
    if args.ssid is not None and args.pc_ip is None and args.zenoh_connect is None:
        detected = detect_lan_ip()
        if detected is None or detected.startswith(AP_SUBNET_PREFIX):
            sys.exit(
                "Cannot auto-detect your IP on the target network (you are on the robot AP).\n"
                "Check it while connected to that network (ipconfig getifaddr en0), then\n"
                "re-run with --pc-ip <ip>, or pass --zenoh-connect explicitly."
            )
        args.pc_ip = detected
        print(f"Auto-detected laptop IP {detected}; zenoh_connect=tcp/{detected}:{args.zenoh_port}")

    config = build_config(args)
    if not config:
        parser.error("nothing to configure; pass --status or at least one setting")

    print(f"POST {base}/config: "
          f"{json.dumps({k: ('***' if k == 'wifi_password' else v) for k, v in config.items()})}")
    result = http_json("POST", f"{base}/config", config)
    print(f"Robot response: {json.dumps(result)}")
    if not result.get("ok"):
        sys.exit(1)

    if not result.get("reboot"):
        print_status(http_json("GET", f"{base}/status"))
        return

    print("Robot is rebooting to join the network; polling for it to come back...")
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(3)
        try:
            status = http_json("GET", f"{base}/status", timeout=3)
        except (urllib.error.URLError, OSError):
            continue
        if status.get("sta_connected") or time.time() > deadline - 15:
            print_status(status)
            return
    print("Robot did not come back within 60 s. If your laptop dropped off the AP,\n"
          "rejoin KIWI-MASTER and run --status, or find the robot on the target network.")
    sys.exit(1)


if __name__ == "__main__":
    main()
