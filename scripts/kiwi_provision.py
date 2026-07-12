#!/usr/bin/env python3
"""Provision the kiwi master over its soft-AP -- no reflash, no USB.

Run this from your NORMAL WiFi network (the one the robot should join). The
script auto-detects your laptop IP and current SSID, switches this Mac onto
the KIWI-MASTER AP, uploads the config, waits for the robot to join your
network, prints the robot's new IP, and switches your WiFi back.

One command on a new network:
  python3 scripts/kiwi_provision.py --password <wifi-password>
(--ssid defaults to the network you are currently on, --pc-ip to your
current IP.)

Calibration once the robot is on your network (no AP dance, applies live):
  python3 scripts/kiwi_provision.py --host <robot-ip> --wheel-radius 0.024
  python3 scripts/kiwi_provision.py --host <robot-ip> --status

The master serves HTTP on http://192.168.4.1/ from the KIWI-MASTER AP
(password: seeedstudio) and on its station IP once connected. Settings
persist in NVS on the robot; drive parameters are forwarded to the follower
over UART and persist there too.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

AP_SSID = "KIWI-MASTER"
AP_PASSWORD = "seeedstudio"
AP_HOST = "192.168.4.1"
AP_SUBNET_PREFIX = "192.168.4."


def run(cmd, timeout=15):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def http_json(method, url, body=None, timeout=5):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wifi_device():
    out = run(["networksetup", "-listallhardwareports"]).stdout
    port = None
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and port in ("Wi-Fi", "AirPort"):
            return line.split(":", 1)[1].strip()
    return "en0"


def current_ssid(device):
    out = run(["networksetup", "-getairportnetwork", device]).stdout
    if ":" in out and "not associated" not in out.lower():
        ssid = out.split(":", 1)[1].strip()
        if ssid:
            return ssid
    # networksetup is unreliable on recent macOS; ipconfig still knows.
    out = run(["ipconfig", "getsummary", device]).stdout
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("SSID") and ":" in stripped:
            ssid = stripped.split(":", 1)[1].strip()
            if ssid and ssid != "<redacted>":
                return ssid
    return None


def interface_ip(device):
    ip = run(["ipconfig", "getifaddr", device]).stdout.strip()
    return ip or None


def join_wifi(device, ssid, password, timeout_s=30):
    print(f"Switching {device} to '{ssid}'...")
    cmd = ["networksetup", "-setairportnetwork", device, ssid]
    if password:
        cmd.append(password)
    result = run(cmd, timeout=timeout_s)
    failure = (result.stdout + result.stderr).strip()
    if "Failed" in failure or "Error" in failure:
        return False, failure
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if interface_ip(device):
            return True, None
        time.sleep(1)
    return False, "no IP address acquired"


def wait_for_ap_status(timeout_s, want_sta=False):
    """Poll /status over the AP; optionally hold out for sta_connected."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            last = http_json("GET", f"http://{AP_HOST}/status", timeout=3)
            if not want_sta or last.get("sta_connected"):
                return last
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return last


def build_config(args):
    config = {}
    if args.ssid is not None:
        config["wifi_ssid"] = args.ssid
    if args.password is not None:
        config["wifi_password"] = args.password
    if args.zenoh_connect is not None:
        config["zenoh_connect"] = args.zenoh_connect
    elif args.pc_ip is not None:
        # UDP: zenoh-pico's TCP transport on the ESP32 starves under load
        # (silently drops most payloads >47 B); UDP streams at full rate.
        config["zenoh_connect"] = f"udp/{args.pc_ip}:{args.zenoh_port}"
    if args.zenoh_mode is not None:
        config["zenoh_mode"] = args.zenoh_mode
    elif "zenoh_connect" in config:
        # Connecting out to a zenohd router implies client mode.
        config["zenoh_mode"] = "client"
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
    if args.encoder_polarity is not None:
        enc = [int(p) for p in args.encoder_polarity.split(",")]
        if len(enc) != 3 or any(p not in (-1, 1) for p in enc):
            sys.exit("--encoder-polarity must be three comma-separated values of 1 or -1")
        config["encoder_polarity"] = enc
    if args.motor_deadband is not None:
        deadband = [int(d) for d in args.motor_deadband.split(",")]
        if len(deadband) != 3 or any(not 0 <= d <= 90 for d in deadband):
            sys.exit("--motor-deadband must be three comma-separated percents (0-90)")
        config["motor_deadband_pct"] = deadband
    if args.pid_kp is not None:
        config["pid_kp"] = args.pid_kp
    if args.pid_ki is not None:
        config["pid_ki"] = args.pid_ki
    if args.closed_loop is not None:
        config["closed_loop"] = args.closed_loop
    return config


def post_config(host, config):
    redacted = {k: ("***" if k == "wifi_password" else v) for k, v in config.items()}
    print(f"POST http://{host}/config: {json.dumps(redacted)}")
    result = http_json("POST", f"http://{host}/config", config, timeout=8)
    print(f"Robot response: {json.dumps(result)}")
    if not result.get("ok"):
        sys.exit(1)
    return result


def print_status(status):
    if status is None:
        print("No status available.")
        return
    print(json.dumps(status, indent=2))
    if status.get("sta_connected"):
        sta_ip = status.get("sta_ip")
        print(f"\nRobot is on '{status.get('wifi_ssid')}' at {sta_ip}")
        print(f"Reprovision/calibrate without the AP: --host {sta_ip}")
    else:
        print("\nRobot STA is NOT connected; it is reachable only on the AP.")
    if status.get("zenoh_connect"):
        port = status["zenoh_connect"].rsplit(":", 1)[-1]
        print(f"Run the router on your laptop: "
              f"zenohd --listen udp/0.0.0.0:{port} --listen tcp/0.0.0.0:{port}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host",
                        help="robot address; skips the WiFi dance and talks directly "
                        "(use the robot's STA IP, or 192.168.4.1 if already on the AP)")
    parser.add_argument("--status", action="store_true",
                        help="print robot status and exit (requires --host or the AP)")
    parser.add_argument("--ssid", help="WiFi the robot should join "
                        "(default: the network this Mac is on right now)")
    parser.add_argument("--password", help="WiFi password for --ssid")
    parser.add_argument("--pc-ip", help="this laptop's IP on the target network "
                        "(default: auto-detected before switching to the AP)")
    parser.add_argument("--zenoh-port", type=int, default=7447)
    parser.add_argument("--zenoh-connect",
                        help="full zenoh locator, overrides --pc-ip (e.g. tcp/192.168.8.42:7447)")
    parser.add_argument("--zenoh-mode", choices=["client", "peer"])
    parser.add_argument("--wheel-radius", type=float, help="wheel radius in meters")
    parser.add_argument("--base-radius", type=float, help="drive base radius in meters")
    parser.add_argument("--max-speed", type=float, help="max wheel surface speed in m/s")
    parser.add_argument("--cmd-timeout-ms", type=int, help="velocity command timeout")
    parser.add_argument("--motor-polarity", help="three comma-separated 1/-1, e.g. '1,-1,1'")
    parser.add_argument("--encoder-polarity",
                        help="three comma-separated 1/-1; flip together with the matching "
                        "motor_polarity entry when a wheel is physically reversed")
    parser.add_argument("--motor-deadband",
                        help="three comma-separated breakaway percents, e.g. '32,28,45'")
    parser.add_argument("--pid-kp", type=float, help="wheel PI proportional gain, %% per m/s")
    parser.add_argument("--pid-ki", type=float, help="wheel PI integral gain, %% per m/s*s")
    parser.add_argument("--closed-loop", type=int, choices=[0, 1],
                        help="1 = feedforward + PI on wheel speed, 0 = feedforward only")
    args = parser.parse_args()

    # Direct mode: talk to the given host, no WiFi switching.
    if args.host:
        if args.status:
            print_status(http_json("GET", f"http://{args.host}/status"))
            return
        config = build_config(args)
        if not config:
            parser.error("nothing to configure; pass --status or at least one setting")
        result = post_config(args.host, config)
        if result.get("reboot"):
            print("Robot is rebooting to apply network settings.")
        else:
            print_status(http_json("GET", f"http://{args.host}/status"))
        return

    device = wifi_device()
    ip = interface_ip(device)
    ssid = current_ssid(device)
    on_ap = ssid == AP_SSID or (ip or "").startswith(AP_SUBNET_PREFIX)

    if args.status:
        if not on_ap:
            ok, err = join_wifi(device, AP_SSID, AP_PASSWORD)
            if not ok:
                sys.exit(f"Could not join {AP_SSID}: {err}")
        try:
            print_status(wait_for_ap_status(20))
        finally:
            if not on_ap and ssid:
                join_wifi(device, ssid, None)
        return

    if on_ap:
        sys.exit(
            f"You are on {AP_SSID}. Run this from the network the robot should join\n"
            "so your laptop IP can be detected, or pass --host 192.168.4.1 with\n"
            "explicit --ssid/--password/--pc-ip values."
        )

    # Fill in defaults from the network we are on right now.
    if args.ssid is None:
        if args.password is None and not build_config(args):
            parser.error("nothing to configure; pass --status or at least one setting")
        if args.password is not None:
            if ssid is None:
                sys.exit("Could not detect the current SSID; pass --ssid explicitly.")
            args.ssid = ssid
            print(f"Using current network as target: '{ssid}'")
    elif args.password is None:
        sys.exit("--ssid requires --password (use --password '' for an open network)")

    if args.pc_ip is None and args.zenoh_connect is None:
        if ip is None:
            sys.exit("Could not detect this Mac's IP; pass --pc-ip explicitly.")
        args.pc_ip = ip
        print(f"Using laptop IP {ip}: zenoh_connect=tcp/{ip}:{args.zenoh_port}")

    config = build_config(args)
    if not config:
        parser.error("nothing to configure; pass --status or at least one setting")

    ok, err = join_wifi(device, AP_SSID, AP_PASSWORD)
    if not ok:
        sys.exit(f"Could not join {AP_SSID}: {err}\nIs the robot powered on?")

    status = None
    try:
        result = post_config(AP_HOST, config)
        if result.get("reboot"):
            print("Robot is rebooting to join the network; waiting on the AP...")
            time.sleep(8)
            # The AP drops during reboot; macOS may wander off it. Re-join.
            join_wifi(device, AP_SSID, AP_PASSWORD)
            status = wait_for_ap_status(60, want_sta=True)
        else:
            status = wait_for_ap_status(15)
    finally:
        if ssid:
            print(f"Switching {device} back to '{ssid}'...")
            join_wifi(device, ssid, None)

    print()
    print_status(status)
    if status and not status.get("sta_connected"):
        print("\nThe robot did not report joining the network -- check the SSID/password\n"
              "and re-run, or watch its USB serial output.")
        sys.exit(1)

    # Final proof from the target network: reach the robot on its STA IP.
    sta_ip = status.get("sta_ip") if status else None
    if sta_ip:
        for _ in range(10):
            try:
                http_json("GET", f"http://{sta_ip}/status", timeout=3)
                print(f"Verified: robot reachable at http://{sta_ip}/ from this network.")
                return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        print(f"Robot claims {sta_ip} but is not reachable from here yet; "
              "give it a moment or check that both are on the same network.")


if __name__ == "__main__":
    main()
