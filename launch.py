#!/usr/bin/env python3
"""Launch and supervise Kiwi's Zenoh, SLAM, dashboard, navigation, and teleop.

With no map arguments, an interactive terminal lists saved maps and offers to
create a new one. Closing this process with Ctrl-C gracefully stops every
managed robot process, lets SLAM save, and publishes a final zero drive
command. A healthy Zenoh router is reused and left running so the physical
robot's UDP client stays connected across mapping sessions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_MAP_DIR = PROJECT_ROOT / "maps"
RUNTIME_SCRIPT_NAMES = {
    "kiwi_command_mux.py",
    "kiwi_dashboard.py",
    "kiwi_image_navigation.py",
    "kiwi_navigation.py",
    "kiwi_sim_bridge.py",
    "kiwi_slam.py",
    "kiwi_teleop.py",
}
ZENOH_LISTENERS = ("udp/0.0.0.0:7447", "tcp/0.0.0.0:7447")


@dataclass(frozen=True)
class MapSelection:
    prefix: Path
    resume: bool


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    isolated_group: bool
    descendant_groups: set[int] = field(default_factory=set)


def normalize_map_prefix(value: str | Path) -> Path:
    text = str(Path(value).expanduser())
    for suffix in (".graph.json", ".slam.npz", ".pgm", ".yaml"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def map_state_exists(prefix: Path) -> bool:
    return Path(f"{prefix}.graph.json").is_file() and \
        Path(f"{prefix}.slam.npz").is_file()


def discover_saved_maps(map_dir: Path = DEFAULT_MAP_DIR) -> list[Path]:
    if not map_dir.is_dir():
        return []
    prefixes = []
    for graph in map_dir.rglob("*.graph.json"):
        prefix = normalize_map_prefix(graph)
        if map_state_exists(prefix):
            prefixes.append(prefix)
    return sorted(set(prefixes), key=lambda path: str(path).lower())


def map_keyframe_count(prefix: Path) -> int | None:
    try:
        document = json.loads(
            Path(f"{prefix}.graph.json").read_text(encoding="utf-8"))
        nodes = document.get("nodes")
        return len(nodes) if isinstance(nodes, list) else None
    except (OSError, json.JSONDecodeError):
        return None


def new_map_prefix(value: str) -> Path:
    value = value.strip()
    if not value:
        value = "kiwi_map_" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
    candidate = Path(value).expanduser()
    if len(candidate.parts) == 1:
        candidate = Path("maps") / candidate
    return normalize_map_prefix(candidate)


def prompt_for_map() -> MapSelection:
    maps = discover_saved_maps()
    print("\nSaved SLAM maps:")
    for index, prefix in enumerate(maps, 1):
        relative = prefix.relative_to(PROJECT_ROOT)
        count = map_keyframe_count(prefix)
        detail = f" ({count} keyframes)" if count is not None else ""
        print(f"  {index}. {relative}{detail}")
    print("  n. Create a new map")

    default = "1" if maps else "n"
    while True:
        answer = input(f"Select [{default}]: ").strip().lower() or default
        if answer in ("n", "new"):
            name = input(
                "New map name/path [timestamped name under maps/]: ").strip()
            prefix = new_map_prefix(name)
            if map_state_exists(prefix) or Path(f"{prefix}.images").exists():
                print(f"{prefix} already contains map data; choose another name.")
                continue
            return MapSelection(prefix, resume=False)
        try:
            index = int(answer) - 1
            if 0 <= index < len(maps):
                return MapSelection(maps[index], resume=True)
        except ValueError:
            pass
        print("Enter a listed number or n.")


def select_map(args) -> MapSelection:
    if args.map:
        prefix = normalize_map_prefix(args.map)
        if not map_state_exists(prefix):
            raise RuntimeError(
                f"saved map needs both {prefix}.graph.json and {prefix}.slam.npz")
        return MapSelection(prefix, resume=True)
    if args.new_map:
        prefix = new_map_prefix(args.new_map)
        if map_state_exists(prefix) or Path(f"{prefix}.images").exists():
            raise RuntimeError(
                f"new map target already contains data: {prefix}; use --map to resume")
        return MapSelection(prefix, resume=False)
    if sys.stdin.isatty():
        return prompt_for_map()
    raise RuntimeError("use --map PREFIX or --new-map PREFIX outside a terminal")


def compatible_manifest(prefix: Path) -> Path | None:
    """Use the same compatibility check as resumed SLAM."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from kiwi_image_map import discover_compatible_image_manifest
        from kiwi_slam_core import PoseGraphSlam

        slam = PoseGraphSlam.load(prefix)
        return discover_compatible_image_manifest(prefix, slam.keyframes)
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def selected_manifest(args, selection: MapSelection) -> Path | None:
    if args.no_image_navigation:
        return None
    if args.image_manifest:
        candidate = Path(args.image_manifest).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_dir():
            candidate = candidate / "manifest.json"
        if not candidate.is_file():
            raise RuntimeError(f"image manifest not found: {candidate}")
        return candidate.resolve()
    return compatible_manifest(selection.prefix) if selection.resume else None


def image_manifests(prefix: Path) -> set[Path]:
    root = Path(f"{prefix}.images")
    return {path.resolve() for path in root.glob("*/manifest.json")
            if path.is_file()}


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def process_has_runtime_script(command: str) -> bool:
    return any(Path(token).name in RUNTIME_SCRIPT_NAMES
               for token in _command_tokens(command))


def existing_runtime_processes() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True,
        check=False)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        if pid != os.getpid() and process_has_runtime_script(command):
            found.append((pid, command))
    return found


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def descendant_process_groups(root_pids: set[int]) -> set[int]:
    """Find child-created groups without ever targeting a root's shell group."""
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid="], capture_output=True, text=True,
        check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(value.isdigit() for value in fields):
            rows.append(tuple(map(int, fields)))
    descendants = set(root_pids)
    changed = True
    while changed:
        changed = False
        for pid, parent, _group in rows:
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    root_groups = {
        group for pid, _parent, group in rows if pid in root_pids
    }
    return {
        group for pid, _parent, group in rows
        if pid in descendants and pid not in root_pids and
        group not in root_groups and group != os.getpgrp()
    }


def process_group_is_running(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_existing_runtime(processes: list[tuple[int, str]], timeout_s=8.0) -> None:
    if not processes:
        return
    descendant_groups = descendant_process_groups({
        pid for pid, _command in processes})
    print("Stopping existing Kiwi runtime gracefully (saving SLAM if needed)...",
          flush=True)
    for pid, _command in processes:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    for group in descendant_groups:
        try:
            os.killpg(group, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_s
    remaining = [pid for pid, _command in processes]
    live_groups = set(descendant_groups)
    while (remaining or live_groups) and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = [pid for pid in remaining if pid_is_running(pid)]
        live_groups = {
            group for group in live_groups if process_group_is_running(group)}
    if not remaining and not live_groups:
        print("Existing Kiwi runtime stopped.", flush=True)
        return

    if remaining:
        print("  graceful shutdown timed out; terminating PID(s) " +
              ", ".join(map(str, remaining)), flush=True)
    if live_groups:
        print("  graceful shutdown timed out; terminating process group(s) " +
              ", ".join(map(str, sorted(live_groups))), flush=True)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for group in live_groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if remaining or live_groups:
        deadline = time.monotonic() + 2.0
        while (remaining or live_groups) and time.monotonic() < deadline:
            time.sleep(0.1)
            remaining = [pid for pid in remaining if pid_is_running(pid)]
            live_groups = {
                group for group in live_groups
                if process_group_is_running(group)}
    if remaining:
        print("  terminate timed out; killing PID(s) " +
              ", ".join(map(str, remaining)), flush=True)
    if live_groups:
        print("  terminate timed out; killing process group(s) " +
              ", ".join(map(str, sorted(live_groups))), flush=True)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for group in live_groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_runtime_preflight(processes: list[tuple[int, str]],
                           timeout_s: float) -> None:
    """Stop old runtime processes and catch late or respawned orphans."""
    pending = processes
    for sweep in range(3):
        stop_existing_runtime(
            pending, timeout_s if sweep == 0 else min(timeout_s, 3.0))
        time.sleep(0.1)
        pending = existing_runtime_processes()
        if not pending:
            print("Kiwi runtime preflight is clear.", flush=True)
            return
        if sweep < 2:
            print("Additional Kiwi runtime process(es) appeared during "
                  "cleanup:", flush=True)
            for pid, command in pending:
                print(f"  PID {pid}: {command}", flush=True)
    detail = "; ".join(f"PID {pid}: {command}"
                       for pid, command in pending)
    raise RuntimeError(
        "could not clear existing Kiwi runtime processes: " + detail)


def loopback_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def loopback_listener_pids(port: int) -> set[int]:
    """Return processes currently listening on one loopback TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True, text=True, check=False)
    except OSError:
        return set()
    return {
        int(line[1:]) for line in result.stdout.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }


def process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True, text=True, check=False)
    return result.stdout.strip()


def reusable_zenoh_router_pid() -> int | None:
    """Return the healthy local Kiwi router without disrupting robot data."""
    candidates = []
    for pid in loopback_listener_pids(7447):
        command = process_command(pid)
        try:
            executable = Path(_command_tokens(command)[0]).name
        except IndexError:
            continue
        if executable == "zenohd" and all(
                listener in command for listener in ZENOH_LISTENERS):
            candidates.append(pid)
    return candidates[0] if len(candidates) == 1 else None


def launchd_labels_for_pids(pids: set[int]) -> set[str]:
    """Find user launchd jobs whose active process is in ``pids``."""
    if not pids:
        return set()
    try:
        result = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True,
            check=False)
    except OSError:
        return set()
    labels = set()
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3 and fields[0].isdigit() and \
                int(fields[0]) in pids:
            labels.add(fields[2])
    return labels


def remove_launchd_jobs(labels: set[str]) -> None:
    """Unregister verified stale user jobs so keepalive cannot respawn them."""
    for label in sorted(labels):
        result = subprocess.run(
            ["launchctl", "remove", label], capture_output=True, text=True,
            check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"could not stop stale launchd job {label}{suffix}")


def preflight_image_port(port: int, label: str, timeout_s: float) -> None:
    """Clear an orphaned Kiwi gallery listener without killing other apps."""
    if loopback_port_available(port):
        return

    owners = loopback_listener_pids(port)
    stale = [
        (pid, command)
        for pid, command in existing_runtime_processes()
        if pid in owners and any(
            Path(token).name == "kiwi_image_navigation.py"
            for token in _command_tokens(command)
        )
    ]
    if stale:
        print(f"Preflight: stopping stale Kiwi image navigation on "
              f"{label} port {port}:")
        for pid, command in stale:
            print(f"  PID {pid}: {command}")
        launchd_labels = launchd_labels_for_pids({pid for pid, _ in stale})
        if launchd_labels:
            print("  removing launchd keepalive job(s): " +
                  ", ".join(sorted(launchd_labels)))
            remove_launchd_jobs(launchd_labels)
        stop_existing_runtime(stale, timeout_s)

    if loopback_port_available(port):
        return

    remaining = loopback_listener_pids(port)
    detail = (f" (listener PID{'s' if len(remaining) != 1 else ''} "
              f"{', '.join(map(str, sorted(remaining)))})"
              if remaining else "")
    raise RuntimeError(f"{label} port {port} is already in use{detail}")


def preflight_image_ports(args) -> None:
    if args.no_image_navigation:
        return
    preflight_image_port(
        args.gallery_port, "gallery", args.shutdown_timeout)
    if not args.no_mcp:
        preflight_image_port(args.mcp_port, "MCP", args.shutdown_timeout)


def wait_for_port(host: str, port: int, process: subprocess.Popen,
                  timeout_s: float, owner_pid: int | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Zenoh exited during startup with code {process.returncode}")
        # start_zenoh.py may be replacing an old router that still owns this
        # port. A successful connection alone would then release the rest of
        # the stack too early; require the newly spawned (and later exec'd)
        # Zenoh process itself to own the listener.
        if owner_pid is not None and \
                owner_pid not in loopback_listener_pids(port):
            time.sleep(0.1)
            continue
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for Zenoh on {host}:{port}")


def _watchdog_target_running(target: int, use_process_group: bool) -> bool:
    return (process_group_is_running(target) if use_process_group
            else pid_is_running(target))


def _watchdog_signal(target: int, use_process_group: bool,
                     sig: signal.Signals) -> None:
    try:
        if use_process_group:
            os.killpg(target, sig)
        else:
            os.kill(target, sig)
    except ProcessLookupError:
        pass


def process_watchdog(parent_pid: int, target: int,
                     use_process_group: bool) -> int:
    """Stop one managed child if the launcher disappears unexpectedly."""
    while os.getppid() == parent_pid and \
            _watchdog_target_running(target, use_process_group):
        time.sleep(0.1)
    if not _watchdog_target_running(target, use_process_group):
        return 0
    for sig, timeout_s in (
            (signal.SIGINT, 8.0),
            (signal.SIGTERM, 2.0),
            (signal.SIGKILL, 0.0)):
        _watchdog_signal(target, use_process_group, sig)
        deadline = time.monotonic() + timeout_s
        while timeout_s and \
                _watchdog_target_running(target, use_process_group) and \
                time.monotonic() < deadline:
            time.sleep(0.1)
        if not _watchdog_target_running(target, use_process_group):
            return 0
    return 1


def ensure_zenoh_router(command: list[str], timeout_s: float):
    """Reuse a healthy router, or start one that persists across launches."""
    existing_pid = reusable_zenoh_router_pid()
    if existing_pid is not None:
        print(f"[zenoh] reusing healthy router PID {existing_pid}; "
              "it will remain running after shutdown", flush=True)
        return None

    print(f"[zenoh] {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command, cwd=PROJECT_ROOT, start_new_session=True)
    wait_for_port(
        "127.0.0.1", 7447, process, timeout_s, owner_pid=process.pid)
    print(f"[zenoh] router PID {process.pid} is ready and will remain running "
          "after shutdown", flush=True)
    return process


class Supervisor:
    def __init__(self, shutdown_timeout_s: float):
        self.shutdown_timeout_s = shutdown_timeout_s
        self.processes: list[ManagedProcess] = []
        self.watchdogs: list[subprocess.Popen] = []
        self._cleaned = False

    def start(self, name: str, command: list[str], *, interactive=False):
        print(f"[{name}] {shlex.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            start_new_session=not interactive,
        )
        self.processes.append(ManagedProcess(
            name=name, process=process, isolated_group=not interactive))
        watchdog = subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--process-watchdog", str(os.getpid()), str(process.pid),
                "group" if not interactive else "pid",
            ],
            cwd=PROJECT_ROOT,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.watchdogs.append(watchdog)
        return process

    def reap_watchdogs(self) -> None:
        self.watchdogs = [watchdog for watchdog in self.watchdogs
                          if watchdog.poll() is None]

    @staticmethod
    def _group_running(process_group: int) -> bool:
        return process_group_is_running(process_group)

    def refresh_descendants(self) -> None:
        """Remember child-created groups such as Rerun's viewer process."""
        self.reap_watchdogs()
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid="], capture_output=True,
            text=True, check=False)
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 3 and all(value.isdigit() for value in fields):
                rows.append(tuple(map(int, fields)))
        for item in self.processes:
            if not item.isolated_group:
                continue
            descendants = {item.process.pid}
            changed = True
            while changed:
                changed = False
                for pid, parent, _group in rows:
                    if parent in descendants and pid not in descendants:
                        descendants.add(pid)
                        changed = True
            item.descendant_groups.update(
                group for pid, _parent, group in rows
                if pid in descendants and group != item.process.pid)

    @staticmethod
    def _running(item: ManagedProcess) -> bool:
        if not item.isolated_group:
            return item.process.poll() is None
        # The group can still contain a viewer or navigator after its direct
        # child has exited, so process lifetime alone is not sufficient.
        groups = {item.process.pid, *item.descendant_groups}
        return any(Supervisor._group_running(group) for group in groups)

    @staticmethod
    def _signal(item: ManagedProcess, sig: signal.Signals) -> None:
        if item.isolated_group:
            for group in {item.process.pid, *item.descendant_groups}:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        elif item.process.poll() is None:
            try:
                item.process.send_signal(sig)
            except ProcessLookupError:
                pass

    @classmethod
    def _wait_stopped(cls, item: ManagedProcess, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while cls._running(item) and time.monotonic() < deadline:
            item.process.poll()
            time.sleep(0.05)
        item.process.poll()
        return not cls._running(item)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if not self.processes:
            return
        self.refresh_descendants()
        print("\nStopping Kiwi runtime...", flush=True)
        for item in reversed(self.processes):
            if not self._running(item):
                continue
            print(f"  stopping {item.name}", flush=True)
            self._signal(item, signal.SIGINT)
            if not self._wait_stopped(item, self.shutdown_timeout_s):
                print(f"  {item.name} did not stop; terminating", flush=True)
                self._signal(item, signal.SIGTERM)
                if not self._wait_stopped(item, 2.0):
                    print(f"  {item.name} did not terminate; killing", flush=True)
                    self._signal(item, signal.SIGKILL)
                    self._wait_stopped(item, 2.0)
            if item.process.poll() is None:
                try:
                    item.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        deadline = time.monotonic() + 1.0
        while self.watchdogs and time.monotonic() < deadline:
            self.reap_watchdogs()
            if self.watchdogs:
                time.sleep(0.05)
        print("Kiwi runtime stopped.", flush=True)


def common_args(args) -> list[str]:
    return [
        "--connect", args.connect,
        "--namespace", args.namespace,
        "--robot-yaw-deg", repr(args.robot_yaw_deg),
    ]


def runtime_commands(args, selection: MapSelection,
                     manifest: Path | None) -> dict[str, list[str]]:
    python = sys.executable
    commands = {
        "zenoh": [python, "-u", str(SCRIPTS_DIR / "start_zenoh.py"),
                  "--restart"],
        "mux": [python, "-u", str(SCRIPTS_DIR / "kiwi_command_mux.py"),
                "--connect", args.connect, "--namespace", args.namespace],
        "slam": [python, "-u", str(SCRIPTS_DIR / "kiwi_slam.py")],
        "dashboard": [python, "-u", str(SCRIPTS_DIR / "kiwi_dashboard.py")],
        "images": [python, "-u",
                   str(SCRIPTS_DIR / "kiwi_image_navigation.py")],
        "teleop": [python, "-u", str(SCRIPTS_DIR / "kiwi_teleop.py")],
    }
    commands["slam"] += common_args(args)
    commands["slam"] += [
        "--yaw-estimator", getattr(args, "yaw_estimator", "legacy")]
    calibration = getattr(args, "calibration", None)
    if calibration:
        commands["slam"] += ["--calibration", calibration]
        commands["dashboard"] += ["--calibration", calibration]
        commands["images"] += ["--calibration", calibration]
    if selection.resume:
        commands["slam"] += ["--resume", str(selection.prefix)]
        # Select once in the supervisor and pin both SLAM and image navigation
        # to the same manifest. Independent auto-discovery can otherwise expose
        # different live and agent-facing image sessions.
        if manifest is not None:
            commands["slam"] += ["--resume-image-manifest", str(manifest)]
        if args.resume_global:
            commands["slam"].append("--resume-global")
        if args.resume_pose:
            commands["slam"] += [
                "--resume-pose", *(repr(value) for value in args.resume_pose)]
        resume_search_distance = getattr(
            args, "resume_search_distance", None)
        if resume_search_distance is not None:
            commands["slam"] += [
                "--resume-search-distance",
                repr(resume_search_distance),
            ]
    else:
        commands["slam"] += ["--output", str(selection.prefix)]
    commands["dashboard"] += common_args(args)
    commands["images"] += [
        "--manifest", str(manifest or Path("<created-by-slam>")),
        "--port", str(args.gallery_port),
        "--mcp-port", str(getattr(args, "mcp_port", 8766)),
        "--agent-max-travel-distance",
        repr(getattr(args, "agent_max_travel_distance", 5.0)),
        "--runtime-collision-radius",
        repr(getattr(args, "runtime_collision_radius", 0.18)),
        "--command-topic", "cmd_vel/navigation",
        *common_args(args),
    ]
    if getattr(args, "no_mcp", False):
        commands["images"].append("--no-mcp")
    if args.no_open_gallery:
        commands["images"].append("--no-open")
    commands["teleop"] += [
        "--command-topic", "cmd_vel/teleop",
        *common_args(args),
    ]
    if args.gamepad:
        commands["teleop"].append("--gamepad")
    return commands


def wait_for_manifest(prefix: Path, before: set[Path], slam_process,
                      timeout_s: float) -> Path:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if slam_process.poll() is not None:
            raise RuntimeError(
                f"SLAM exited before creating an image map (code "
                f"{slam_process.returncode})")
        candidates = image_manifests(prefix) - before
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime_ns)
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {prefix}.images/*/manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--map", help="resume a saved map prefix or state file")
    maps.add_argument("--new-map", help="create a new map prefix")
    parser.add_argument("--resume-global", action="store_true",
                        help="globally relocalize a resumed map")
    parser.add_argument(
        "--resume-pose", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
        help="approximate pose for resumed-map relocalization")
    parser.add_argument(
        "--resume-search-distance", type=float, metavar="METERS",
        help=("translation radius around the resume pose to search during "
              "relocalization (default: saved map setting, normally 1 m)"))
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/xiao")
    parser.add_argument("--robot-yaw-deg", type=float, default=60.0)
    parser.add_argument(
        "--calibration", help="yaw/LiDAR calibration YAML/JSON file")
    parser.add_argument(
        "--yaw-estimator", choices=("legacy", "fused"), default="legacy")
    parser.add_argument("--gallery-port", type=int, default=8767)
    parser.add_argument("--mcp-port", type=int, default=8766)
    parser.add_argument("--no-mcp", action="store_true")
    parser.add_argument("--agent-max-travel-distance", type=float, default=5.0)
    parser.add_argument(
        "--runtime-collision-radius", type=float, default=0.18,
        help=("hard live-navigation collision radius in meters; reduce only "
              "for controlled demos (default: 0.18)"))
    parser.add_argument(
        "--image-manifest",
        help="explicit image-map manifest or session directory for the gallery")
    parser.add_argument("--no-open-gallery", action="store_true")
    parser.add_argument(
        "--gamepad", action="store_true",
        help=("use gamepad teleop; centered sticks release navigation and "
              "Start/Menu button 6 toggles TELEOP/AGENT control; stick input "
              "always reclaims TELEOP"))
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-image-navigation", action="store_true")
    parser.add_argument("--no-teleop", action="store_true")
    parser.add_argument(
        "--replace-existing", action="store_true",
        help="gracefully stop existing Kiwi runtime scripts before launch")
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--shutdown-timeout", type=float, default=20.0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="select the map and print commands without changing processes")
    return parser


def validate_args(parser, args) -> None:
    if args.resume_global and args.resume_pose:
        parser.error("--resume-global and --resume-pose are mutually exclusive")
    if (args.resume_global or args.resume_pose or
            args.resume_search_distance is not None) and args.new_map:
        parser.error("resume localization options cannot be used with --new-map")
    if args.image_manifest and args.new_map:
        parser.error("--image-manifest can only be used with a resumed map")
    if args.image_manifest and args.no_image_navigation:
        parser.error("--image-manifest conflicts with --no-image-navigation")
    if not math.isfinite(args.robot_yaw_deg):
        parser.error("--robot-yaw-deg must be finite")
    if (not math.isfinite(args.runtime_collision_radius) or
            not 0.0 <= args.runtime_collision_radius <= 0.25):
        parser.error(
            "--runtime-collision-radius must be finite and in [0, 0.25]")
    if args.resume_pose and not all(math.isfinite(value)
                                    for value in args.resume_pose):
        parser.error("--resume-pose values must be finite")
    if (args.resume_search_distance is not None and
            (not math.isfinite(args.resume_search_distance) or
             args.resume_search_distance <= 0.0)):
        parser.error("--resume-search-distance must be positive and finite")
    if not 1 <= args.gallery_port <= 65535:
        parser.error("--gallery-port must be in [1, 65535]")
    if not 1 <= args.mcp_port <= 65535:
        parser.error("--mcp-port must be in [1, 65535]")
    if (not args.no_image_navigation and not args.no_mcp and
            args.mcp_port == args.gallery_port):
        parser.error("--mcp-port and --gallery-port must differ")
    if (not math.isfinite(args.agent_max_travel_distance) or
            args.agent_max_travel_distance <= 0.0):
        parser.error("--agent-max-travel-distance must be positive")
    if args.startup_timeout <= 0.0 or args.shutdown_timeout <= 0.0:
        parser.error("startup and shutdown timeouts must be positive")
    if not args.no_teleop and not args.dry_run and not sys.stdin.isatty():
        parser.error("interactive teleop needs a terminal; use --no-teleop")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        selection = select_map(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.image_manifest and not selection.resume:
        parser.error("--image-manifest can only be used with a resumed map")

    mode = "resume" if selection.resume else "new"
    print(f"Map: {selection.prefix} ({mode})")
    enabled = ["zenoh", "mux", "slam"]
    if not args.no_dashboard:
        enabled.append("dashboard")
    if not args.no_image_navigation:
        enabled.append("images")
    if not args.no_teleop:
        enabled.append("teleop")
    if args.dry_run:
        try:
            manifest = selected_manifest(args, selection)
        except RuntimeError as exc:
            parser.error(str(exc))
        commands = runtime_commands(args, selection, manifest)
        print("\nDry run; processes were not changed:")
        for name in enabled:
            print(f"[{name}] {shlex.join(commands[name])}")
        return 0

    existing = existing_runtime_processes()
    if existing:
        print("Existing Kiwi runtime processes:")
        for pid, command in existing:
            print(f"  PID {pid}: {command}")
        replace = args.replace_existing
        if not replace and sys.stdin.isatty():
            answer = input("Stop them and continue? [Y/n]: ").strip().lower()
            replace = answer in ("", "y", "yes")
        if not replace:
            parser.error("stop existing runtime processes or use --replace-existing")
        try:
            stop_runtime_preflight(existing, args.shutdown_timeout)
        except RuntimeError as exc:
            parser.error(str(exc))

    # A live SLAM process can have appended image captures whose keyframes are
    # not in the on-disk graph yet. Recompute only after takeover saved it.
    try:
        manifest = selected_manifest(args, selection)
    except RuntimeError as exc:
        parser.error(str(exc))
    commands = runtime_commands(args, selection, manifest)

    try:
        preflight_image_ports(args)
    except RuntimeError as exc:
        parser.error(str(exc))

    supervisor = Supervisor(args.shutdown_timeout)
    previous_manifests = image_manifests(selection.prefix)

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_shutdown)
    zenoh_process = None
    try:
        zenoh_process = ensure_zenoh_router(
            commands["zenoh"], args.startup_timeout)
        supervisor.start("mux", commands["mux"])
        slam = supervisor.start("slam", commands["slam"])
        if not args.no_dashboard:
            supervisor.start("dashboard", commands["dashboard"])
        if not args.no_image_navigation:
            if manifest is None:
                manifest = wait_for_manifest(
                    selection.prefix, previous_manifests, slam,
                    args.startup_timeout)
                commands = runtime_commands(args, selection, manifest)
            # Repeat immediately before binding. Another launcher can leave an
            # orphaned gallery in the gap between the initial process scan and
            # SLAM producing a new image manifest.
            preflight_image_ports(args)
            supervisor.start("images", commands["images"])
        if not args.no_teleop:
            supervisor.start("teleop", commands["teleop"], interactive=True)

        supervisor.refresh_descendants()
        print("\nKiwi runtime is up.")
        if not args.no_image_navigation:
            print(f"Image navigation: http://127.0.0.1:{args.gallery_port}/")
            if not args.no_mcp:
                print(f"Kiwi MCP: http://127.0.0.1:{args.mcp_port}/mcp")
        print("Teleop overrides navigation while active; Space releases it.")
        print("Press Ctrl-C here to save the map and stop everything.")
        next_descendant_refresh = time.monotonic() + 1.0
        while True:
            if zenoh_process is not None and zenoh_process.poll() is not None:
                raise RuntimeError(
                    f"Zenoh exited unexpectedly with code "
                    f"{zenoh_process.returncode}")
            for item in supervisor.processes:
                code = item.process.poll()
                if code is not None:
                    raise RuntimeError(
                        f"{item.name} exited unexpectedly with code {code}")
            if time.monotonic() >= next_descendant_refresh:
                supervisor.refresh_descendants()
                next_descendant_refresh = time.monotonic() + 1.0
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
        return 0
    except RuntimeError as exc:
        print(f"\nLaunch failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # A second Ctrl-C or terminal SIGHUP must not interrupt the ordered
        # shutdown half-way through and orphan the gallery/SLAM processes.
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
            signal.signal(shutdown_signal, signal.SIG_IGN)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        supervisor.cleanup()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--process-watchdog":
        raise SystemExit(process_watchdog(
            int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "group"))
    raise SystemExit(main())
