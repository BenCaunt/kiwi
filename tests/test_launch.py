import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("kiwi_launch", ROOT / "launch.py")
launch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launch
SPEC.loader.exec_module(launch)


class LaunchMapTests(unittest.TestCase):
    def test_discovers_only_complete_saved_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "good.graph.json").write_text('{"nodes": []}')
            (root / "good.slam.npz").write_bytes(b"state")
            (root / "incomplete.graph.json").write_text('{"nodes": []}')

            maps = launch.discover_saved_maps(root)

            self.assertEqual(maps, [(root / "good").resolve()])

    def test_normalizes_any_saved_state_filename_to_prefix(self):
        prefix = launch.normalize_map_prefix("maps/example.slam.npz")
        self.assertEqual(prefix, (ROOT / "maps/example").resolve())

    def test_keyframe_count_reads_graph_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory) / "map"
            pathlib.Path(f"{prefix}.graph.json").write_text(json.dumps({
                "nodes": [{}, {}, {}],
            }))
            self.assertEqual(launch.map_keyframe_count(prefix), 3)

    def test_descendant_groups_exclude_the_roots_shared_shell_group(self):
        table = """\
100 50 50
101 100 101
102 101 101
103 100 50
"""
        result = types.SimpleNamespace(stdout=table)
        with mock.patch.object(launch.subprocess, "run", return_value=result), \
                mock.patch.object(launch.os, "getpgrp", return_value=50):
            groups = launch.descendant_process_groups({100})

        self.assertEqual(groups, {101})

    def test_runtime_preflight_rescans_for_late_orphan(self):
        first = (100, "python3 scripts/kiwi_slam.py")
        late = (101, "python3 scripts/kiwi_image_navigation.py")
        with mock.patch.object(launch, "stop_existing_runtime") as stop, \
                mock.patch.object(
                    launch, "existing_runtime_processes",
                    side_effect=([late], [])), \
                mock.patch.object(launch.time, "sleep"):
            launch.stop_runtime_preflight([first], 8.0)

        self.assertEqual(stop.call_args_list, [
            mock.call([first], 8.0),
            mock.call([late], 3.0),
        ])

    def test_runtime_preflight_fails_if_process_keeps_respawning(self):
        stale = (100, "python3 scripts/kiwi_slam.py")
        with mock.patch.object(launch, "stop_existing_runtime"), \
                mock.patch.object(
                    launch, "existing_runtime_processes",
                    side_effect=([stale], [stale], [stale])), \
                mock.patch.object(launch.time, "sleep"):
            with self.assertRaisesRegex(
                    RuntimeError, "could not clear.*PID 100"):
                launch.stop_runtime_preflight([stale], 8.0)


class LaunchPortPreflightTests(unittest.TestCase):
    def test_reuses_router_with_required_udp_and_tcp_listeners(self):
        command = (
            "/usr/local/bin/zenohd --listen udp/0.0.0.0:7447 "
            "--listen tcp/0.0.0.0:7447")
        with mock.patch.object(
                launch, "loopback_listener_pids", return_value={42}), \
                mock.patch.object(
                    launch, "process_command", return_value=command):
            pid = launch.reusable_zenoh_router_pid()

        self.assertEqual(pid, 42)

    def test_ensure_zenoh_does_not_restart_healthy_router(self):
        with mock.patch.object(
                launch, "reusable_zenoh_router_pid", return_value=42), \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            process = launch.ensure_zenoh_router(["start-zenoh"], 5.0)

        self.assertIsNone(process)
        popen.assert_not_called()

    def test_watchdog_interrupts_target_after_parent_disappears(self):
        with mock.patch.object(launch.os, "getppid", return_value=1), \
                mock.patch.object(
                    launch, "_watchdog_target_running",
                    side_effect=(True, False, False)), \
                mock.patch.object(launch, "_watchdog_signal") as send:
            code = launch.process_watchdog(500, 600, True)

        self.assertEqual(code, 0)
        send.assert_called_once_with(600, True, launch.signal.SIGINT)

    def test_zenoh_readiness_ignores_listener_owned_by_old_router(self):
        process = types.SimpleNamespace(
            pid=222,
            returncode=None,
            poll=mock.Mock(return_value=None),
        )
        connection = mock.MagicMock()
        with mock.patch.object(
                launch, "loopback_listener_pids",
                side_effect=({111}, {222})), \
                mock.patch.object(launch.time, "sleep"), \
                mock.patch.object(
                    launch.socket, "create_connection",
                    return_value=connection) as connect:
            launch.wait_for_port(
                "127.0.0.1", 7447, process, 1.0, owner_pid=222)

        connect.assert_called_once_with(("127.0.0.1", 7447), timeout=0.2)

    def test_listener_pid_parser_uses_lsof_machine_output(self):
        result = types.SimpleNamespace(stdout="p4471\nf16\np5512\nf21\n")
        with mock.patch.object(
                launch.subprocess, "run", return_value=result) as run:
            owners = launch.loopback_listener_pids(8767)

        self.assertEqual(owners, {4471, 5512})
        self.assertEqual(
            run.call_args.args[0],
            ["lsof", "-nP", "-iTCP:8767", "-sTCP:LISTEN", "-Fp"],
        )

    def test_launchd_label_parser_matches_active_pid(self):
        result = types.SimpleNamespace(stdout=(
            "PID\tStatus\tLabel\n"
            "6994\t0\tcom.kiwi.agent-mcp\n"
            "-\t0\tcom.example.idle\n"
            "7000\t0\tcom.example.other\n"
        ))
        with mock.patch.object(
                launch.subprocess, "run", return_value=result):
            labels = launch.launchd_labels_for_pids({6994})

        self.assertEqual(labels, {"com.kiwi.agent-mcp"})

    def test_preflight_stops_only_stale_image_navigation_owner(self):
        image = (4471, "python3 scripts/kiwi_image_navigation.py --port 8767")
        dashboard = (4472, "python3 scripts/kiwi_dashboard.py")
        with mock.patch.object(
                launch, "loopback_port_available",
                side_effect=(False, True)), \
                mock.patch.object(
                    launch, "loopback_listener_pids", return_value={4471}), \
                mock.patch.object(
                    launch, "existing_runtime_processes",
                    return_value=[image, dashboard]), \
                mock.patch.object(
                    launch, "launchd_labels_for_pids", return_value=set()), \
                mock.patch.object(launch, "stop_existing_runtime") as stop:
            launch.preflight_image_port(8767, "gallery", 5.0)

        stop.assert_called_once_with([image], 5.0)

    def test_preflight_removes_image_navigation_keepalive_job(self):
        image = (6994, "python3 scripts/kiwi_image_navigation.py --port 8767")
        with mock.patch.object(
                launch, "loopback_port_available",
                side_effect=(False, True)), \
                mock.patch.object(
                    launch, "loopback_listener_pids", return_value={6994}), \
                mock.patch.object(
                    launch, "existing_runtime_processes", return_value=[image]), \
                mock.patch.object(
                    launch, "launchd_labels_for_pids",
                    return_value={"com.kiwi.agent-mcp"}), \
                mock.patch.object(launch, "remove_launchd_jobs") as remove, \
                mock.patch.object(launch, "stop_existing_runtime") as stop:
            launch.preflight_image_port(8767, "gallery", 5.0)

        remove.assert_called_once_with({"com.kiwi.agent-mcp"})
        stop.assert_called_once_with([image], 5.0)

    def test_preflight_refuses_to_kill_unrelated_listener(self):
        with mock.patch.object(
                launch, "loopback_port_available",
                side_effect=(False, False)), \
                mock.patch.object(
                    launch, "loopback_listener_pids", return_value={9001}), \
                mock.patch.object(
                    launch, "existing_runtime_processes", return_value=[]), \
                mock.patch.object(launch, "stop_existing_runtime") as stop:
            with self.assertRaisesRegex(
                    RuntimeError, "gallery port 8767.*PID 9001"):
                launch.preflight_image_port(8767, "gallery", 5.0)

        stop.assert_not_called()

    def test_image_preflight_checks_gallery_and_optional_mcp(self):
        args = types.SimpleNamespace(
            no_image_navigation=False,
            no_mcp=False,
            gallery_port=8767,
            mcp_port=8766,
            shutdown_timeout=8.0,
        )
        with mock.patch.object(launch, "preflight_image_port") as check:
            launch.preflight_image_ports(args)

        self.assertEqual(check.call_args_list, [
            mock.call(8767, "gallery", 8.0),
            mock.call(8766, "MCP", 8.0),
        ])


class LaunchCommandTests(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "connect": "tcp/127.0.0.1:7447",
            "namespace": "kiwi/test",
            "robot_yaw_deg": 60.0,
            "resume_global": False,
            "resume_pose": None,
            "resume_search_distance": None,
            "gallery_port": 8766,
            "image_manifest": None,
            "no_open_gallery": True,
            "gamepad": False,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_resumed_stack_uses_headless_slam_and_mux_topics(self):
        selection = launch.MapSelection(ROOT / "maps/kiwi_map", resume=True)
        manifest = ROOT / "maps/kiwi_map.images/session/manifest.json"

        commands = launch.runtime_commands(self.args(), selection, manifest)

        self.assertIn("--resume", commands["slam"])
        self.assertNotIn("--viewer", commands["slam"])
        self.assertEqual(
            commands["images"][commands["images"].index("--command-topic") + 1],
            "cmd_vel/navigation",
        )
        self.assertEqual(
            commands["images"][commands["images"].index("--mcp-port") + 1],
            "8766",
        )
        self.assertEqual(
            commands["teleop"][commands["teleop"].index("--command-topic") + 1],
            "cmd_vel/teleop",
        )
        self.assertIn("--restart", commands["zenoh"])

    def test_new_stack_writes_output_instead_of_resuming(self):
        selection = launch.MapSelection(ROOT / "maps/new", resume=False)

        commands = launch.runtime_commands(self.args(), selection, None)

        self.assertIn("--output", commands["slam"])
        self.assertNotIn("--resume", commands["slam"])

    def test_resume_search_distance_is_forwarded_to_slam(self):
        selection = launch.MapSelection(ROOT / "maps/kiwi_map", resume=True)

        commands = launch.runtime_commands(
            self.args(resume_search_distance=3.0), selection, None)

        option = commands["slam"].index("--resume-search-distance")
        self.assertEqual(commands["slam"][option + 1], "3.0")

    def test_runtime_collision_radius_is_forwarded_to_image_navigation(self):
        selection = launch.MapSelection(ROOT / "maps/kiwi_map", resume=True)

        commands = launch.runtime_commands(
            self.args(runtime_collision_radius=0.12), selection, None)

        option = commands["images"].index("--runtime-collision-radius")
        self.assertEqual(commands["images"][option + 1], "0.12")

    def test_explicit_manifest_is_shared_by_slam_and_gallery(self):
        selection = launch.MapSelection(ROOT / "maps/kiwi_map", resume=True)
        manifest = ROOT / "maps/kiwi_map.images/session/manifest.json"

        commands = launch.runtime_commands(
            self.args(image_manifest=str(manifest)), selection, manifest)

        self.assertEqual(
            commands["slam"][
                commands["slam"].index("--resume-image-manifest") + 1],
            str(manifest),
        )
        self.assertEqual(
            commands["images"][commands["images"].index("--manifest") + 1],
            str(manifest),
        )

    def test_auto_discovered_manifest_is_shared_by_slam_and_gallery(self):
        selection = launch.MapSelection(ROOT / "maps/kiwi_map", resume=True)
        manifest = ROOT / "maps/kiwi_map.images/session/manifest.json"

        commands = launch.runtime_commands(
            self.args(image_manifest=None), selection, manifest)

        self.assertEqual(
            commands["slam"][
                commands["slam"].index("--resume-image-manifest") + 1],
            str(manifest),
        )
        self.assertEqual(
            commands["images"][commands["images"].index("--manifest") + 1],
            str(manifest),
        )


if __name__ == "__main__":
    unittest.main()
