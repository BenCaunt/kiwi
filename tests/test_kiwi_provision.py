import importlib.util
from pathlib import Path
import unittest
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kiwi_provision.py"
SPEC = importlib.util.spec_from_file_location("kiwi_provision", SCRIPT_PATH)
kiwi_provision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kiwi_provision)


class WifiJoinedTests(unittest.TestCase):
    @mock.patch.object(kiwi_provision, "interface_ip", return_value="192.168.8.109")
    @mock.patch.object(kiwi_provision, "current_ssid", return_value="SpectrumSetup-86")
    def test_rejects_stale_ip_from_previous_network(self, _ssid, _ip):
        self.assertIsNone(
            kiwi_provision.wifi_joined(
                "en0", "KIWI-MASTER", kiwi_provision.AP_SUBNET_PREFIX
            )
        )

    @mock.patch.object(kiwi_provision, "interface_ip", return_value="192.168.8.109")
    @mock.patch.object(kiwi_provision, "current_ssid", return_value="KIWI-MASTER")
    def test_rejects_wrong_subnet_on_robot_ap(self, _ssid, _ip):
        self.assertIsNone(
            kiwi_provision.wifi_joined(
                "en0", "KIWI-MASTER", kiwi_provision.AP_SUBNET_PREFIX
            )
        )

    @mock.patch.object(kiwi_provision, "interface_ip", return_value="192.168.4.2")
    @mock.patch.object(kiwi_provision, "current_ssid", return_value="KIWI-MASTER")
    def test_accepts_requested_ssid_and_subnet(self, _ssid, _ip):
        self.assertEqual(
            kiwi_provision.wifi_joined(
                "en0", "KIWI-MASTER", kiwi_provision.AP_SUBNET_PREFIX
            ),
            "192.168.4.2",
        )

    @mock.patch.object(kiwi_provision, "interface_ip", return_value="192.168.8.109")
    @mock.patch.object(kiwi_provision, "current_ssid", return_value="SpectrumSetup-86")
    def test_accepts_restored_network_with_any_acquired_subnet(self, _ssid, _ip):
        self.assertEqual(
            kiwi_provision.wifi_joined("en0", "SpectrumSetup-86"),
            "192.168.8.109",
        )


class JoinWifiTests(unittest.TestCase):
    @mock.patch.object(kiwi_provision.time, "sleep")
    @mock.patch.object(kiwi_provision.time, "time", side_effect=[0, 0, 1])
    @mock.patch.object(kiwi_provision, "interface_ip", return_value="192.168.8.109")
    @mock.patch.object(kiwi_provision, "current_ssid", return_value="SpectrumSetup-86")
    @mock.patch.object(kiwi_provision, "wifi_joined", return_value=None)
    @mock.patch.object(kiwi_provision, "run")
    def test_does_not_report_success_from_stale_ip(
        self, run, _joined, _ssid, _ip, _time, _sleep
    ):
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

        ok, error = kiwi_provision.join_wifi(
            "en0",
            "KIWI-MASTER",
            "seeedstudio",
            timeout_s=1,
            attempts=1,
            expected_subnet_prefix=kiwi_provision.AP_SUBNET_PREFIX,
        )

        self.assertFalse(ok)
        self.assertIn("SpectrumSetup-86", error)
        self.assertIn("192.168.8.109", error)

    @mock.patch.object(kiwi_provision, "interface_ip", return_value="192.168.1.238")
    @mock.patch.object(kiwi_provision, "current_ssid", return_value="SpectrumSetup-86")
    @mock.patch.object(kiwi_provision.time, "sleep")
    @mock.patch.object(kiwi_provision.time, "time", side_effect=[0, 2])
    @mock.patch.object(kiwi_provision, "run")
    def test_networksetup_timeout_is_reported_without_traceback(
        self, run, _time, _sleep, _ssid, _ip
    ):
        run.side_effect = kiwi_provision.subprocess.TimeoutExpired(
            ["networksetup", "-setairportnetwork"], 30
        )

        ok, error = kiwi_provision.join_wifi(
            "en0", "KIWI-MASTER", "seeedstudio", timeout_s=1, attempts=1
        )

        self.assertFalse(ok)
        self.assertIn("timed out after 1 seconds", error)

    @mock.patch.object(kiwi_provision.time, "sleep")
    @mock.patch.object(kiwi_provision.time, "time", side_effect=[0, 0])
    @mock.patch.object(kiwi_provision, "wifi_joined", return_value="192.168.4.2")
    @mock.patch.object(kiwi_provision, "run")
    def test_accepts_successful_join_even_when_networksetup_times_out(
        self, run, _joined, _time, _sleep
    ):
        run.side_effect = kiwi_provision.subprocess.TimeoutExpired(
            ["networksetup", "-setairportnetwork"], 30
        )

        ok, error = kiwi_provision.join_wifi(
            "en0", "KIWI-MASTER", "seeedstudio", timeout_s=30, attempts=1
        )

        self.assertTrue(ok)
        self.assertIsNone(error)


class PostConfigTests(unittest.TestCase):
    @mock.patch.object(kiwi_provision, "http_json", side_effect=TimeoutError("timed out"))
    def test_reboot_timeout_has_actionable_message_instead_of_traceback(self, _http):
        with self.assertRaises(SystemExit) as raised:
            kiwi_provision.post_config(
                "192.168.1.157", {"zenoh_connect": "udp/192.168.1.238:7447"}
            )

        message = str(raised.exception)
        self.assertIn("result is indeterminate", message)
        self.assertIn("Do not retry blindly", message)


if __name__ == "__main__":
    unittest.main()
