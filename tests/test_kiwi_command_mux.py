import json
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_command_mux import CommandMux, decode_command  # noqa: E402


class CommandMuxTests(unittest.TestCase):
    def test_teleop_preempts_navigation_and_release_resumes_it(self):
        mux = CommandMux(teleop_timeout_s=0.35, navigation_timeout_s=0.35)
        mux.update("navigation", json.dumps({
            "vx": 0.2, "vy": 0.0, "omega": 0.1,
        }), now=10.0)
        mux.update("teleop", json.dumps({
            "vx": 0.0, "vy": -0.1, "omega": 0.0, "active": True,
        }), now=10.1)

        source, command = mux.select(now=10.2)
        self.assertEqual(source, "teleop")
        self.assertEqual(command["vy"], -0.1)

        mux.update("teleop", json.dumps({
            "vx": 0.0, "vy": 0.0, "omega": 0.0, "active": False,
        }), now=10.21)
        source, command = mux.select(now=10.22)
        self.assertEqual(source, "navigation")
        self.assertEqual(command["vx"], 0.2)

    def test_stale_sources_force_zero(self):
        mux = CommandMux(teleop_timeout_s=0.3, navigation_timeout_s=0.3)
        mux.update("navigation", '{"vx":0.2,"vy":0,"omega":0}', now=1.0)

        source, command = mux.select(now=1.31)

        self.assertEqual(source, "idle")
        self.assertEqual(command, {"vx": 0.0, "vy": 0.0, "omega": 0.0})

    def test_legacy_zero_teleop_command_releases_lease(self):
        mux = CommandMux()
        mux.update("teleop", '{"vx":0,"vy":0,"omega":0}', now=2.0)

        self.assertEqual(mux.select(now=2.1)[0], "idle")

    def test_rejects_nonfinite_or_invalid_active_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_command('{"vx":NaN,"vy":0,"omega":0}')
        with self.assertRaisesRegex(ValueError, "boolean"):
            decode_command('{"vx":0,"vy":0,"omega":0,"active":1}')


if __name__ == "__main__":
    unittest.main()
