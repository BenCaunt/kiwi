import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_teleop import (  # noqa: E402
    GAMEPAD_DEADZONE,
    GamepadHandoff,
    _deadzone,
)


class GamepadTeleopTests(unittest.TestCase):
    def test_centered_stick_drift_is_zeroed(self):
        self.assertEqual(_deadzone(GAMEPAD_DEADZONE * 0.75), 0.0)
        self.assertEqual(_deadzone(-GAMEPAD_DEADZONE), 0.0)
        self.assertGreater(_deadzone(0.5), 0.0)

    def test_handoff_toggles_only_on_button_press_edge(self):
        handoff = GamepadHandoff()

        self.assertTrue(handoff.teleop_enabled)
        self.assertTrue(handoff.update(True))
        self.assertFalse(handoff.teleop_enabled)
        self.assertFalse(handoff.update(True))
        self.assertFalse(handoff.teleop_enabled)
        self.assertFalse(handoff.update(False))
        self.assertTrue(handoff.update(True))
        self.assertTrue(handoff.teleop_enabled)

    def test_stick_input_always_reclaims_teleop(self):
        handoff = GamepadHandoff()
        handoff.update(True)
        self.assertFalse(handoff.teleop_enabled)

        self.assertFalse(handoff.reclaim_for_stick_input(False))
        self.assertFalse(handoff.teleop_enabled)
        self.assertTrue(handoff.reclaim_for_stick_input(True))
        self.assertTrue(handoff.teleop_enabled)


if __name__ == "__main__":
    unittest.main()
