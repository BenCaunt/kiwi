import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kiwi_lidar.py"
SPEC = importlib.util.spec_from_file_location("kiwi_lidar", SCRIPT_PATH)
kiwi_lidar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kiwi_lidar)


def valid_frame():
    raw = bytearray(kiwi_lidar.FRAME_LEN)
    raw[0] = 0x54
    raw[1] = 0x2C
    raw[-1] = kiwi_lidar.crc8(raw[:-1])
    return bytes(raw)


class ParseFramesTests(unittest.TestCase):
    def test_accepts_legacy_single_frame_sample(self):
        decoded = kiwi_lidar.parse_frames(valid_frame())

        self.assertEqual(len(decoded), 1)
        self.assertIsNotNone(decoded[0])

    def test_decodes_concatenated_batch(self):
        raw = valid_frame()
        decoded = kiwi_lidar.parse_frames(raw * 20)

        self.assertEqual(len(decoded), 20)
        self.assertTrue(all(frame is not None for frame in decoded))

    def test_rejects_partial_batch(self):
        self.assertEqual(kiwi_lidar.parse_frames(valid_frame() + b"partial"), [])


if __name__ == "__main__":
    unittest.main()
