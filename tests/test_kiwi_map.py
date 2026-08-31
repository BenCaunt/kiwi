import pathlib
import sys
import unittest

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_map import decode_occupancy_map, encode_occupancy_map  # noqa: E402
from kiwi_slam_core import OccupancyMap  # noqa: E402


class LiveOccupancyMapTests(unittest.TestCase):
    def test_round_trips_map_metadata_and_cells(self):
        source = OccupancyMap(
            data=np.array(((-1, 0, 100), (25, 65, 80)), dtype=np.int16),
            resolution_m=0.05,
            origin_x=-1.25,
            origin_y=2.5,
        )

        decoded = decode_occupancy_map(encode_occupancy_map(source, 17))

        np.testing.assert_array_equal(decoded.data, source.data)
        self.assertAlmostEqual(decoded.resolution_m, source.resolution_m)
        self.assertAlmostEqual(decoded.origin_x, source.origin_x)
        self.assertAlmostEqual(decoded.origin_y, source.origin_y)
        self.assertEqual(decoded.keyframes, 17)

    def test_rejects_truncated_and_corrupt_payloads(self):
        source = OccupancyMap(
            data=np.array(((0, 100),), dtype=np.int16),
            resolution_m=0.1,
            origin_x=0.0,
            origin_y=0.0,
        )
        payload = encode_occupancy_map(source, 1)

        for invalid in (payload[:10], payload[:-1], payload + b"trailing"):
            with self.subTest(size=len(invalid)):
                with self.assertRaises(ValueError):
                    decode_occupancy_map(invalid)


if __name__ == "__main__":
    unittest.main()
