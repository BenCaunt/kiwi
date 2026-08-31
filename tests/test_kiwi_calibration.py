import json
import math
import pathlib
import tempfile
import sys
import unittest

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_calibration import (  # noqa: E402
    CalibrationLogWriter,
    RotationObservation,
    aggregate_rotation_segments,
    fit_planar_calibration,
    fit_rotation_calibration,
    load_calibration,
    read_lidar_records,
)


class CalibrationFileTests(unittest.TestCase):
    def test_loads_documented_yaml_mapping_without_pyyaml(self):
        source = """
format: kiwi-slam-calibration-v1
created_at: 2026-08-25T00:00:00Z
source_run: calibration/test
yaw_estimator:
  wheel_yaw_scale: 0.98
  imu_yaw_scale: 1.01
  initial_rate_bias_deg_s: 0.05
  imu_weight: 0.8
lidar:
  time_offset_ms: 8.0
  x_m: 0.03
  y_m: -0.01
  yaw_deg: 1.5
validation:
  wall_residual_p90_m: 0.06
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "calibration.yaml"
            path.write_text(source, encoding="utf-8")

            calibration = load_calibration(path)

        self.assertAlmostEqual(calibration.yaw_estimator.wheel_yaw_scale, 0.98)
        self.assertAlmostEqual(calibration.lidar.time_offset_ms, 8.0)
        self.assertEqual(calibration.validation["wall_residual_p90_m"], 0.06)

    def test_raw_log_round_trips_exact_lidar_payloads(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = pathlib.Path(parent) / "run"
            with CalibrationLogWriter(directory, {"namespace": "kiwi/sim"}) as log:
                log.write_odometry({"follower_time_us": 10}, 100)
                log.write_lidar(b"\x00\x01raw", 200)
                log.write_event("slow clockwise", 300)

            odometry = json.loads(
                (directory / "odom.jsonl").read_text().strip())
            lidar = list(read_lidar_records(directory / "lidar.bin"))

        self.assertEqual(odometry["report"]["follower_time_us"], 10)
        self.assertEqual(lidar, [(200, b"\x00\x01raw")])


class CalibrationSolverTests(unittest.TestCase):
    def test_aggregates_same_direction_scan_increments_into_turns(self):
        increments = [
            RotationObservation(0.1, direction * 0.2, direction * 0.2,
                                direction * 0.2)
            for direction in (1, 1, 1, -1, -1, -1, 1, 1, 1)
        ]

        segments = aggregate_rotation_segments(
            increments, min_rotation_rad=0.5)

        self.assertEqual(len(segments), 3)
        self.assertAlmostEqual(segments[0].wheel_delta_rad, 0.6)
        self.assertAlmostEqual(segments[1].wheel_delta_rad, -0.6)
        self.assertAlmostEqual(segments[2].dt_s, 0.3)

    def test_recovers_rotation_scale_and_bias_with_an_outlier(self):
        rng = np.random.default_rng(3)
        wheel_scale = 0.975
        imu_scale = 1.012
        imu_bias = math.radians(0.055)
        observations = []
        for index in range(40):
            dt = 0.4 + 0.02 * (index % 7)
            scan_delta = math.radians((-1 if index % 2 else 1) *
                                      (4.0 + index % 5))
            wheel = scan_delta / wheel_scale
            imu = (scan_delta + imu_bias * dt) / imu_scale
            scan_delta += rng.normal(0.0, math.radians(0.03))
            if index == 7:
                scan_delta += math.radians(8.0)
            observations.append(RotationObservation(
                dt, wheel, imu, scan_delta))

        fit = fit_rotation_calibration(observations)

        self.assertAlmostEqual(fit.wheel_yaw_scale, wheel_scale, delta=0.004)
        self.assertAlmostEqual(fit.imu_yaw_scale, imu_scale, delta=0.006)
        self.assertAlmostEqual(
            math.degrees(fit.imu_rate_bias_rad_s),
            math.degrees(imu_bias), delta=0.015)

    def test_rotation_fit_weights_the_more_repeatable_heading_sensor(self):
        observations = []
        for index in range(30):
            scan = math.radians((-1 if index % 2 else 1) * 10.0)
            wheel = scan + math.radians((index % 3 - 1) * 0.2)
            imu = scan + math.radians((index % 5 - 2) * 2.0)
            observations.append(RotationObservation(
                0.5, wheel, imu, scan))

        fit = fit_rotation_calibration(observations)

        self.assertLess(fit.wheel_rmse_rad, fit.imu_rmse_rad)
        self.assertLess(fit.imu_weight, 0.5)

    def test_recovers_planar_time_and_translation_minimum(self):
        expected = (0.011, 0.043, -0.026)

        def objective(time_s, x_m, y_m):
            return ((time_s - expected[0]) / 0.01) ** 2 + \
                ((x_m - expected[1]) / 0.05) ** 2 + \
                ((y_m - expected[2]) / 0.05) ** 2

        fit = fit_planar_calibration(objective)

        self.assertAlmostEqual(fit.time_offset_s, expected[0], delta=0.0011)
        self.assertAlmostEqual(fit.x_m, expected[1], delta=0.002)
        self.assertAlmostEqual(fit.y_m, expected[2], delta=0.002)


if __name__ == "__main__":
    unittest.main()
