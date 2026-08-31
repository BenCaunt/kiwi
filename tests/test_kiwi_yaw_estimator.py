import math
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from kiwi_yaw_estimator import (  # noqa: E402
    YawEstimator,
    YawEstimatorConfig,
)


class YawEstimatorTests(unittest.TestCase):
    def test_incremental_imu_crosses_wrap_without_a_jump(self):
        estimator = YawEstimator()
        estimator.update_odometry(
            time_s=0.0, wheel_omega_rad_s=0.0,
            imu_yaw_rad=math.radians(179.0), imu_valid=True)

        estimate = estimator.update_odometry(
            time_s=0.1, wheel_omega_rad_s=math.radians(20.0),
            imu_yaw_rad=math.radians(-179.0), imu_valid=True)

        self.assertAlmostEqual(math.degrees(estimate.yaw), 2.0, places=6)
        self.assertEqual(estimate.source, "wheel+imu")

    def test_imu_discontinuity_uses_wheels_and_resets_feedback(self):
        estimator = YawEstimator()
        estimator.update_odometry(
            time_s=0.0, imu_yaw_rad=0.0, imu_valid=True)

        estimate = estimator.update_odometry(
            time_s=0.1, wheel_omega_rad_s=1.0,
            imu_yaw_rad=2.0, imu_valid=True, imu_discontinuity=True)

        self.assertAlmostEqual(estimate.yaw, 0.1)
        self.assertEqual(estimate.source, "wheel")
        self.assertIn("imu_discontinuity", estimate.feedback_status)

    def test_missing_sensor_fallbacks_are_explicit(self):
        estimator = YawEstimator()
        estimator.update_odometry(time_s=0.0, wheel_valid=False)
        held = estimator.update_odometry(
            time_s=0.1, wheel_valid=False, imu_valid=False)
        wheel = estimator.update_odometry(
            time_s=0.2, wheel_omega_rad_s=1.0,
            wheel_valid=True, imu_valid=False)

        self.assertEqual(held.source, "hold")
        self.assertEqual(wheel.source, "wheel")
        self.assertAlmostEqual(wheel.yaw, 0.1)
        self.assertGreater(wheel.uncertainty_rad, held.uncertainty_rad)

    def test_wheel_slip_is_reduced_by_incremental_imu(self):
        estimator = YawEstimator(YawEstimatorConfig(imu_weight=0.85))
        estimator.update_odometry(
            time_s=0.0, imu_yaw_rad=0.0, imu_valid=True)

        estimate = estimator.update_odometry(
            time_s=0.1, wheel_omega_rad_s=10.0,
            imu_yaw_rad=0.0, imu_valid=True)

        self.assertLess(abs(estimate.yaw), 0.2)
        self.assertGreater(abs(estimate.yaw), 0.0)

    def test_trusted_scan_slope_learns_session_rate_bias(self):
        config = YawEstimatorConfig(
            bias_time_constant_s=10.0,
            feedback_window_s=12.0,
            feedback_min_span_s=3.0,
        )
        estimator = YawEstimator(config)
        imu_yaw = 0.0
        injected_bias = math.radians(0.05)
        estimator.update_odometry(
            time_s=0.0, imu_yaw_rad=imu_yaw, imu_valid=True)
        accepted = 0
        for index in range(1, 361):
            time_s = index * 0.5
            imu_yaw += injected_bias * 0.5
            estimate = estimator.update_odometry(
                time_s=time_s, wheel_omega_rad_s=0.0,
                imu_yaw_rad=imu_yaw, imu_valid=True)
            accepted += estimator.observe_scan(
                time_s=time_s,
                heading_disagreement_rad=-estimate.yaw,
                scan_matched=True, score=0.98, hit_ratio=0.90,
                rmse_m=0.03, wall_support_ratio=0.8,
            )

        learned_deg_s = math.degrees(
            estimator.snapshot().learned_rate_bias_rad_s)
        # With the default 0.5 blend, only half of the injected IMU drift
        # enters the fused estimate.
        self.assertGreater(accepted, 10)
        self.assertAlmostEqual(learned_deg_s, 0.025, delta=0.012)

    def test_feedback_quality_and_loop_gates_do_not_change_yaw(self):
        estimator = YawEstimator(YawEstimatorConfig(
            feedback_min_span_s=0.1, feedback_window_s=2.0))
        estimator.update_odometry(time_s=0.0)
        initial_yaw = estimator.snapshot().yaw

        accepted = estimator.observe_scan(
            time_s=1.0, heading_disagreement_rad=0.2,
            scan_matched=True, score=0.2, hit_ratio=0.9,
            rmse_m=0.02, wall_support_ratio=0.8)
        loop = estimator.observe_scan(
            time_s=2.0, heading_disagreement_rad=0.2,
            scan_matched=True, score=1.0, hit_ratio=1.0,
            rmse_m=0.0, wall_support_ratio=1.0, loop_closed=True)

        self.assertFalse(accepted)
        self.assertFalse(loop)
        self.assertEqual(estimator.snapshot().yaw, initial_yaw)
        self.assertIn("loop_closure", estimator.snapshot().feedback_status)

    def test_turning_motion_cannot_be_mislearned_as_rate_bias(self):
        estimator = YawEstimator()
        estimator.update_odometry(
            time_s=0.0, imu_yaw_rad=0.0, imu_valid=True)
        estimator.update_odometry(
            time_s=0.1, wheel_omega_rad_s=1.0,
            imu_yaw_rad=0.1, imu_valid=True)

        accepted = estimator.observe_scan(
            time_s=0.1, heading_disagreement_rad=0.02,
            scan_matched=True, score=1.0, hit_ratio=1.0,
            rmse_m=0.0, wall_support_ratio=1.0)

        self.assertFalse(accepted)
        self.assertEqual(
            estimator.snapshot().feedback_status,
            "rejected:high_yaw_rate")
        self.assertEqual(estimator.snapshot().learned_rate_bias_rad_s, 0.0)

    def test_translation_cannot_be_mislearned_as_rate_bias(self):
        estimator = YawEstimator()
        estimator.update_odometry(
            time_s=0.0, imu_yaw_rad=0.0, imu_valid=True)
        estimator.update_odometry(
            time_s=0.1, wheel_omega_rad_s=0.0,
            imu_yaw_rad=0.0, imu_valid=True, linear_speed_m_s=0.2)

        accepted = estimator.observe_scan(
            time_s=0.1, heading_disagreement_rad=0.02,
            scan_matched=True, score=1.0, hit_ratio=1.0,
            rmse_m=0.0, wall_support_ratio=1.0)

        self.assertFalse(accepted)
        self.assertEqual(
            estimator.snapshot().feedback_status,
            "rejected:translating")


if __name__ == "__main__":
    unittest.main()
