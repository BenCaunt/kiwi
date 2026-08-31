"""Continuous wheel/IMU yaw fusion with cautiously gated LiDAR feedback.

This module deliberately has no Zenoh, viewer, or simulator dependencies.  It
can therefore be exercised on retained odometry/SLAM reports and used unchanged
by the live robot and simulator paths.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
import math
import threading

import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class YawEstimatorConfig:
    wheel_yaw_scale: float = 1.0
    imu_yaw_scale: float = 1.0
    initial_rate_bias_deg_s: float = 0.0
    # Equal influence is the uncalibrated fallback. Calibration replaces this
    # with an inverse-variance weight measured against LiDAR scan rotation.
    imu_weight: float = 0.5
    bias_time_constant_s: float = 30.0
    max_rate_bias_deg_s: float = 0.3
    innovation_base_deg: float = 3.0
    max_interval_s: float = 0.2
    feedback_window_s: float = 15.0
    feedback_min_span_s: float = 3.0
    feedback_min_score: float = 0.90
    feedback_min_hit_ratio: float = 0.75
    feedback_max_rmse_m: float = 0.10
    feedback_min_wall_support_ratio: float = 0.25
    feedback_max_abs_yaw_rate_deg_s: float = 15.0
    feedback_max_linear_speed_m_s: float = 0.03
    initial_uncertainty_deg: float = 1.0
    healthy_noise_deg_sqrt_s: float = 0.25
    degraded_noise_deg_sqrt_s: float = 1.5

    @classmethod
    def from_mapping(cls, values):
        if values is None:
            return cls()
        if not isinstance(values, dict):
            raise ValueError("yaw_estimator calibration must be a mapping")
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items()
                      if key in known})

    def validate(self):
        numeric = asdict(self)
        if not all(_finite(value) for value in numeric.values()):
            raise ValueError("yaw estimator calibration values must be finite")
        for name, value in numeric.items():
            setattr(self, name, float(value))
        if self.wheel_yaw_scale <= 0.0 or self.imu_yaw_scale <= 0.0:
            raise ValueError("yaw scale factors must be positive")
        if not 0.0 <= self.imu_weight <= 1.0:
            raise ValueError("imu_weight must be between zero and one")
        if self.bias_time_constant_s <= 0.0:
            raise ValueError("bias_time_constant_s must be positive")
        if self.max_interval_s <= 0.0:
            raise ValueError("max_interval_s must be positive")
        if self.feedback_window_s < self.feedback_min_span_s:
            raise ValueError("feedback window must include its minimum span")
        return self


@dataclass(frozen=True)
class YawEstimate:
    yaw: float
    yaw_rate_rad_s: float
    uncertainty_rad: float
    learned_rate_bias_rad_s: float
    wheel_delta_rad: float
    imu_delta_rad: float
    imu_weight: float
    dt_s: float
    source: str
    feedback_status: str

    def diagnostics(self):
        return {
            "fused_yaw": self.yaw,
            "fused_yaw_rate": self.yaw_rate_rad_s,
            "yaw_uncertainty_deg": math.degrees(self.uncertainty_rad),
            "learned_rate_bias_deg_s": math.degrees(
                self.learned_rate_bias_rad_s),
            "wheel_delta_deg": math.degrees(self.wheel_delta_rad),
            "imu_delta_deg": math.degrees(self.imu_delta_rad),
            "imu_weight": self.imu_weight,
            "source": self.source,
            "scan_feedback": self.feedback_status,
        }


class YawEstimator:
    """Fuse incremental wheel and IMU yaw without imposing an absolute axis."""

    def __init__(self, config=None):
        self.config = (config or YawEstimatorConfig()).validate()
        self._lock = threading.RLock()
        self._yaw = 0.0
        self._last_time_s = None
        self._last_imu_yaw = None
        self._learned_bias = math.radians(
            self.config.initial_rate_bias_deg_s)
        self._uncertainty = math.radians(
            self.config.initial_uncertainty_deg)
        self._feedback = deque()
        self._last_feedback_bias_time_s = None
        self._last_discontinuity_time_s = None
        self._feedback_status = "idle"
        self._source_counts = Counter({"initializing": 1})
        self._feedback_counts = Counter()
        self._latest_linear_speed_m_s = 0.0
        self._latest = YawEstimate(
            yaw=0.0,
            yaw_rate_rad_s=0.0,
            uncertainty_rad=self._uncertainty,
            learned_rate_bias_rad_s=self._learned_bias,
            wheel_delta_rad=0.0,
            imu_delta_rad=0.0,
            imu_weight=0.0,
            dt_s=0.0,
            source="initializing",
            feedback_status=self._feedback_status,
        )

    @staticmethod
    def _robust_innovation(innovation, limit):
        """Huber influence for one angular innovation."""
        return max(-limit, min(limit, innovation))

    def update_odometry(self, *, time_s, wheel_omega_rad_s=0.0,
                        imu_yaw_rad=None, imu_valid=False,
                        imu_discontinuity=False, wheel_valid=True,
                        linear_speed_m_s=0.0):
        with self._lock:
            time_s = float(time_s)
            if not math.isfinite(time_s):
                raise ValueError("odometry time must be finite")
            imu_healthy = bool(imu_valid and _finite(imu_yaw_rad))
            wheel_healthy = bool(
                wheel_valid and _finite(wheel_omega_rad_s))
            self._latest_linear_speed_m_s = (
                abs(float(linear_speed_m_s))
                if _finite(linear_speed_m_s) else math.inf)

            if self._last_time_s is None:
                self._last_time_s = time_s
                if imu_healthy:
                    self._last_imu_yaw = float(imu_yaw_rad)
                self._latest = self._make_estimate(
                    0.0, 0.0, 0.0, 0.0, "initializing")
                return self._latest

            elapsed = time_s - self._last_time_s
            self._last_time_s = time_s
            if elapsed <= 0.0:
                self._latest = self._make_estimate(
                    0.0, 0.0, 0.0, 0.0, "nonmonotonic_time")
                return self._latest
            dt = min(elapsed, self.config.max_interval_s)

            wheel_delta = (
                self.config.wheel_yaw_scale * float(wheel_omega_rad_s) * dt
                if wheel_healthy else None)
            imu_delta = None
            if imu_healthy and self._last_imu_yaw is not None and \
                    not imu_discontinuity:
                imu_delta = self.config.imu_yaw_scale * wrap_angle(
                    float(imu_yaw_rad) - self._last_imu_yaw)
                # Match the legacy integrator's bounded-gap behavior. An IMU
                # delta spans the full report gap, while wheel/translation
                # integration intentionally caps that gap for safety.
                if elapsed > dt:
                    imu_delta *= dt / elapsed
            if imu_healthy:
                self._last_imu_yaw = float(imu_yaw_rad)
            if imu_discontinuity:
                self._last_discontinuity_time_s = time_s
                self._feedback.clear()
                self._feedback_status = "rejected:imu_discontinuity"
                self._feedback_counts[self._feedback_status] += 1

            blend_weight = 0.0
            if wheel_delta is not None and imu_delta is not None:
                innovation = wrap_angle(imu_delta - wheel_delta)
                limit = math.radians(self.config.innovation_base_deg) + \
                    abs(wheel_delta)
                blend_weight = self.config.imu_weight
                fused_delta = wheel_delta + blend_weight * \
                    self._robust_innovation(innovation, limit)
                source = "wheel+imu"
            elif imu_delta is not None:
                fused_delta = imu_delta
                blend_weight = 1.0
                source = "imu"
            elif wheel_delta is not None:
                fused_delta = wheel_delta
                source = "wheel"
            else:
                fused_delta = 0.0
                source = "hold"

            if source != "hold":
                fused_delta -= self._learned_bias * dt
                self._yaw += fused_delta
            yaw_rate = fused_delta / dt if dt > 0.0 else 0.0

            noise_deg = (
                self.config.healthy_noise_deg_sqrt_s
                if source == "wheel+imu"
                else self.config.degraded_noise_deg_sqrt_s)
            variance = self._uncertainty ** 2 + \
                math.radians(noise_deg) ** 2 * dt
            if source == "wheel+imu":
                variance *= math.exp(-0.15 * dt)
            self._uncertainty = math.sqrt(max(variance, 1.0e-12))

            self._latest = self._make_estimate(
                dt,
                0.0 if wheel_delta is None else wheel_delta,
                0.0 if imu_delta is None else imu_delta,
                blend_weight,
                source,
                yaw_rate=yaw_rate,
            )
            return self._latest

    def _make_estimate(self, dt, wheel_delta, imu_delta, imu_weight, source,
                       yaw_rate=0.0):
        self._source_counts[source] += 1
        return YawEstimate(
            yaw=self._yaw,
            yaw_rate_rad_s=yaw_rate,
            uncertainty_rad=self._uncertainty,
            learned_rate_bias_rad_s=self._learned_bias,
            wheel_delta_rad=wheel_delta,
            imu_delta_rad=imu_delta,
            imu_weight=imu_weight,
            dt_s=dt,
            source=source,
            feedback_status=self._feedback_status,
        )

    @staticmethod
    def _robust_slope(samples):
        times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        values = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        centered = times - times.mean()
        design = np.column_stack((np.ones(len(times)), centered))
        weights = np.ones(len(times), dtype=np.float64)
        coefficients = np.zeros(2, dtype=np.float64)
        for _ in range(5):
            weighted = design * np.sqrt(weights)[:, None]
            target = values * np.sqrt(weights)
            coefficients = np.linalg.lstsq(weighted, target, rcond=None)[0]
            residual = values - design @ coefficients
            scale = max(1.4826 * float(np.median(
                np.abs(residual - np.median(residual)))), math.radians(0.05))
            normalized = np.abs(residual) / (1.5 * scale)
            weights = np.ones_like(normalized)
            outliers = normalized > 1.0
            weights[outliers] = 1.0 / normalized[outliers]
        return float(coefficients[1])

    def observe_scan(self, *, time_s, heading_disagreement_rad,
                     scan_matched, score, hit_ratio, rmse_m,
                     wall_support_ratio=0.0, geometry_quality=False,
                     loop_closed=False, relocalized=False,
                     session_changed=False):
        """Observe one SLAM correction and return whether bias was updated."""
        with self._lock:
            time_s = float(time_s)
            if loop_closed or relocalized or session_changed:
                self._feedback.clear()
                reason = ("loop_closure" if loop_closed else
                          "relocalization" if relocalized else
                          "session_change")
                self._feedback_status = f"rejected:{reason}"
                self._feedback_counts[self._feedback_status] += 1
                self._refresh_latest()
                return False
            if abs(self._latest.yaw_rate_rad_s) > math.radians(
                    self.config.feedback_max_abs_yaw_rate_deg_s):
                # During a turn, yaw scale error and constant rate bias are
                # not separately observable. Never bridge a regression window
                # across that motion; scale is solved offline from CW/CCW data.
                self._feedback.clear()
                self._feedback_status = "rejected:high_yaw_rate"
                self._feedback_counts[self._feedback_status] += 1
                self._refresh_latest()
                return False
            if self._latest_linear_speed_m_s > \
                    self.config.feedback_max_linear_speed_m_s:
                # Translation couples corridor/map alignment error into yaw.
                # A stationary scan sequence is the observable condition for
                # a constant rate bias; motion scale is calibrated offline.
                self._feedback.clear()
                self._feedback_status = "rejected:translating"
                self._feedback_counts[self._feedback_status] += 1
                self._refresh_latest()
                return False
            gates = (
                (scan_matched, "scan_unmatched"),
                (_finite(score) and float(score) >=
                 self.config.feedback_min_score, "low_score"),
                (_finite(hit_ratio) and float(hit_ratio) >=
                 self.config.feedback_min_hit_ratio, "low_hit_ratio"),
                (_finite(rmse_m) and float(rmse_m) <=
                 self.config.feedback_max_rmse_m, "high_rmse"),
                (_finite(heading_disagreement_rad), "invalid_heading"),
                (_finite(wall_support_ratio) and
                 (float(wall_support_ratio) >=
                  self.config.feedback_min_wall_support_ratio or
                  geometry_quality), "weak_geometry"),
            )
            for passed, reason in gates:
                if not passed:
                    self._feedback_status = f"rejected:{reason}"
                    self._feedback_counts[self._feedback_status] += 1
                    self._refresh_latest()
                    return False
            if (self._last_discontinuity_time_s is not None and
                    time_s - self._last_discontinuity_time_s <
                    self.config.feedback_window_s):
                self._feedback_status = "rejected:recent_imu_discontinuity"
                self._feedback_counts[self._feedback_status] += 1
                self._refresh_latest()
                return False

            heading = float(heading_disagreement_rad)
            if self._feedback:
                heading = self._feedback[-1][1] + wrap_angle(
                    heading - self._feedback[-1][1])
            self._feedback.append((time_s, heading))
            oldest = time_s - self.config.feedback_window_s
            while self._feedback and self._feedback[0][0] < oldest:
                self._feedback.popleft()
            span = self._feedback[-1][0] - self._feedback[0][0]
            if span < self.config.feedback_min_span_s or len(self._feedback) < 3:
                self._feedback_status = "warming"
                self._feedback_counts[self._feedback_status] += 1
                self._refresh_latest()
                return False

            observed_bias = -self._robust_slope(self._feedback)
            max_bias = math.radians(self.config.max_rate_bias_deg_s)
            observed_bias = max(-max_bias, min(max_bias, observed_bias))
            elapsed = (
                span if self._last_feedback_bias_time_s is None
                else max(0.0, time_s - self._last_feedback_bias_time_s))
            alpha = 1.0 - math.exp(
                -elapsed / self.config.bias_time_constant_s)
            # The disagreement slope is the *remaining* rate error after the
            # currently learned bias has already been applied to propagation.
            # Integrate that residual instead of blending toward it, otherwise
            # the estimate settles at roughly half the true bias.
            self._learned_bias += alpha * observed_bias
            self._learned_bias = max(
                -max_bias, min(max_bias, self._learned_bias))
            self._last_feedback_bias_time_s = time_s
            # Start a fresh slope interval after changing propagation. Keeping
            # pre-update samples in the regression would make the controller
            # repeatedly integrate a residual caused by the old bias value.
            latest_feedback = self._feedback[-1]
            self._feedback.clear()
            self._feedback.append(latest_feedback)
            self._feedback_status = "accepted"
            self._feedback_counts[self._feedback_status] += 1
            self._refresh_latest()
            return True

    def _refresh_latest(self):
        current = self._latest
        self._latest = YawEstimate(
            yaw=current.yaw,
            yaw_rate_rad_s=current.yaw_rate_rad_s,
            uncertainty_rad=current.uncertainty_rad,
            learned_rate_bias_rad_s=self._learned_bias,
            wheel_delta_rad=current.wheel_delta_rad,
            imu_delta_rad=current.imu_delta_rad,
            imu_weight=current.imu_weight,
            dt_s=current.dt_s,
            source=current.source,
            feedback_status=self._feedback_status,
        )

    def reset_feedback(self, reason="reset"):
        with self._lock:
            self._feedback.clear()
            self._feedback_status = f"rejected:{reason}"
            self._feedback_counts[self._feedback_status] += 1
            self._refresh_latest()

    def snapshot(self):
        with self._lock:
            return self._latest

    def summary(self):
        with self._lock:
            return {
                **self._latest.diagnostics(),
                "odometry_sources": dict(self._source_counts),
                "scan_feedback_counts": dict(self._feedback_counts),
            }
