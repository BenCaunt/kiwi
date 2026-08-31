#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace kiwi {

enum class MessageType : uint8_t {
  VelocityCommand = 1,
  TwistReport = 2,
  Heartbeat = 3,
  DriveParams = 4,
  DriveParamsAck = 5,
};

enum class VelocityMode : uint8_t {
  BodyTwist = 0,
  Stop = 1,
};

struct __attribute__((packed)) VelocityCommandPayload {
  uint64_t master_time_us;
  float vx_mps;
  float vy_mps;
  float omega_radps;
  uint16_t timeout_ms;
  uint8_t mode;
  uint8_t reserved;
};

struct __attribute__((packed)) TwistReportPayload {
  uint64_t follower_time_us;
  uint32_t report_seq;
  float measured_vx_mps;
  float measured_vy_mps;
  float measured_omega_radps;
  float command_vx_mps;
  float command_vy_mps;
  float command_omega_radps;
  float wheel_speed_mps[3];
  float wheel_angle_rad[3];
  int64_t encoder_count[3];
  uint8_t imu_ready;
  uint8_t encoder_ready_mask;
  uint16_t status_flags;
  // BNO08x magnetometer-free Game Rotation Vector (i, j, k, real) and
  // calibrated accelerometer. SLAM uses the quaternion relative to startup.
  float imu_quat_ijkr[4];
  float imu_accel_mps2[3];
};

// Runtime-tunable drive parameters, master -> follower. The follower applies
// them immediately, persists them to NVS, and replies with DriveParamsAck.
// Compiled robot_config.h values are first-boot defaults only. The master
// resends until the acked version matches its stored version.
struct __attribute__((packed)) DriveParamsPayload {
  uint32_t version;
  float wheel_radius_m;
  float drive_base_radius_m;
  // Doubles as the velocity feedforward gain: percent = 100 * target / max.
  // Calibrate to the true no-load top wheel speed for accurate feedforward.
  float max_wheel_surface_speed_mps;
  int8_t motor_polarity[3];
  // ESC breakaway compensation: commands are remapped so |percent| starts at
  // this value. 0 disables. Measured per motor (they vary 30-70%).
  uint8_t motor_deadband_pct[3];
  uint16_t velocity_command_timeout_ms;
  // PI feedback on encoder-measured wheel speed, on top of the feedforward.
  float pid_kp;  // percent per (m/s) of speed error
  float pid_ki;  // percent per (m/s * s) of integrated error
  uint8_t closed_loop;  // 0 = feedforward only, 1 = feedforward + PI
  // Encoder count sign per wheel. Flip together with motor_polarity (same
  // packet = atomic) when a wheel is physically reversed, or the closed
  // loop's feedback inverts and fights itself.
  int8_t encoder_polarity[3];
};

struct __attribute__((packed)) DriveParamsAckPayload {
  uint32_t version;
};

static_assert(sizeof(VelocityCommandPayload) == 24, "Unexpected velocity command size");
static_assert(sizeof(TwistReportPayload) == 116, "Unexpected twist report size");
static_assert(sizeof(DriveParamsPayload) == 36, "Unexpected drive params size");
static_assert(sizeof(DriveParamsAckPayload) == 4, "Unexpected drive params ack size");

// Shared sanity bounds so master and follower reject the same nonsense.
inline bool driveParamsValid(const DriveParamsPayload &params) {
  if (!isfinite(params.wheel_radius_m) ||
      params.wheel_radius_m <= 0.001f || params.wheel_radius_m > 0.5f) {
    return false;
  }
  if (!isfinite(params.drive_base_radius_m) ||
      params.drive_base_radius_m <= 0.01f || params.drive_base_radius_m > 1.0f) {
    return false;
  }
  if (!isfinite(params.max_wheel_surface_speed_mps) ||
      params.max_wheel_surface_speed_mps <= 0.05f ||
      params.max_wheel_surface_speed_mps > 10.0f) {
    return false;
  }
  for (uint8_t i = 0; i < 3; ++i) {
    if (params.motor_polarity[i] != 1 && params.motor_polarity[i] != -1) {
      return false;
    }
    if (params.motor_deadband_pct[i] > 90) {
      return false;
    }
    if (params.encoder_polarity[i] != 1 && params.encoder_polarity[i] != -1) {
      return false;
    }
  }
  if (params.velocity_command_timeout_ms < 20 ||
      params.velocity_command_timeout_ms > 10000) {
    return false;
  }
  if (!isfinite(params.pid_kp) || params.pid_kp < 0.0f || params.pid_kp > 500.0f) {
    return false;
  }
  if (!isfinite(params.pid_ki) || params.pid_ki < 0.0f || params.pid_ki > 2000.0f) {
    return false;
  }
  if (params.closed_loop > 1) {
    return false;
  }
  return true;
}

}  // namespace kiwi
