#pragma once

#include <stddef.h>
#include <stdint.h>

namespace kiwi {

enum class MessageType : uint8_t {
  VelocityCommand = 1,
  TwistReport = 2,
  Heartbeat = 3,
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
};

static_assert(sizeof(VelocityCommandPayload) == 24, "Unexpected velocity command size");
static_assert(sizeof(TwistReportPayload) == 88, "Unexpected twist report size");

}  // namespace kiwi
