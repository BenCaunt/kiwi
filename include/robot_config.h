#pragma once

#include <Arduino.h>

#ifndef ROBOT_NAMESPACE
#define ROBOT_NAMESPACE "kiwi/xiao"
#endif

namespace kiwi_config {

// Both boards use the same logical UART pins. Cross TX/RX between boards.
constexpr uint8_t kFollowerUartRxPin = D6;
constexpr uint8_t kFollowerUartTxPin = D7;
constexpr uint32_t kFollowerUartBaud = 460800;

// Master-only LD19/FHL-LD19 lidar serial. LD19-style units commonly stream
// fixed 47-byte frames at 230400 8N1.
constexpr uint8_t kLidarRxPin = D4;
constexpr uint8_t kLidarTxPin = D5;
constexpr uint32_t kLidarBaud = 230400;

// Follower I2C bus: TCA9548A mux plus BNO08x on the root bus.
constexpr uint8_t kI2cSdaPin = D4;
constexpr uint8_t kI2cSclPin = D5;
constexpr uint32_t kI2cClockHz = 400000;
constexpr uint8_t kTca9548aAddress = 0x70;
constexpr uint8_t kAs5600Address = 0x36;
// Measured 2026-07-06 with the motor_encoder_map test: motor i drives the
// wheel whose AS5600 sits on mux channel kEncoderTcaChannels[i]. Polarity is
// chosen so a positive motor command increases the reported count.
constexpr uint8_t kEncoderTcaChannels[3] = {1, 0, 2};
constexpr int8_t kEncoderPolarity[3] = {-1, 1, -1};

// Follower motor outputs (same mapping run): the second Dominion's DRIVEN
// input is on D3 and its arming input is on D1, so D3 is a motor pin here.
constexpr uint8_t kMotorPins[3] = {D0, D2, D3};
constexpr uint8_t kMotorPwmChannels[3] = {0, 1, 2};
// The second Dominion's unused channel input is wired to D1. Dominion dual
// ESCs only arm once BOTH channel inputs see a valid centered pulse, so this
// pin must always output 1500 us.
constexpr uint8_t kEscAuxNeutralPin = D1;
constexpr uint8_t kEscAuxNeutralPwmChannel = 3;
constexpr int8_t kMotorPolarity[3] = {1, 1, 1};
constexpr uint32_t kEscPwmFrequencyHz = 50;
constexpr uint8_t kEscPwmResolutionBits = 14;
constexpr uint16_t kEscMinPulseUs = 1000;
constexpr uint16_t kEscNeutralPulseUs = 1500;
constexpr uint16_t kEscMaxPulseUs = 2000;

// Kiwi geometry. Wheel angles are the wheel module positions around the robot,
// measured from robot +X, with each omni wheel driving tangentially.
constexpr float kWheelRadiusM = 0.025f;
constexpr float kDriveBaseRadiusM = 0.090f;
constexpr float kMaxWheelSurfaceSpeedMps = 1.0f;
constexpr float kWheelAnglesRad[3] = {
    0.0f,
    2.09439510239f,
    4.18879020479f,
};

// Follower control/report periods.
constexpr uint32_t kEncoderSamplePeriodMs = 5;
constexpr uint32_t kFollowerReportPeriodMs = 20;
constexpr uint32_t kVelocityCommandTimeoutMs = 250;

// BNO08x defaults from the existing robot project.
constexpr uint8_t kImuPrimaryAddress = 0x4B;
constexpr uint8_t kImuSecondaryAddress = 0x4A;
constexpr uint8_t kImuIntPin = D9;
constexpr uint8_t kImuResetPin = D8;
constexpr uint16_t kImuReportIntervalMs = 20;

// Master publish periods.
constexpr uint32_t kCameraPublishPeriodMs = 100;
constexpr uint32_t kMasterStatusPeriodMs = 1000;

}  // namespace kiwi_config
