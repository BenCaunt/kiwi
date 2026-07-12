#include <Arduino.h>
#include <Preferences.h>
#include <SparkFun_BNO08x_Arduino_Library.h>
#include <Wire.h>
#include <esp_timer.h>

#include <math.h>
#include <string.h>

#include "kiwi_messages.h"
#include "kiwi_uart_protocol.h"
#include "robot_config.h"

namespace {

using kiwi::DriveParamsAckPayload;
using kiwi::DriveParamsPayload;
using kiwi::MessageType;
using kiwi::Packet;
using kiwi::PacketReader;
using kiwi::TwistReportPayload;
using kiwi::VelocityCommandPayload;
using kiwi::VelocityMode;

constexpr float kTwoPi = 6.28318530718f;
constexpr int32_t kAs5600CountsPerRev = 4096;
constexpr int32_t kAs5600HalfCounts = kAs5600CountsPerRev / 2;

HardwareSerial MasterUart(1);
PacketReader masterReader;
BNO08x imu;

struct EncoderState {
  bool ready = false;
  uint16_t raw = 0;
  uint16_t lastRaw = 0;
  int64_t count = 0;
  float wheelSpeedMps = 0.0f;
  float filteredSpeedMps = 0.0f;  // EMA for the control loop
  float wheelAngleRad = 0.0f;
  uint32_t updatedUs = 0;
  uint32_t samples = 0;
  uint32_t readErrors = 0;
};

EncoderState encoders[3];
// Runtime drive parameters. Compiled robot_config.h values are first-boot
// defaults; the master can update these over UART and they persist in NVS.
DriveParamsPayload driveParams = {};
uint32_t persistedDriveParamsVersion = 0;
Preferences configPrefs;
VelocityCommandPayload activeCommand = {};
uint32_t lastCommandMs = 0;
bool commandActive = false;
bool imuReady = false;
bool imuReportsEnabled = false;
uint8_t imuAddress = 0;
uint16_t reportTxSeq = 0;
uint32_t reportSeq = 0;
uint32_t commandsReceived = 0;
uint32_t badPackets = 0;
uint32_t lastEncoderSampleMs = 0;
uint32_t lastReportMs = 0;
uint32_t lastSerialStatusMs = 0;

// Wheel velocity control: feedforward from max_wheel_surface_speed plus an
// optional PI on the encoder-measured speed (driveParams.closed_loop).
constexpr uint32_t kControlPeriodMs = 10;
constexpr float kIntegralTermLimitPct = 40.0f;
float wheelTargetsMps[3] = {};
float wheelIntegrals[3] = {};
uint32_t lastControlMs = 0;

void loadDriveParams() {
  driveParams.version = 0;
  driveParams.wheel_radius_m = kiwi_config::kWheelRadiusM;
  driveParams.drive_base_radius_m = kiwi_config::kDriveBaseRadiusM;
  driveParams.max_wheel_surface_speed_mps = kiwi_config::kMaxWheelSurfaceSpeedMps;
  for (uint8_t i = 0; i < 3; ++i) {
    driveParams.motor_polarity[i] = kiwi_config::kMotorPolarity[i];
    driveParams.motor_deadband_pct[i] = 0;
    driveParams.encoder_polarity[i] = kiwi_config::kEncoderPolarity[i];
  }
  driveParams.velocity_command_timeout_ms = kiwi_config::kVelocityCommandTimeoutMs;
  driveParams.pid_kp = 0.0f;
  driveParams.pid_ki = 0.0f;
  driveParams.closed_loop = 0;

  configPrefs.begin("kiwi", true);
  DriveParamsPayload stored = {};
  const size_t len = configPrefs.getBytes("drive", &stored, sizeof(stored));
  configPrefs.end();
  if (len == sizeof(stored) && driveParamsValid(stored)) {
    driveParams = stored;
    persistedDriveParamsVersion = stored.version;
    Serial.printf("Drive params loaded from NVS, version=%lu\n",
                  static_cast<unsigned long>(stored.version));
  } else {
    Serial.println("Drive params: using compiled defaults (no valid NVS entry).");
  }
  Serial.printf("Drive params: wheel_r=%.4f base_r=%.4f max_v=%.2f pol=%d/%d/%d timeout=%u\n",
                driveParams.wheel_radius_m,
                driveParams.drive_base_radius_m,
                driveParams.max_wheel_surface_speed_mps,
                driveParams.motor_polarity[0],
                driveParams.motor_polarity[1],
                driveParams.motor_polarity[2],
                driveParams.velocity_command_timeout_ms);
}

void persistDriveParams() {
  configPrefs.begin("kiwi", false);
  configPrefs.putBytes("drive", &driveParams, sizeof(driveParams));
  configPrefs.end();
  persistedDriveParamsVersion = driveParams.version;
}

uint32_t pulseUsToDuty(uint16_t pulseUs) {
  const uint32_t maxDuty = (1UL << kiwi_config::kEscPwmResolutionBits) - 1;
  const uint32_t periodUs = 1000000UL / kiwi_config::kEscPwmFrequencyHz;
  return (static_cast<uint32_t>(pulseUs) * maxDuty + (periodUs / 2)) / periodUs;
}

uint16_t motorPercentToPulseUs(float percent) {
  percent = constrain(percent, -100.0f, 100.0f);
  if (percent >= 0.0f) {
    return kiwi_config::kEscNeutralPulseUs +
           static_cast<uint16_t>(((kiwi_config::kEscMaxPulseUs - kiwi_config::kEscNeutralPulseUs) * percent) / 100.0f);
  }
  return kiwi_config::kEscNeutralPulseUs -
         static_cast<uint16_t>(((kiwi_config::kEscNeutralPulseUs - kiwi_config::kEscMinPulseUs) * -percent) / 100.0f);
}

void setMotorPercent(uint8_t index, float percent) {
  if (index >= 3) {
    return;
  }
  // Deadband compensation: remap nonzero commands to start at the motor's
  // measured breakaway percentage so slow twists actually move every wheel.
  const float deadband = driveParams.motor_deadband_pct[index];
  if (deadband > 0.0f && percent != 0.0f) {
    const float magnitude = constrain(fabsf(percent), 0.0f, 100.0f);
    percent = copysignf(deadband + ((100.0f - deadband) * magnitude) / 100.0f,
                        percent);
  }
  percent *= driveParams.motor_polarity[index] >= 0 ? 1.0f : -1.0f;
  const uint16_t pulseUs = motorPercentToPulseUs(percent);
  ledcWrite(kiwi_config::kMotorPwmChannels[index], pulseUsToDuty(pulseUs));
}

void stopMotors() {
  for (uint8_t i = 0; i < 3; ++i) {
    setMotorPercent(i, 0.0f);
  }
}

void initMotors() {
  for (uint8_t i = 0; i < 3; ++i) {
    ledcSetup(kiwi_config::kMotorPwmChannels[i],
              kiwi_config::kEscPwmFrequencyHz,
              kiwi_config::kEscPwmResolutionBits);
    ledcAttachPin(kiwi_config::kMotorPins[i], kiwi_config::kMotorPwmChannels[i]);
  }

  // The second Dominion only arms when its unused channel input also sees a
  // valid centered pulse; D3 permanently holds neutral for it.
  ledcSetup(kiwi_config::kEscAuxNeutralPwmChannel,
            kiwi_config::kEscPwmFrequencyHz,
            kiwi_config::kEscPwmResolutionBits);
  ledcAttachPin(kiwi_config::kEscAuxNeutralPin, kiwi_config::kEscAuxNeutralPwmChannel);
  ledcWrite(kiwi_config::kEscAuxNeutralPwmChannel,
            pulseUsToDuty(kiwi_config::kEscNeutralPulseUs));

  stopMotors();
  Serial.printf("Motor PWM ready on GPIO%u/GPIO%u/GPIO%u, aux neutral on GPIO%u\n",
                kiwi_config::kMotorPins[0],
                kiwi_config::kMotorPins[1],
                kiwi_config::kMotorPins[2],
                kiwi_config::kEscAuxNeutralPin);
}

bool tcaSelect(uint8_t channel) {
  if (channel > 7) {
    return false;
  }
  Wire.beginTransmission(kiwi_config::kTca9548aAddress);
  Wire.write(static_cast<uint8_t>(1U << channel));
  return Wire.endTransmission() == 0;
}

void tcaDisable() {
  Wire.beginTransmission(kiwi_config::kTca9548aAddress);
  Wire.write(static_cast<uint8_t>(0));
  Wire.endTransmission();
}

bool readAs5600Raw(uint8_t channel, uint16_t *raw) {
  if (!tcaSelect(channel)) {
    return false;
  }

  Wire.beginTransmission(kiwi_config::kAs5600Address);
  Wire.write(static_cast<uint8_t>(0x0c));
  if (Wire.endTransmission(false) != 0) {
    tcaDisable();
    return false;
  }

  const uint8_t requested = Wire.requestFrom(kiwi_config::kAs5600Address, static_cast<uint8_t>(2));
  if (requested != 2) {
    tcaDisable();
    return false;
  }

  const uint8_t high = Wire.read();
  const uint8_t low = Wire.read();
  *raw = static_cast<uint16_t>(((high & 0x0f) << 8) | low);
  tcaDisable();
  return true;
}

int32_t signedAs5600Delta(uint16_t current, uint16_t previous) {
  int32_t delta = static_cast<int32_t>(current) - static_cast<int32_t>(previous);
  if (delta > kAs5600HalfCounts) {
    delta -= kAs5600CountsPerRev;
  } else if (delta < -kAs5600HalfCounts) {
    delta += kAs5600CountsPerRev;
  }
  return delta;
}

float wheelAngleFromRaw(uint8_t index, uint16_t raw) {
  if (driveParams.encoder_polarity[index] < 0) {
    raw = static_cast<uint16_t>((kAs5600CountsPerRev - raw) % kAs5600CountsPerRev);
  }
  return (static_cast<float>(raw) * kTwoPi) / kAs5600CountsPerRev;
}

void seedEncoders() {
  for (uint8_t i = 0; i < 3; ++i) {
    uint16_t raw = 0;
    if (readAs5600Raw(kiwi_config::kEncoderTcaChannels[i], &raw)) {
      encoders[i].ready = true;
      encoders[i].raw = raw;
      encoders[i].lastRaw = raw;
      encoders[i].count = 0;
      encoders[i].wheelSpeedMps = 0.0f;
      encoders[i].wheelAngleRad = wheelAngleFromRaw(i, raw);
      encoders[i].updatedUs = static_cast<uint32_t>(esp_timer_get_time());
      encoders[i].samples = 1;
    } else {
      ++encoders[i].readErrors;
    }
  }
}

void updateEncoders() {
  const uint32_t nowUs = static_cast<uint32_t>(esp_timer_get_time());
  for (uint8_t i = 0; i < 3; ++i) {
    uint16_t raw = 0;
    if (!readAs5600Raw(kiwi_config::kEncoderTcaChannels[i], &raw)) {
      ++encoders[i].readErrors;
      encoders[i].wheelSpeedMps = 0.0f;
      continue;
    }

    EncoderState &encoder = encoders[i];
    if (!encoder.ready) {
      encoder.ready = true;
      encoder.raw = raw;
      encoder.lastRaw = raw;
      encoder.wheelAngleRad = wheelAngleFromRaw(i, raw);
      encoder.updatedUs = nowUs;
      encoder.samples = 1;
      continue;
    }

    int32_t delta = signedAs5600Delta(raw, encoder.lastRaw);
    delta *= driveParams.encoder_polarity[i] >= 0 ? 1 : -1;
    const uint32_t dtUs = nowUs - encoder.updatedUs;
    if (dtUs > 0) {
      const float deltaRad = (static_cast<float>(delta) * kTwoPi) / kAs5600CountsPerRev;
      encoder.wheelSpeedMps = (deltaRad * driveParams.wheel_radius_m) / (static_cast<float>(dtUs) * 1.0e-6f);
      encoder.filteredSpeedMps += 0.3f * (encoder.wheelSpeedMps - encoder.filteredSpeedMps);
    }
    encoder.count += delta;
    encoder.raw = raw;
    encoder.lastRaw = raw;
    encoder.wheelAngleRad = wheelAngleFromRaw(i, raw);
    encoder.updatedUs = nowUs;
    ++encoder.samples;
  }
}

uint8_t encoderReadyMask() {
  uint8_t mask = 0;
  for (uint8_t i = 0; i < 3; ++i) {
    if (encoders[i].ready) {
      mask |= static_cast<uint8_t>(1U << i);
    }
  }
  return mask;
}

void wheelSpeedsFromTwist(float vx, float vy, float omega, float *wheelMps) {
  for (uint8_t i = 0; i < 3; ++i) {
    const float theta = kiwi_config::kWheelAnglesRad[i];
    wheelMps[i] = (-sinf(theta) * vx) + (cosf(theta) * vy) +
                  (driveParams.drive_base_radius_m * omega);
  }
}

void twistFromWheelSpeeds(const float *wheelMps, float *vx, float *vy, float *omega) {
  float sumVx = 0.0f;
  float sumVy = 0.0f;
  float sumOmega = 0.0f;
  for (uint8_t i = 0; i < 3; ++i) {
    const float theta = kiwi_config::kWheelAnglesRad[i];
    sumVx += -sinf(theta) * wheelMps[i];
    sumVy += cosf(theta) * wheelMps[i];
    sumOmega += wheelMps[i];
  }
  *vx = (2.0f / 3.0f) * sumVx;
  *vy = (2.0f / 3.0f) * sumVy;
  *omega = sumOmega / (3.0f * driveParams.drive_base_radius_m);
}

void applyActiveCommand() {
  if (!commandActive ||
      millis() - lastCommandMs > max<uint32_t>(activeCommand.timeout_ms, 1)) {
    commandActive = false;
  }

  if (!commandActive ||
      activeCommand.mode == static_cast<uint8_t>(VelocityMode::Stop)) {
    for (uint8_t i = 0; i < 3; ++i) {
      wheelTargetsMps[i] = 0.0f;
      wheelIntegrals[i] = 0.0f;
    }
    stopMotors();
    return;
  }

  wheelSpeedsFromTwist(activeCommand.vx_mps,
                       activeCommand.vy_mps,
                       activeCommand.omega_radps,
                       wheelTargetsMps);
}

void updateWheelControl(uint32_t nowMs) {
  if (nowMs - lastControlMs < kControlPeriodMs) {
    return;
  }
  const float dtS = static_cast<float>(nowMs - lastControlMs) * 1.0e-3f;
  lastControlMs = nowMs;

  if (!commandActive) {
    return;  // applyActiveCommand already forced neutral + cleared integrals
  }

  for (uint8_t i = 0; i < 3; ++i) {
    const float target = wheelTargetsMps[i];
    if (fabsf(target) < 0.01f) {
      wheelIntegrals[i] = 0.0f;
      setMotorPercent(i, 0.0f);
      continue;
    }

    float percent = (target / driveParams.max_wheel_surface_speed_mps) * 100.0f;
    if (driveParams.closed_loop != 0) {
      const float error = target - encoders[i].filteredSpeedMps;
      if (driveParams.pid_ki > 0.0f) {
        const float limit = kIntegralTermLimitPct / driveParams.pid_ki;
        wheelIntegrals[i] = constrain(wheelIntegrals[i] + error * dtS, -limit, limit);
      }
      percent += driveParams.pid_kp * error + driveParams.pid_ki * wheelIntegrals[i];
    }
    setMotorPercent(i, constrain(percent, -100.0f, 100.0f));
  }
}

bool tryBeginImuAt(uint8_t address) {
  tcaDisable();
  Wire.beginTransmission(address);
  const uint8_t err = Wire.endTransmission();
  if (err != 0) {
    Serial.printf("BNO08x probe 0x%02x: i2c err=%u (2=NACK, 5=timeout)\n", address, err);
    return false;
  }

  // Reset is handled manually in initImu (the library's own hardware reset
  // probes the chip again before it finishes booting). Passing -1 for INT and
  // RESET keeps the library in polled mode.
  if (!imu.begin(address, Wire, -1, -1)) {
    return false;
  }
  imuAddress = address;
  return true;
}

bool initImu() {
  pinMode(kiwi_config::kImuIntPin, INPUT_PULLUP);
  pinMode(kiwi_config::kImuResetPin, OUTPUT);
  // Clean reset pulse, then wait out the BNO08x boot time (~100-150 ms) before
  // probing; probing too early NACKs even with correct wiring.
  digitalWrite(kiwi_config::kImuResetPin, LOW);
  delay(20);
  digitalWrite(kiwi_config::kImuResetPin, HIGH);
  delay(1000);

  bool found = false;
  for (uint8_t attempt = 0; attempt < 3 && !found; ++attempt) {
    if (attempt > 0) {
      delay(150);
    }
    found = tryBeginImuAt(kiwi_config::kImuPrimaryAddress) ||
            tryBeginImuAt(kiwi_config::kImuSecondaryAddress);
  }
  if (!found) {
    Serial.println("BNO08x not found on follower I2C root bus.");
    return false;
  }

  const bool rotationOk = imu.enableRotationVector(kiwi_config::kImuReportIntervalMs);
  delay(10);
  const bool accelOk = imu.enableAccelerometer(kiwi_config::kImuReportIntervalMs);
  imuReportsEnabled = rotationOk || accelOk;
  Serial.printf("BNO08x ready at 0x%02x, reports=%s\n",
                imuAddress,
                imuReportsEnabled ? "enabled" : "failed");
  return true;
}

void updateImu() {
  if (!imuReady) {
    return;
  }
  if (imu.wasReset()) {
    imuReportsEnabled = imu.enableRotationVector(kiwi_config::kImuReportIntervalMs);
    imuReportsEnabled = imu.enableAccelerometer(kiwi_config::kImuReportIntervalMs) || imuReportsEnabled;
  }
  for (uint8_t i = 0; i < 4 && imu.getSensorEvent(); ++i) {
    // Reports are consumed here so the BNO08x FIFO does not back up. The first
    // robot-facing telemetry contract is wheel odometry; IMU fields can be
    // added to TwistReportPayload once mounting orientation is fixed.
  }
}

void handleVelocityCommand(const Packet &packet) {
  if (packet.payloadLength != sizeof(VelocityCommandPayload)) {
    ++badPackets;
    return;
  }

  memcpy(&activeCommand, packet.payload, sizeof(activeCommand));
  if (!isfinite(activeCommand.vx_mps) ||
      !isfinite(activeCommand.vy_mps) ||
      !isfinite(activeCommand.omega_radps)) {
    ++badPackets;
    commandActive = false;
    stopMotors();
    return;
  }

  if (activeCommand.timeout_ms == 0) {
    activeCommand.timeout_ms = driveParams.velocity_command_timeout_ms;
  }
  lastCommandMs = millis();
  commandActive = activeCommand.mode != static_cast<uint8_t>(VelocityMode::Stop);
  ++commandsReceived;
  applyActiveCommand();
}

void sendDriveParamsAck() {
  DriveParamsAckPayload ack = {};
  ack.version = driveParams.version;
  kiwi::writePacket(MasterUart,
                    MessageType::DriveParamsAck,
                    reportTxSeq++,
                    &ack,
                    sizeof(ack));
}

void handleDriveParams(const Packet &packet) {
  if (packet.payloadLength != sizeof(DriveParamsPayload)) {
    ++badPackets;
    return;
  }

  DriveParamsPayload incoming = {};
  memcpy(&incoming, packet.payload, sizeof(incoming));
  if (!driveParamsValid(incoming)) {
    ++badPackets;
    // Ack the currently active version so the master sees the rejection.
    sendDriveParamsAck();
    return;
  }

  driveParams = incoming;
  if (driveParams.version != persistedDriveParamsVersion) {
    persistDriveParams();
    Serial.printf("Drive params updated + persisted, version=%lu\n",
                  static_cast<unsigned long>(driveParams.version));
  }
  sendDriveParamsAck();
}

void processMasterUart() {
  Packet packet;
  while (masterReader.readFrom(MasterUart, packet)) {
    if (packet.type == MessageType::VelocityCommand) {
      handleVelocityCommand(packet);
    } else if (packet.type == MessageType::DriveParams) {
      handleDriveParams(packet);
    } else {
      ++badPackets;
    }
  }
}

void sendTwistReport() {
  float wheelMps[3] = {
      encoders[0].wheelSpeedMps,
      encoders[1].wheelSpeedMps,
      encoders[2].wheelSpeedMps,
  };
  float measuredVx = 0.0f;
  float measuredVy = 0.0f;
  float measuredOmega = 0.0f;
  twistFromWheelSpeeds(wheelMps, &measuredVx, &measuredVy, &measuredOmega);

  TwistReportPayload report = {};
  report.follower_time_us = static_cast<uint64_t>(esp_timer_get_time());
  report.report_seq = reportSeq++;
  report.measured_vx_mps = measuredVx;
  report.measured_vy_mps = measuredVy;
  report.measured_omega_radps = measuredOmega;
  if (commandActive) {
    report.command_vx_mps = activeCommand.vx_mps;
    report.command_vy_mps = activeCommand.vy_mps;
    report.command_omega_radps = activeCommand.omega_radps;
  }
  for (uint8_t i = 0; i < 3; ++i) {
    report.wheel_speed_mps[i] = encoders[i].wheelSpeedMps;
    report.wheel_angle_rad[i] = encoders[i].wheelAngleRad;
    report.encoder_count[i] = encoders[i].count;
  }
  report.imu_ready = imuReady ? 1 : 0;
  report.encoder_ready_mask = encoderReadyMask();
  report.status_flags = 0;
  if (!commandActive) {
    report.status_flags |= 0x0001;
  }
  if (report.encoder_ready_mask != 0x07) {
    report.status_flags |= 0x0002;
  }
  if (!imuReady) {
    report.status_flags |= 0x0004;
  }

  kiwi::writePacket(MasterUart,
                    MessageType::TwistReport,
                    reportTxSeq++,
                    &report,
                    sizeof(report));
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("Booting kiwi follower: motors + AS5600 encoders + BNO08x IMU + master UART");

  MasterUart.begin(kiwi_config::kFollowerUartBaud,
                   SERIAL_8N1,
                   kiwi_config::kFollowerUartRxPin,
                   kiwi_config::kFollowerUartTxPin);
  Serial.printf("Master UART RX=GPIO%u TX=GPIO%u baud=%lu\n",
                kiwi_config::kFollowerUartRxPin,
                kiwi_config::kFollowerUartTxPin,
                static_cast<unsigned long>(kiwi_config::kFollowerUartBaud));

  loadDriveParams();

  Wire.begin(kiwi_config::kI2cSdaPin, kiwi_config::kI2cSclPin);
  Wire.setClock(kiwi_config::kI2cClockHz);
  Wire.setTimeOut(50);
  tcaDisable();

  initMotors();
  seedEncoders();
  imuReady = initImu();
}

void loop() {
  processMasterUart();

  const uint32_t nowMs = millis();
  if (nowMs - lastEncoderSampleMs >= kiwi_config::kEncoderSamplePeriodMs) {
    lastEncoderSampleMs = nowMs;
    updateEncoders();
  }

  updateImu();
  applyActiveCommand();
  updateWheelControl(nowMs);

  if (nowMs - lastReportMs >= kiwi_config::kFollowerReportPeriodMs) {
    lastReportMs = nowMs;
    sendTwistReport();
  }

  if (nowMs - lastSerialStatusMs >= 1000) {
    lastSerialStatusMs = nowMs;
    Serial.printf("follower status cmd=%lu bad=%lu enc_mask=0x%02x imu=%s dpv=%lu counts=%lld/%lld/%lld enc_err=%lu/%lu/%lu heap=%lu\n",
                  static_cast<unsigned long>(commandsReceived),
                  static_cast<unsigned long>(badPackets),
                  encoderReadyMask(),
                  imuReady ? "ok" : "off",
                  static_cast<unsigned long>(driveParams.version),
                  static_cast<long long>(encoders[0].count),
                  static_cast<long long>(encoders[1].count),
                  static_cast<long long>(encoders[2].count),
                  static_cast<unsigned long>(encoders[0].readErrors),
                  static_cast<unsigned long>(encoders[1].readErrors),
                  static_cast<unsigned long>(encoders[2].readErrors),
                  static_cast<unsigned long>(ESP.getFreeHeap()));
  }

  delay(1);
}
