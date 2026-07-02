#include <Arduino.h>
#include <SparkFun_BNO08x_Arduino_Library.h>
#include <Wire.h>
#include <esp_timer.h>

#include <math.h>
#include <string.h>

#include "kiwi_messages.h"
#include "kiwi_uart_protocol.h"
#include "robot_config.h"

namespace {

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
  float wheelAngleRad = 0.0f;
  uint32_t updatedUs = 0;
  uint32_t samples = 0;
  uint32_t readErrors = 0;
};

EncoderState encoders[3];
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
  percent *= kiwi_config::kMotorPolarity[index] >= 0 ? 1.0f : -1.0f;
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

void seedEncoders() {
  for (uint8_t i = 0; i < 3; ++i) {
    uint16_t raw = 0;
    if (readAs5600Raw(kiwi_config::kEncoderTcaChannels[i], &raw)) {
      encoders[i].ready = true;
      encoders[i].raw = raw;
      encoders[i].lastRaw = raw;
      encoders[i].count = 0;
      encoders[i].wheelSpeedMps = 0.0f;
      encoders[i].wheelAngleRad = (static_cast<float>(raw) * kTwoPi) / kAs5600CountsPerRev;
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
      encoder.wheelAngleRad = (static_cast<float>(raw) * kTwoPi) / kAs5600CountsPerRev;
      encoder.updatedUs = nowUs;
      encoder.samples = 1;
      continue;
    }

    int32_t delta = signedAs5600Delta(raw, encoder.lastRaw);
    delta *= kiwi_config::kEncoderPolarity[i] >= 0 ? 1 : -1;
    const uint32_t dtUs = nowUs - encoder.updatedUs;
    if (dtUs > 0) {
      const float deltaRad = (static_cast<float>(delta) * kTwoPi) / kAs5600CountsPerRev;
      encoder.wheelSpeedMps = (deltaRad * kiwi_config::kWheelRadiusM) / (static_cast<float>(dtUs) * 1.0e-6f);
    }
    encoder.count += delta;
    encoder.raw = raw;
    encoder.lastRaw = raw;
    encoder.wheelAngleRad = (static_cast<float>(raw) * kTwoPi) / kAs5600CountsPerRev;
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
                  (kiwi_config::kDriveBaseRadiusM * omega);
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
  *omega = sumOmega / (3.0f * kiwi_config::kDriveBaseRadiusM);
}

void applyActiveCommand() {
  if (!commandActive ||
      millis() - lastCommandMs > max<uint32_t>(activeCommand.timeout_ms, 1)) {
    commandActive = false;
    stopMotors();
    return;
  }

  if (activeCommand.mode == static_cast<uint8_t>(VelocityMode::Stop)) {
    stopMotors();
    return;
  }

  float wheelMps[3] = {};
  wheelSpeedsFromTwist(activeCommand.vx_mps,
                       activeCommand.vy_mps,
                       activeCommand.omega_radps,
                       wheelMps);
  for (uint8_t i = 0; i < 3; ++i) {
    const float percent = (wheelMps[i] / kiwi_config::kMaxWheelSurfaceSpeedMps) * 100.0f;
    setMotorPercent(i, percent);
  }
}

bool tryBeginImuAt(uint8_t address) {
  tcaDisable();
  Wire.beginTransmission(address);
  if (Wire.endTransmission() != 0) {
    return false;
  }

  if (!imu.begin(address, Wire, kiwi_config::kImuIntPin, kiwi_config::kImuResetPin)) {
    return false;
  }
  imuAddress = address;
  return true;
}

bool initImu() {
  pinMode(kiwi_config::kImuIntPin, INPUT_PULLUP);
  pinMode(kiwi_config::kImuResetPin, OUTPUT);
  digitalWrite(kiwi_config::kImuResetPin, HIGH);
  delay(50);

  if (!tryBeginImuAt(kiwi_config::kImuPrimaryAddress) &&
      !tryBeginImuAt(kiwi_config::kImuSecondaryAddress)) {
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
    activeCommand.timeout_ms = kiwi_config::kVelocityCommandTimeoutMs;
  }
  lastCommandMs = millis();
  commandActive = activeCommand.mode != static_cast<uint8_t>(VelocityMode::Stop);
  ++commandsReceived;
  applyActiveCommand();
}

void processMasterUart() {
  Packet packet;
  while (masterReader.readFrom(MasterUart, packet)) {
    if (packet.type == MessageType::VelocityCommand) {
      handleVelocityCommand(packet);
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

  if (nowMs - lastReportMs >= kiwi_config::kFollowerReportPeriodMs) {
    lastReportMs = nowMs;
    sendTwistReport();
  }

  if (nowMs - lastSerialStatusMs >= 1000) {
    lastSerialStatusMs = nowMs;
    Serial.printf("follower status cmd=%lu bad=%lu enc_mask=0x%02x imu=%s counts=%lld/%lld/%lld heap=%lu\n",
                  static_cast<unsigned long>(commandsReceived),
                  static_cast<unsigned long>(badPackets),
                  encoderReadyMask(),
                  imuReady ? "ok" : "off",
                  static_cast<long long>(encoders[0].count),
                  static_cast<long long>(encoders[1].count),
                  static_cast<long long>(encoders[2].count),
                  static_cast<unsigned long>(ESP.getFreeHeap()));
  }

  delay(1);
}
