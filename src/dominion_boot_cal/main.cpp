#include <Arduino.h>

#include "robot_config.h"

namespace {

constexpr uint32_t kMaxHoldMs = 12000;
constexpr uint32_t kMinHoldMs = 7000;
constexpr uint32_t kCenterHoldMs = 8000;
constexpr uint32_t kStatusPeriodMs = 500;

uint16_t currentPulseUs = kiwi_config::kEscNeutralPulseUs;
uint32_t phaseStartedMs = 0;
uint32_t lastStatusMs = 0;

enum class Phase : uint8_t {
  Max,
  Min,
  Center,
  Done,
};

Phase phase = Phase::Max;

uint32_t pulseUsToDuty(uint16_t pulseUs) {
  const uint32_t maxDuty = (1UL << kiwi_config::kEscPwmResolutionBits) - 1;
  const uint32_t periodUs = 1000000UL / kiwi_config::kEscPwmFrequencyHz;
  return (static_cast<uint32_t>(pulseUs) * maxDuty + (periodUs / 2)) / periodUs;
}

// The calibration pulse train goes to all four ESC inputs: the three motor
// pins plus the aux input, so both Dominions see valid signal on both of
// their channels throughout the sequence.
void writeAll(uint16_t pulseUs) {
  currentPulseUs = constrain(pulseUs, kiwi_config::kEscMinPulseUs, kiwi_config::kEscMaxPulseUs);
  for (uint8_t i = 0; i < 3; ++i) {
    ledcWrite(kiwi_config::kMotorPwmChannels[i], pulseUsToDuty(currentPulseUs));
  }
  ledcWrite(kiwi_config::kEscAuxNeutralPwmChannel, pulseUsToDuty(currentPulseUs));
}

const char *phaseName() {
  switch (phase) {
    case Phase::Max:
      return "max_2000";
    case Phase::Min:
      return "min_1000";
    case Phase::Center:
      return "center_1500";
    case Phase::Done:
      return "done_neutral";
    default:
      return "unknown";
  }
}

void printStatus() {
  Serial.printf("boot_cal phase=%s pulse_us=%u elapsed_ms=%lu motor_gpios=%u/%u/%u aux_gpio=%u\n",
                phaseName(),
                currentPulseUs,
                static_cast<unsigned long>(millis() - phaseStartedMs),
                kiwi_config::kMotorPins[0],
                kiwi_config::kMotorPins[1],
                kiwi_config::kMotorPins[2],
                kiwi_config::kEscAuxNeutralPin);
}

void initMotorOutputs() {
  for (uint8_t i = 0; i < 3; ++i) {
    ledcSetup(kiwi_config::kMotorPwmChannels[i],
              kiwi_config::kEscPwmFrequencyHz,
              kiwi_config::kEscPwmResolutionBits);
    ledcAttachPin(kiwi_config::kMotorPins[i], kiwi_config::kMotorPwmChannels[i]);
  }
  ledcSetup(kiwi_config::kEscAuxNeutralPwmChannel,
            kiwi_config::kEscPwmFrequencyHz,
            kiwi_config::kEscPwmResolutionBits);
  ledcAttachPin(kiwi_config::kEscAuxNeutralPin, kiwi_config::kEscAuxNeutralPwmChannel);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  initMotorOutputs();

  // Deliberately start at max immediately. This firmware is only for ESC
  // calibration when the ESP32 and ESCs share the same power cycle.
  phase = Phase::Max;
  phaseStartedMs = millis();
  writeAll(kiwi_config::kEscMaxPulseUs);

  delay(250);
  Serial.println();
  Serial.println("Boot Dominion calibration firmware");
  Serial.println("Sequence: 2000 us -> 1000 us -> 1500 us -> hold neutral.");
  printStatus();
}

void loop() {
  const uint32_t nowMs = millis();
  const uint32_t phaseAgeMs = nowMs - phaseStartedMs;

  if (phase == Phase::Max && phaseAgeMs >= kMaxHoldMs) {
    phase = Phase::Min;
    phaseStartedMs = nowMs;
    writeAll(kiwi_config::kEscMinPulseUs);
    Serial.println("Switching to 1000 us minimum.");
    printStatus();
  } else if (phase == Phase::Min && phaseAgeMs >= kMinHoldMs) {
    phase = Phase::Center;
    phaseStartedMs = nowMs;
    writeAll(kiwi_config::kEscNeutralPulseUs);
    Serial.println("Switching to 1500 us center.");
    printStatus();
  } else if (phase == Phase::Center && phaseAgeMs >= kCenterHoldMs) {
    phase = Phase::Done;
    phaseStartedMs = nowMs;
    writeAll(kiwi_config::kEscNeutralPulseUs);
    Serial.println("Calibration timing complete. Power-cycle to save/apply, then flash follower_motor_ramp again.");
    printStatus();
  }

  if (nowMs - lastStatusMs >= kStatusPeriodMs) {
    lastStatusMs = nowMs;
    printStatus();
  }

  delay(1);
}
