#include <Arduino.h>

#include "robot_config.h"

namespace {

// All four candidate ESC signal pins are driven as equal channels. The exact
// pin-to-motor/ESC mapping is unknown, so every pin gets a valid neutral pulse
// at boot (Dominion dual ESCs only arm when BOTH channel inputs see a valid
// centered pulse) and every pin can be pulsed individually to map the wiring.
constexpr uint8_t kNumChannels = 4;
constexpr uint8_t kChannelPins[kNumChannels] = {D0, D1, D2, D3};
constexpr uint8_t kChannelPwmChannels[kNumChannels] = {0, 1, 2, 3};
constexpr uint8_t kAllChannelsMask = 0x0F;

constexpr float kDefaultRampLimitPercent = 20.0f;
constexpr float kRampStepPercent = 1.0f;
constexpr uint32_t kRampStepPeriodMs = 75;
constexpr uint32_t kStatusPeriodMs = 500;
constexpr uint32_t kEscArmHoldMs = 7000;
// Failsafe: a direct pulse hold reverts to neutral after this long unless the
// command is re-sent. Prevents a runaway motor if the USB link dies mid-test.
constexpr uint32_t kDirectPulseTimeoutMs = 8000;
// The Dominion needs ~5 s at max to enter calibration plus ~5 s to latch the
// max point, measured from ESC power-on. Leave slack for the human flipping
// the battery switch after the max hold begins.
constexpr uint32_t kCalibrationMaxHoldMs = 16000;
constexpr uint32_t kCalibrationMinHoldMs = 7000;
constexpr uint32_t kCalibrationCenterHoldMs = 8000;

enum class RampPhase : uint8_t {
  Idle,
  PositiveUp,
  PositiveDown,
  NegativeDown,
  NegativeUp,
  DirectPulse,
  CalibrationMax,
  CalibrationMin,
  CalibrationCenter,
};

RampPhase rampPhase = RampPhase::Idle;
float rampLimitPercent = kDefaultRampLimitPercent;
float currentPercent = 0.0f;
uint8_t activeChannelMask = kAllChannelsMask;
uint16_t currentPulseUs[kNumChannels] = {
    kiwi_config::kEscNeutralPulseUs,
    kiwi_config::kEscNeutralPulseUs,
    kiwi_config::kEscNeutralPulseUs,
    kiwi_config::kEscNeutralPulseUs,
};
uint32_t lastRampStepMs = 0;
uint32_t lastStatusMs = 0;
uint32_t phaseStartedMs = 0;
String commandLine;

uint32_t pulseUsToDuty(uint16_t pulseUs) {
  const uint32_t maxDuty = (1UL << kiwi_config::kEscPwmResolutionBits) - 1;
  const uint32_t periodUs = 1000000UL / kiwi_config::kEscPwmFrequencyHz;
  return (static_cast<uint32_t>(pulseUs) * maxDuty + (periodUs / 2)) / periodUs;
}

uint16_t percentToPulseUs(float percent) {
  percent = constrain(percent, -100.0f, 100.0f);
  if (percent >= 0.0f) {
    return kiwi_config::kEscNeutralPulseUs +
           static_cast<uint16_t>(((kiwi_config::kEscMaxPulseUs - kiwi_config::kEscNeutralPulseUs) * percent) / 100.0f);
  }
  return kiwi_config::kEscNeutralPulseUs -
         static_cast<uint16_t>(((kiwi_config::kEscNeutralPulseUs - kiwi_config::kEscMinPulseUs) * -percent) / 100.0f);
}

void writeChannelPulseUs(uint8_t channelIndex, uint16_t pulseUs) {
  if (channelIndex >= kNumChannels) {
    return;
  }
  pulseUs = constrain(pulseUs, kiwi_config::kEscMinPulseUs, kiwi_config::kEscMaxPulseUs);
  currentPulseUs[channelIndex] = pulseUs;
  ledcWrite(kChannelPwmChannels[channelIndex], pulseUsToDuty(pulseUs));
}

void applyPercentToMask(float percent, uint8_t channelMask) {
  for (uint8_t i = 0; i < kNumChannels; ++i) {
    const bool active = (channelMask & (1U << i)) != 0;
    writeChannelPulseUs(i, active ? percentToPulseUs(percent) : kiwi_config::kEscNeutralPulseUs);
  }
}

void applyPulseUsToMask(uint16_t pulseUs, uint8_t channelMask = kAllChannelsMask) {
  channelMask &= kAllChannelsMask;
  for (uint8_t i = 0; i < kNumChannels; ++i) {
    writeChannelPulseUs(i, (channelMask & (1U << i)) != 0 ? pulseUs : kiwi_config::kEscNeutralPulseUs);
  }
}

void stopAll() {
  rampPhase = RampPhase::Idle;
  currentPercent = 0.0f;
  activeChannelMask = kAllChannelsMask;
  applyPulseUsToMask(kiwi_config::kEscNeutralPulseUs);
}

const char *phaseName(RampPhase phase) {
  switch (phase) {
    case RampPhase::Idle:
      return "idle";
    case RampPhase::PositiveUp:
      return "positive_up";
    case RampPhase::PositiveDown:
      return "positive_down";
    case RampPhase::NegativeDown:
      return "negative_down";
    case RampPhase::NegativeUp:
      return "negative_up";
    case RampPhase::DirectPulse:
      return "direct_pulse";
    case RampPhase::CalibrationMax:
      return "calibration_max";
    case RampPhase::CalibrationMin:
      return "calibration_min";
    case RampPhase::CalibrationCenter:
      return "calibration_center";
    default:
      return "unknown";
  }
}

void printHelp() {
  Serial.println();
  Serial.println("Kiwi follower Dominion/RC ESC bringup test (4 channels: D0/D1/D2/D3)");
  Serial.println("Commands:");
  Serial.println("  a | ramp       ramp all channels +/- limit");
  Serial.println("  1 | 2 | 3 | 4  ramp one channel only (1=D0 2=D1 3=D2 4=D3)");
  Serial.println("  + | -          adjust ramp limit by 5%");
  Serial.println("  limit <pct>    set ramp limit, 5..80");
  Serial.println("  s | stop       return all outputs to neutral");
  Serial.println("  n | neutral    hold all outputs at 1500 us");
  Serial.println("  low            hold all outputs at 1000 us");
  Serial.println("  high           hold all outputs at 2000 us");
  Serial.println("  pulse <us>     hold all outputs at a pulse width, e.g. pulse 1500");
  Serial.println("  pulse <c> <us> hold channel 1/2/3/4 at a pulse width, others neutral");
  Serial.println("  pulses <a> <b> <c> [d] hold D0/D1/D2[/D3] at independent pulse widths");
  Serial.println("  cal            Dominion calibration sequence on ALL four channels");
  Serial.println("  status         print current state");
  Serial.println("  ? | help       print this help");
  Serial.printf("Current ramp limit: %.0f%%\n", rampLimitPercent);
  Serial.println("Boot and stop hold neutral at 1500 us on all four pins. No movement until USB serial input.");
  Serial.println("Dominion calibration: send cal, then immediately power-cycle ESC power while signal is held high.");
  Serial.println();
}

void printStatus() {
  Serial.printf("phase=%s percent=%.1f limit=%.0f mask=0x%02x pulses_us=%u/%u/%u/%u pins=D0/GPIO%u,D1/GPIO%u,D2/GPIO%u,D3/GPIO%u\n",
                phaseName(rampPhase),
                currentPercent,
                rampLimitPercent,
                activeChannelMask,
                currentPulseUs[0],
                currentPulseUs[1],
                currentPulseUs[2],
                currentPulseUs[3],
                kChannelPins[0],
                kChannelPins[1],
                kChannelPins[2],
                kChannelPins[3]);
}

void startRamp(uint8_t channelMask) {
  activeChannelMask = channelMask & kAllChannelsMask;
  if (activeChannelMask == 0) {
    activeChannelMask = kAllChannelsMask;
  }
  currentPercent = 0.0f;
  rampPhase = RampPhase::PositiveUp;
  lastRampStepMs = millis();
  phaseStartedMs = millis();
  applyPercentToMask(currentPercent, activeChannelMask);
  Serial.printf("Starting ramp mask=0x%02x limit=%.0f%%\n", activeChannelMask, rampLimitPercent);
}

void holdPulse(uint16_t pulseUs, uint8_t channelMask = kAllChannelsMask) {
  rampPhase = RampPhase::DirectPulse;
  activeChannelMask = channelMask & kAllChannelsMask;
  if (activeChannelMask == 0) {
    activeChannelMask = kAllChannelsMask;
  }
  currentPercent = 0.0f;
  phaseStartedMs = millis();
  applyPulseUsToMask(pulseUs, activeChannelMask);
  Serial.printf("Holding pulse %u us on mask=0x%02x\n", pulseUs, activeChannelMask);
}

void holdPulses(const uint16_t *pulsesUs, uint8_t count) {
  rampPhase = RampPhase::DirectPulse;
  activeChannelMask = kAllChannelsMask;
  currentPercent = 0.0f;
  phaseStartedMs = millis();
  for (uint8_t i = 0; i < kNumChannels; ++i) {
    writeChannelPulseUs(i, i < count ? pulsesUs[i] : kiwi_config::kEscNeutralPulseUs);
  }
  Serial.printf("Holding pulses D0=%u us D1=%u us D2=%u us D3=%u us\n",
                currentPulseUs[0],
                currentPulseUs[1],
                currentPulseUs[2],
                currentPulseUs[3]);
}

void startCalibration() {
  rampPhase = RampPhase::CalibrationMax;
  activeChannelMask = kAllChannelsMask;
  currentPercent = 0.0f;
  phaseStartedMs = millis();
  applyPulseUsToMask(kiwi_config::kEscMaxPulseUs);
  Serial.println("Dominion calibration started on all four channels.");
  Serial.println("Outputs are now 2000 us. Power-cycle the ESCs now if they were already powered.");
  Serial.println("Sequence: 2000 us for 16s, 1000 us for 7s, 1500 us for 8s, then neutral idle.");
}

void updateRamp() {
  if (rampPhase == RampPhase::DirectPulse &&
      millis() - phaseStartedMs >= kDirectPulseTimeoutMs) {
    stopAll();
    Serial.println("Direct pulse failsafe timeout; all outputs neutral. Re-send the command to continue.");
    return;
  }

  if (rampPhase == RampPhase::CalibrationMax &&
      millis() - phaseStartedMs >= kCalibrationMaxHoldMs) {
    rampPhase = RampPhase::CalibrationMin;
    phaseStartedMs = millis();
    applyPulseUsToMask(kiwi_config::kEscMinPulseUs);
    Serial.println("Calibration step: outputs now 1000 us minimum.");
    return;
  }

  if (rampPhase == RampPhase::CalibrationMin &&
      millis() - phaseStartedMs >= kCalibrationMinHoldMs) {
    rampPhase = RampPhase::CalibrationCenter;
    phaseStartedMs = millis();
    applyPulseUsToMask(kiwi_config::kEscNeutralPulseUs);
    Serial.println("Calibration step: outputs now 1500 us center.");
    return;
  }

  if (rampPhase == RampPhase::CalibrationCenter &&
      millis() - phaseStartedMs >= kCalibrationCenterHoldMs) {
    stopAll();
    Serial.println("Calibration sequence complete. Power-cycle the ESCs to save/apply calibration.");
    return;
  }

  if (rampPhase == RampPhase::Idle ||
      rampPhase == RampPhase::DirectPulse ||
      rampPhase == RampPhase::CalibrationMax ||
      rampPhase == RampPhase::CalibrationMin ||
      rampPhase == RampPhase::CalibrationCenter ||
      millis() - lastRampStepMs < kRampStepPeriodMs) {
    return;
  }
  lastRampStepMs = millis();

  switch (rampPhase) {
    case RampPhase::PositiveUp:
      currentPercent += kRampStepPercent;
      if (currentPercent >= rampLimitPercent) {
        currentPercent = rampLimitPercent;
        rampPhase = RampPhase::PositiveDown;
      }
      break;

    case RampPhase::PositiveDown:
      currentPercent -= kRampStepPercent;
      if (currentPercent <= 0.0f) {
        currentPercent = 0.0f;
        rampPhase = RampPhase::NegativeDown;
      }
      break;

    case RampPhase::NegativeDown:
      currentPercent -= kRampStepPercent;
      if (currentPercent <= -rampLimitPercent) {
        currentPercent = -rampLimitPercent;
        rampPhase = RampPhase::NegativeUp;
      }
      break;

    case RampPhase::NegativeUp:
      currentPercent += kRampStepPercent;
      if (currentPercent >= 0.0f) {
        stopAll();
        Serial.println("Ramp complete; all outputs neutral.");
        return;
      }
      break;

    default:
      break;
  }

  applyPercentToMask(currentPercent, activeChannelMask);
}

bool parsePulseCommand(const String &command) {
  int channel = 0;
  int pulseUs = 0;
  const int parsedTwo = sscanf(command.c_str(), "pulse %d %d", &channel, &pulseUs);
  if (parsedTwo == 2) {
    if (channel < 1 || channel > kNumChannels) {
      Serial.println("Channel must be 1, 2, 3, or 4.");
      return true;
    }
    if (pulseUs < kiwi_config::kEscMinPulseUs || pulseUs > kiwi_config::kEscMaxPulseUs) {
      Serial.println("Pulse must be between 1000 and 2000 us.");
      return true;
    }
    holdPulse(static_cast<uint16_t>(pulseUs), static_cast<uint8_t>(1U << (channel - 1)));
    return true;
  }

  const int parsedOne = sscanf(command.c_str(), "pulse %d", &pulseUs);
  if (parsedOne == 1) {
    if (pulseUs < kiwi_config::kEscMinPulseUs || pulseUs > kiwi_config::kEscMaxPulseUs) {
      Serial.println("Pulse must be between 1000 and 2000 us.");
      return true;
    }
    holdPulse(static_cast<uint16_t>(pulseUs));
    return true;
  }

  return false;
}

bool parsePulsesCommand(const String &command) {
  int values[kNumChannels] = {0, 0, 0, 0};
  const int parsed = sscanf(command.c_str(), "pulses %d %d %d %d",
                            &values[0], &values[1], &values[2], &values[3]);
  if (parsed < 3) {
    return false;
  }

  uint16_t pulsesUs[kNumChannels];
  for (int i = 0; i < parsed; ++i) {
    if (values[i] < kiwi_config::kEscMinPulseUs || values[i] > kiwi_config::kEscMaxPulseUs) {
      Serial.println("Pulses must be between 1000 and 2000 us.");
      return true;
    }
    pulsesUs[i] = static_cast<uint16_t>(values[i]);
  }

  holdPulses(pulsesUs, static_cast<uint8_t>(parsed));
  return true;
}

void handleCommand(String command) {
  command.trim();
  command.toLowerCase();
  if (command.length() == 0) {
    return;
  }

  if (command == "a" || command == "r" || command == "ramp") {
    startRamp(kAllChannelsMask);
  } else if (command == "1") {
    startRamp(0x01);
  } else if (command == "2") {
    startRamp(0x02);
  } else if (command == "3") {
    startRamp(0x04);
  } else if (command == "4") {
    startRamp(0x08);
  } else if (command == "+" || command == "=") {
    rampLimitPercent = constrain(rampLimitPercent + 5.0f, 5.0f, 80.0f);
    Serial.printf("Ramp limit now %.0f%%\n", rampLimitPercent);
  } else if (command == "-" || command == "_") {
    rampLimitPercent = constrain(rampLimitPercent - 5.0f, 5.0f, 80.0f);
    Serial.printf("Ramp limit now %.0f%%\n", rampLimitPercent);
  } else if (command.startsWith("limit ")) {
    rampLimitPercent = constrain(command.substring(6).toFloat(), 5.0f, 80.0f);
    Serial.printf("Ramp limit now %.0f%%\n", rampLimitPercent);
  } else if (command == "s" || command == "stop") {
    stopAll();
    Serial.println("Stopped; all outputs neutral.");
  } else if (command == "n" || command == "neutral" || command == "center") {
    holdPulse(kiwi_config::kEscNeutralPulseUs);
  } else if (command == "low" || command == "min") {
    holdPulse(kiwi_config::kEscMinPulseUs);
  } else if (command == "high" || command == "max") {
    Serial.println("Warning: 2000 us may spin motors if ESCs are armed.");
    holdPulse(kiwi_config::kEscMaxPulseUs);
  } else if (command == "cal") {
    startCalibration();
  } else if (command == "status") {
    printStatus();
  } else if (command == "?" || command == "h" || command == "help") {
    printHelp();
  } else if (command.startsWith("pulses ")) {
    parsePulsesCommand(command);
  } else if (command.startsWith("pulse ")) {
    parsePulseCommand(command);
  } else {
    Serial.printf("Unknown command '%s'. Send ? for help.\n", command.c_str());
  }
}

void initOutputs() {
  for (uint8_t i = 0; i < kNumChannels; ++i) {
    ledcSetup(kChannelPwmChannels[i],
              kiwi_config::kEscPwmFrequencyHz,
              kiwi_config::kEscPwmResolutionBits);
    ledcAttachPin(kChannelPins[i], kChannelPwmChannels[i]);
  }
  applyPulseUsToMask(kiwi_config::kEscNeutralPulseUs);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("Booting kiwi follower Dominion/RC ESC bringup test (4 channels)");
  initOutputs();
  Serial.printf("Holding all ESC outputs neutral at %u us for %lu ms...\n",
                kiwi_config::kEscNeutralPulseUs,
                static_cast<unsigned long>(kEscArmHoldMs));
  delay(kEscArmHoldMs);
  stopAll();
  printHelp();
}

void loop() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r' || c == '\n') {
      handleCommand(commandLine);
      commandLine = "";
    } else if (isPrintable(c)) {
      commandLine += c;
      if (commandLine.length() > 80) {
        commandLine = "";
        Serial.println("Command too long; cleared input.");
      }
    }
  }

  updateRamp();
  if (millis() - lastStatusMs >= kStatusPeriodMs) {
    lastStatusMs = millis();
    printStatus();
  }

  delay(1);
}
