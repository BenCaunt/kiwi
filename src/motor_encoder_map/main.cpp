#include <Arduino.h>
#include <Wire.h>
#include <esp_timer.h>

#include "robot_config.h"

// Motor -> encoder mapping test. Drives each configured motor output one at a
// time at low power while sampling all three AS5600s behind the PCA9548A, then
// reports which encoder channel moved, in which direction, and the suggested
// kEncoderTcaChannels / kEncoderPolarity values for robot_config.h. Runs
// automatically after the ESC arm hold; re-run with "map" over USB serial.

namespace {

constexpr int32_t kAs5600CountsPerRev = 4096;
constexpr int32_t kAs5600HalfCounts = kAs5600CountsPerRev / 2;

constexpr float kDrivePercent = 20.0f;
constexpr uint32_t kEscArmHoldMs = 7000;
constexpr uint32_t kPreSettleMs = 1000;
constexpr uint32_t kDriveMs = 2000;
constexpr uint32_t kPostSettleMs = 1500;
constexpr uint32_t kEncoderSamplePeriodMs = 5;
constexpr uint32_t kLiveStatusPeriodMs = 500;
// A wheel that actually spins at 20% for 2 s moves thousands of counts; treat
// anything below this as vibration/cross-talk.
constexpr int64_t kMovedThresholdCounts = 300;

enum class Phase : uint8_t {
  ArmHold,
  PreSettle,
  DrivePos,
  MidSettle,
  DriveNeg,
  PostSettle,
  Done,
  Idle,
};

struct EncoderState {
  bool ready = false;
  uint16_t lastRaw = 0;
  int64_t count = 0;
  uint32_t readErrors = 0;
};

EncoderState encoders[3];
Phase phase = Phase::ArmHold;
uint8_t activeMotor = 0;
uint32_t phaseStartedMs = 0;
uint32_t lastEncoderSampleMs = 0;
uint32_t lastLiveStatusMs = 0;
int64_t phaseStartCounts[3] = {};
int64_t posDelta[3][3] = {};  // [motor][encoder]
int64_t negDelta[3][3] = {};
float drivePercent = kDrivePercent;
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

// Raw output percent, deliberately ignoring kMotorPolarity: the point of this
// test is to observe the unmapped hardware response.
void setMotorPercentRaw(uint8_t index, float percent) {
  if (index >= 3) {
    return;
  }
  ledcWrite(kiwi_config::kMotorPwmChannels[index], pulseUsToDuty(percentToPulseUs(percent)));
}

void allMotorsNeutral() {
  for (uint8_t i = 0; i < 3; ++i) {
    setMotorPercentRaw(i, 0.0f);
  }
}

void initMotors() {
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
  ledcWrite(kiwi_config::kEscAuxNeutralPwmChannel,
            pulseUsToDuty(kiwi_config::kEscNeutralPulseUs));
  allMotorsNeutral();
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

// AS5600 STATUS register: MD=magnet detected, ML=too weak, MH=too strong.
bool readAs5600Status(uint8_t channel, uint8_t *status) {
  if (!tcaSelect(channel)) {
    return false;
  }
  Wire.beginTransmission(kiwi_config::kAs5600Address);
  Wire.write(static_cast<uint8_t>(0x0b));
  if (Wire.endTransmission(false) != 0) {
    tcaDisable();
    return false;
  }
  if (Wire.requestFrom(kiwi_config::kAs5600Address, static_cast<uint8_t>(1)) != 1) {
    tcaDisable();
    return false;
  }
  *status = Wire.read();
  tcaDisable();
  return true;
}

void probeEncoders() {
  for (uint8_t i = 0; i < 3; ++i) {
    uint16_t raw = 0;
    uint8_t status = 0;
    const bool rawOk = readAs5600Raw(kiwi_config::kEncoderTcaChannels[i], &raw);
    const bool statusOk = readAs5600Status(kiwi_config::kEncoderTcaChannels[i], &status);
    Serial.printf("encoder ch%u: %s raw=%u status=0x%02x md=%d ml=%d mh=%d\n",
                  kiwi_config::kEncoderTcaChannels[i],
                  rawOk ? "detected" : "NOT RESPONDING",
                  raw,
                  statusOk ? status : 0,
                  statusOk ? ((status >> 5) & 1) : -1,
                  statusOk ? ((status >> 4) & 1) : -1,
                  statusOk ? ((status >> 3) & 1) : -1);
  }
}

void scanI2cBus() {
  // A healthy idle I2C bus has both lines pulled high. A line stuck low means
  // wiring/power trouble that no address scan can see past.
  Serial.printf("bus lines: SDA(GPIO%u)=%d SCL(GPIO%u)=%d (1=high/idle, 0=stuck low)\n",
                kiwi_config::kI2cSdaPin,
                digitalRead(kiwi_config::kI2cSdaPin),
                kiwi_config::kI2cSclPin,
                digitalRead(kiwi_config::kI2cSclPin));
  Serial.printf("imu lines: INT(GPIO%u)=%d RST(GPIO%u)=%d (INT=0 means BNO08x is alive and asserting data-ready)\n",
                kiwi_config::kImuIntPin,
                digitalRead(kiwi_config::kImuIntPin),
                kiwi_config::kImuResetPin,
                digitalRead(kiwi_config::kImuResetPin));

  uint8_t found = 0;
  for (uint8_t addr = 1; addr < 127; ++addr) {
    Wire.beginTransmission(addr);
    const uint8_t err = Wire.endTransmission();
    if (err == 0) {
      ++found;
      const char *note = "";
      if (addr >= 0x70 && addr <= 0x77) {
        note = " <- PCA9548A range";
      } else if (addr == kiwi_config::kAs5600Address) {
        note = " <- AS5600 (leaked past mux?)";
      } else if (addr == 0x4a || addr == 0x4b) {
        note = " <- BNO08x";
      }
      Serial.printf("  found device at 0x%02x%s\n", addr, note);
    }
    // Progress markers so a wedged bus shows exactly where it died.
    if ((addr & 0x0f) == 0) {
      Serial.printf("  ...scanned through 0x%02x (last err=%u)\n", addr, err);
    }
  }
  Serial.printf("scan complete: %u device(s)\n", found);
}

// Bit-level bus wiring test. Releases the I2C peripheral, open-drain drives
// each line low in turn, and reads both back. Detects: a pin that cannot pull
// its own line low (damaged driver / short to supply) and lines shorted to
// each other (driving one drags the other low).
void testBusPins() {
  Wire.end();
  const uint8_t sda = kiwi_config::kI2cSdaPin;
  const uint8_t scl = kiwi_config::kI2cSclPin;
  pinMode(sda, OUTPUT_OPEN_DRAIN | PULLUP);
  pinMode(scl, OUTPUT_OPEN_DRAIN | PULLUP);

  digitalWrite(sda, HIGH);
  digitalWrite(scl, HIGH);
  delay(2);
  Serial.printf("released:      SDA=%d SCL=%d (expect 1 1)\n",
                digitalRead(sda), digitalRead(scl));

  digitalWrite(sda, LOW);
  delay(2);
  Serial.printf("SDA driven low: SDA=%d SCL=%d (expect 0 1; SDA=1 -> pin cannot sink, SCL=0 -> lines shorted)\n",
                digitalRead(sda), digitalRead(scl));
  digitalWrite(sda, HIGH);
  delay(2);

  digitalWrite(scl, LOW);
  delay(2);
  Serial.printf("SCL driven low: SDA=%d SCL=%d (expect 1 0; SCL=1 -> pin cannot sink, SDA=0 -> lines shorted)\n",
                digitalRead(sda), digitalRead(scl));
  digitalWrite(scl, HIGH);
  delay(2);

  Wire.begin(sda, scl);
  Wire.setClock(100000);
  Wire.setTimeOut(20);
  Serial.println("pin test done; I2C re-initialised at 100 kHz");
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

void updateEncoders() {
  for (uint8_t i = 0; i < 3; ++i) {
    uint16_t raw = 0;
    if (!readAs5600Raw(kiwi_config::kEncoderTcaChannels[i], &raw)) {
      ++encoders[i].readErrors;
      continue;
    }
    if (!encoders[i].ready) {
      encoders[i].ready = true;
      encoders[i].lastRaw = raw;
      continue;
    }
    encoders[i].count += signedAs5600Delta(raw, encoders[i].lastRaw);
    encoders[i].lastRaw = raw;
  }
}

void snapshotCounts() {
  for (uint8_t i = 0; i < 3; ++i) {
    phaseStartCounts[i] = encoders[i].count;
  }
}

void enterPhase(Phase next) {
  phase = next;
  phaseStartedMs = millis();
}

void printLiveStatus(const char *label) {
  Serial.printf("[%s m%u] counts=%lld/%lld/%lld raw=%u/%u/%u err=%lu/%lu/%lu\n",
                label,
                activeMotor,
                static_cast<long long>(encoders[0].count),
                static_cast<long long>(encoders[1].count),
                static_cast<long long>(encoders[2].count),
                encoders[0].lastRaw,
                encoders[1].lastRaw,
                encoders[2].lastRaw,
                static_cast<unsigned long>(encoders[0].readErrors),
                static_cast<unsigned long>(encoders[1].readErrors),
                static_cast<unsigned long>(encoders[2].readErrors));
}

void startMapping() {
  for (uint8_t m = 0; m < 3; ++m) {
    for (uint8_t e = 0; e < 3; ++e) {
      posDelta[m][e] = 0;
      negDelta[m][e] = 0;
    }
  }
  activeMotor = 0;
  allMotorsNeutral();
  enterPhase(Phase::PreSettle);
  Serial.printf("Mapping run started: each motor +/-%.0f%% for %lu ms\n",
                drivePercent,
                static_cast<unsigned long>(kDriveMs));
}

void printResults() {
  Serial.println();
  Serial.println("=== motor -> encoder mapping results ===");
  Serial.println("deltas are AS5600 counts over the drive window (4096/rev)");

  int8_t motorToEncoder[3] = {-1, -1, -1};
  int8_t motorDirection[3] = {0, 0, 0};

  for (uint8_t m = 0; m < 3; ++m) {
    Serial.printf("motor %u (GPIO%u):\n", m, kiwi_config::kMotorPins[m]);
    int8_t best = -1;
    int64_t bestMag = 0;
    for (uint8_t e = 0; e < 3; ++e) {
      Serial.printf("  enc ch%u: pos_cmd_delta=%lld neg_cmd_delta=%lld\n",
                    kiwi_config::kEncoderTcaChannels[e],
                    static_cast<long long>(posDelta[m][e]),
                    static_cast<long long>(negDelta[m][e]));
      const int64_t mag = llabs(posDelta[m][e]);
      if (mag > bestMag) {
        bestMag = mag;
        best = static_cast<int8_t>(e);
      }
    }
    if (best < 0 || bestMag < kMovedThresholdCounts) {
      Serial.println("  -> NO ENCODER MOVED (check ESC power/arming and mux wiring)");
      continue;
    }
    motorToEncoder[m] = best;
    motorDirection[m] = posDelta[m][best] > 0 ? 1 : -1;
    const bool negConsistent =
        negDelta[m][best] != 0 &&
        ((posDelta[m][best] > 0) != (negDelta[m][best] > 0)) &&
        llabs(negDelta[m][best]) >= kMovedThresholdCounts;
    Serial.printf("  -> encoder ch%u, positive command makes counts %s%s\n",
                  kiwi_config::kEncoderTcaChannels[best],
                  motorDirection[m] > 0 ? "INCREASE" : "DECREASE",
                  negConsistent ? "" : " (WARNING: reverse drive did not mirror)");
  }

  bool complete = true;
  for (uint8_t m = 0; m < 3; ++m) {
    if (motorToEncoder[m] < 0) {
      complete = false;
    }
  }
  for (uint8_t m = 0; m < 3 && complete; ++m) {
    for (uint8_t n = static_cast<uint8_t>(m + 1); n < 3; ++n) {
      if (motorToEncoder[m] == motorToEncoder[n]) {
        Serial.printf("WARNING: motors %u and %u map to the same encoder\n", m, n);
        complete = false;
      }
    }
  }

  if (complete) {
    Serial.println();
    Serial.println("suggested robot_config.h (encoder index i belongs to motor i,");
    Serial.println("polarity chosen so positive motor command -> increasing counts):");
    Serial.printf("  constexpr uint8_t kEncoderTcaChannels[3] = {%u, %u, %u};\n",
                  kiwi_config::kEncoderTcaChannels[motorToEncoder[0]],
                  kiwi_config::kEncoderTcaChannels[motorToEncoder[1]],
                  kiwi_config::kEncoderTcaChannels[motorToEncoder[2]]);
    Serial.printf("  constexpr int8_t kEncoderPolarity[3] = {%d, %d, %d};\n",
                  motorDirection[0],
                  motorDirection[1],
                  motorDirection[2]);
  } else {
    Serial.println("Mapping incomplete; fix the flagged issues and re-run with 'map'.");
  }
  Serial.println("=== end of results ===");
  Serial.println("Commands: map (re-run) | s (stop) | power <pct> | status");
}

void updateStateMachine() {
  const uint32_t elapsed = millis() - phaseStartedMs;
  switch (phase) {
    case Phase::ArmHold:
      if (elapsed >= kEscArmHoldMs) {
        startMapping();
      }
      break;

    case Phase::PreSettle:
      if (elapsed >= kPreSettleMs) {
        snapshotCounts();
        setMotorPercentRaw(activeMotor, drivePercent);
        Serial.printf("motor %u (GPIO%u) driving +%.0f%%\n",
                      activeMotor, kiwi_config::kMotorPins[activeMotor], drivePercent);
        enterPhase(Phase::DrivePos);
      }
      break;

    case Phase::DrivePos:
      if (elapsed >= kDriveMs) {
        for (uint8_t e = 0; e < 3; ++e) {
          posDelta[activeMotor][e] = encoders[e].count - phaseStartCounts[e];
        }
        allMotorsNeutral();
        enterPhase(Phase::MidSettle);
      }
      break;

    case Phase::MidSettle:
      if (elapsed >= kPostSettleMs) {
        snapshotCounts();
        setMotorPercentRaw(activeMotor, -drivePercent);
        Serial.printf("motor %u (GPIO%u) driving -%.0f%%\n",
                      activeMotor, kiwi_config::kMotorPins[activeMotor], drivePercent);
        enterPhase(Phase::DriveNeg);
      }
      break;

    case Phase::DriveNeg:
      if (elapsed >= kDriveMs) {
        for (uint8_t e = 0; e < 3; ++e) {
          negDelta[activeMotor][e] = encoders[e].count - phaseStartCounts[e];
        }
        allMotorsNeutral();
        enterPhase(Phase::PostSettle);
      }
      break;

    case Phase::PostSettle:
      if (elapsed >= kPostSettleMs) {
        if (activeMotor + 1 < 3) {
          ++activeMotor;
          enterPhase(Phase::PreSettle);
        } else {
          enterPhase(Phase::Done);
          printResults();
        }
      }
      break;

    case Phase::Done:
    case Phase::Idle:
      break;
  }
}

void handleCommand(String command) {
  command.trim();
  command.toLowerCase();
  if (command.length() == 0) {
    return;
  }
  if (command == "map") {
    startMapping();
  } else if (command == "s" || command == "stop") {
    allMotorsNeutral();
    enterPhase(Phase::Idle);
    Serial.println("Stopped; outputs neutral. Send 'map' to run again.");
  } else if (command.startsWith("power ")) {
    drivePercent = constrain(command.substring(6).toFloat(), 5.0f, 50.0f);
    Serial.printf("Drive power now %.0f%%\n", drivePercent);
  } else if (command == "status") {
    printLiveStatus("status");
  } else if (command == "probe") {
    probeEncoders();
  } else if (command == "scan") {
    scanI2cBus();
  } else if (command == "pintest") {
    testBusPins();
  } else if (command.startsWith("aux ")) {
    // Drive the D3 aux line (normally pinned at neutral) to test whether a
    // Dominion has its driven input wired there instead of a motor pin.
    const float pct = constrain(command.substring(4).toFloat(), -50.0f, 50.0f);
    int64_t start[3];
    for (uint8_t i = 0; i < 3; ++i) {
      start[i] = encoders[i].count;
    }
    allMotorsNeutral();
    Serial.printf("driving D3 aux at %.0f%% for 2 s\n", pct);
    ledcWrite(kiwi_config::kEscAuxNeutralPwmChannel, pulseUsToDuty(percentToPulseUs(pct)));
    const uint32_t auxStartMs = millis();
    while (millis() - auxStartMs < 2000) {
      updateEncoders();
      delay(2);
    }
    ledcWrite(kiwi_config::kEscAuxNeutralPwmChannel,
              pulseUsToDuty(kiwi_config::kEscNeutralPulseUs));
    Serial.printf("aux deltas: ch0=%lld ch1=%lld ch2=%lld\n",
                  static_cast<long long>(encoders[0].count - start[0]),
                  static_cast<long long>(encoders[1].count - start[1]),
                  static_cast<long long>(encoders[2].count - start[2]));
  } else if (command.startsWith("watch")) {
    long seconds = command.substring(5).toInt();
    seconds = constrain(seconds, 5, 120);
    Serial.printf("watching encoders for %ld s; spin wheels by hand\n", seconds);
    allMotorsNeutral();
    const uint32_t startMs = millis();
    uint32_t lastPrintMs = 0;
    while (millis() - startMs < static_cast<uint32_t>(seconds) * 1000UL) {
      updateEncoders();
      const uint32_t nowWatch = millis();
      if (nowWatch - lastPrintMs >= 300) {
        lastPrintMs = nowWatch;
        Serial.printf("w t=%5.1f counts=%lld/%lld/%lld raw=%u/%u/%u\n",
                      (nowWatch - startMs) / 1000.0f,
                      static_cast<long long>(encoders[0].count),
                      static_cast<long long>(encoders[1].count),
                      static_cast<long long>(encoders[2].count),
                      encoders[0].lastRaw,
                      encoders[1].lastRaw,
                      encoders[2].lastRaw);
      }
      delay(2);
    }
    Serial.println("watch done");
  } else if (command.startsWith("clock ")) {
    const long khz = command.substring(6).toInt();
    if (khz >= 1 && khz <= 1000) {
      Wire.setClock(static_cast<uint32_t>(khz) * 1000UL);
      Serial.printf("I2C clock now %ld kHz\n", khz);
    } else {
      Serial.println("clock <khz>, 1..1000");
    }
  } else {
    Serial.println("Commands: map | s | power <pct> | status | probe | scan | clock <khz>");
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("Booting kiwi motor->encoder mapping test");

  // INT/RESET are now wired (2026-07-07): release the BNO08x from reset so it
  // shows up in bus scans, and observe its INT line.
  pinMode(kiwi_config::kImuResetPin, OUTPUT);
  digitalWrite(kiwi_config::kImuResetPin, HIGH);
  pinMode(kiwi_config::kImuIntPin, INPUT_PULLUP);

  Wire.begin(kiwi_config::kI2cSdaPin, kiwi_config::kI2cSclPin);
  Wire.setClock(100000);  // conservative clock while debugging wiring
  Wire.setTimeOut(20);
  tcaDisable();

  probeEncoders();

  initMotors();
  enterPhase(Phase::ArmHold);
  Serial.printf("Holding neutral %lu ms for ESC arming, then mapping starts automatically...\n",
                static_cast<unsigned long>(kEscArmHoldMs));
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
      }
    }
  }

  // Only poll encoders while mapping is active; keeps the bus quiet for
  // scan/probe commands and keeps serial responsive if the bus is sick.
  const bool mapping = phase == Phase::PreSettle || phase == Phase::DrivePos ||
                       phase == Phase::MidSettle || phase == Phase::DriveNeg ||
                       phase == Phase::PostSettle;
  const uint32_t nowMs = millis();
  if (mapping && nowMs - lastEncoderSampleMs >= kEncoderSamplePeriodMs) {
    lastEncoderSampleMs = nowMs;
    updateEncoders();
  }

  if (nowMs - lastLiveStatusMs >= kLiveStatusPeriodMs) {
    lastLiveStatusMs = nowMs;
    if (phase == Phase::DrivePos || phase == Phase::DriveNeg) {
      printLiveStatus(phase == Phase::DrivePos ? "drive+" : "drive-");
    }
  }

  updateStateMachine();
  delay(1);
}
