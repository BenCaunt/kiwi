#include <Arduino.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <WebServer.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_timer.h>
#include <zenoh-pico.h>

#include <ctype.h>
#include <math.h>
#include <string.h>

#include "camera_pins.h"
#include "kiwi_messages.h"
#include "kiwi_uart_protocol.h"
#include "robot_config.h"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#if __has_include("local_zenoh.h")
#include "local_zenoh.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD ""
#endif

#ifndef ZENOH_MODE
#define ZENOH_MODE "peer"
#endif

#ifndef ZENOH_CONNECT
#define ZENOH_CONNECT ""
#endif

namespace {

using kiwi::MessageType;
using kiwi::Packet;
using kiwi::PacketReader;
using kiwi::TwistReportPayload;
using kiwi::VelocityCommandPayload;
using kiwi::VelocityMode;

constexpr char kApSsid[] = "KIWI-MASTER";
constexpr char kApPassword[] = "seeedstudio";
constexpr uint32_t kDriveParamsResendMs = 2000;

// Runtime configuration, provisioned over the soft-AP HTTP endpoint and
// persisted in NVS. Compiled secrets.h/local_zenoh.h/robot_config.h values
// are first-boot defaults only.
struct RuntimeConfig {
  char wifiSsid[33];
  char wifiPassword[65];
  char zenohMode[8];
  char zenohConnect[96];
  kiwi::DriveParamsPayload drive;
};

RuntimeConfig runtimeConfig;
Preferences configPrefs;
WebServer httpServer(80);
bool httpServerStarted = false;
uint32_t driveParamsAckedVersion = 0xffffffff;
uint32_t lastDriveParamsSendMs = 0;
bool rebootPending = false;
uint32_t rebootAtMs = 0;
// STA joins are scan-gated: we only call WiFi.begin when a scan actually
// sees the stored SSID. Blind retries (plus the driver's auto-reconnect)
// drag the radio off-channel continuously, starving the soft-AP's beacons --
// which made KIWI-MASTER invisible precisely when away from the home
// network, the one situation provisioning exists for.
constexpr uint32_t kStaScanIntervalMs = 30000;
bool staScanInFlight = false;
constexpr uint32_t kZenohRetryMs = 10000;
// The soft-AP pins the radio to its channel, which can break STA auth against
// a router on a different channel (endless AUTH_EXPIRE). So: STA joins first
// and the AP comes up on the STA's channel; if the join hasn't landed by this
// deadline, start the AP anyway so provisioning stays reachable.
constexpr uint32_t kApFallbackMs = 15000;
bool apStarted = false;
bool staWasConnected = false;
uint32_t lastStaAttemptMs = 0;
uint32_t lastZenohAttemptMs = 0;
constexpr size_t kCameraHeaderBytes = 32;
constexpr size_t kLd19FrameBytes = 47;
// The LD19 produces roughly 450-500 small UART frames/s. Sending each 47-byte
// frame as a separate reliable Zenoh operation starves every lower-rate topic.
// Twenty frames remain comfortably below the 1500-byte IP MTU while reducing
// transport calls by about 20x. Consumers accept one or more concatenated
// frames per sample.
constexpr size_t kLd19FramesPerBatch = 20;
constexpr size_t kLd19BatchBytes = kLd19FrameBytes * kLd19FramesPerBatch;
constexpr size_t kMaxLd19FramesPerLoop = 32;
constexpr size_t kMaxFollowerPacketsPerLoop = 16;
constexpr uint32_t kLd19BatchMaxAgeMs = 50;
constexpr uint32_t kTwistPublishPeriodMs = 50;  // latest report at <=20 Hz

constexpr char kZenohCameraKey[] = ROBOT_NAMESPACE "/camera/jpeg";
constexpr char kZenohLidarRawKey[] = ROBOT_NAMESPACE "/lidar/ld19/raw";
constexpr char kZenohTwistKey[] = ROBOT_NAMESPACE "/odom/twist";
constexpr char kZenohStatusKey[] = ROBOT_NAMESPACE "/status/master";
constexpr char kZenohCmdVelKey[] = ROBOT_NAMESPACE "/cmd_vel";

HardwareSerial FollowerUart(1);
HardwareSerial LidarUart(2);
PacketReader followerReader;

z_owned_session_t zenohSession;
z_owned_publisher_t cameraPub;
z_owned_publisher_t lidarPub;
z_owned_publisher_t twistPub;
z_owned_publisher_t statusPub;
z_owned_subscriber_t cmdVelSub;

bool zenohReady = false;
bool cameraReady = false;
uint16_t followerTxSeq = 0;
uint32_t cameraSeq = 0;
uint32_t lidarFrames = 0;
uint32_t lidarBadFrames = 0;
uint32_t lidarBytes = 0;
uint32_t followerReports = 0;
uint32_t followerBadPackets = 0;
uint32_t velocityCommands = 0;
uint32_t cameraPublished = 0;
uint32_t cameraPublishErrors = 0;
uint32_t lidarPublished = 0;
uint32_t lidarPublishErrors = 0;
uint32_t twistPublished = 0;
uint32_t twistPublishErrors = 0;
uint32_t lidarBatchesPublished = 0;
uint32_t lastCameraPublishMs = 0;
uint32_t lastStatusPublishMs = 0;
uint32_t lastTwistPublishMs = 0;

uint8_t lidarBatch[kLd19BatchBytes] = {};
size_t lidarBatchFrames = 0;
uint32_t lidarBatchStartedMs = 0;
TwistReportPayload latestTwistReport = {};
bool twistReportPending = false;

uint32_t lastLoopStartUs = 0;
uint32_t maxLoopGapUs = 0;
uint32_t maxPublishUs = 0;
uint32_t lidarRxHighWater = 0;
uint32_t followerRxHighWater = 0;

struct Ld19Reader {
  uint8_t frame[kLd19FrameBytes] = {};
  size_t index = 0;

  bool readFrom(Stream &stream, uint8_t *out) {
    while (stream.available() > 0) {
      const int raw = stream.read();
      if (raw < 0) {
        continue;
      }
      const uint8_t byte = static_cast<uint8_t>(raw);
      ++lidarBytes;

      if (index == 0) {
        if (byte != 0x54) {
          continue;
        }
        frame[index++] = byte;
        continue;
      }

      if (index == 1 && byte != 0x2c) {
        index = 0;
        ++lidarBadFrames;
        continue;
      }

      frame[index++] = byte;
      if (index == kLd19FrameBytes) {
        memcpy(out, frame, kLd19FrameBytes);
        index = 0;
        return true;
      }
    }

    return false;
  }
};

Ld19Reader lidarReader;

// LD19 CRC8: poly 0x4D, MSB-first, init 0, over the first 46 frame bytes.
// Validated against live sensor data (~1.5% of frames arrive corrupted).
uint8_t ld19CrcTable[256];

void initLd19CrcTable() {
  for (uint16_t i = 0; i < 256; ++i) {
    uint8_t crc = static_cast<uint8_t>(i);
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0x4d)
                         : static_cast<uint8_t>(crc << 1);
    }
    ld19CrcTable[i] = crc;
  }
}

bool ld19FrameCrcOk(const uint8_t *frame) {
  uint8_t crc = 0;
  for (size_t i = 0; i + 1 < kLd19FrameBytes; ++i) {
    crc = ld19CrcTable[crc ^ frame[i]];
  }
  return crc == frame[kLd19FrameBytes - 1];
}

void writeLe16(uint8_t *dst, uint16_t value) {
  dst[0] = value & 0xff;
  dst[1] = (value >> 8) & 0xff;
}

void writeLe32(uint8_t *dst, uint32_t value) {
  dst[0] = value & 0xff;
  dst[1] = (value >> 8) & 0xff;
  dst[2] = (value >> 16) & 0xff;
  dst[3] = (value >> 24) & 0xff;
}

void writeLe64(uint8_t *dst, uint64_t value) {
  for (uint8_t i = 0; i < 8; ++i) {
    dst[i] = (value >> (8 * i)) & 0xff;
  }
}

void loadRuntimeConfig() {
  strlcpy(runtimeConfig.wifiSsid, WIFI_SSID, sizeof(runtimeConfig.wifiSsid));
  strlcpy(runtimeConfig.wifiPassword, WIFI_PASSWORD, sizeof(runtimeConfig.wifiPassword));
  strlcpy(runtimeConfig.zenohMode, ZENOH_MODE, sizeof(runtimeConfig.zenohMode));
  strlcpy(runtimeConfig.zenohConnect, ZENOH_CONNECT, sizeof(runtimeConfig.zenohConnect));
  runtimeConfig.drive.version = 0;
  runtimeConfig.drive.wheel_radius_m = kiwi_config::kWheelRadiusM;
  runtimeConfig.drive.drive_base_radius_m = kiwi_config::kDriveBaseRadiusM;
  runtimeConfig.drive.max_wheel_surface_speed_mps = kiwi_config::kMaxWheelSurfaceSpeedMps;
  for (uint8_t i = 0; i < 3; ++i) {
    runtimeConfig.drive.motor_polarity[i] = kiwi_config::kMotorPolarity[i];
    runtimeConfig.drive.motor_deadband_pct[i] = 0;
    runtimeConfig.drive.encoder_polarity[i] = kiwi_config::kEncoderPolarity[i];
  }
  runtimeConfig.drive.velocity_command_timeout_ms = kiwi_config::kVelocityCommandTimeoutMs;
  runtimeConfig.drive.pid_kp = 0.0f;
  runtimeConfig.drive.pid_ki = 0.0f;
  runtimeConfig.drive.closed_loop = 0;

  configPrefs.begin("kiwi", true);
  if (configPrefs.isKey("ssid")) {
    configPrefs.getString("ssid", runtimeConfig.wifiSsid, sizeof(runtimeConfig.wifiSsid));
  }
  if (configPrefs.isKey("pass")) {
    configPrefs.getString("pass", runtimeConfig.wifiPassword, sizeof(runtimeConfig.wifiPassword));
  }
  if (configPrefs.isKey("zmode")) {
    configPrefs.getString("zmode", runtimeConfig.zenohMode, sizeof(runtimeConfig.zenohMode));
  }
  if (configPrefs.isKey("zconn")) {
    configPrefs.getString("zconn", runtimeConfig.zenohConnect, sizeof(runtimeConfig.zenohConnect));
  }
  kiwi::DriveParamsPayload stored = {};
  if (configPrefs.getBytes("drive", &stored, sizeof(stored)) == sizeof(stored) &&
      kiwi::driveParamsValid(stored)) {
    runtimeConfig.drive = stored;
  }
  configPrefs.end();

  Serial.printf("Runtime config: ssid=%s zenoh=%s@%s drive_version=%lu\n",
                strlen(runtimeConfig.wifiSsid) > 0 ? runtimeConfig.wifiSsid : "(unset)",
                runtimeConfig.zenohMode,
                strlen(runtimeConfig.zenohConnect) > 0 ? runtimeConfig.zenohConnect : "(scout)",
                static_cast<unsigned long>(runtimeConfig.drive.version));
}

void saveRuntimeConfig() {
  configPrefs.begin("kiwi", false);
  configPrefs.putString("ssid", runtimeConfig.wifiSsid);
  configPrefs.putString("pass", runtimeConfig.wifiPassword);
  configPrefs.putString("zmode", runtimeConfig.zenohMode);
  configPrefs.putString("zconn", runtimeConfig.zenohConnect);
  configPrefs.putBytes("drive", &runtimeConfig.drive, sizeof(runtimeConfig.drive));
  configPrefs.end();
}

void sendDriveParamsToFollower() {
  kiwi::writePacket(FollowerUart,
                    MessageType::DriveParams,
                    followerTxSeq++,
                    &runtimeConfig.drive,
                    sizeof(runtimeConfig.drive));
}

camera_config_t makeCameraConfig() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  return config;
}

bool initCamera() {
  camera_config_t config = makeCameraConfig();
  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    Serial.println("Camera init returned a null sensor handle.");
    return false;
  }

  if (sensor->id.PID == OV3660_PID) {
    sensor->set_vflip(sensor, 1);
    sensor->set_brightness(sensor, 1);
    sensor->set_saturation(sensor, -2);
  }
  sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  Serial.printf("Camera ready, PID=0x%04x\n", sensor->id.PID);
  return true;
}

void startAccessPoint(bool staConnected) {
  if (apStarted) {
    return;
  }
  WiFi.mode(WIFI_AP_STA);
  if (!WiFi.softAP(kApSsid, kApPassword)) {
    Serial.println("Unable to start the master provisioning AP.");
    return;
  }
  // Re-assert after the mode change; modem power-save throttles TCP badly.
  WiFi.setSleep(false);
  apStarted = true;
  Serial.printf("Master AP: %s / %s at %s%s\n",
                kApSsid,
                kApPassword,
                WiFi.softAPIP().toString().c_str(),
                staConnected ? " (sharing STA channel)" : " (STA not connected)");

  // When a visible STA network fails authentication, setup() has already
  // started the HTTP listener in WIFI_STA mode before the fallback AP is
  // created. The STA -> AP_STA mode transition can invalidate that socket:
  // DHCP on KIWI-MASTER still works, but port 80 never answers. Rebind after
  // the AP netif exists. Routes remain registered across WebServer::begin().
  if (httpServerStarted) {
    httpServer.begin();
    Serial.printf("Provisioning HTTP server rebound on http://%s/\n",
                  WiFi.softAPIP().toString().c_str());
  }
}

void startWifi() {
  WiFi.persistent(false);

  if (strlen(runtimeConfig.wifiSsid) == 0) {
    Serial.println("No Wi-Fi configured; provision over the AP (POST /config) or add secrets.h.");
    startAccessPoint(false);
    WiFi.setSleep(false);
    return;
  }

  WiFi.mode(WIFI_STA);
  // Must come after mode(): modem power-save otherwise stays on and
  // throttles TCP throughput to a crawl.
  WiFi.setSleep(false);

  // One boot-time scan: proves the target is visible on 2.4 GHz and shows its
  // auth mode -- the difference between "bad password" and "can't even see it".
  bool targetVisible = false;
  const int16_t found = WiFi.scanNetworks();
  for (int16_t i = 0; i < found; ++i) {
    const bool isTarget = WiFi.SSID(i) == runtimeConfig.wifiSsid;
    targetVisible |= isTarget;
    Serial.printf("  scan: %-32s ch=%2d rssi=%d auth=%d%s\n",
                  WiFi.SSID(i).c_str(),
                  WiFi.channel(i),
                  WiFi.RSSI(i),
                  static_cast<int>(WiFi.encryptionType(i)),
                  isTarget ? "  <-- target" : "");
  }
  WiFi.scanDelete();

  WiFi.setAutoReconnect(false);  // rejoins are scan-gated in maintainNetwork
  if (targetVisible) {
    Serial.printf("Connecting STA Wi-Fi to %s (AP follows on its channel)\n",
                  runtimeConfig.wifiSsid);
    WiFi.begin(runtimeConfig.wifiSsid, runtimeConfig.wifiPassword);
  } else {
    Serial.printf("'%s' not in range; starting AP now, rescanning every %lus\n",
                  runtimeConfig.wifiSsid,
                  static_cast<unsigned long>(kStaScanIntervalMs / 1000));
    startAccessPoint(false);
  }
  lastStaAttemptMs = millis();
}

bool publishZenohBytes(z_owned_publisher_t *publisher, const uint8_t *data, size_t len) {
  if (!zenohReady) {
    return false;
  }

  const uint32_t startedUs = micros();
  z_owned_bytes_t payload;
  if (z_bytes_copy_from_buf(&payload, data, len) < 0) {
    return false;
  }

  const bool ok = z_publisher_put(z_publisher_loan(publisher), z_bytes_move(&payload), NULL) >= 0;
  const uint32_t elapsedUs = static_cast<uint32_t>(micros() - startedUs);
  maxPublishUs = max(maxPublishUs, elapsedUs);
  return ok;
}

bool declareZenohPublisher(const char *key,
                           z_owned_publisher_t *publisher,
                           z_priority_t priority,
                           z_congestion_control_t congestion) {
  z_view_keyexpr_t keyexpr;
  z_view_keyexpr_from_str_unchecked(&keyexpr, key);

  z_publisher_options_t options;
  z_publisher_options_default(&options);
  options.congestion_control = congestion;
  options.priority = priority;
  options.is_express = true;

  Serial.printf("Declaring Zenoh publisher: %s\n", key);
  return z_declare_publisher(z_session_loan(&zenohSession),
                             publisher,
                             z_view_keyexpr_loan(&keyexpr),
                             &options) >= 0;
}

bool declareZenohSubscriber(const char *key,
                            z_owned_subscriber_t *subscriber,
                            z_closure_sample_callback_t handler) {
  z_owned_closure_sample_t callback;
  z_closure_sample(&callback, handler, NULL, NULL);
  z_view_keyexpr_t keyexpr;
  z_view_keyexpr_from_str_unchecked(&keyexpr, key);

  Serial.printf("Declaring Zenoh subscriber: %s\n", key);
  return z_declare_subscriber(z_session_loan(&zenohSession),
                              subscriber,
                              z_view_keyexpr_loan(&keyexpr),
                              z_closure_sample_move(&callback),
                              NULL) >= 0;
}

size_t readZenohSamplePayload(z_loaned_sample_t *sample, uint8_t *buffer, size_t capacity) {
  if (capacity == 0) {
    return 0;
  }
  z_bytes_reader_t reader = z_bytes_get_reader(z_sample_payload(sample));
  return z_bytes_reader_read(&reader, buffer, capacity);
}

const char *skipSpace(const char *text) {
  while (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n') {
    ++text;
  }
  return text;
}

bool parseJsonFloatField(const char *payload, const char *key, float *value) {
  const char *field = strstr(payload, key);
  if (field == nullptr) {
    return false;
  }
  const char *colon = strchr(field, ':');
  if (colon == nullptr) {
    return false;
  }
  char *end = nullptr;
  const float parsed = strtof(colon + 1, &end);
  if (end == colon + 1 || !isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

bool parseNextFloat(const char **cursor, float *value) {
  const char *p = *cursor;
  while (*p != '\0' && *p != '-' && *p != '+' && *p != '.' && !isdigit(*p)) {
    ++p;
  }
  if (*p == '\0') {
    return false;
  }

  char *end = nullptr;
  const float parsed = strtof(p, &end);
  if (end == p || !isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  *cursor = end;
  return true;
}

bool parseVelocityPayload(z_loaned_sample_t *sample, VelocityCommandPayload *command) {
  uint8_t raw[128] = {};
  const size_t len = readZenohSamplePayload(sample, raw, sizeof(raw) - 1);

  if (len == sizeof(VelocityCommandPayload)) {
    memcpy(command, raw, sizeof(*command));
    command->master_time_us = static_cast<uint64_t>(esp_timer_get_time());
    return true;
  }

  char *text = reinterpret_cast<char *>(raw);
  text[len] = '\0';
  const char *trimmed = skipSpace(text);
  command->master_time_us = static_cast<uint64_t>(esp_timer_get_time());
  command->timeout_ms = kiwi_config::kVelocityCommandTimeoutMs;
  command->mode = static_cast<uint8_t>(VelocityMode::BodyTwist);
  command->reserved = 0;

  if (*trimmed == '{') {
    return parseJsonFloatField(trimmed, "\"vx\"", &command->vx_mps) &&
           parseJsonFloatField(trimmed, "\"vy\"", &command->vy_mps) &&
           parseJsonFloatField(trimmed, "\"omega\"", &command->omega_radps);
  }

  const char *cursor = trimmed;
  return parseNextFloat(&cursor, &command->vx_mps) &&
         parseNextFloat(&cursor, &command->vy_mps) &&
         parseNextFloat(&cursor, &command->omega_radps);
}

void sendVelocityCommand(const VelocityCommandPayload &command) {
  if (kiwi::writePacket(FollowerUart,
                        MessageType::VelocityCommand,
                        followerTxSeq++,
                        &command,
                        sizeof(command))) {
    ++velocityCommands;
  }
}

void cmdVelHandler(z_loaned_sample_t *sample, void *arg) {
  (void)arg;
  VelocityCommandPayload command = {};
  if (!parseVelocityPayload(sample, &command)) {
    Serial.println("Invalid cmd_vel payload. Expected binary VelocityCommandPayload, JSON, or 'vx vy omega'.");
    return;
  }
  sendVelocityCommand(command);
}

bool startZenohSession() {
  z_owned_config_t config;
  z_config_default(&config);
  zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_MODE_KEY, runtimeConfig.zenohMode);
  if (strlen(runtimeConfig.zenohConnect) > 0) {
    zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_CONNECT_KEY, runtimeConfig.zenohConnect);
  }

  Serial.printf("Opening Zenoh session mode=%s connect=%s\n",
                runtimeConfig.zenohMode,
                strlen(runtimeConfig.zenohConnect) > 0 ? runtimeConfig.zenohConnect : "(default)");
  if (z_open(&zenohSession, z_config_move(&config), NULL) < 0) {
    Serial.println("Unable to open Zenoh session.");
    return false;
  }

  // zenoh-pico needs its background tasks: the read task delivers subscriber
  // callbacks (cmd_vel) and the lease task keeps the session alive.
  if (zp_start_read_task(z_session_loan_mut(&zenohSession), NULL) < 0 ||
      zp_start_lease_task(z_session_loan_mut(&zenohSession), NULL) < 0) {
    Serial.println("Unable to start Zenoh read/lease tasks.");
    z_session_drop(z_session_move(&zenohSession));
    return false;
  }

  if (!declareZenohPublisher(kZenohCameraKey, &cameraPub, Z_PRIORITY_DATA_HIGH,
                             Z_CONGESTION_CONTROL_DROP) ||
      !declareZenohPublisher(kZenohLidarRawKey, &lidarPub, Z_PRIORITY_REAL_TIME,
                             Z_CONGESTION_CONTROL_DROP) ||
      !declareZenohPublisher(kZenohTwistKey, &twistPub, Z_PRIORITY_REAL_TIME,
                             Z_CONGESTION_CONTROL_DROP) ||
      !declareZenohPublisher(kZenohStatusKey, &statusPub, Z_PRIORITY_INTERACTIVE_LOW,
                             Z_CONGESTION_CONTROL_DROP) ||
      !declareZenohSubscriber(kZenohCmdVelKey, &cmdVelSub, cmdVelHandler)) {
    Serial.println("Unable to declare one or more Zenoh resources.");
    z_session_drop(z_session_move(&zenohSession));
    return false;
  }

  zenohReady = true;
  Serial.println("Zenoh ready.");
  return true;
}

void publishCameraFrame() {
  if (!cameraReady) {
    ++cameraPublishErrors;
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (fb == nullptr) {
    ++cameraPublishErrors;
    Serial.println("Camera capture failed.");
    return;
  }

  const size_t totalLen = kCameraHeaderBytes + fb->len;
  uint8_t *payload = static_cast<uint8_t *>(malloc(totalLen));
  if (payload == nullptr) {
    esp_camera_fb_return(fb);
    ++cameraPublishErrors;
    return;
  }

  memset(payload, 0, kCameraHeaderBytes);
  memcpy(payload, "KVC1", 4);
  payload[4] = 1;
  payload[5] = static_cast<uint8_t>(fb->format);
  writeLe16(payload + 6, fb->width);
  writeLe16(payload + 8, fb->height);
  writeLe16(payload + 10, kCameraHeaderBytes);
  writeLe32(payload + 12, cameraSeq++);
  writeLe64(payload + 16, static_cast<uint64_t>(esp_timer_get_time()));
  writeLe32(payload + 24, fb->len);
  memcpy(payload + kCameraHeaderBytes, fb->buf, fb->len);

  const bool ok = publishZenohBytes(&cameraPub, payload, totalLen);
  free(payload);
  esp_camera_fb_return(fb);

  if (ok) {
    ++cameraPublished;
  } else {
    ++cameraPublishErrors;
  }
}

void publishLidarBatch() {
  if (lidarBatchFrames == 0) {
    return;
  }

  const uint32_t frameCount = static_cast<uint32_t>(lidarBatchFrames);
  const size_t byteCount = lidarBatchFrames * kLd19FrameBytes;
  if (publishZenohBytes(&lidarPub, lidarBatch, byteCount)) {
    lidarPublished += frameCount;
    ++lidarBatchesPublished;
  } else {
    lidarPublishErrors += frameCount;
  }
  lidarBatchFrames = 0;
  lidarBatchStartedMs = 0;
}

void queueLidarFrame(const uint8_t *frame) {
  if (lidarBatchFrames == 0) {
    lidarBatchStartedMs = millis();
  }
  memcpy(lidarBatch + lidarBatchFrames * kLd19FrameBytes, frame, kLd19FrameBytes);
  ++lidarBatchFrames;
  if (lidarBatchFrames == kLd19FramesPerBatch) {
    publishLidarBatch();
  }
}

void publishTwistReport(const TwistReportPayload &report) {
  char payload[768];
  const int written = snprintf(
      payload,
      sizeof(payload),
      "{\"follower_time_us\":%llu,\"seq\":%lu,"
      "\"measured\":{\"vx\":%.6f,\"vy\":%.6f,\"omega\":%.6f},"
      "\"command\":{\"vx\":%.6f,\"vy\":%.6f,\"omega\":%.6f},"
      "\"wheel_speed_mps\":[%.6f,%.6f,%.6f],"
      "\"wheel_angle_rad\":[%.6f,%.6f,%.6f],"
      "\"encoder_count\":[%lld,%lld,%lld],"
      "\"imu_ready\":%s,\"encoder_ready_mask\":%u,\"status_flags\":%u,"
      "\"imu_quat_ijkr\":[%.6f,%.6f,%.6f,%.6f],"
      "\"imu_accel_mps2\":[%.4f,%.4f,%.4f]}",
      static_cast<unsigned long long>(report.follower_time_us),
      static_cast<unsigned long>(report.report_seq),
      report.measured_vx_mps,
      report.measured_vy_mps,
      report.measured_omega_radps,
      report.command_vx_mps,
      report.command_vy_mps,
      report.command_omega_radps,
      report.wheel_speed_mps[0],
      report.wheel_speed_mps[1],
      report.wheel_speed_mps[2],
      report.wheel_angle_rad[0],
      report.wheel_angle_rad[1],
      report.wheel_angle_rad[2],
      static_cast<long long>(report.encoder_count[0]),
      static_cast<long long>(report.encoder_count[1]),
      static_cast<long long>(report.encoder_count[2]),
      report.imu_ready != 0 ? "true" : "false",
      report.encoder_ready_mask,
      report.status_flags,
      report.imu_quat_ijkr[0],
      report.imu_quat_ijkr[1],
      report.imu_quat_ijkr[2],
      report.imu_quat_ijkr[3],
      report.imu_accel_mps2[0],
      report.imu_accel_mps2[1],
      report.imu_accel_mps2[2]);

  if (written <= 0) {
    ++twistPublishErrors;
    return;
  }

  if (publishZenohBytes(&twistPub,
                        reinterpret_cast<const uint8_t *>(payload),
                        min(static_cast<size_t>(written), sizeof(payload) - 1))) {
    ++twistPublished;
  } else {
    ++twistPublishErrors;
  }
}

void publishStatus() {
  char payload[768];
  const int written = snprintf(
      payload,
      sizeof(payload),
      "{\"esp_ms\":%lu,\"sta_connected\":%s,\"sta_ip\":\"%s\",\"rssi\":%d,"
      "\"camera_ready\":%s,\"zenoh_ready\":%s,"
      "\"lidar_frames\":%lu,\"lidar_bad_frames\":%lu,"
      "\"follower_reports\":%lu,\"follower_bad_packets\":%lu,"
      "\"velocity_commands\":%lu,\"camera_published\":%lu,\"camera_errors\":%lu,"
      "\"lidar_published\":%lu,\"lidar_errors\":%lu,"
      "\"lidar_batches\":%lu,\"twist_published\":%lu,\"twist_errors\":%lu,"
      "\"loop_gap_max_us\":%lu,\"publish_max_us\":%lu,"
      "\"lidar_rx_high_water\":%lu,\"follower_rx_high_water\":%lu,\"free_heap\":%lu}",
      static_cast<unsigned long>(millis()),
      WiFi.status() == WL_CONNECTED ? "true" : "false",
      WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : "",
      WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0,
      cameraReady ? "true" : "false",
      zenohReady ? "true" : "false",
      static_cast<unsigned long>(lidarFrames),
      static_cast<unsigned long>(lidarBadFrames),
      static_cast<unsigned long>(followerReports),
      static_cast<unsigned long>(followerBadPackets),
      static_cast<unsigned long>(velocityCommands),
      static_cast<unsigned long>(cameraPublished),
      static_cast<unsigned long>(cameraPublishErrors),
      static_cast<unsigned long>(lidarPublished),
      static_cast<unsigned long>(lidarPublishErrors),
      static_cast<unsigned long>(lidarBatchesPublished),
      static_cast<unsigned long>(twistPublished),
      static_cast<unsigned long>(twistPublishErrors),
      static_cast<unsigned long>(maxLoopGapUs),
      static_cast<unsigned long>(maxPublishUs),
      static_cast<unsigned long>(lidarRxHighWater),
      static_cast<unsigned long>(followerRxHighWater),
      static_cast<unsigned long>(ESP.getFreeHeap()));

  if (written > 0) {
    publishZenohBytes(&statusPub,
                      reinterpret_cast<const uint8_t *>(payload),
                      min(static_cast<size_t>(written), sizeof(payload) - 1));
  }
}

void processFollowerUart() {
  Packet packet;
  followerRxHighWater = max(followerRxHighWater,
                            static_cast<uint32_t>(FollowerUart.available()));
  size_t processed = 0;
  while (processed < kMaxFollowerPacketsPerLoop &&
         followerReader.readFrom(FollowerUart, packet)) {
    ++processed;
    if (packet.type == MessageType::TwistReport &&
        packet.payloadLength == sizeof(TwistReportPayload)) {
      memcpy(&latestTwistReport, packet.payload, sizeof(latestTwistReport));
      ++followerReports;
      twistReportPending = true;
    } else if (packet.type == MessageType::DriveParamsAck &&
               packet.payloadLength == sizeof(kiwi::DriveParamsAckPayload)) {
      kiwi::DriveParamsAckPayload ack = {};
      memcpy(&ack, packet.payload, sizeof(ack));
      if (ack.version != driveParamsAckedVersion) {
        Serial.printf("Follower acked drive params version=%lu\n",
                      static_cast<unsigned long>(ack.version));
      }
      driveParamsAckedVersion = ack.version;
    } else {
      ++followerBadPackets;
    }
  }
}

void handleHttpStatus() {
  JsonDocument doc;
  doc["ap_ip"] = WiFi.softAPIP().toString();
  doc["sta_connected"] = WiFi.status() == WL_CONNECTED;
  doc["sta_ip"] = WiFi.localIP().toString();
  doc["wifi_ssid"] = runtimeConfig.wifiSsid;
  doc["zenoh_mode"] = runtimeConfig.zenohMode;
  doc["zenoh_connect"] = runtimeConfig.zenohConnect;
  doc["zenoh_ready"] = zenohReady;
  doc["camera_ready"] = cameraReady;
  doc["lidar_frames"] = lidarFrames;
  doc["follower_reports"] = followerReports;
  JsonObject drive = doc["drive"].to<JsonObject>();
  drive["version"] = runtimeConfig.drive.version;
  drive["wheel_radius_m"] = runtimeConfig.drive.wheel_radius_m;
  drive["drive_base_radius_m"] = runtimeConfig.drive.drive_base_radius_m;
  drive["max_wheel_surface_speed_mps"] = runtimeConfig.drive.max_wheel_surface_speed_mps;
  JsonArray polarity = drive["motor_polarity"].to<JsonArray>();
  for (uint8_t i = 0; i < 3; ++i) {
    polarity.add(runtimeConfig.drive.motor_polarity[i]);
  }
  drive["velocity_command_timeout_ms"] = runtimeConfig.drive.velocity_command_timeout_ms;
  JsonArray deadband = drive["motor_deadband_pct"].to<JsonArray>();
  for (uint8_t i = 0; i < 3; ++i) {
    deadband.add(runtimeConfig.drive.motor_deadband_pct[i]);
  }
  drive["pid_kp"] = runtimeConfig.drive.pid_kp;
  drive["pid_ki"] = runtimeConfig.drive.pid_ki;
  drive["closed_loop"] = runtimeConfig.drive.closed_loop;
  JsonArray encPolarity = drive["encoder_polarity"].to<JsonArray>();
  for (uint8_t i = 0; i < 3; ++i) {
    encPolarity.add(runtimeConfig.drive.encoder_polarity[i]);
  }
  drive["acked_by_follower"] = driveParamsAckedVersion == runtimeConfig.drive.version;

  String out;
  serializeJson(doc, out);
  httpServer.send(200, "application/json", out);
}

void sendHttpError(int code, const char *message) {
  String out = "{\"ok\":false,\"error\":\"";
  out += message;
  out += "\"}";
  httpServer.send(code, "application/json", out);
}

bool copyJsonString(JsonDocument &doc, const char *key, char *dst, size_t dstSize, bool *changed) {
  if (!doc[key].is<const char *>()) {
    return true;
  }
  const char *value = doc[key];
  if (strlen(value) >= dstSize) {
    return false;
  }
  if (strcmp(dst, value) != 0) {
    strlcpy(dst, value, dstSize);
    *changed = true;
  }
  return true;
}

void handleHttpConfig() {
  if (!httpServer.hasArg("plain")) {
    sendHttpError(400, "missing JSON body");
    return;
  }

  JsonDocument doc;
  if (deserializeJson(doc, httpServer.arg("plain"))) {
    sendHttpError(400, "invalid JSON");
    return;
  }

  bool networkChanged = false;
  if (!copyJsonString(doc, "wifi_ssid", runtimeConfig.wifiSsid,
                      sizeof(runtimeConfig.wifiSsid), &networkChanged) ||
      !copyJsonString(doc, "wifi_password", runtimeConfig.wifiPassword,
                      sizeof(runtimeConfig.wifiPassword), &networkChanged) ||
      !copyJsonString(doc, "zenoh_connect", runtimeConfig.zenohConnect,
                      sizeof(runtimeConfig.zenohConnect), &networkChanged)) {
    sendHttpError(400, "string field too long");
    return;
  }
  if (doc["zenoh_mode"].is<const char *>()) {
    const char *mode = doc["zenoh_mode"];
    if (strcmp(mode, "client") != 0 && strcmp(mode, "peer") != 0) {
      sendHttpError(400, "zenoh_mode must be client or peer");
      return;
    }
    if (strcmp(runtimeConfig.zenohMode, mode) != 0) {
      strlcpy(runtimeConfig.zenohMode, mode, sizeof(runtimeConfig.zenohMode));
      networkChanged = true;
    }
  }

  kiwi::DriveParamsPayload drive = runtimeConfig.drive;
  bool driveChanged = false;
  if (doc["wheel_radius_m"].is<float>()) {
    drive.wheel_radius_m = doc["wheel_radius_m"];
    driveChanged = true;
  }
  if (doc["drive_base_radius_m"].is<float>()) {
    drive.drive_base_radius_m = doc["drive_base_radius_m"];
    driveChanged = true;
  }
  if (doc["max_wheel_surface_speed_mps"].is<float>()) {
    drive.max_wheel_surface_speed_mps = doc["max_wheel_surface_speed_mps"];
    driveChanged = true;
  }
  if (doc["velocity_command_timeout_ms"].is<unsigned int>()) {
    drive.velocity_command_timeout_ms = doc["velocity_command_timeout_ms"];
    driveChanged = true;
  }
  if (doc["motor_polarity"].is<JsonArray>()) {
    JsonArray polarity = doc["motor_polarity"];
    if (polarity.size() != 3) {
      sendHttpError(400, "motor_polarity must have 3 entries");
      return;
    }
    for (uint8_t i = 0; i < 3; ++i) {
      drive.motor_polarity[i] = polarity[i].as<int>();
    }
    driveChanged = true;
  }
  if (doc["encoder_polarity"].is<JsonArray>()) {
    JsonArray encPolarity = doc["encoder_polarity"];
    if (encPolarity.size() != 3) {
      sendHttpError(400, "encoder_polarity must have 3 entries");
      return;
    }
    for (uint8_t i = 0; i < 3; ++i) {
      drive.encoder_polarity[i] = encPolarity[i].as<int>();
    }
    driveChanged = true;
  }
  if (doc["motor_deadband_pct"].is<JsonArray>()) {
    JsonArray deadband = doc["motor_deadband_pct"];
    if (deadband.size() != 3) {
      sendHttpError(400, "motor_deadband_pct must have 3 entries");
      return;
    }
    for (uint8_t i = 0; i < 3; ++i) {
      drive.motor_deadband_pct[i] = deadband[i].as<unsigned int>();
    }
    driveChanged = true;
  }
  if (doc["pid_kp"].is<float>()) {
    drive.pid_kp = doc["pid_kp"];
    driveChanged = true;
  }
  if (doc["pid_ki"].is<float>()) {
    drive.pid_ki = doc["pid_ki"];
    driveChanged = true;
  }
  if (doc["closed_loop"].is<unsigned int>() || doc["closed_loop"].is<bool>()) {
    drive.closed_loop = doc["closed_loop"].as<bool>() ? 1 : 0;
    driveChanged = true;
  }

  if (driveChanged) {
    drive.version = runtimeConfig.drive.version + 1;
    if (!kiwi::driveParamsValid(drive)) {
      sendHttpError(400, "drive parameter out of range");
      return;
    }
    runtimeConfig.drive = drive;
    lastDriveParamsSendMs = 0;  // push to the follower immediately
  }

  saveRuntimeConfig();

  JsonDocument resp;
  resp["ok"] = true;
  resp["reboot"] = networkChanged;
  resp["drive_version"] = runtimeConfig.drive.version;
  String out;
  serializeJson(resp, out);
  httpServer.send(200, "application/json", out);

  if (networkChanged) {
    Serial.println("Network config changed via /config; rebooting to apply.");
    rebootPending = true;
    rebootAtMs = millis() + 1500;
  }
}

// Keep the STA link and Zenoh session alive from the main loop: retry the
// join until it lands (routers can be slow right after a reboot) and
// re-establish both after any drop.
void maintainNetwork(uint32_t nowMs) {
  if (strlen(runtimeConfig.wifiSsid) == 0) {
    return;
  }

  const bool staConnected = WiFi.status() == WL_CONNECTED;
  if (staConnected && !staWasConnected) {
    WiFi.setSleep(false);
    Serial.printf("STA Wi-Fi ready: %s RSSI=%d\n",
                  WiFi.localIP().toString().c_str(),
                  WiFi.RSSI());
    lastZenohAttemptMs = nowMs - kZenohRetryMs;  // try Zenoh right away
  } else if (!staConnected && staWasConnected) {
    Serial.println("STA Wi-Fi lost; retrying in the background.");
    zenohReady = false;
  }
  staWasConnected = staConnected;

  if (staConnected || nowMs >= kApFallbackMs) {
    startAccessPoint(staConnected);
  }

  if (!staConnected) {
    // Scan-gated rejoin: an async scan every 30 s, and WiFi.begin only when
    // the SSID is actually visible. Keeps the AP's beacons stable otherwise.
    if (staScanInFlight) {
      const int16_t n = WiFi.scanComplete();
      if (n >= 0) {
        bool visible = false;
        for (int16_t i = 0; i < n; ++i) {
          if (WiFi.SSID(i) == runtimeConfig.wifiSsid) {
            visible = true;
            break;
          }
        }
        WiFi.scanDelete();
        staScanInFlight = false;
        if (visible) {
          Serial.printf("'%s' in range; attempting STA join\n", runtimeConfig.wifiSsid);
          WiFi.begin(runtimeConfig.wifiSsid, runtimeConfig.wifiPassword);
        }
      } else if (n == WIFI_SCAN_FAILED) {
        staScanInFlight = false;
      }
    } else if (nowMs - lastStaAttemptMs >= kStaScanIntervalMs) {
      lastStaAttemptMs = nowMs;
      WiFi.scanNetworks(true /* async */);
      staScanInFlight = true;
    }
    return;
  }

  if (!zenohReady && nowMs - lastZenohAttemptMs >= kZenohRetryMs) {
    lastZenohAttemptMs = nowMs;
    zenohReady = startZenohSession();
  }
}

void startHttpServer() {
  httpServer.on("/status", HTTP_GET, handleHttpStatus);
  httpServer.on("/config", HTTP_POST, handleHttpConfig);
  httpServer.onNotFound([]() { sendHttpError(404, "not found"); });
  httpServer.begin();
  httpServerStarted = true;
  Serial.printf("Provisioning HTTP server on http://%s/ (also on STA IP when connected)\n",
                WiFi.softAPIP().toString().c_str());
}

void processLidar() {
  lidarRxHighWater = max(lidarRxHighWater,
                         static_cast<uint32_t>(LidarUart.available()));
  uint8_t frame[kLd19FrameBytes] = {};
  size_t processed = 0;
  while (processed < kMaxLd19FramesPerLoop &&
         lidarReader.readFrom(LidarUart, frame)) {
    ++processed;
    if (!ld19FrameCrcOk(frame)) {
      ++lidarBadFrames;
      continue;
    }
    ++lidarFrames;
    queueLidarFrame(frame);
  }
  if (lidarBatchFrames > 0 &&
      millis() - lidarBatchStartedMs >= kLd19BatchMaxAgeMs) {
    publishLidarBatch();
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("Booting kiwi master: camera + LD19 lidar + Zenoh + follower UART");
  Serial.printf("Namespace: %s\n", ROBOT_NAMESPACE);

  initLd19CrcTable();
  loadRuntimeConfig();

  // Generous RX buffers: blocking Zenoh publishes can stall the loop for
  // tens of milliseconds, and both streams must survive that without loss.
  FollowerUart.setRxBufferSize(4096);
  FollowerUart.begin(kiwi_config::kFollowerUartBaud,
                     SERIAL_8N1,
                     kiwi_config::kFollowerUartRxPin,
                     kiwi_config::kFollowerUartTxPin);
  // LD19 is transmit-only on serial; no TX pin needed toward the lidar.
  LidarUart.setRxBufferSize(8192);
  LidarUart.begin(kiwi_config::kLidarBaud,
                  SERIAL_8N1,
                  kiwi_config::kLidarRxPin,
                  -1);
  ledcSetup(kiwi_config::kLidarPwmChannel,
            kiwi_config::kLidarPwmFrequencyHz,
            kiwi_config::kLidarPwmResolutionBits);
  ledcAttachPin(kiwi_config::kLidarPwmPin, kiwi_config::kLidarPwmChannel);
  ledcWrite(kiwi_config::kLidarPwmChannel, kiwi_config::kLidarPwmDuty);
  Serial.printf("Follower UART RX=GPIO%u TX=GPIO%u baud=%lu\n",
                kiwi_config::kFollowerUartRxPin,
                kiwi_config::kFollowerUartTxPin,
                static_cast<unsigned long>(kiwi_config::kFollowerUartBaud));
  Serial.printf("LD19 UART RX=GPIO%u PWM=GPIO%u baud=%lu\n",
                kiwi_config::kLidarRxPin,
                kiwi_config::kLidarPwmPin,
                static_cast<unsigned long>(kiwi_config::kLidarBaud));

  cameraReady = initCamera();
  startWifi();
  startHttpServer();
  // Zenoh starts from maintainNetwork() once the STA link is up.
}

void loop() {
  const uint32_t loopStartUs = micros();
  if (lastLoopStartUs != 0) {
    maxLoopGapUs = max(maxLoopGapUs, loopStartUs - lastLoopStartUs);
  }
  lastLoopStartUs = loopStartUs;

  processFollowerUart();
  processLidar();
  httpServer.handleClient();

  const uint32_t nowMs = millis();
  maintainNetwork(nowMs);

  if (driveParamsAckedVersion != runtimeConfig.drive.version &&
      nowMs - lastDriveParamsSendMs >= kDriveParamsResendMs) {
    lastDriveParamsSendMs = nowMs;
    sendDriveParamsToFollower();
  }

  if (rebootPending && static_cast<int32_t>(nowMs - rebootAtMs) >= 0) {
    Serial.println("Rebooting now.");
    Serial.flush();
    ESP.restart();
  }
  if (twistReportPending && nowMs - lastTwistPublishMs >= kTwistPublishPeriodMs) {
    lastTwistPublishMs = nowMs;
    twistReportPending = false;
    publishTwistReport(latestTwistReport);
  }
  if (nowMs - lastCameraPublishMs >= kiwi_config::kCameraPublishPeriodMs) {
    lastCameraPublishMs = nowMs;
    publishCameraFrame();
  }

  if (nowMs - lastStatusPublishMs >= kiwi_config::kMasterStatusPeriodMs) {
    lastStatusPublishMs = nowMs;
    publishStatus();
    Serial.printf("master status zenoh=%s wifi=%s lidar=%lu lidar_bytes=%lu lidar_bad=%lu "
                  "follower=%lu follower_bad=%lu cmd=%lu cam=%lu/%lu lid_pub=%lu/%lu "
                  "lid_batch=%lu tw_pub=%lu/%lu loop_max=%luus pub_max=%luus "
                  "rx_hi=%lu/%lu heap=%lu\n",
                  zenohReady ? "ok" : "off",
                  WiFi.status() == WL_CONNECTED ? "ok" : "off",
                  static_cast<unsigned long>(lidarFrames),
                  static_cast<unsigned long>(lidarBytes),
                  static_cast<unsigned long>(lidarBadFrames),
                  static_cast<unsigned long>(followerReports),
                  static_cast<unsigned long>(followerBadPackets),
                  static_cast<unsigned long>(velocityCommands),
                  static_cast<unsigned long>(cameraPublished),
                  static_cast<unsigned long>(cameraPublishErrors),
                  static_cast<unsigned long>(lidarPublished),
                  static_cast<unsigned long>(lidarPublishErrors),
                  static_cast<unsigned long>(lidarBatchesPublished),
                  static_cast<unsigned long>(twistPublished),
                  static_cast<unsigned long>(twistPublishErrors),
                  static_cast<unsigned long>(maxLoopGapUs),
                  static_cast<unsigned long>(maxPublishUs),
                  static_cast<unsigned long>(lidarRxHighWater),
                  static_cast<unsigned long>(followerRxHighWater),
                  static_cast<unsigned long>(ESP.getFreeHeap()));
    maxLoopGapUs = 0;
    maxPublishUs = 0;
    lidarRxHighWater = 0;
    followerRxHighWater = 0;
  }

  delay(1);
}
