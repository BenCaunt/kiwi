#include <Arduino.h>
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
constexpr uint32_t kWifiConnectTimeoutMs = 15000;
constexpr size_t kCameraHeaderBytes = 32;
constexpr size_t kLd19FrameBytes = 47;

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
SemaphoreHandle_t zenohPublishMutex = nullptr;

bool zenohReady = false;
bool cameraReady = false;
uint16_t followerTxSeq = 0;
uint32_t cameraSeq = 0;
uint32_t lidarFrames = 0;
uint32_t lidarBadFrames = 0;
uint32_t followerReports = 0;
uint32_t followerBadPackets = 0;
uint32_t velocityCommands = 0;
uint32_t cameraPublished = 0;
uint32_t cameraPublishErrors = 0;
uint32_t lidarPublished = 0;
uint32_t lidarPublishErrors = 0;
uint32_t twistPublished = 0;
uint32_t twistPublishErrors = 0;
uint32_t lastCameraPublishMs = 0;
uint32_t lastStatusPublishMs = 0;

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

bool connectWifi() {
  WiFi.persistent(false);
  WiFi.mode(WIFI_AP_STA);
  WiFi.setSleep(false);
  WiFi.softAP(kApSsid, kApPassword);
  Serial.printf("Master AP: %s / %s at %s\n",
                kApSsid,
                kApPassword,
                WiFi.softAPIP().toString().c_str());

  if (strlen(WIFI_SSID) == 0) {
    Serial.println("No WIFI_SSID configured; Zenoh will stay disabled until secrets.h is added.");
    return false;
  }

  Serial.printf("Connecting STA Wi-Fi to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const uint32_t startMs = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startMs < kWifiConnectTimeoutMs) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("STA Wi-Fi timed out, status=%d\n", static_cast<int>(WiFi.status()));
    return false;
  }

  Serial.printf("STA Wi-Fi ready: %s RSSI=%d\n",
                WiFi.localIP().toString().c_str(),
                WiFi.RSSI());
  return true;
}

bool publishZenohBytes(z_owned_publisher_t *publisher, const uint8_t *data, size_t len) {
  if (!zenohReady || zenohPublishMutex == nullptr) {
    return false;
  }

  z_owned_bytes_t payload;
  if (z_bytes_copy_from_buf(&payload, data, len) < 0) {
    return false;
  }

  if (xSemaphoreTake(zenohPublishMutex, pdMS_TO_TICKS(250)) != pdTRUE) {
    z_bytes_drop(z_bytes_move(&payload));
    return false;
  }

  const bool ok = z_publisher_put(z_publisher_loan(publisher), z_bytes_move(&payload), NULL) >= 0;
  xSemaphoreGive(zenohPublishMutex);
  return ok;
}

bool declareZenohPublisher(const char *key, z_owned_publisher_t *publisher, z_priority_t priority) {
  z_view_keyexpr_t keyexpr;
  z_view_keyexpr_from_str_unchecked(&keyexpr, key);

  z_publisher_options_t options;
  z_publisher_options_default(&options);
  options.congestion_control = Z_CONGESTION_CONTROL_DROP;
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
  zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_MODE_KEY, ZENOH_MODE);
  if (strlen(ZENOH_CONNECT) > 0) {
    zp_config_insert(z_config_loan_mut(&config), Z_CONFIG_CONNECT_KEY, ZENOH_CONNECT);
  }

  Serial.printf("Opening Zenoh session mode=%s connect=%s\n",
                ZENOH_MODE,
                strlen(ZENOH_CONNECT) > 0 ? ZENOH_CONNECT : "(default)");
  if (z_open(&zenohSession, z_config_move(&config), NULL) < 0) {
    Serial.println("Unable to open Zenoh session.");
    return false;
  }

  if (!declareZenohPublisher(kZenohCameraKey, &cameraPub, Z_PRIORITY_DATA_HIGH) ||
      !declareZenohPublisher(kZenohLidarRawKey, &lidarPub, Z_PRIORITY_REAL_TIME) ||
      !declareZenohPublisher(kZenohTwistKey, &twistPub, Z_PRIORITY_REAL_TIME) ||
      !declareZenohPublisher(kZenohStatusKey, &statusPub, Z_PRIORITY_INTERACTIVE_LOW) ||
      !declareZenohSubscriber(kZenohCmdVelKey, &cmdVelSub, cmdVelHandler)) {
    Serial.println("Unable to declare one or more Zenoh resources.");
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

void publishLidarFrame(const uint8_t *frame) {
  if (publishZenohBytes(&lidarPub, frame, kLd19FrameBytes)) {
    ++lidarPublished;
  } else {
    ++lidarPublishErrors;
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
      "\"imu_ready\":%s,\"encoder_ready_mask\":%u,\"status_flags\":%u}",
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
      report.status_flags);

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
  char payload[512];
  const int written = snprintf(
      payload,
      sizeof(payload),
      "{\"esp_ms\":%lu,\"sta_connected\":%s,\"sta_ip\":\"%s\",\"rssi\":%d,"
      "\"camera_ready\":%s,\"zenoh_ready\":%s,"
      "\"lidar_frames\":%lu,\"lidar_bad_frames\":%lu,"
      "\"follower_reports\":%lu,\"follower_bad_packets\":%lu,"
      "\"velocity_commands\":%lu,\"camera_published\":%lu,\"camera_errors\":%lu,"
      "\"lidar_published\":%lu,\"lidar_errors\":%lu,"
      "\"twist_published\":%lu,\"twist_errors\":%lu,\"free_heap\":%lu}",
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
      static_cast<unsigned long>(twistPublished),
      static_cast<unsigned long>(twistPublishErrors),
      static_cast<unsigned long>(ESP.getFreeHeap()));

  if (written > 0) {
    publishZenohBytes(&statusPub,
                      reinterpret_cast<const uint8_t *>(payload),
                      min(static_cast<size_t>(written), sizeof(payload) - 1));
  }
}

void processFollowerUart() {
  Packet packet;
  while (followerReader.readFrom(FollowerUart, packet)) {
    if (packet.type != MessageType::TwistReport ||
        packet.payloadLength != sizeof(TwistReportPayload)) {
      ++followerBadPackets;
      continue;
    }

    TwistReportPayload report = {};
    memcpy(&report, packet.payload, sizeof(report));
    ++followerReports;
    publishTwistReport(report);
  }
}

void processLidar() {
  uint8_t frame[kLd19FrameBytes] = {};
  while (lidarReader.readFrom(LidarUart, frame)) {
    ++lidarFrames;
    publishLidarFrame(frame);
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("Booting kiwi master: camera + LD19 lidar + Zenoh + follower UART");
  Serial.printf("Namespace: %s\n", ROBOT_NAMESPACE);

  FollowerUart.begin(kiwi_config::kFollowerUartBaud,
                     SERIAL_8N1,
                     kiwi_config::kFollowerUartRxPin,
                     kiwi_config::kFollowerUartTxPin);
  LidarUart.begin(kiwi_config::kLidarBaud,
                  SERIAL_8N1,
                  kiwi_config::kLidarRxPin,
                  kiwi_config::kLidarTxPin);
  Serial.printf("Follower UART RX=GPIO%u TX=GPIO%u baud=%lu\n",
                kiwi_config::kFollowerUartRxPin,
                kiwi_config::kFollowerUartTxPin,
                static_cast<unsigned long>(kiwi_config::kFollowerUartBaud));
  Serial.printf("LD19 UART RX=GPIO%u TX=GPIO%u baud=%lu\n",
                kiwi_config::kLidarRxPin,
                kiwi_config::kLidarTxPin,
                static_cast<unsigned long>(kiwi_config::kLidarBaud));

  cameraReady = initCamera();
  const bool wifiReady = connectWifi();
  zenohPublishMutex = xSemaphoreCreateMutex();
  if (wifiReady && zenohPublishMutex != nullptr) {
    zenohReady = startZenohSession();
  }
}

void loop() {
  processFollowerUart();
  processLidar();

  const uint32_t nowMs = millis();
  if (nowMs - lastCameraPublishMs >= kiwi_config::kCameraPublishPeriodMs) {
    lastCameraPublishMs = nowMs;
    publishCameraFrame();
  }

  if (nowMs - lastStatusPublishMs >= kiwi_config::kMasterStatusPeriodMs) {
    lastStatusPublishMs = nowMs;
    publishStatus();
    Serial.printf("master status zenoh=%s wifi=%s lidar=%lu follower=%lu cmd=%lu heap=%lu\n",
                  zenohReady ? "ok" : "off",
                  WiFi.status() == WL_CONNECTED ? "ok" : "off",
                  static_cast<unsigned long>(lidarFrames),
                  static_cast<unsigned long>(followerReports),
                  static_cast<unsigned long>(velocityCommands),
                  static_cast<unsigned long>(ESP.getFreeHeap()));
  }

  delay(1);
}
