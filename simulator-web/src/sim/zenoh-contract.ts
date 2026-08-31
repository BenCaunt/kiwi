import { alignedToRaw } from "./robot";
import type { RobotState } from "./robot";
import {
  RETAINED_ROBOT_PROFILE,
  type HardwareSensorProfile,
} from "./hardware-profile";
import type { LidarSample, Twist2 } from "./types";
import { deriveSeed, SeededRandom } from "./random";

export const BRIDGE_CHANNEL = Object.freeze({
  ODOMETRY: 1,
  LIDAR: 2,
  CAMERA: 3,
  STATUS: 4,
});

const LD19_FRAME_LENGTH = 47;
const LD19_POINTS_PER_FRAME = 12;
const LD19_FRAMES_PER_REVOLUTION = 40;
const LD19_ROTATION_HZ = 10;
const textEncoder = new TextEncoder();
const wheelAngles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];

export type LidarRangeSampler = (
  localAngle: number,
  acquisitionTime: number,
) => number;

const crcTable = new Uint8Array(256);
for (let tableIndex = 0; tableIndex < 256; tableIndex += 1) {
  let crc = tableIndex;
  for (let bit = 0; bit < 8; bit += 1) {
    crc = ((crc & 0x80) !== 0 ? (crc << 1) ^ 0x4d : crc << 1) & 0xff;
  }
  crcTable[tableIndex] = crc;
}

export function crc8(data: Uint8Array): number {
  let crc = 0;
  for (const byte of data) crc = crcTable[crc ^ byte] ?? 0;
  return crc;
}

function zeroTwist(): Twist2 {
  return { vx: 0, vy: 0, omega: 0 };
}

function wrapAngle(angle: number): number {
  return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

function wheelSpeeds(twist: Twist2, driveBaseRadius = 0.09): number[] {
  return wheelAngles.map(
    (angle) =>
      -Math.sin(angle) * twist.vx +
      Math.cos(angle) * twist.vy +
      driveBaseRadius * twist.omega,
  );
}

function jsonPayload(document: unknown): Uint8Array {
  return textEncoder.encode(JSON.stringify(document));
}

function sampleRange(scan: LidarSample[], localAngle: number): number {
  if (scan.length === 0) return 0;
  const twoPi = 2 * Math.PI;
  const normalized = ((localAngle % twoPi) + twoPi) % twoPi;
  const index = Math.round((normalized / twoPi) * scan.length) % scan.length;
  return scan[index]?.hit ? (scan[index]?.distance ?? 0) : 0;
}

export function buildCameraPayload(
  jpeg: Uint8Array,
  sequence: number,
  timestampUs: bigint,
  width = 320,
  height = 240,
): Uint8Array {
  const headerLength = 32;
  const payload = new Uint8Array(headerLength + jpeg.length);
  payload.set([0x4b, 0x56, 0x43, 0x31], 0); // KVC1
  payload[4] = 1;
  payload[5] = 4; // esp_camera PIXFORMAT_JPEG
  const view = new DataView(payload.buffer);
  view.setUint16(6, width, true);
  view.setUint16(8, height, true);
  view.setUint16(10, headerLength, true);
  view.setUint32(12, sequence, true);
  view.setBigUint64(16, timestampUs, true);
  view.setUint32(24, jpeg.length, true);
  payload.set(jpeg, headerLength);
  return payload;
}

export class FirmwareContract {
  readonly profile: Readonly<HardwareSensorProfile>;
  private readonly seed: number;
  private odometryRandom: SeededRandom;
  private lidarRandom: SeededRandom;
  private reportSequence = 0;
  private cameraSequence = 0;
  private lidarFrameIndex = 0;
  private lidarTimestampMs = 0;
  private wheelSpeedMps = [0, 0, 0];
  private wheelAngleRad = [0, 0, 0];
  private encoderFloat = [0, 0, 0];
  private encoderCount = [0, 0, 0];
  private acceleration = [0, 0, 9.81];
  private lastMeasuredRaw = zeroTwist();
  private measuredRaw = zeroTwist();
  private imuYaw = 0;
  private lastTrueYaw: number | null = null;
  private lidarBlindSectorCenterRad = 0;
  private lidarBlindSectorWidthRad = 0;
  private lidarDropoutProbability = 0;
  private commandCount = 0;
  private cameraPublished = 0;
  private lidarFramesPublished = 0;
  private lidarBatchesPublished = 0;
  private twistPublished = 0;

  constructor(
    profile: Readonly<HardwareSensorProfile> = RETAINED_ROBOT_PROFILE,
    seed = 1,
  ) {
    this.profile = profile;
    this.seed = seed;
    this.odometryRandom = new SeededRandom(deriveSeed(seed, "odometry_imu"));
    this.lidarRandom = new SeededRandom(deriveSeed(seed, "lidar"));
  }

  reset(): void {
    this.odometryRandom = new SeededRandom(deriveSeed(this.seed, "odometry_imu"));
    this.lidarRandom = new SeededRandom(deriveSeed(this.seed, "lidar"));
    this.reportSequence = 0;
    this.cameraSequence = 0;
    this.lidarFrameIndex = 0;
    this.lidarTimestampMs = 0;
    this.wheelSpeedMps = [0, 0, 0];
    this.wheelAngleRad = [0, 0, 0];
    this.encoderFloat = [0, 0, 0];
    this.encoderCount = [0, 0, 0];
    this.acceleration = [0, 0, 9.81];
    this.lastMeasuredRaw = zeroTwist();
    this.measuredRaw = zeroTwist();
    this.imuYaw = 0;
    this.lastTrueYaw = null;
    this.lidarBlindSectorCenterRad = 0;
    this.lidarBlindSectorWidthRad = 0;
    this.lidarDropoutProbability = this.profile.lidarRandomDropoutProbability;
    this.commandCount = 0;
    this.cameraPublished = 0;
    this.lidarFramesPublished = 0;
    this.lidarBatchesPublished = 0;
    this.twistPublished = 0;
  }

  recordCommand(): void {
    this.commandCount += 1;
  }

  step(dt: number, state: RobotState): void {
    const trueRaw = alignedToRaw(state.velocityAligned);
    const skew = (this.profile.odometryAxisSkewDeg * Math.PI) / 180;
    const c = Math.cos(skew);
    const s = Math.sin(skew);
    const moving = Math.hypot(trueRaw.vx, trueRaw.vy) > 0.002;
    const noiseStd = moving ? this.profile.odometryVelocityNoiseStdMps : 0;
    this.measuredRaw = {
      vx:
        this.profile.odometryLinearScale * (c * trueRaw.vx - s * trueRaw.vy) +
        noiseStd * this.odometryRandom.normal(),
      vy:
        this.profile.odometryLinearScale * (s * trueRaw.vx + c * trueRaw.vy) +
        noiseStd * this.odometryRandom.normal(),
      omega: this.profile.odometryAngularScale * trueRaw.omega,
    };

    if (this.lastTrueYaw === null) {
      this.imuYaw = state.pose.yaw;
    } else {
      const trueYawDelta = wrapAngle(state.pose.yaw - this.lastTrueYaw);
      const drift =
        (this.profile.imuYawDriftDegPerSecond * Math.PI) / 180 * dt;
      const randomWalk =
        (this.profile.imuYawRandomWalkDegPerSqrtSecond * Math.PI) / 180 *
        Math.sqrt(dt) *
        this.odometryRandom.normal();
      this.imuYaw = wrapAngle(
        this.imuYaw + this.profile.imuYawScale * trueYawDelta + drift + randomWalk,
      );
    }
    this.lastTrueYaw = state.pose.yaw;

    this.wheelSpeedMps = wheelSpeeds(this.measuredRaw);
    this.wheelSpeedMps.forEach((speed, index) => {
      const deltaAngle = (speed / 0.025) * dt;
      this.wheelAngleRad[index] =
        ((this.wheelAngleRad[index] ?? 0) + deltaAngle) % (2 * Math.PI);
      this.encoderFloat[index] =
        (this.encoderFloat[index] ?? 0) +
        (deltaAngle / (2 * Math.PI)) * 4096;
      this.encoderCount[index] = Math.round(this.encoderFloat[index] ?? 0);
    });
    this.acceleration = [
      (this.measuredRaw.vx - this.lastMeasuredRaw.vx) / dt,
      (this.measuredRaw.vy - this.lastMeasuredRaw.vy) / dt,
      9.81,
    ];
    this.lastMeasuredRaw = { ...this.measuredRaw };
  }

  odometry(state: RobotState, simulationTime: number): Uint8Array {
    const command = state.commandActive
      ? alignedToRaw(state.commandAligned)
      : zeroTwist();
    const imuYaw = wrapAngle(
      this.imuYaw +
        (this.profile.imuYawNoiseStdDeg * Math.PI) / 180 * this.odometryRandom.normal(),
    );
    const payload = jsonPayload({
      follower_time_us: Math.round(simulationTime * 1_000_000),
      seq: this.reportSequence,
      measured: this.measuredRaw,
      command,
      wheel_speed_mps: this.wheelSpeedMps,
      wheel_angle_rad: this.wheelAngleRad,
      encoder_count: this.encoderCount,
      imu_ready: true,
      encoder_ready_mask: 7,
      status_flags: state.commandActive ? 0 : 1,
      imu_quat_ijkr: [
        0,
        0,
        Math.sin(imuYaw / 2),
        Math.cos(imuYaw / 2),
      ],
      imu_accel_mps2: this.acceleration,
    });
    this.reportSequence += 1;
    this.twistPublished += 1;
    return payload;
  }

  lidar(scan: LidarSample[] | LidarRangeSampler, frameCount = 20): Uint8Array {
    const payload = new Uint8Array(frameCount * LD19_FRAME_LENGTH);
    const view = new DataView(payload.buffer);
    const frameSpanDeg = 360 / LD19_FRAMES_PER_REVOLUTION;
    const framePeriodMs =
      1000 / (LD19_ROTATION_HZ * LD19_FRAMES_PER_REVOLUTION);

    for (let frameNumber = 0; frameNumber < frameCount; frameNumber += 1) {
      const offset = frameNumber * LD19_FRAME_LENGTH;
      const startDeg = this.lidarFrameIndex * frameSpanDeg;
      // Twelve evenly spaced samples per 9-degree packet interval yields 480
      // distinct samples per 10 Hz revolution, without duplicating packet
      // boundary points.
      const pointStepDeg = frameSpanDeg / LD19_POINTS_PER_FRAME;
      const endDeg =
        (startDeg + pointStepDeg * (LD19_POINTS_PER_FRAME - 1)) % 360;
      const frameTime = this.lidarTimestampMs / 1000;
      if (this.lidarFrameIndex === 0) {
        const minimum = this.profile.lidarBlindSectorMinDeg;
        const maximum = this.profile.lidarBlindSectorMaxDeg;
        this.lidarBlindSectorCenterRad = this.lidarRandom.uniform() * 2 * Math.PI;
        this.lidarBlindSectorWidthRad =
          ((minimum + (maximum - minimum) * this.lidarRandom.uniform()) * Math.PI) / 180;
        this.lidarDropoutProbability = Math.min(
          Math.max(
            this.profile.lidarRandomDropoutProbability +
              this.profile.lidarDropoutVariationStd * this.lidarRandom.normal(),
            0,
          ),
          0.8,
        );
      }
      payload[offset] = 0x54;
      payload[offset + 1] = 0x2c;
      view.setUint16(offset + 2, Math.round(360 * LD19_ROTATION_HZ), true);
      view.setUint16(offset + 4, Math.round(startDeg * 100) % 36000, true);

      for (let pointIndex = 0; pointIndex < LD19_POINTS_PER_FRAME; pointIndex += 1) {
        const rawAngle =
          ((startDeg + pointStepDeg * pointIndex) * Math.PI) / 180;
        const localAngle = -rawAngle;
        const acquisitionTime =
          frameTime + (framePeriodMs / 1000) * pointIndex / LD19_POINTS_PER_FRAME;
        const idealDistance =
          typeof scan === "function"
            ? scan(localAngle, acquisitionTime)
            : sampleRange(scan, localAngle);
        const inBlindSector =
          Math.abs(wrapAngle(localAngle - this.lidarBlindSectorCenterRad)) <=
          this.lidarBlindSectorWidthRad / 2;
        const dropped =
          inBlindSector ||
          this.lidarRandom.uniform() < this.lidarDropoutProbability;
        const distance =
          idealDistance > 0 &&
          idealDistance <= this.profile.lidarMaxRangeM &&
          !dropped
            ? Math.max(
                Math.min(
                  idealDistance +
                    this.profile.lidarRangeNoiseStdM * this.lidarRandom.normal(),
                  this.profile.lidarMaxRangeM,
                ),
                0.001,
              )
            : 0;
        const distanceMm =
          distance > 0 ? Math.min(Math.max(Math.round(distance * 1000), 1), 12000) : 0;
        const intensity =
          distance > 0
            ? Math.min(Math.max(Math.round(230 / (1 + 0.12 * distance)), 20), 255)
            : 0;
        view.setUint16(offset + 6 + pointIndex * 3, distanceMm, true);
        payload[offset + 8 + pointIndex * 3] = intensity;
      }

      view.setUint16(offset + 42, Math.round(endDeg * 100) % 36000, true);
      // Only the 16-bit wire value wraps. Keep the internal acquisition clock
      // monotonic so ray poses still resolve after the LD19's 30 s rollover.
      view.setUint16(offset + 44, Math.round(this.lidarTimestampMs) % 30000, true);
      payload[offset + 46] = crc8(payload.subarray(offset, offset + 46));
      this.lidarFrameIndex =
        (this.lidarFrameIndex + 1) % LD19_FRAMES_PER_REVOLUTION;
      this.lidarTimestampMs += framePeriodMs;
    }

    this.lidarFramesPublished += frameCount;
    this.lidarBatchesPublished += 1;
    return payload;
  }

  camera(jpeg: Uint8Array, simulationTime: number): Uint8Array {
    const payload = buildCameraPayload(
      jpeg,
      this.cameraSequence,
      BigInt(Math.round(simulationTime * 1_000_000)),
    );
    this.cameraSequence += 1;
    this.cameraPublished += 1;
    return payload;
  }

  status(worldName: string, simulationTime: number): Uint8Array {
    return jsonPayload({
      esp_ms: Math.round(simulationTime * 1000),
      sta_connected: true,
      sta_ip: "127.0.0.1",
      rssi: -25,
      camera_ready: true,
      zenoh_ready: true,
      lidar_frames: this.lidarFramesPublished,
      lidar_bad_frames: 0,
      follower_reports: this.twistPublished,
      follower_bad_packets: 0,
      velocity_commands: this.commandCount,
      camera_published: this.cameraPublished,
      camera_errors: 0,
      lidar_published: this.lidarFramesPublished,
      lidar_errors: 0,
      lidar_batches: this.lidarBatchesPublished,
      twist_published: this.twistPublished,
      twist_errors: 0,
      loop_gap_max_us: 0,
      publish_max_us: 0,
      lidar_rx_high_water: 940,
      follower_rx_high_water: 116,
      free_heap: 2_000_000,
      simulator: true,
      simulator_frontend: "threejs",
      environment: worldName,
      simulator_sensor_profile: this.profile.id,
    });
  }
}
