import { describe, expect, it } from "vitest";

import type { RobotState } from "./robot";
import { IDEAL_SENSOR_PROFILE, RETAINED_ROBOT_PROFILE } from "./hardware-profile";
import { buildCameraPayload, crc8, FirmwareContract } from "./zenoh-contract";
import type { LidarSample } from "./types";

const decoder = new TextDecoder();

function state(): RobotState {
  return {
    pose: { x: 1, y: -2, yaw: Math.PI / 3 },
    commandAligned: { vx: 0.4, vy: 0, omega: 0.2 },
    velocityAligned: { vx: 0.2, vy: -0.1, omega: 0.15 },
    commandActive: true,
    collided: false,
  };
}

describe("firmware Zenoh contract", () => {
  it("keeps LiDAR randomness independent from odometry publication", () => {
    const first = new FirmwareContract(RETAINED_ROBOT_PROFILE, 99);
    const second = new FirmwareContract(RETAINED_ROBOT_PROFILE, 99);
    const robotState = state();
    second.step(0.01, robotState);
    for (let index = 0; index < 20; index += 1) {
      second.odometry(robotState, index / 20);
    }
    expect(first.lidar(() => 2)).toEqual(second.lidar(() => 2));
  });

  it("emits the firmware odometry JSON fields in the raw drivetrain frame", () => {
    const contract = new FirmwareContract(IDEAL_SENSOR_PROFILE);
    const robotState = state();
    contract.step(0.01, robotState);
    const report = JSON.parse(decoder.decode(contract.odometry(robotState, 1.25)));

    expect(Object.keys(report).sort()).toEqual(
      [
        "command",
        "encoder_count",
        "encoder_ready_mask",
        "follower_time_us",
        "imu_accel_mps2",
        "imu_quat_ijkr",
        "imu_ready",
        "measured",
        "seq",
        "status_flags",
        "wheel_angle_rad",
        "wheel_speed_mps",
      ].sort(),
    );
    expect(report.follower_time_us).toBe(1_250_000);
    expect(report.measured.vx).toBeCloseTo(0.01339746);
    expect(report.measured.vy).toBeCloseTo(-0.22320508);
    expect(report.wheel_speed_mps).toHaveLength(3);
    expect(report.encoder_count).toHaveLength(3);
  });

  it("emits twenty CRC-valid LD19 frames per batch", () => {
    const contract = new FirmwareContract(IDEAL_SENSOR_PROFILE);
    const scan: LidarSample[] = Array.from({ length: 180 }, (_, index) => ({
      angle: (index / 180) * 2 * Math.PI,
      distance: 1 + index / 180,
      hit: true,
    }));
    const payload = contract.lidar(scan);

    expect(payload).toHaveLength(20 * 47);
    for (let offset = 0; offset < payload.length; offset += 47) {
      const frame = payload.subarray(offset, offset + 47);
      expect(Array.from(frame.subarray(0, 2))).toEqual([0x54, 0x2c]);
      expect(frame[46]).toBe(crc8(frame.subarray(0, 46)));
    }
  });

  it("acquires 480 distinct rolling LD19 rays over one 10 Hz revolution", () => {
    const contract = new FirmwareContract(IDEAL_SENSOR_PROFILE);
    const samples: Array<{ angle: number; time: number }> = [];
    const sampler = (angle: number, time: number) => {
      samples.push({ angle, time });
      return 2;
    };

    contract.lidar(sampler);
    contract.lidar(sampler);

    expect(samples).toHaveLength(480);
    expect(new Set(samples.map(({ angle }) => angle.toFixed(8))).size).toBe(480);
    expect(samples[0]?.time).toBeCloseTo(0);
    expect(samples.at(-1)?.time).toBeCloseTo(0.09979, 4);
    expect(samples.every((sample, index) =>
      index === 0 || sample.time > (samples[index - 1]?.time ?? -1),
    )).toBe(true);
  });

  it("keeps acquisition time monotonic across the 30 second wire rollover", () => {
    const contract = new FirmwareContract(IDEAL_SENSOR_PROFILE);
    let lastAcquisitionTime = 0;
    let payload: Uint8Array<ArrayBufferLike> = new Uint8Array();
    for (let batch = 0; batch < 601; batch += 1) {
      payload = contract.lidar((_angle, time) => {
        lastAcquisitionTime = time;
        return 1;
      });
    }

    const finalFrameOffset = payload.length - 47;
    const wireTimestamp = new DataView(
      payload.buffer,
      payload.byteOffset,
      payload.byteLength,
    ).getUint16(finalFrameOffset + 44, true);
    expect(lastAcquisitionTime).toBeGreaterThan(30);
    expect(wireTimestamp).toBeLessThan(100);
  });

  it("builds the 32-byte KVC1 camera header", () => {
    const jpeg = new Uint8Array([0xff, 0xd8, 0x01, 0x02, 0xff, 0xd9]);
    const payload = buildCameraPayload(jpeg, 7, 123_456n);
    const view = new DataView(payload.buffer);

    expect(decoder.decode(payload.subarray(0, 4))).toBe("KVC1");
    expect(view.getUint16(6, true)).toBe(320);
    expect(view.getUint16(8, true)).toBe(240);
    expect(view.getUint16(10, true)).toBe(32);
    expect(view.getUint32(12, true)).toBe(7);
    expect(view.getBigUint64(16, true)).toBe(123_456n);
    expect(view.getUint32(24, true)).toBe(jpeg.length);
    expect(Array.from(payload.subarray(32))).toEqual(Array.from(jpeg));
  });

  it("adds residual missing returns without inventing packet corruption", () => {
    const contract = new FirmwareContract(RETAINED_ROBOT_PROFILE, 7);
    const sample = () => 2;
    const payload = new Uint8Array([
      ...contract.lidar(sample),
      ...contract.lidar(sample),
    ]);
    const view = new DataView(payload.buffer);
    let valid = 0;
    let maximumMm = 0;
    for (let offset = 0; offset < payload.length; offset += 47) {
      for (let point = 0; point < 12; point += 1) {
        const distanceMm = view.getUint16(offset + 6 + point * 3, true);
        if (distanceMm > 0) valid += 1;
        maximumMm = Math.max(maximumMm, distanceMm);
      }
    }

    expect(valid).toBeGreaterThanOrEqual(420);
    expect(valid).toBeLessThanOrEqual(470);
    expect(maximumMm).toBeLessThanOrEqual(8_000);
  });
});
