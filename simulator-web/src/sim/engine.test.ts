import { describe, expect, it } from "vitest";

import { KiwiSimEngine } from "./engine";
import { IDEAL_SENSOR_PROFILE } from "./hardware-profile";
import type { WorldDefinition } from "./types";

const square: WorldDefinition = {
  id: "engine-square",
  name: "Engine square",
  description: "Deterministic engine fixture",
  spawn: { x: 0, y: 0, yaw: 0 },
  walls: [
    { start: { x: -2, y: -2 }, end: { x: 2, y: -2 } },
    { start: { x: 2, y: -2 }, end: { x: 2, y: 2 } },
    { start: { x: 2, y: 2 }, end: { x: -2, y: 2 } },
    { start: { x: -2, y: 2 }, end: { x: -2, y: -2 } },
  ],
};

describe("KiwiSimEngine", () => {
  it("advances exactly thirty physics ticks for a 4 Hz policy interval", () => {
    const engine = new KiwiSimEngine(square, {
      hardwareProfile: IDEAL_SENSOR_PROFILE,
    });
    engine.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, 1);

    const result = engine.advanceTicks(30);

    expect(result.ticks).toBe(30);
    expect(engine.simulationTime).toBeCloseTo(0.25, 12);
    expect(engine.robot.state.pose.x).toBeGreaterThan(0);
  });

  it("resets all time, motion, sensor, and watchdog state", () => {
    const engine = new KiwiSimEngine(square, {
      hardwareProfile: IDEAL_SENSOR_PROFILE,
    });
    engine.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, 1);
    engine.advanceTicks(60);

    const snapshot = engine.reset();

    expect(snapshot.simulationTime).toBe(0);
    expect(snapshot.robot.pose).toEqual(square.spawn);
    expect(snapshot.robot.velocityAligned).toEqual({ vx: 0, vy: 0, omega: 0 });
    expect(snapshot.robot.commandActive).toBe(false);
    expect(snapshot.lidar).toHaveLength(180);
  });

  it("emits deterministic sensor deadlines without a connected transport", () => {
    const engine = new KiwiSimEngine(square, {
      hardwareProfile: IDEAL_SENSOR_PROFILE,
    });

    const result = engine.advanceTicks(12);
    const eventTypes = result.events.map(({ type }) => type);

    expect(eventTypes.filter((type) => type === "odometry")).toHaveLength(3);
    expect(eventTypes.filter((type) => type === "lidar")).toHaveLength(2);
    expect(eventTypes.filter((type) => type === "camera_due")).toHaveLength(2);
    expect(eventTypes.filter((type) => type === "status")).toHaveLength(1);
    expect(eventTypes.filter((type) => type === "ground_truth")).toHaveLength(3);
  });

  it("returns copies instead of leaking mutable snapshots", () => {
    const engine = new KiwiSimEngine(square, {
      hardwareProfile: IDEAL_SENSOR_PROFILE,
    });
    const snapshot = engine.snapshot();
    snapshot.robot.pose.x = 99;
    snapshot.lidar[0]!.distance = 99;

    expect(engine.robot.state.pose.x).toBe(0);
    expect(engine.lidar[0]?.distance).not.toBe(99);
  });

  it("replays identical state and event payloads after a seeded reset", () => {
    const engine = new KiwiSimEngine(square);
    const run = () => {
      engine.reset(square, { seed: 123 });
      engine.setAlignedCommand({ vx: 0.2, vy: 0.05, omega: 0.1 }, 1);
      const result = engine.advanceTicks(30);
      return {
        snapshot: engine.snapshot(),
        events: result.events.map((event) => ({
          ...event,
          payload: "payload" in event ? Array.from(event.payload) : undefined,
        })),
      };
    };
    expect(run()).toEqual(run());
  });
});
