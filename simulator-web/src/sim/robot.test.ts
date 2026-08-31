import { describe, expect, it } from "vitest";

import { scanLidar } from "./lidar";
import { alignedToRaw, KiwiRobot, rawToAligned } from "./robot";
import type { WorldDefinition } from "./types";

const square: WorldDefinition = {
  id: "square",
  name: "Square",
  description: "Test world",
  spawn: { x: 0, y: 0, yaw: 0 },
  walls: [
    { start: { x: -2, y: -2 }, end: { x: 2, y: -2 } },
    { start: { x: 2, y: -2 }, end: { x: 2, y: 2 } },
    { start: { x: 2, y: 2 }, end: { x: -2, y: 2 } },
    { start: { x: -2, y: 2 }, end: { x: -2, y: -2 } },
  ],
};

describe("KiwiRobot", () => {
  it("keeps raw and sensor-aligned frame transforms invertible", () => {
    const command = { vx: 0.4, vy: -0.2, omega: 0.7 };
    const roundTrip = rawToAligned(alignedToRaw(command));
    expect(roundTrip.vx).toBeCloseTo(command.vx);
    expect(roundTrip.vy).toBeCloseTo(command.vy);
    expect(roundTrip.omega).toBe(command.omega);
  });

  it("moves forward along the sensor-facing axis", () => {
    const robot = new KiwiRobot(square);
    let now = 0;
    for (let index = 0; index < 120; index += 1) {
      robot.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, now);
      now += 1 / 120;
      robot.step(1 / 120, now);
    }
    expect(robot.state.pose.x).toBeGreaterThan(0.42);
    expect(robot.state.pose.y).toBeCloseTo(0);
  });

  it("stops at walls using the circular robot footprint", () => {
    const wallWorld: WorldDefinition = {
      ...square,
      walls: [{ start: { x: 0.5, y: -2 }, end: { x: 0.5, y: 2 } }],
    };
    const robot = new KiwiRobot(wallWorld);
    let now = 0;
    for (let index = 0; index < 600; index += 1) {
      robot.setAlignedCommand({ vx: 0.8, vy: 0, omega: 0 }, now);
      now += 1 / 120;
      robot.step(1 / 120, now);
    }
    expect(robot.state.pose.x).toBeLessThanOrEqual(0.5 - robot.config.radius + 1e-6);
    expect(robot.state.collided).toBe(true);
  });

  it("drives beneath a table top through the gap between its legs", () => {
    const furnishedWorld: WorldDefinition = {
      ...square,
      objects: [
        {
          id: "table",
          kind: "table",
          position: { x: 0.7, y: 0 },
          size: { x: 0.8, y: 0.8 },
          height: 0.6,
        },
      ],
    };
    const robot = new KiwiRobot(furnishedWorld);
    let now = 0;
    for (let index = 0; index < 420; index += 1) {
      robot.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, now);
      now += 1 / 120;
      robot.step(1 / 120, now);
    }
    expect(robot.state.pose.x).toBeGreaterThan(1.1 + robot.config.radius);
    expect(robot.state.collided).toBe(false);
  });

  it("still collides with a visible table leg", () => {
    const furnishedWorld: WorldDefinition = {
      ...square,
      spawn: { x: 0, y: 0.28, yaw: 0 },
      objects: [
        {
          id: "table",
          kind: "table",
          position: { x: 0.7, y: 0 },
          size: { x: 0.8, y: 0.8 },
          height: 0.6,
        },
      ],
    };
    const robot = new KiwiRobot(furnishedWorld);
    let now = 0;
    for (let index = 0; index < 300; index += 1) {
      robot.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, now);
      now += 1 / 120;
      robot.step(1 / 120, now);
    }
    expect(robot.state.pose.x).toBeLessThanOrEqual(0.385 - robot.config.radius + 1e-6);
    expect(robot.state.collided).toBe(true);
  });

  it("collides with a table top below the robot's clearance height", () => {
    const furnishedWorld: WorldDefinition = {
      ...square,
      objects: [
        {
          id: "low-table",
          kind: "low-table",
          position: { x: 0.6, y: 0 },
          size: { x: 0.6, y: 0.8 },
          height: 0.2,
        },
      ],
    };
    const robot = new KiwiRobot(furnishedWorld);
    let now = 0;
    for (let index = 0; index < 300; index += 1) {
      robot.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, now);
      now += 1 / 120;
      robot.step(1 / 120, now);
    }
    expect(robot.state.pose.x).toBeLessThanOrEqual(0.3 - robot.config.radius + 1e-6);
    expect(robot.state.collided).toBe(true);
  });

  it("expires a stale command", () => {
    const robot = new KiwiRobot(square);
    robot.setAlignedCommand({ vx: 0.5, vy: 0, omega: 0 }, 0);
    robot.step(0.01, 0.01);
    robot.step(0.1, 0.31);
    expect(robot.state.commandActive).toBe(false);
    expect(robot.state.velocityAligned.vx).toBeLessThan(0.05);
  });
});

describe("scanLidar", () => {
  it("returns wall ranges around a closed room", () => {
    const scan = scanLidar(square, square.spawn, { rays: 4, maxRange: 12 });
    expect(scan).toHaveLength(4);
    expect(scan.every((sample) => sample.hit)).toBe(true);
    expect(scan.map((sample) => sample.distance)).toEqual([2, 2, 2, 2]);
  });

  it("returns furniture before the room wall", () => {
    const furnishedWorld: WorldDefinition = {
      ...square,
      objects: [
        {
          id: "island",
          kind: "island",
          position: { x: 1, y: 0 },
          size: { x: 0.4, y: 0.8 },
          height: 0.9,
        },
      ],
    };
    const scan = scanLidar(furnishedWorld, square.spawn, { rays: 4, maxRange: 12 });
    expect(scan[0]?.distance).toBeCloseTo(0.8);
  });

  it("casts through the open center beneath a table top", () => {
    const furnishedWorld: WorldDefinition = {
      ...square,
      objects: [
        {
          id: "table",
          kind: "table",
          position: { x: 1, y: 0 },
          size: { x: 0.8, y: 0.8 },
          height: 0.7,
        },
      ],
    };
    const scan = scanLidar(furnishedWorld, square.spawn, { rays: 4, maxRange: 12 });
    expect(scan[0]?.distance).toBeCloseTo(2);
  });
});
