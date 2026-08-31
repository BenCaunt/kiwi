import { describe, expect, it } from "vitest";

import { KiwiSimEngine } from "./engine";
import { IDEAL_SENSOR_PROFILE } from "./hardware-profile";
import { KiwiRlEnvironment } from "./rl-environment";
import type { WorldDefinition } from "./types";

const square: WorldDefinition = {
  id: "rl-square",
  name: "RL square",
  description: "RL environment fixture",
  spawn: { x: 0, y: 0, yaw: 0 },
  walls: [
    { start: { x: -4, y: -4 }, end: { x: 4, y: -4 } },
    { start: { x: 4, y: -4 }, end: { x: 4, y: 4 } },
    { start: { x: 4, y: 4 }, end: { x: -4, y: 4 } },
    { start: { x: -4, y: 4 }, end: { x: -4, y: -4 } },
  ],
};

function engine(): KiwiSimEngine {
  return new KiwiSimEngine(square, { hardwareProfile: IDEAL_SENSOR_PROFILE });
}

describe("KiwiRlEnvironment", () => {
  it("rejects out-of-range actions instead of clipping silently", () => {
    const environment = new KiwiRlEnvironment(engine(), {
      actionMode: "relative_pose_v1",
    });
    environment.reset();
    expect(() =>
      environment.step({ kind: "relative_pose", dx: 2.1, dy: 0, dyaw: 0 }),
    ).toThrow(/translation limit/);
  });

  it("tracks an anchored robot-relative pose for a fixed policy interval", () => {
    const environment = new KiwiRlEnvironment(engine(), {
      actionMode: "relative_pose_v1",
    });

    const result = environment.step({
      kind: "relative_pose",
      dx: 1,
      dy: 0,
      dyaw: 0,
    });

    expect(result.ticks).toBe(30);
    expect(result.controllerCommands).toHaveLength(5);
    expect(result.controllerCommands.every(({ target }) => target.x === 1)).toBe(true);
    expect(result.current.robot.pose.x).toBeGreaterThan(0);
    expect(result.current.robot.pose.y).toBeCloseTo(0);
    expect(result.events.some(({ type }) => type === "relative_target_accepted")).toBe(true);
  });

  it("anchors every trajectory waypoint to one rotated action-start frame", () => {
    const environment = new KiwiRlEnvironment(engine(), {
      actionMode: "relative_trajectory_v1",
      trajectoryLookaheadIndex: 1,
    });
    environment.reset({ pose: { x: 1, y: 2, yaw: Math.PI / 2 } });

    const result = environment.step({
      kind: "relative_trajectory",
      waypoints: [
        { dx: 0.5, dy: 0, dyaw: 0 },
        { dx: 1, dy: 0.5, dyaw: Math.PI / 2 },
      ],
    });

    expect(result.controllerCommands[0]?.target.x).toBeCloseTo(0.5);
    expect(result.controllerCommands[0]?.target.y).toBeCloseTo(3);
    expect(result.controllerCommands[0]?.target.yaw).toBeCloseTo(-Math.PI);
  });

  it("preserves direct aligned twist control as a separate mode", () => {
    const environment = new KiwiRlEnvironment(engine(), {
      actionMode: "twist_aligned_v1",
      policyHz: 20,
    });

    const result = environment.step({
      kind: "twist",
      vx: 0.5,
      vy: 0,
      omega: 0,
    });

    expect(result.ticks).toBe(6);
    expect(result.controllerCommands).toHaveLength(0);
    expect(result.current.robot.pose.x).toBeGreaterThan(0);
  });

  it("reports target replacement and clears controller state on reset", () => {
    const environment = new KiwiRlEnvironment(engine());
    environment.step({ kind: "relative_pose", dx: 1, dy: 0, dyaw: 0 });
    const replaced = environment.step({
      kind: "relative_pose",
      dx: 0,
      dy: 1,
      dyaw: 0,
    });

    expect(replaced.events.some(({ type }) => type === "relative_target_replaced")).toBe(true);
    const reset = environment.reset({ seed: 19 });
    expect(reset.seed).toBe(19);
    expect(environment.relativeController.active).toBe(false);
    expect(environment.episodeSteps).toBe(0);
  });

  it("retains contact from any physics substep in the policy result", () => {
    const environment = new KiwiRlEnvironment(engine());
    environment.reset({ pose: { x: 3.86, y: 0, yaw: 0 } });

    const result = environment.step({
      kind: "relative_pose",
      dx: 0.5,
      dy: 0,
      dyaw: 0,
    });

    expect(result.collisionTickCount).toBeGreaterThan(0);
    expect(result.events.some(({ type }) => type === "contact_started")).toBe(true);
  });

  it("rejects actions from a different configured level", () => {
    const environment = new KiwiRlEnvironment(engine(), {
      actionMode: "relative_pose_v1",
    });

    expect(() =>
      environment.step({ kind: "twist", vx: 0, vy: 0, omega: 0 }),
    ).toThrow(/does not match/);
  });
});
