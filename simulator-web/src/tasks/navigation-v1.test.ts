import { describe, expect, it } from "vitest";

import { KiwiSimEngine } from "../sim/engine";
import { IDEAL_SENSOR_PROFILE } from "../sim/hardware-profile";
import { KiwiRlEnvironment } from "../sim/rl-environment";
import type { Pose2, WorldDefinition } from "../sim/types";
import { cameraCalibration, type KiwiVisionRenderer } from "../vision/camera";
import { KiwiVisualEnvironment } from "../vision/visual-environment";
import { NavigationTaskEnvironment, validateAuthoredNavigationPairs } from "./navigation-v1";

class FlatRenderer implements KiwiVisionRenderer {
  readonly calibration = cameraCalibration(2, 2, 72);
  loadWorld(_world: WorldDefinition): void {}
  captureRgb(pose: Pose2): Uint8Array {
    return new Uint8Array(12).fill(Math.round((pose.x + pose.y + 10) * 5));
  }
}

function task(): NavigationTaskEnvironment {
  const world: WorldDefinition = {
    id: "room",
    name: "Task fixture",
    description: "Task fixture",
    spawn: { x: -1.8, y: 0, yaw: 0 },
    walls: [
      { start: { x: -3, y: -2.2 }, end: { x: 3, y: -2.2 } },
      { start: { x: 3, y: -2.2 }, end: { x: 3, y: 2.2 } },
      { start: { x: 3, y: 2.2 }, end: { x: -3, y: 2.2 } },
      { start: { x: -3, y: 2.2 }, end: { x: -3, y: -2.2 } },
    ],
  };
  const engine = new KiwiSimEngine(world, { hardwareProfile: IDEAL_SENSOR_PROFILE });
  const control = new KiwiRlEnvironment(engine, { actionMode: "relative_pose_v1" });
  return new NavigationTaskEnvironment(
    new KiwiVisualEnvironment(control, new FlatRenderer(), { contextLength: 2, contextStride: 1 }),
  );
}

describe("NavigationTaskEnvironment", () => {
  it("keeps goal coordinates privileged and exposes named rewards", () => {
    const environment = task();
    const reset = environment.reset(4);
    expect(reset.observation.schema).toBe("vision_goal_v1");
    expect(reset.info).not.toHaveProperty("privileged");

    const step = environment.step({ kind: "relative_pose", dx: 0.2, dy: 0, dyaw: 0 });
    expect(step.reward).toBe(Object.values(step.info.reward_terms).reduce((a, b) => a + b, 0));
    expect(step.info.reward_terms.progress).toBeGreaterThan(0);
    expect(step.info).not.toHaveProperty("privileged");
  });

  it("validates every authored pair as reachable", () => {
    expect(validateAuthoredNavigationPairs()).toEqual([]);
  });
});
