import { describe, expect, it } from "vitest";

import { KiwiSimEngine } from "../sim/engine";
import { IDEAL_SENSOR_PROFILE } from "../sim/hardware-profile";
import { KiwiRlEnvironment } from "../sim/rl-environment";
import type { Pose2, WorldDefinition } from "../sim/types";
import {
  cameraCalibration,
  type KiwiVisionRenderer,
} from "./camera";
import { KiwiVisualEnvironment } from "./visual-environment";

const world: WorldDefinition = {
  id: "visual-square",
  name: "Visual square",
  description: "Visual environment fixture",
  spawn: { x: 0, y: 0, yaw: 0 },
  walls: [
    { start: { x: -2, y: -2 }, end: { x: 2, y: -2 } },
    { start: { x: 2, y: -2 }, end: { x: 2, y: 2 } },
    { start: { x: 2, y: 2 }, end: { x: -2, y: 2 } },
    { start: { x: -2, y: 2 }, end: { x: -2, y: -2 } },
  ],
};

class FakeRenderer implements KiwiVisionRenderer {
  readonly calibration = cameraCalibration(2, 1, 60);
  loadedWorld?: string;
  captures: Pose2[] = [];

  loadWorld(definition: WorldDefinition): void {
    this.loadedWorld = definition.id;
  }

  captureRgb(pose: Pose2): Uint8Array {
    this.captures.push({ ...pose });
    const value = Math.min(Math.max(Math.round((pose.x + 2) * 40), 0), 255);
    return new Uint8Array(2 * 1 * 3).fill(value);
  }
}

function visualEnvironment(): { environment: KiwiVisualEnvironment; renderer: FakeRenderer } {
  const engine = new KiwiSimEngine(world, { hardwareProfile: IDEAL_SENSOR_PROFILE });
  const control = new KiwiRlEnvironment(engine, { actionMode: "relative_pose_v1" });
  const renderer = new FakeRenderer();
  return {
    environment: new KiwiVisualEnvironment(control, renderer, {
      contextLength: 3,
      contextStride: 1,
    }),
    renderer,
  };
}

describe("KiwiVisualEnvironment", () => {
  it("returns current temporal RGB and a hidden-pose goal image on reset", () => {
    const { environment, renderer } = visualEnvironment();

    const reset = environment.reset({ goalPose: { x: 1, y: 0, yaw: 0 } });

    expect(renderer.loadedWorld).toBe(world.id);
    expect(renderer.captures).toHaveLength(2);
    expect(reset.observation.schema).toBe("vision_goal_v1");
    expect(reset.observation.rgbShape).toEqual([3, 1, 2, 3]);
    expect(Array.from(reset.observation.rgbValid)).toEqual([0, 0, 1]);
    expect(reset.observation.goalRgb?.[0]).toBe(120);
  });

  it("captures every due simulation-time camera frame during a policy step", () => {
    const { environment, renderer } = visualEnvironment();
    environment.reset();

    const result = environment.step({
      kind: "relative_pose",
      dx: 1,
      dy: 0,
      dyaw: 0,
    });

    expect(renderer.captures).toHaveLength(3); // reset, 0.1 s, 0.2 s
    expect(Array.from(result.observation.rgbValid)).toEqual([1, 1, 1]);
    expect(Array.from(result.observation.rgbTimeS)).toEqual([0, 0.1, 0.2]);
    expect(result.observation.goalRgbValid).toBe(0);
  });

  it("requires reset so goal and temporal state cannot leak across episodes", () => {
    const { environment } = visualEnvironment();
    expect(() =>
      environment.step({ kind: "relative_pose", dx: 0, dy: 0, dyaw: 0 }),
    ).toThrow(/reset/);
  });
});
