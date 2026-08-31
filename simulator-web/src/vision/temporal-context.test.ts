import { describe, expect, it } from "vitest";

import type { Pose2 } from "../sim/types";
import { KIWI_FRONT_RENDER_CAMERA, type VisionFrame } from "./camera";
import { TemporalVisionContext } from "./temporal-context";

const pose: Pose2 = { x: 0, y: 0, yaw: 0 };

function frame(sequence: number, value: number): VisionFrame {
  return {
    rgb: new Uint8Array(
      KIWI_FRONT_RENDER_CAMERA.width * KIWI_FRONT_RENDER_CAMERA.height * 3,
    ).fill(value),
    width: KIWI_FRONT_RENDER_CAMERA.width,
    height: KIWI_FRONT_RENDER_CAMERA.height,
    simulationTime: sequence * 0.1,
    sequence,
    pose,
    calibration: KIWI_FRONT_RENDER_CAMERA,
  };
}

describe("TemporalVisionContext", () => {
  it("returns fixed current-plus-history RGB with reset validity masks", () => {
    const context = new TemporalVisionContext({ contextLength: 3, contextStride: 1 });
    context.reset(frame(0, 7));

    const observation = context.observation();

    expect(observation.rgbShape).toEqual([3, 240, 320, 3]);
    expect(Array.from(observation.rgbValid)).toEqual([0, 0, 1]);
    expect(observation.rgb[0]).toBe(7);
    expect(observation.rgb.at(-1)).toBe(7);
  });

  it("selects chronological strided history and carries a goal image", () => {
    const context = new TemporalVisionContext({ contextLength: 3, contextStride: 2 });
    context.reset(frame(0, 0), frame(99, 99));
    for (let index = 1; index <= 4; index += 1) context.push(frame(index, index));

    const observation = context.observation();
    const frameBytes = 320 * 240 * 3;

    expect(Array.from(observation.rgbValid)).toEqual([1, 1, 1]);
    expect(observation.rgb[0]).toBe(0);
    expect(observation.rgb[frameBytes]).toBe(2);
    expect(observation.rgb[frameBytes * 2]).toBe(4);
    expect(Array.from(observation.rgbTimeS)).toEqual([0, 0.2, 0.4]);
    expect(Array.from(observation.rgbSequence)).toEqual([0, 2, 4]);
    expect(observation.goalRgb?.[0]).toBe(99);
    expect(observation.schema).toBe("vision_goal_v1");
    expect(observation.goalRgbSequence).toBe(99);
  });

  it("rejects non-monotonic capture times", () => {
    const context = new TemporalVisionContext({ contextLength: 2, contextStride: 1 });
    context.reset(frame(1, 1));
    expect(() => context.push(frame(1, 2))).toThrow(/increase monotonically/);
  });
});
