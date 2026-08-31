import { describe, expect, it } from "vitest";

import { TimedPoseHistory } from "./pose-history";

describe("TimedPoseHistory", () => {
  it("interpolates rolling-sensor poses across the yaw wrap", () => {
    const history = new TimedPoseHistory();
    history.reset(0, { x: 0, y: 1, yaw: (170 * Math.PI) / 180 });
    history.append(0.1, { x: 1, y: 3, yaw: (-170 * Math.PI) / 180 });

    const pose = history.interpolate(0.05);
    expect(pose?.x).toBeCloseTo(0.5);
    expect(pose?.y).toBeCloseTo(2);
    expect(Math.abs(pose?.yaw ?? 0)).toBeCloseTo(Math.PI);
  });

  it("rejects acquisition times outside the retained history", () => {
    const history = new TimedPoseHistory();
    history.reset(1, { x: 0, y: 0, yaw: 0 });
    history.append(1.1, { x: 0, y: 0, yaw: 0 });
    expect(history.interpolate(0.9)).toBeUndefined();
    expect(history.interpolate(1.2)).toBeUndefined();
  });
});
