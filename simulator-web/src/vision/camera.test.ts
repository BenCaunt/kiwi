import { describe, expect, it } from "vitest";

import {
  cameraCalibration,
  rgbaBottomUpToUprightRgb,
  rotateRgb180,
} from "./camera";

describe("cameraCalibration", () => {
  it("records the current Three.js vertical FOV without calling it horizontal", () => {
    const calibration = cameraCalibration(320, 240, 72);

    expect(calibration.verticalFovDeg).toBe(72);
    expect(calibration.horizontalFovDeg).toBeCloseTo(88.1796649);
    expect(calibration.fx).toBeCloseTo(calibration.fy);
    expect(calibration.policyOrientation).toBe("upright");
    expect(calibration.firmwareOrientation).toBe("rotated_180");
  });

  it("freezes upright policy and rotated firmware pixel orientation", () => {
    // WebGL order: bottom-left red, bottom-right green, top-left blue, top-right white.
    const rgba = new Uint8Array([
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 255, 255, 255, 255,
    ]);
    const upright = rgbaBottomUpToUprightRgb(rgba, 2, 2);
    expect(Array.from(upright)).toEqual([
      0, 0, 255, 255, 255, 255,
      255, 0, 0, 0, 255, 0,
    ]);
    expect(Array.from(rotateRgb180(upright, 2, 2))).toEqual([
      0, 255, 0, 255, 0, 0,
      255, 255, 255, 0, 0, 255,
    ]);
  });
});
