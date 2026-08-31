import type { Pose2, WorldDefinition } from "../sim/types";

export interface CameraCalibration {
  profile: string;
  width: number;
  height: number;
  verticalFovDeg: number;
  horizontalFovDeg: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  nearM: number;
  farM: number;
  robotToCamera: {
    forwardM: number;
    leftM: number;
    heightM: number;
    yawRad: number;
    pitchRad: number;
    rollRad: number;
  };
  policyOrientation: "upright";
  firmwareOrientation: "rotated_180";
}

export interface VisionFrame {
  rgb: Uint8Array;
  width: number;
  height: number;
  simulationTime: number;
  sequence: number;
  pose: Pose2;
  calibration: Readonly<CameraCalibration>;
}

export interface KiwiVisionRenderer {
  readonly calibration: Readonly<CameraCalibration>;
  loadWorld(world: WorldDefinition): void;
  captureRgb(pose: Pose2): Uint8Array;
}

export function cameraCalibration(
  width = 320,
  height = 240,
  verticalFovDeg = 72,
  nearM = 0.025,
  farM = 20,
): CameraCalibration {
  if (![width, height].every(Number.isInteger) || width <= 0 || height <= 0) {
    throw new Error("Camera dimensions must be positive integers");
  }
  if (!Number.isFinite(verticalFovDeg) || verticalFovDeg <= 0 || verticalFovDeg >= 180) {
    throw new Error("Camera vertical FOV must be in (0, 180) degrees");
  }
  const verticalFovRad = (verticalFovDeg * Math.PI) / 180;
  const aspect = width / height;
  const horizontalFovRad = 2 * Math.atan(Math.tan(verticalFovRad / 2) * aspect);
  const fy = height / (2 * Math.tan(verticalFovRad / 2));
  const fx = width / (2 * Math.tan(horizontalFovRad / 2));
  return {
    profile: "kiwi-front-render-v1",
    width,
    height,
    verticalFovDeg,
    horizontalFovDeg: (horizontalFovRad * 180) / Math.PI,
    fx,
    fy,
    cx: (width - 1) / 2,
    cy: (height - 1) / 2,
    nearM,
    farM,
    robotToCamera: {
      forwardM: 0,
      leftM: 0,
      heightM: 0.22,
      yawRad: 0,
      pitchRad: 0,
      rollRad: 0,
    },
    policyOrientation: "upright",
    firmwareOrientation: "rotated_180",
  };
}

export const KIWI_FRONT_RENDER_CAMERA: Readonly<CameraCalibration> =
  Object.freeze(cameraCalibration());

export function rgbaBottomUpToUprightRgb(
  pixels: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  if (pixels.length !== width * height * 4) throw new Error("RGBA buffer size mismatch");
  const rgb = new Uint8Array(width * height * 3);
  for (let sourceY = 0; sourceY < height; sourceY += 1) {
    const destinationY = height - 1 - sourceY;
    for (let sourceX = 0; sourceX < width; sourceX += 1) {
      const source = (sourceY * width + sourceX) * 4;
      const destination = (destinationY * width + sourceX) * 3;
      rgb[destination] = pixels[source] ?? 0;
      rgb[destination + 1] = pixels[source + 1] ?? 0;
      rgb[destination + 2] = pixels[source + 2] ?? 0;
    }
  }
  return rgb;
}

export function rotateRgb180(rgb: Uint8Array, width: number, height: number): Uint8Array {
  if (rgb.length !== width * height * 3) throw new Error("RGB buffer size mismatch");
  const rotated = new Uint8Array(rgb.length);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const destination = width * height - 1 - pixel;
    rotated.set(rgb.subarray(pixel * 3, pixel * 3 + 3), destination * 3);
  }
  return rotated;
}
