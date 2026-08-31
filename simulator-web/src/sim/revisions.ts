import { DEFAULT_POSE_CONTROLLER_CONFIG } from "../control/pose-controller";
import { KIWI_FRONT_RENDER_CAMERA } from "../vision/camera";
import { DEFAULT_ENGINE_CONFIG } from "./engine";
import type { WorldDefinition } from "./types";

export const PROTOCOL_VERSION = 1;
export const ENGINE_VERSION = "0.1.0";

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const document = value as Record<string, unknown>;
  return `{${Object.keys(document)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonical(document[key])}`)
    .join(",")}}`;
}

/** Stable 64-bit FNV-1a content identifier, represented as a fixed hex string. */
export function contentRevision(value: unknown): string {
  const bytes = new TextEncoder().encode(canonical(value));
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
}

export const PHYSICS_REVISION = contentRevision({
  engine: DEFAULT_ENGINE_CONFIG,
  integration: "fixed_tick_semi_implicit_v1",
  collision: "height_aware_furniture_components_slide_v2",
});

export const CONTROLLER_REVISION = contentRevision({
  controller: DEFAULT_POSE_CONTROLLER_CONFIG,
  relativePose: "anchored_se2_v1",
  trajectory: "cumulative_fixed_origin_index_lookahead_v1",
});

export const RENDERER_REVISION = contentRevision({
  renderer: "three-webgl-shared-scene-v1",
  canonicalOrientation: "upright_rgb8",
  firmwareOrientation: "rotated_180_jpeg",
  camera: KIWI_FRONT_RENDER_CAMERA,
});

export function worldRevision(world: WorldDefinition): string {
  return contentRevision(world);
}
