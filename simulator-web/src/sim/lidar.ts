import { raySegmentDistance } from "./math";
import type { LidarSample, Pose2, WorldDefinition } from "./types";
import { worldLidarSegments } from "./world-geometry";

export interface LidarConfig {
  rays: number;
  maxRange: number;
}

export const DEFAULT_LIDAR_CONFIG: Readonly<LidarConfig> = Object.freeze({
  rays: 180,
  maxRange: 12,
});

export type LidarRaycaster = (pose: Pose2, localAngle: number) => LidarSample;

export function createLidarRaycaster(
  world: WorldDefinition,
  maxRange = DEFAULT_LIDAR_CONFIG.maxRange,
): LidarRaycaster {
  const segments = worldLidarSegments(world);
  return (pose, localAngle) => {
    const worldAngle = pose.yaw + localAngle;
    let closest = maxRange;
    let hit = false;
    for (const segment of segments) {
      const distance = raySegmentDistance(
        pose,
        worldAngle,
        segment.start,
        segment.end,
      );
      if (distance !== null && distance < closest) {
        closest = distance;
        hit = true;
      }
    }
    return { angle: localAngle, distance: closest, hit };
  };
}

export function scanLidar(
  world: WorldDefinition,
  pose: Pose2,
  config: LidarConfig = DEFAULT_LIDAR_CONFIG,
): LidarSample[] {
  const samples: LidarSample[] = [];
  const cast = createLidarRaycaster(world, config.maxRange);
  for (let index = 0; index < config.rays; index += 1) {
    const localAngle = (index / config.rays) * Math.PI * 2;
    samples.push(cast(pose, localAngle));
  }
  return samples;
}
