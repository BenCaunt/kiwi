import { describe, expect, it } from "vitest";

import { worldCollisionSegments } from "./world-geometry";
import type { Vec2, WallSegment } from "./types";
import { WORLD_LIST, worldById } from "./worlds";

function distanceToSegment(point: Vec2, segment: WallSegment): number {
  const dx = segment.end.x - segment.start.x;
  const dy = segment.end.y - segment.start.y;
  const denominator = dx * dx + dy * dy;
  const projection = denominator === 0
    ? 0
    : Math.max(
        0,
        Math.min(
          1,
          ((point.x - segment.start.x) * dx + (point.y - segment.start.y) * dy) /
            denominator,
        ),
      );
  return Math.hypot(
    point.x - (segment.start.x + projection * dx),
    point.y - (segment.start.y + projection * dy),
  );
}

describe("planar home collection", () => {
  const homes = WORLD_LIST.filter((world) => world.category === "home");

  it("offers several culturally distinct single-level layouts", () => {
    expect(homes.map((world) => world.id)).toEqual([
      "home",
      "home-machiya",
      "home-riad",
      "home-kerala",
    ]);
    expect(homes.every((world) => world.tags?.includes("single level"))).toBe(true);
    expect(new Set(homes.map((world) => world.style))).toHaveLength(homes.length);
  });

  it("gives every home a furnished, lit, multi-room interior", () => {
    for (const home of homes) {
      expect(home.floorZones?.length, home.id).toBeGreaterThanOrEqual(6);
      expect(home.objects?.length, home.id).toBeGreaterThanOrEqual(14);
      expect(home.lights?.length, home.id).toBeGreaterThanOrEqual(4);
      expect(home.ambience, home.id).toBeDefined();
    }
  });

  it("keeps object, room, and fixture identifiers unique within each home", () => {
    for (const home of homes) {
      for (const collection of [home.floorZones, home.objects, home.lights]) {
        const ids = collection?.map(({ id }) => id) ?? [];
        expect(new Set(ids).size, home.id).toBe(ids.length);
      }
    }
  });

  it("spawns the robot clear of walls and collidable furniture", () => {
    for (const home of homes) {
      const nearest = Math.min(
        ...worldCollisionSegments(home).map((segment) =>
          distanceToSegment(home.spawn, segment),
        ),
      );
      expect(nearest, home.id).toBeGreaterThan(0.13);
    }
  });

  it("uses a varied procedural surface palette without external assets", () => {
    const patterns = new Set(
      homes.flatMap((home) => [
        ...(home.floorZones ?? []).map((zone) => zone.pattern),
        ...(home.objects ?? []).map((object) => object.pattern),
      ]),
    );
    for (const pattern of [
      "carpet",
      "mosaic",
      "stone",
      "tatami",
      "terrazzo",
      "tile",
      "wood",
    ] as const) {
      expect(patterns.has(pattern), pattern).toBe(true);
    }
  });

  it("falls back to the calibrated family home for unknown ids", () => {
    expect(worldById("missing").id).toBe("home");
  });
});
