import { describe, expect, it } from "vitest";

import type { WorldDefinition } from "../sim/types";
import { NavigationGrid } from "./navigation-grid";

const dividedWorld: WorldDefinition = {
  id: "divided",
  name: "Divided",
  description: "Geodesic fixture",
  spawn: { x: -1, y: 0, yaw: 0 },
  walls: [
    { start: { x: -2, y: -2 }, end: { x: 2, y: -2 } },
    { start: { x: 2, y: -2 }, end: { x: 2, y: 2 } },
    { start: { x: 2, y: 2 }, end: { x: -2, y: 2 } },
    { start: { x: -2, y: 2 }, end: { x: -2, y: -2 } },
    { start: { x: 0, y: -2 }, end: { x: 0, y: 1 } },
  ],
};

describe("NavigationGrid", () => {
  it("uses a collision-aware path rather than distance through a wall", () => {
    const field = new NavigationGrid(dividedWorld, { resolutionM: 0.1 }).createDistanceField({
      x: 1,
      y: 0,
    });
    const distance = field.distance({ x: -1, y: 0 });
    expect(distance).toBeGreaterThan(3.2);
    expect(distance).toBeLessThan(8);
  });

  it("is deterministic for repeated queries", () => {
    const grid = new NavigationGrid(dividedWorld);
    const a = grid.createDistanceField({ x: 1, y: 0 }).distance({ x: -1, y: 0 });
    const b = grid.createDistanceField({ x: 1, y: 0 }).distance({ x: -1, y: 0 });
    expect(a).toBe(b);
  });
});
