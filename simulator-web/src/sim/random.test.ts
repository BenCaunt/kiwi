import { describe, expect, it } from "vitest";

import { deriveSeed, SeededRandom } from "./random";

describe("named random streams", () => {
  it("are repeatable and isolated by name", () => {
    expect(deriveSeed(42, "lidar")).toBe(deriveSeed(42, "lidar"));
    expect(deriveSeed(42, "lidar")).not.toBe(deriveSeed(42, "odometry_imu"));
    const first = new SeededRandom(deriveSeed(42, "lidar"));
    const second = new SeededRandom(deriveSeed(42, "lidar"));
    expect([first.uniform(), first.normal()]).toEqual([second.uniform(), second.normal()]);
  });
});
