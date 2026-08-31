import { describe, expect, it } from "vitest";

import { FixedStepClock } from "./fixed-step";

describe("FixedStepClock", () => {
  it("advances an exact integer number of ticks without wall time", () => {
    const clock = new FixedStepClock(120);
    const samples: number[] = [];

    clock.advanceTicks(30, (_dt, now) => samples.push(now));

    expect(samples).toHaveLength(30);
    expect(clock.simulationTime).toBeCloseTo(0.25, 12);
    expect(samples[0]).toBeCloseTo(1 / 120, 12);
    expect(samples.at(-1)).toBeCloseTo(0.25, 12);
  });

  it("rejects fractional and negative tick counts", () => {
    const clock = new FixedStepClock();
    expect(() => clock.advanceTicks(1.5, () => undefined)).toThrow();
    expect(() => clock.advanceTicks(-1, () => undefined)).toThrow();
  });
});
