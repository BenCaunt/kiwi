export class FixedStepClock {
  readonly stepHz: number;
  readonly stepSeconds: number;
  readonly maxFrameSeconds: number;
  private tickCount = 0;
  private accumulator = 0;

  constructor(stepHz = 120, maxFrameSeconds = 0.1) {
    this.stepHz = stepHz;
    this.stepSeconds = 1 / stepHz;
    this.maxFrameSeconds = maxFrameSeconds;
  }

  get simulationTime(): number {
    return this.tickCount / this.stepHz;
  }

  reset(): void {
    this.accumulator = 0;
    this.tickCount = 0;
  }

  advanceTicks(count: number, step: (dt: number, now: number) => void): void {
    if (!Number.isInteger(count) || count < 0) {
      throw new Error("Fixed-step tick count must be a non-negative integer");
    }
    for (let index = 0; index < count; index += 1) {
      this.tickCount += 1;
      step(this.stepSeconds, this.simulationTime);
    }
  }

  advance(frameSeconds: number, step: (dt: number, now: number) => void): void {
    this.accumulator += Math.min(Math.max(frameSeconds, 0), this.maxFrameSeconds);
    while (this.accumulator >= this.stepSeconds) {
      this.advanceTicks(1, step);
      this.accumulator -= this.stepSeconds;
    }
  }
}
