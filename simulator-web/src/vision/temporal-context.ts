import type { CameraCalibration, VisionFrame } from "./camera";

export interface VisionObservation {
  schema: "vision_v1" | "vision_goal_v1";
  rgb: Uint8Array;
  rgbShape: readonly [number, number, number, 3];
  rgbValid: Uint8Array;
  rgbTimeS: Float64Array;
  rgbSequence: Uint32Array;
  goalRgb?: Uint8Array;
  goalRgbShape?: readonly [number, number, 3];
  goalRgbValid: number;
  goalRgbSequence?: number;
  calibration: Readonly<CameraCalibration>;
}

export interface TemporalVisionConfig {
  contextLength: number;
  contextStride: number;
}

function validateFrame(frame: VisionFrame, calibration?: CameraCalibration): void {
  const expected = frame.width * frame.height * 3;
  if (frame.rgb.length !== expected) {
    throw new Error(`RGB frame has ${frame.rgb.length} bytes; expected ${expected}`);
  }
  if (
    calibration &&
    (frame.width !== calibration.width || frame.height !== calibration.height)
  ) {
    throw new Error("RGB frame dimensions changed within one temporal context");
  }
}

/** Fixed-shape chronological visual history for policy observations. */
export class TemporalVisionContext {
  readonly config: TemporalVisionConfig;
  private frames: VisionFrame[] = [];
  private goal?: VisionFrame;

  constructor(config: TemporalVisionConfig) {
    if (!Number.isInteger(config.contextLength) || config.contextLength <= 0) {
      throw new Error("contextLength must be a positive integer");
    }
    if (!Number.isInteger(config.contextStride) || config.contextStride <= 0) {
      throw new Error("contextStride must be a positive integer");
    }
    this.config = { ...config };
  }

  reset(initial: VisionFrame, goal?: VisionFrame): void {
    validateFrame(initial);
    if (goal) validateFrame(goal, initial.calibration);
    this.frames = [initial];
    this.goal = goal;
  }

  push(frame: VisionFrame): void {
    const first = this.frames[0];
    if (!first) throw new Error("Temporal vision context must be reset before push");
    validateFrame(frame, first.calibration);
    const latest = this.frames.at(-1);
    if (latest && frame.simulationTime <= latest.simulationTime) {
      throw new Error("Vision frame times must increase monotonically");
    }
    this.frames.push(frame);
    const maximum = (this.config.contextLength - 1) * this.config.contextStride + 1;
    if (this.frames.length > maximum) this.frames.splice(0, this.frames.length - maximum);
  }

  observation(): VisionObservation {
    const first = this.frames[0];
    const latestIndex = this.frames.length - 1;
    if (!first || latestIndex < 0) {
      throw new Error("Temporal vision context must be reset before observation");
    }
    const frameBytes = first.width * first.height * 3;
    const rgb = new Uint8Array(this.config.contextLength * frameBytes);
    const valid = new Uint8Array(this.config.contextLength);
    const times = new Float64Array(this.config.contextLength);
    const sequences = new Uint32Array(this.config.contextLength);

    for (let slot = 0; slot < this.config.contextLength; slot += 1) {
      const historyOffset =
        (this.config.contextLength - 1 - slot) * this.config.contextStride;
      const index = latestIndex - historyOffset;
      const frame = index >= 0 ? this.frames[index] : first;
      if (!frame) throw new Error("Temporal vision frame is unavailable");
      rgb.set(frame.rgb, slot * frameBytes);
      valid[slot] = index >= 0 ? 1 : 0;
      times[slot] = frame.simulationTime;
      sequences[slot] = frame.sequence;
    }

    const observation: VisionObservation = {
      schema: this.goal ? "vision_goal_v1" : "vision_v1",
      rgb,
      rgbShape: [
        this.config.contextLength,
        first.height,
        first.width,
        3,
      ],
      rgbValid: valid,
      rgbTimeS: times,
      rgbSequence: sequences,
      goalRgbValid: this.goal ? 1 : 0,
      calibration: first.calibration,
    };
    if (this.goal) {
      observation.goalRgb = this.goal.rgb.slice();
      observation.goalRgbShape = [first.height, first.width, 3];
      observation.goalRgbSequence = this.goal.sequence;
    }
    return observation;
  }
}
