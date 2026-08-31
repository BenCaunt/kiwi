import {
  KiwiRlEnvironment,
  type EnvironmentAction,
  type RlEnvironmentConfig,
  type RlResetOptions,
  type RlStepResult,
} from "../sim/rl-environment";
import type { EngineSnapshot } from "../sim/engine";
import type { Pose2 } from "../sim/types";
import type { KiwiVisionRenderer, VisionFrame } from "./camera";
import {
  TemporalVisionContext,
  type TemporalVisionConfig,
  type VisionObservation,
} from "./temporal-context";

export interface VisualEnvironmentConfig extends TemporalVisionConfig {
  control?: Partial<RlEnvironmentConfig>;
}

export interface VisualResetOptions extends RlResetOptions {
  goalPose?: Pose2;
}

export interface VisualResetResult {
  snapshot: EngineSnapshot;
  observation: VisionObservation;
}

export interface VisualStepResult extends RlStepResult {
  observation: VisionObservation;
}

function copyPose(pose: Pose2): Pose2 {
  return { x: pose.x, y: pose.y, yaw: pose.yaw };
}

/** Couples deterministic control with a renderer and fixed temporal RGB schema. */
export class KiwiVisualEnvironment {
  readonly control: KiwiRlEnvironment;
  readonly renderer: KiwiVisionRenderer;
  readonly context: TemporalVisionContext;
  private nextSequence = 0;
  private initialized = false;

  constructor(
    control: KiwiRlEnvironment,
    renderer: KiwiVisionRenderer,
    config: TemporalVisionConfig,
  ) {
    this.control = control;
    this.renderer = renderer;
    this.context = new TemporalVisionContext(config);
  }

  reset(options: VisualResetOptions = {}): VisualResetResult {
    const snapshot = this.control.reset(options);
    this.renderer.loadWorld(this.control.engine.world);
    this.nextSequence = 0;
    const initial = this.capture(snapshot.robot.pose, snapshot.simulationTime);
    const goal = options.goalPose
      ? this.capture(options.goalPose, snapshot.simulationTime, false)
      : undefined;
    this.context.reset(initial, goal);
    this.initialized = true;
    return { snapshot, observation: this.context.observation() };
  }

  step(action: EnvironmentAction): VisualStepResult {
    if (!this.initialized) {
      throw new Error("Visual environment must be reset before step");
    }
    const result = this.control.step(action);
    for (const event of result.events) {
      // reset() already captures the t=0 frame. The firmware schedule retains
      // its immediate first due event for backward compatibility.
      if (event.type === "camera_due" && event.scheduledTime > 0) {
        this.context.push(this.capture(event.pose, event.scheduledTime));
      }
    }
    return { ...result, observation: this.context.observation() };
  }

  private capture(
    pose: Pose2,
    simulationTime: number,
    advanceSequence = true,
  ): VisionFrame {
    const frame: VisionFrame = {
      rgb: this.renderer.captureRgb(pose),
      width: this.renderer.calibration.width,
      height: this.renderer.calibration.height,
      simulationTime,
      sequence: this.nextSequence,
      pose: copyPose(pose),
      calibration: this.renderer.calibration,
    };
    if (advanceSequence) this.nextSequence += 1;
    return frame;
  }
}
