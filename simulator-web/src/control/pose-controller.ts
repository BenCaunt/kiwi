import { rotate, wrapAngle } from "../sim/math";
import type { Pose2, Twist2 } from "../sim/types";

export interface RelativePoseDelta {
  dx: number;
  dy: number;
  dyaw: number;
}

export interface PoseControllerConfig {
  kpX: number;
  kpY: number;
  kpYaw: number;
  maxLinearSpeed: number;
  maxAngularSpeed: number;
  positionTolerance: number;
  yawTolerance: number;
}

export const DEFAULT_POSE_CONTROLLER_CONFIG: Readonly<PoseControllerConfig> =
  Object.freeze({
    kpX: 0.8,
    kpY: 0.8,
    kpYaw: 1.5,
    maxLinearSpeed: 0.25,
    maxAngularSpeed: 1,
    positionTolerance: 0.04,
    yawTolerance: (2 * Math.PI) / 180,
  });

function finiteValues(values: readonly number[], name: string): void {
  if (!values.every(Number.isFinite)) {
    throw new Error(`${name} values must be finite`);
  }
}

function nonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be finite and non-negative`);
  }
  return value;
}

function positive(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be finite and positive`);
  }
  return value;
}

function copyPose(pose: Pose2): Pose2 {
  return { x: pose.x, y: pose.y, yaw: pose.yaw };
}

export function composeRelativePose(
  origin: Pose2,
  relative: RelativePoseDelta,
): Pose2 {
  finiteValues(
    [origin.x, origin.y, origin.yaw, relative.dx, relative.dy, relative.dyaw],
    "Pose",
  );
  const offset = rotate({ x: relative.dx, y: relative.dy }, origin.yaw);
  return {
    x: origin.x + offset.x,
    y: origin.y + offset.y,
    yaw: wrapAngle(origin.yaw + relative.dyaw),
  };
}

export function poseError(current: Pose2, target: Pose2): Pose2 {
  return {
    x: target.x - current.x,
    y: target.y - current.y,
    yaw: wrapAngle(target.yaw - current.yaw),
  };
}

/** Matches scripts/kiwi_pose_controller.py in the sensor-aligned body frame. */
export class PoseStabilizingController {
  readonly config: PoseControllerConfig;

  constructor(config: Partial<PoseControllerConfig> = {}) {
    this.config = { ...DEFAULT_POSE_CONTROLLER_CONFIG, ...config };
    nonNegative(this.config.kpX, "kpX");
    nonNegative(this.config.kpY, "kpY");
    nonNegative(this.config.kpYaw, "kpYaw");
    positive(this.config.maxLinearSpeed, "maxLinearSpeed");
    positive(this.config.maxAngularSpeed, "maxAngularSpeed");
    nonNegative(this.config.positionTolerance, "positionTolerance");
    nonNegative(this.config.yawTolerance, "yawTolerance");
  }

  withinPositionTolerance(current: Pose2, target: Pose2): boolean {
    return (
      Math.hypot(current.x - target.x, current.y - target.y) <=
      this.config.positionTolerance
    );
  }

  atTarget(current: Pose2, target: Pose2): boolean {
    return (
      this.withinPositionTolerance(current, target) &&
      Math.abs(poseError(current, target).yaw) <= this.config.yawTolerance
    );
  }

  command(current: Pose2, target: Pose2): Twist2 {
    const error = poseError(current, target);
    const mapCommand = {
      x: this.config.kpX * error.x,
      y: this.config.kpY * error.y,
    };
    const bodyCommand = rotate(mapCommand, -current.yaw);
    const speed = Math.hypot(bodyCommand.x, bodyCommand.y);
    const scale =
      speed > this.config.maxLinearSpeed
        ? this.config.maxLinearSpeed / speed
        : 1;
    const settled = this.withinPositionTolerance(current, target);
    return {
      vx: settled ? 0 : bodyCommand.x * scale,
      vy: settled ? 0 : bodyCommand.y * scale,
      omega: Math.max(
        -this.config.maxAngularSpeed,
        Math.min(this.config.kpYaw * error.yaw, this.config.maxAngularSpeed),
      ),
    };
  }
}

/** Anchors relative policy actions once, then tracks a fixed world-frame pose. */
export class RelativePoseController {
  readonly controller: PoseStabilizingController;
  private activeTarget?: Pose2;
  private activeTrajectory: Pose2[] = [];
  private activeLookaheadIndex = 0;

  constructor(config: Partial<PoseControllerConfig> = {}) {
    this.controller = new PoseStabilizingController(config);
  }

  get target(): Pose2 | undefined {
    return this.activeTarget ? copyPose(this.activeTarget) : undefined;
  }

  get trajectory(): readonly Pose2[] {
    return this.activeTrajectory.map(copyPose);
  }

  get lookaheadIndex(): number {
    return this.activeLookaheadIndex;
  }

  get active(): boolean {
    return this.activeTarget !== undefined;
  }

  clear(): void {
    this.activeTarget = undefined;
    this.activeTrajectory = [];
    this.activeLookaheadIndex = 0;
  }

  setRelativeTarget(origin: Pose2, relative: RelativePoseDelta): Pose2 {
    const target = composeRelativePose(origin, relative);
    this.activeTrajectory = [target];
    this.activeLookaheadIndex = 0;
    this.activeTarget = target;
    return copyPose(target);
  }

  setRelativeTrajectory(
    origin: Pose2,
    relatives: readonly RelativePoseDelta[],
    lookaheadIndex: number,
  ): Pose2 {
    if (relatives.length === 0) {
      throw new Error("Relative trajectory must contain at least one waypoint");
    }
    if (!Number.isInteger(lookaheadIndex) || lookaheadIndex < 0) {
      throw new Error("Trajectory lookahead index must be a non-negative integer");
    }
    if (lookaheadIndex >= relatives.length) {
      throw new Error("Trajectory lookahead index exceeds the waypoint horizon");
    }
    this.activeTrajectory = relatives.map((relative) =>
      composeRelativePose(origin, relative),
    );
    this.activeLookaheadIndex = lookaheadIndex;
    const selected = this.activeTrajectory[lookaheadIndex];
    if (!selected) throw new Error("Trajectory lookahead target is unavailable");
    this.activeTarget = selected;
    return copyPose(selected);
  }

  atTarget(current: Pose2): boolean {
    return this.activeTarget !== undefined &&
      this.controller.atTarget(current, this.activeTarget);
  }

  command(current: Pose2): Twist2 {
    return this.activeTarget
      ? this.controller.command(current, this.activeTarget)
      : { vx: 0, vy: 0, omega: 0 };
  }
}
