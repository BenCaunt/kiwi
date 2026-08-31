import {
  RelativePoseController,
  type PoseControllerConfig,
  type RelativePoseDelta,
} from "../control/pose-controller";
import {
  KiwiSimEngine,
  type EngineAdvanceResult,
  type EngineEvent,
  type EngineResetOptions,
  type EngineSnapshot,
} from "./engine";
import type { Pose2, Twist2, WorldDefinition } from "./types";

export type ActionMode =
  | "relative_pose_v1"
  | "relative_trajectory_v1"
  | "twist_aligned_v1";

export interface RelativePoseAction extends RelativePoseDelta {
  kind: "relative_pose";
}

export interface RelativeTrajectoryAction {
  kind: "relative_trajectory";
  waypoints: RelativePoseDelta[];
}

export interface AlignedTwistAction extends Twist2 {
  kind: "twist";
}

export type EnvironmentAction =
  | RelativePoseAction
  | RelativeTrajectoryAction
  | AlignedTwistAction;

export interface RlEnvironmentConfig {
  actionMode: ActionMode;
  policyHz: number;
  controllerHz: number;
  trajectoryLookaheadIndex: number;
  maxEpisodeSteps: number;
  maxRelativeTranslationM: number;
  maxRelativeYawRad: number;
  maxTrajectoryWaypoints: number;
  controller: Partial<PoseControllerConfig>;
}

export const DEFAULT_RL_ENVIRONMENT_CONFIG: Readonly<RlEnvironmentConfig> =
  Object.freeze({
    actionMode: "relative_pose_v1",
    policyHz: 4,
    controllerHz: 20,
    trajectoryLookaheadIndex: 0,
    maxEpisodeSteps: 400,
    maxRelativeTranslationM: 2,
    maxRelativeYawRad: Math.PI,
    maxTrajectoryWaypoints: 32,
    controller: Object.freeze({}),
  });

export interface ControllerCommandSample {
  simulationTime: number;
  target: Pose2;
  command: Twist2;
}

export interface RelativeTargetEvent {
  type:
    | "relative_target_accepted"
    | "relative_target_replaced"
    | "relative_target_reached";
  simulationTime: number;
  target: Pose2;
}

export type RlEnvironmentEvent = EngineEvent | RelativeTargetEvent;

export interface RlStepResult {
  action: EnvironmentAction;
  previous: EngineSnapshot;
  current: EngineSnapshot;
  ticks: number;
  collisionTickCount: number;
  controllerCommands: ControllerCommandSample[];
  events: RlEnvironmentEvent[];
  terminated: boolean;
  truncated: boolean;
}

export interface RlResetOptions extends EngineResetOptions {
  world?: WorldDefinition;
}

function positiveRate(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite rate`);
  }
}

function exactTickCount(physicsHz: number, rate: number, name: string): number {
  positiveRate(rate, name);
  const count = physicsHz / rate;
  if (!Number.isInteger(count)) {
    throw new Error(`${name} must divide physicsHz exactly`);
  }
  return count;
}

function copyTwist(twist: Twist2): Twist2 {
  return { vx: twist.vx, vy: twist.vy, omega: twist.omega };
}

function copyDelta(delta: RelativePoseDelta): RelativePoseDelta {
  return { dx: delta.dx, dy: delta.dy, dyaw: delta.dyaw };
}

function copyAction(action: EnvironmentAction): EnvironmentAction {
  if (action.kind === "relative_pose") return { kind: action.kind, ...copyDelta(action) };
  if (action.kind === "relative_trajectory") {
    return { kind: action.kind, waypoints: action.waypoints.map(copyDelta) };
  }
  return { kind: action.kind, ...copyTwist(action) };
}

function validateFinite(values: readonly number[], name: string): void {
  if (!values.every(Number.isFinite)) throw new Error(`${name} values must be finite`);
}

/** Fixed-duration RL control layer above the deterministic simulation engine. */
export class KiwiRlEnvironment {
  readonly engine: KiwiSimEngine;
  readonly config: RlEnvironmentConfig;
  readonly relativeController: RelativePoseController;
  readonly physicsTicksPerPolicyStep: number;
  readonly physicsTicksPerControllerStep: number;
  episodeSteps = 0;
  private needsReset = false;
  private reportedTargetReached = false;

  constructor(
    engine: KiwiSimEngine,
    config: Partial<RlEnvironmentConfig> = {},
  ) {
    this.engine = engine;
    this.config = {
      ...DEFAULT_RL_ENVIRONMENT_CONFIG,
      ...config,
      controller: config.controller ?? DEFAULT_RL_ENVIRONMENT_CONFIG.controller,
    };
    if (!Number.isInteger(this.config.maxEpisodeSteps) || this.config.maxEpisodeSteps <= 0) {
      throw new Error("maxEpisodeSteps must be a positive integer");
    }
    positiveRate(this.config.maxRelativeTranslationM, "maxRelativeTranslationM");
    positiveRate(this.config.maxRelativeYawRad, "maxRelativeYawRad");
    if (!Number.isInteger(this.config.maxTrajectoryWaypoints) || this.config.maxTrajectoryWaypoints <= 0) {
      throw new Error("maxTrajectoryWaypoints must be a positive integer");
    }
    if (
      !Number.isInteger(this.config.trajectoryLookaheadIndex) ||
      this.config.trajectoryLookaheadIndex < 0
    ) {
      throw new Error("trajectoryLookaheadIndex must be a non-negative integer");
    }
    this.physicsTicksPerPolicyStep = exactTickCount(
      engine.config.physicsHz,
      this.config.policyHz,
      "policyHz",
    );
    this.physicsTicksPerControllerStep = exactTickCount(
      engine.config.physicsHz,
      this.config.controllerHz,
      "controllerHz",
    );
    if (this.config.controllerHz < this.config.policyHz) {
      throw new Error("controllerHz must be at least policyHz");
    }
    this.relativeController = new RelativePoseController(this.config.controller);
  }

  reset(options: RlResetOptions = {}): EngineSnapshot {
    this.episodeSteps = 0;
    this.needsReset = false;
    this.reportedTargetReached = false;
    this.relativeController.clear();
    return this.engine.reset(options.world ?? this.engine.world, options);
  }

  step(action: EnvironmentAction): RlStepResult {
    if (this.needsReset) throw new Error("Environment requires reset after truncation");
    this.validateAction(action);
    const appliedAction = copyAction(action);
    const previous = this.engine.snapshot();
    const events: RlEnvironmentEvent[] = [];
    const controllerCommands: ControllerCommandSample[] = [];
    const previousTarget = this.relativeController.target;

    if (action.kind === "relative_pose") {
      const target = this.relativeController.setRelativeTarget(
        previous.robot.pose,
        action,
      );
      this.reportedTargetReached = false;
      if (previousTarget) {
        events.push({
          type: "relative_target_replaced",
          simulationTime: this.engine.simulationTime,
          target: previousTarget,
        });
      }
      events.push({
        type: "relative_target_accepted",
        simulationTime: this.engine.simulationTime,
        target,
      });
    } else if (action.kind === "relative_trajectory") {
      const target = this.relativeController.setRelativeTrajectory(
        previous.robot.pose,
        action.waypoints,
        this.config.trajectoryLookaheadIndex,
      );
      this.reportedTargetReached = false;
      if (previousTarget) {
        events.push({
          type: "relative_target_replaced",
          simulationTime: this.engine.simulationTime,
          target: previousTarget,
        });
      }
      events.push({
        type: "relative_target_accepted",
        simulationTime: this.engine.simulationTime,
        target,
      });
    } else {
      this.relativeController.clear();
      this.engine.recordCommand();
      this.engine.setAlignedCommand(action, 1 / this.config.policyHz + 1e-9);
    }

    let tickIndex = 0;
    const advance = this.engine.advanceTicks(
      this.physicsTicksPerPolicyStep,
      (activeEngine) => {
        if (
          action.kind !== "twist" &&
          tickIndex % this.physicsTicksPerControllerStep === 0
        ) {
          const target = this.relativeController.target;
          if (!target) throw new Error("Relative controller lost its target");
          const command = this.relativeController.command(activeEngine.robot.state.pose);
          activeEngine.recordCommand();
          activeEngine.setAlignedCommand(
            command,
            1 / this.config.controllerHz + 1e-9,
          );
          controllerCommands.push({
            simulationTime: activeEngine.simulationTime,
            target,
            command: copyTwist(command),
          });
        }
        tickIndex += 1;
      },
    );
    events.push(...advance.events);
    this.appendReachedEvent(events);

    this.episodeSteps += 1;
    const truncated = this.episodeSteps >= this.config.maxEpisodeSteps;
    if (truncated) this.needsReset = true;
    return {
      action: appliedAction,
      previous,
      current: this.engine.snapshot(),
      ticks: advance.ticks,
      collisionTickCount: advance.collisionTickCount,
      controllerCommands,
      events,
      terminated: false,
      truncated,
    };
  }

  private appendReachedEvent(events: RlEnvironmentEvent[]): void {
    const target = this.relativeController.target;
    if (
      target &&
      !this.reportedTargetReached &&
      this.relativeController.atTarget(this.engine.robot.state.pose)
    ) {
      events.push({
        type: "relative_target_reached",
        simulationTime: this.engine.simulationTime,
        target,
      });
      this.reportedTargetReached = true;
    }
  }

  private validateAction(action: EnvironmentAction): void {
    const expectedKind =
      this.config.actionMode === "relative_pose_v1"
        ? "relative_pose"
        : this.config.actionMode === "relative_trajectory_v1"
          ? "relative_trajectory"
          : "twist";
    if (action.kind !== expectedKind) {
      throw new Error(
        `Action kind ${action.kind} does not match configured mode ${this.config.actionMode}`,
      );
    }
    if (action.kind === "relative_pose") {
      validateFinite([action.dx, action.dy, action.dyaw], "Relative pose action");
      this.validateRelativeDelta(action);
    } else if (action.kind === "relative_trajectory") {
      if (action.waypoints.length === 0) {
        throw new Error("Relative trajectory action must contain waypoints");
      }
      if (action.waypoints.length > this.config.maxTrajectoryWaypoints) {
        throw new Error(`Relative trajectory exceeds ${this.config.maxTrajectoryWaypoints} waypoints`);
      }
      for (const waypoint of action.waypoints) {
        validateFinite(
          [waypoint.dx, waypoint.dy, waypoint.dyaw],
          "Relative trajectory waypoint",
        );
        this.validateRelativeDelta(waypoint);
      }
    } else {
      validateFinite([action.vx, action.vy, action.omega], "Twist action");
      if (Math.hypot(action.vx, action.vy) > this.engine.config.robotConfig.maxLinearSpeed) {
        throw new Error("Twist action exceeds the configured linear speed limit");
      }
      if (Math.abs(action.omega) > this.engine.config.robotConfig.maxAngularSpeed) {
        throw new Error("Twist action exceeds the configured angular speed limit");
      }
    }
  }

  private validateRelativeDelta(delta: RelativePoseDelta): void {
    if (Math.hypot(delta.dx, delta.dy) > this.config.maxRelativeTranslationM) {
      throw new Error("Relative action exceeds the configured translation limit");
    }
    if (Math.abs(delta.dyaw) > this.config.maxRelativeYawRad) {
      throw new Error("Relative action exceeds the configured yaw limit");
    }
  }
}

export function controllerEffort(samples: readonly ControllerCommandSample[]): number {
  return samples.reduce(
    (total, sample) =>
      total +
      sample.command.vx * sample.command.vx +
      sample.command.vy * sample.command.vy +
      sample.command.omega * sample.command.omega,
    0,
  );
}

export function engineEvents(result: RlStepResult): EngineEvent[] {
  return result.events.filter((event): event is EngineEvent =>
    !event.type.startsWith("relative_target_"),
  );
}

export function engineAdvance(result: RlStepResult): EngineAdvanceResult {
  return {
    ticks: result.ticks,
    collisionTickCount: result.collisionTickCount,
    events: engineEvents(result),
  };
}
