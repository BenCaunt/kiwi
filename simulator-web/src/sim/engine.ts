import { FixedStepClock } from "./fixed-step";
import {
  RETAINED_ROBOT_PROFILE,
  type HardwareSensorProfile,
} from "./hardware-profile";
import { createLidarRaycaster, scanLidar } from "./lidar";
import { TimedPoseHistory } from "./pose-history";
import {
  DEFAULT_ROBOT_CONFIG,
  KiwiRobot,
  type RobotConfig,
  type RobotState,
} from "./robot";
import { FirmwareContract } from "./zenoh-contract";
import type { LidarSample, Pose2, Twist2, WorldDefinition } from "./types";

export interface EngineConfig {
  physicsHz: number;
  displayLidarHz: number;
  odometryHz: number;
  lidarBatchHz: number;
  statusHz: number;
  groundTruthHz: number;
  hardwareProfile: Readonly<HardwareSensorProfile>;
  robotConfig: Readonly<RobotConfig>;
  seed: number;
}

export const DEFAULT_ENGINE_CONFIG: Readonly<EngineConfig> = Object.freeze({
  physicsHz: 120,
  displayLidarHz: 10,
  odometryHz: 20,
  lidarBatchHz: 20,
  statusHz: 1,
  groundTruthHz: 20,
  hardwareProfile: RETAINED_ROBOT_PROFILE,
  robotConfig: DEFAULT_ROBOT_CONFIG,
  seed: 1,
});

interface TimedEngineEvent {
  simulationTime: number;
  scheduledTime: number;
}

export interface OdometryEvent extends TimedEngineEvent {
  type: "odometry";
  payload: Uint8Array;
}

export interface LidarBatchEvent extends TimedEngineEvent {
  type: "lidar";
  payload: Uint8Array;
}

export interface CameraDueEvent extends TimedEngineEvent {
  type: "camera_due";
  pose: Pose2;
}

export interface StatusEvent extends TimedEngineEvent {
  type: "status";
  payload: Uint8Array;
}

export interface GroundTruthEvent extends TimedEngineEvent {
  type: "ground_truth";
  worldId: string;
  pose: Pose2;
}

export interface ContactStartedEvent extends TimedEngineEvent {
  type: "contact_started";
  pose: Pose2;
}

export interface CommandTimedOutEvent extends TimedEngineEvent {
  type: "command_timed_out";
}

export type EngineEvent =
  | OdometryEvent
  | LidarBatchEvent
  | CameraDueEvent
  | StatusEvent
  | GroundTruthEvent
  | ContactStartedEvent
  | CommandTimedOutEvent;

export interface EngineAdvanceResult {
  ticks: number;
  collisionTickCount: number;
  events: EngineEvent[];
}

export interface EngineSnapshot {
  simulationTime: number;
  worldId: string;
  seed: number;
  robot: RobotState;
  lidar: LidarSample[];
}

export interface EngineResetOptions {
  seed?: number;
  pose?: Pose2;
}

export type BeforePhysicsTick = (
  engine: KiwiSimEngine,
  dt: number,
  simulationTime: number,
) => void;

function positiveRate(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite rate`);
  }
  return value;
}

function copyPose(pose: Pose2): Pose2 {
  return { x: pose.x, y: pose.y, yaw: pose.yaw };
}

function copyTwist(twist: Twist2): Twist2 {
  return { vx: twist.vx, vy: twist.vy, omega: twist.omega };
}

function copyRobotState(state: RobotState): RobotState {
  return {
    pose: copyPose(state.pose),
    commandAligned: copyTwist(state.commandAligned),
    velocityAligned: copyTwist(state.velocityAligned),
    commandActive: state.commandActive,
    collided: state.collided,
  };
}

function copyLidar(samples: readonly LidarSample[]): LidarSample[] {
  return samples.map((sample) => ({ ...sample }));
}

/**
 * Renderer- and transport-independent owner of Kiwi simulation time.
 *
 * Browser animation, headless RL stepping, and the firmware bridge all advance
 * this same object. Sensor deadlines are simulation-time based and therefore do
 * not change when a bridge connects or a renderer asks for another frame.
 */
export class KiwiSimEngine {
  readonly config: EngineConfig;
  readonly clock: FixedStepClock;
  readonly robot: KiwiRobot;
  private firmware: FirmwareContract;
  private readonly poseHistory = new TimedPoseHistory();
  private castLidarRay: ReturnType<typeof createLidarRaycaster>;
  private currentLidar: LidarSample[];
  private currentWorld: WorldDefinition;
  private nextDisplayLidarAt = 0;
  private nextOdometryAt = 0;
  private nextLidarBatchAt = 0;
  private nextCameraAt = 0;
  private nextStatusAt = 0;
  private nextGroundTruthAt = 0;
  private wasColliding = false;
  private currentSeed: number;

  constructor(
    world: WorldDefinition,
    config: Partial<EngineConfig> = {},
  ) {
    this.config = {
      ...DEFAULT_ENGINE_CONFIG,
      ...config,
      hardwareProfile: config.hardwareProfile ?? DEFAULT_ENGINE_CONFIG.hardwareProfile,
      robotConfig: config.robotConfig ?? DEFAULT_ENGINE_CONFIG.robotConfig,
    };
    positiveRate(this.config.physicsHz, "physicsHz");
    positiveRate(this.config.displayLidarHz, "displayLidarHz");
    positiveRate(this.config.odometryHz, "odometryHz");
    positiveRate(this.config.lidarBatchHz, "lidarBatchHz");
    positiveRate(this.config.statusHz, "statusHz");
    positiveRate(this.config.groundTruthHz, "groundTruthHz");
    if (!Number.isInteger(this.config.seed)) {
      throw new Error("seed must be an integer");
    }

    this.currentWorld = world;
    this.currentSeed = this.config.seed;
    this.clock = new FixedStepClock(this.config.physicsHz);
    this.robot = new KiwiRobot(world, this.config.robotConfig);
    this.firmware = new FirmwareContract(this.config.hardwareProfile, this.config.seed);
    this.castLidarRay = createLidarRaycaster(
      world,
      this.config.hardwareProfile.lidarMaxRangeM,
    );
    this.currentLidar = this.scanDisplayLidar();
    this.reset(world);
  }

  get world(): WorldDefinition {
    return this.currentWorld;
  }

  get simulationTime(): number {
    return this.clock.simulationTime;
  }

  get lidar(): readonly LidarSample[] {
    return this.currentLidar;
  }

  reset(
    world = this.currentWorld,
    options: EngineResetOptions = {},
  ): EngineSnapshot {
    const seed = options.seed ?? this.config.seed;
    if (!Number.isInteger(seed)) throw new Error("seed must be an integer");
    this.currentSeed = seed;
    this.currentWorld = world;
    this.robot.reset(world, options.pose ?? world.spawn);
    this.clock.reset();
    this.firmware = new FirmwareContract(
      this.config.hardwareProfile,
      this.currentSeed,
    );
    this.firmware.reset();
    this.poseHistory.reset(0, this.robot.state.pose);
    this.castLidarRay = createLidarRaycaster(
      world,
      this.config.hardwareProfile.lidarMaxRangeM,
    );
    this.currentLidar = this.scanDisplayLidar();
    this.nextDisplayLidarAt = 1 / this.config.displayLidarHz;
    this.nextOdometryAt = 0;
    // A real batch is emitted only after its twenty frames have been acquired.
    this.nextLidarBatchAt = 1 / this.config.lidarBatchHz;
    this.nextCameraAt = 0;
    this.nextStatusAt = 0;
    this.nextGroundTruthAt = 0;
    this.wasColliding = false;
    return this.snapshot();
  }

  recordCommand(): void {
    this.firmware.recordCommand();
  }

  setAlignedCommand(command: Twist2, timeoutSeconds?: number): void {
    this.robot.setAlignedCommand(command, this.simulationTime, timeoutSeconds);
  }

  setRawCommand(command: Twist2, timeoutSeconds?: number): void {
    this.robot.setRawCommand(command, this.simulationTime, timeoutSeconds);
  }

  stop(): void {
    this.robot.stop(this.simulationTime);
  }

  cameraPayload(jpeg: Uint8Array, simulationTime = this.simulationTime): Uint8Array {
    return this.firmware.camera(jpeg, simulationTime);
  }

  advanceFrame(
    frameSeconds: number,
    beforeTick?: BeforePhysicsTick,
  ): EngineAdvanceResult {
    return this.collectAdvance((step) => this.clock.advance(frameSeconds, step), beforeTick);
  }

  advanceTicks(count: number, beforeTick?: BeforePhysicsTick): EngineAdvanceResult {
    return this.collectAdvance((step) => this.clock.advanceTicks(count, step), beforeTick);
  }

  snapshot(): EngineSnapshot {
    return {
      simulationTime: this.simulationTime,
      worldId: this.currentWorld.id,
      seed: this.currentSeed,
      robot: copyRobotState(this.robot.state),
      lidar: copyLidar(this.currentLidar),
    };
  }

  private scanDisplayLidar(): LidarSample[] {
    return scanLidar(this.currentWorld, this.robot.state.pose, {
      rays: 180,
      maxRange: this.config.hardwareProfile.lidarMaxRangeM,
    });
  }

  private collectAdvance(
    advance: (step: (dt: number, now: number) => void) => void,
    beforeTick?: BeforePhysicsTick,
  ): EngineAdvanceResult {
    const result: EngineAdvanceResult = {
      ticks: 0,
      collisionTickCount: 0,
      events: [],
    };
    advance((dt, simulationTime) => {
      beforeTick?.(this, dt, simulationTime);
      this.stepPhysics(dt, simulationTime, result);
    });
    return result;
  }

  private stepPhysics(
    dt: number,
    simulationTime: number,
    result: EngineAdvanceResult,
  ): void {
    const commandWasActive = this.robot.state.commandActive;
    this.robot.step(dt, simulationTime);
    this.poseHistory.append(simulationTime, this.robot.state.pose);
    this.firmware.step(dt, this.robot.state);
    result.ticks += 1;

    if (commandWasActive && !this.robot.state.commandActive) {
      result.events.push({
        type: "command_timed_out",
        simulationTime,
        scheduledTime: simulationTime,
      });
    }
    if (this.robot.state.collided) {
      result.collisionTickCount += 1;
      if (!this.wasColliding) {
        result.events.push({
          type: "contact_started",
          simulationTime,
          scheduledTime: simulationTime,
          pose: copyPose(this.robot.state.pose),
        });
      }
    }
    this.wasColliding = this.robot.state.collided;

    while (simulationTime + 1e-12 >= this.nextDisplayLidarAt) {
      this.currentLidar = this.scanDisplayLidar();
      this.nextDisplayLidarAt += 1 / this.config.displayLidarHz;
    }
    while (simulationTime + 1e-12 >= this.nextOdometryAt) {
      const scheduledTime = this.nextOdometryAt;
      result.events.push({
        type: "odometry",
        simulationTime,
        scheduledTime,
        payload: this.firmware.odometry(this.robot.state, simulationTime),
      });
      this.nextOdometryAt += 1 / this.config.odometryHz;
    }
    while (simulationTime + 1e-12 >= this.nextLidarBatchAt) {
      const scheduledTime = this.nextLidarBatchAt;
      const payload = this.firmware.lidar((localAngle, acquisitionTime) => {
        const acquisitionPose = this.poseHistory.interpolate(acquisitionTime);
        if (!acquisitionPose) return 0;
        const sample = this.castLidarRay(acquisitionPose, localAngle);
        return sample.hit ? sample.distance : 0;
      });
      result.events.push({
        type: "lidar",
        simulationTime,
        scheduledTime,
        payload,
      });
      this.nextLidarBatchAt += 1 / this.config.lidarBatchHz;
    }
    while (simulationTime + 1e-12 >= this.nextCameraAt) {
      const scheduledTime = this.nextCameraAt;
      const capturePose = this.poseHistory.interpolate(scheduledTime);
      result.events.push({
        type: "camera_due",
        simulationTime,
        scheduledTime,
        pose: copyPose(capturePose ?? this.robot.state.pose),
      });
      this.nextCameraAt += 1 / this.config.hardwareProfile.cameraHz;
    }
    while (simulationTime + 1e-12 >= this.nextStatusAt) {
      const scheduledTime = this.nextStatusAt;
      result.events.push({
        type: "status",
        simulationTime,
        scheduledTime,
        payload: this.firmware.status(this.currentWorld.id, simulationTime),
      });
      this.nextStatusAt += 1 / this.config.statusHz;
    }
    while (simulationTime + 1e-12 >= this.nextGroundTruthAt) {
      const scheduledTime = this.nextGroundTruthAt;
      result.events.push({
        type: "ground_truth",
        simulationTime,
        scheduledTime,
        worldId: this.currentWorld.id,
        pose: copyPose(this.robot.state.pose),
      });
      this.nextGroundTruthAt += 1 / this.config.groundTruthHz;
    }
  }
}
