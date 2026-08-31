import { clamp, distanceToSegment, rotate, wrapAngle } from "./math";
import type { Pose2, Twist2, WallSegment, WorldDefinition } from "./types";
import {
  DEFAULT_ROBOT_COLLISION_HEIGHT_M,
  worldCollisionSegments,
} from "./world-geometry";

export interface RobotConfig {
  robotYawDeg: number;
  radius: number;
  height: number;
  responseTime: number;
  commandTimeout: number;
  maxLinearSpeed: number;
  maxAngularSpeed: number;
}

export interface RobotState {
  pose: Pose2;
  commandAligned: Twist2;
  velocityAligned: Twist2;
  commandActive: boolean;
  collided: boolean;
}

export const DEFAULT_ROBOT_CONFIG: Readonly<RobotConfig> = Object.freeze({
  robotYawDeg: 60,
  radius: 0.13,
  height: DEFAULT_ROBOT_COLLISION_HEIGHT_M,
  responseTime: 0.12,
  commandTimeout: 0.25,
  maxLinearSpeed: 0.8,
  maxAngularSpeed: 2.5,
});

function copyTwist(twist: Twist2): Twist2 {
  return { vx: twist.vx, vy: twist.vy, omega: twist.omega };
}

function zeroTwist(): Twist2 {
  return { vx: 0, vy: 0, omega: 0 };
}

export function alignedToRaw(twist: Twist2, robotYawDeg = 60): Twist2 {
  const raw = rotate({ x: twist.vx, y: twist.vy }, (-robotYawDeg * Math.PI) / 180);
  return { vx: raw.x, vy: raw.y, omega: twist.omega };
}

export function rawToAligned(twist: Twist2, robotYawDeg = 60): Twist2 {
  const aligned = rotate(
    { x: twist.vx, y: twist.vy },
    (robotYawDeg * Math.PI) / 180,
  );
  return { vx: aligned.x, vy: aligned.y, omega: twist.omega };
}

export class KiwiRobot {
  readonly config: RobotConfig;
  world: WorldDefinition;
  state: RobotState;
  private lastCommandAt = Number.NEGATIVE_INFINITY;
  private activeCommandTimeout: number;
  private collisionSegments: WallSegment[];

  constructor(
    world: WorldDefinition,
    config: RobotConfig = DEFAULT_ROBOT_CONFIG,
  ) {
    this.world = world;
    this.config = { ...config };
    this.state = this.createInitialState(world.spawn);
    this.activeCommandTimeout = this.config.commandTimeout;
    this.collisionSegments = worldCollisionSegments(world, this.config.height);
  }

  private createInitialState(pose: Pose2): RobotState {
    return {
      pose: { ...pose },
      commandAligned: zeroTwist(),
      velocityAligned: zeroTwist(),
      commandActive: false,
      collided: false,
    };
  }

  reset(world = this.world, pose: Pose2 = world.spawn): void {
    if (![pose.x, pose.y, pose.yaw].every(Number.isFinite)) {
      throw new Error("Reset pose values must be finite");
    }
    this.world = world;
    this.state = this.createInitialState(pose);
    this.collisionSegments = worldCollisionSegments(world, this.config.height);
    this.activeCommandTimeout = this.config.commandTimeout;
    this.lastCommandAt = Number.NEGATIVE_INFINITY;
  }

  setAlignedCommand(command: Twist2, now: number, timeoutSeconds?: number): void {
    if (![command.vx, command.vy, command.omega].every(Number.isFinite)) {
      throw new Error("Velocity command values must be finite");
    }
    this.state.commandAligned = copyTwist(command);
    this.state.commandActive = true;
    this.lastCommandAt = now;
    this.activeCommandTimeout =
      timeoutSeconds === undefined
        ? this.config.commandTimeout
        : Math.max(timeoutSeconds, 0.001);
  }

  setRawCommand(command: Twist2, now: number, timeoutSeconds?: number): void {
    this.setAlignedCommand(
      rawToAligned(command, this.config.robotYawDeg),
      now,
      timeoutSeconds,
    );
  }

  stop(now: number): void {
    this.setAlignedCommand(zeroTwist(), now);
  }

  private collides(x: number, y: number): boolean {
    const point = { x, y };
    return this.collisionSegments.some(
      (segment) =>
        distanceToSegment(point, segment.start, segment.end) < this.config.radius,
    );
  }

  private moveWithCollisions(dx: number, dy: number): boolean {
    const distance = Math.hypot(dx, dy);
    const stepLimit = this.config.radius * 0.35;
    const steps = Math.max(1, Math.ceil(distance / stepLimit));
    const stepX = dx / steps;
    const stepY = dy / steps;
    let collided = false;

    for (let index = 0; index < steps; index += 1) {
      const { x, y } = this.state.pose;
      if (!this.collides(x + stepX, y + stepY)) {
        this.state.pose.x += stepX;
        this.state.pose.y += stepY;
        continue;
      }

      collided = true;
      if (!this.collides(x + stepX, y)) this.state.pose.x += stepX;
      if (!this.collides(this.state.pose.x, y + stepY)) this.state.pose.y = y + stepY;
    }
    return collided;
  }

  step(dt: number, now: number): void {
    const safeDt = clamp(dt, 0, 0.1);
    if (safeDt === 0) return;
    if (now - this.lastCommandAt > this.activeCommandTimeout) {
      this.state.commandActive = false;
    }

    const requested = this.state.commandActive
      ? this.state.commandAligned
      : zeroTwist();
    const requestedSpeed = Math.hypot(requested.vx, requested.vy);
    const linearScale =
      requestedSpeed > this.config.maxLinearSpeed
        ? this.config.maxLinearSpeed / requestedSpeed
        : 1;
    const target = {
      vx: requested.vx * linearScale,
      vy: requested.vy * linearScale,
      omega: clamp(
        requested.omega,
        -this.config.maxAngularSpeed,
        this.config.maxAngularSpeed,
      ),
    };

    const alpha = 1 - Math.exp(-safeDt / Math.max(this.config.responseTime, 1e-6));
    const velocity = this.state.velocityAligned;
    velocity.vx += alpha * (target.vx - velocity.vx);
    velocity.vy += alpha * (target.vy - velocity.vy);
    velocity.omega += alpha * (target.omega - velocity.omega);

    const worldVelocity = rotate(
      { x: velocity.vx, y: velocity.vy },
      this.state.pose.yaw,
    );
    this.state.collided = this.moveWithCollisions(
      worldVelocity.x * safeDt,
      worldVelocity.y * safeDt,
    );
    this.state.pose.yaw = wrapAngle(
      this.state.pose.yaw + velocity.omega * safeDt,
    );
  }
}
