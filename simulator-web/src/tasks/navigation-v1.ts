import { controllerEffort, type EnvironmentAction } from "../sim/rl-environment";
import { wrapAngle } from "../sim/math";
import { deriveSeed } from "../sim/random";
import type { Pose2 } from "../sim/types";
import { WORLD_LIST } from "../sim/worlds";
import type { VisionObservation } from "../vision/temporal-context";
import {
  KiwiVisualEnvironment,
  type VisualResetResult,
  type VisualStepResult,
} from "../vision/visual-environment";
import { NavigationGrid, type GeodesicDistanceField } from "./navigation-grid";

export type NavigationTaskId = "image_goal_navigation_v1" | "point_navigation_v1";

export interface NavigationPair {
  id: string;
  start: Pose2;
  goal: Pose2;
}

/**
 * Small, reviewed task set. These are deliberately authored rather than sampled
 * from bounding boxes; grid reachability is validated again at reset.
 */
export const NAVIGATION_PAIRS: Readonly<Record<string, readonly NavigationPair[]>> =
  Object.freeze({
    home: [
      { id: "entry-short", start: { x: 0, y: -3.25, yaw: Math.PI / 2 }, goal: { x: 0, y: -2.9, yaw: Math.PI / 2 } },
      { id: "entry-to-hall", start: { x: 0, y: -3.25, yaw: Math.PI / 2 }, goal: { x: 0, y: 1.5, yaw: Math.PI / 2 } },
    ],
    "home-machiya": [
      { id: "genkan-to-dining", start: { x: 0, y: -4.35, yaw: Math.PI / 2 }, goal: { x: 0, y: -3, yaw: Math.PI / 2 } },
      { id: "genkan-to-hall", start: { x: 0, y: -4.35, yaw: Math.PI / 2 }, goal: { x: 0, y: 3.5, yaw: Math.PI / 2 } },
    ],
    "home-riad": [
      { id: "entry-north", start: { x: 0, y: -4.35, yaw: Math.PI / 2 }, goal: { x: 0.45, y: -3.45, yaw: Math.PI / 2 } },
      { id: "entry-to-courtyard", start: { x: 0, y: -4.35, yaw: Math.PI / 2 }, goal: { x: 0, y: -1.2, yaw: Math.PI / 2 } },
    ],
    "home-kerala": [
      { id: "veranda-to-entry", start: { x: 0, y: -4.8, yaw: Math.PI / 2 }, goal: { x: 0, y: -3.5, yaw: Math.PI / 2 } },
      { id: "veranda-to-courtyard", start: { x: 0, y: -4.8, yaw: Math.PI / 2 }, goal: { x: 0, y: -0.8, yaw: Math.PI / 2 } },
    ],
    room: [
      { id: "west-to-center", start: { x: -1.8, y: 0, yaw: 0 }, goal: { x: -0.5, y: 0, yaw: 0 } },
      { id: "west-to-east", start: { x: -1.8, y: 0, yaw: 0 }, goal: { x: 2.2, y: -1.3, yaw: 0 } },
    ],
    warehouse: [
      { id: "west-aisle", start: { x: -4.25, y: 0, yaw: 0 }, goal: { x: -3, y: 0, yaw: 0 } },
      { id: "cross-center-aisle", start: { x: -4.25, y: 0, yaw: 0 }, goal: { x: 4.25, y: 0, yaw: 0 } },
    ],
    maze: [
      { id: "first-corridor", start: { x: -3.35, y: -2.35, yaw: 0 }, goal: { x: -3, y: -2.35, yaw: 0 } },
      { id: "maze-traverse", start: { x: -3.35, y: -2.35, yaw: 0 }, goal: { x: 3.35, y: -2.35, yaw: 0 } },
    ],
  });

export interface RewardWeights {
  progress: number;
  success: number;
  collision: number;
  time: number;
  controller: number;
  smoothness: number;
}

export const DEFAULT_REWARD_WEIGHTS: Readonly<RewardWeights> = Object.freeze({
  progress: 1,
  success: 5,
  collision: 0.25,
  time: 0.01,
  controller: 0.001,
  smoothness: 0.01,
});

export interface NavigationTaskConfig {
  id: NavigationTaskId;
  successRadiusM: number;
  requireGoalHeading: boolean;
  successYawToleranceRad: number;
  reward: RewardWeights;
  privilegedDebug: boolean;
}

export const DEFAULT_NAVIGATION_TASK_CONFIG: Readonly<NavigationTaskConfig> =
  Object.freeze({
    id: "image_goal_navigation_v1",
    successRadiusM: 0.25,
    requireGoalHeading: false,
    successYawToleranceRad: 0.25,
    reward: DEFAULT_REWARD_WEIGHTS,
    privilegedDebug: false,
  });

export interface NavigationResetInfo {
  task_id: NavigationTaskId;
  task_pair_id: string;
  initial_geodesic_distance_m: number;
  goal_rgb_is_policy_input: boolean;
  privileged?: { start: Pose2; goal: Pose2 };
}

export interface RewardTerms {
  progress: number;
  success: number;
  collision: number;
  time: number;
  controller: number;
  smoothness: number;
}

export interface NavigationStepInfo {
  reward_terms: RewardTerms;
  geodesic_distance_m: number;
  success: boolean;
  termination_reason?: "goal_reached";
  collision_tick_count: number;
  privileged?: { pose: Pose2; goal: Pose2 };
}

export interface NavigationResetResult extends VisualResetResult {
  info: NavigationResetInfo;
}

export interface NavigationStepResult extends VisualStepResult {
  observation: VisionObservation;
  reward: number;
  terminated: boolean;
  info: NavigationStepInfo;
}

function actionVector(action: EnvironmentAction): readonly number[] {
  if (action.kind === "twist") return [action.vx, action.vy, action.omega];
  if (action.kind === "relative_pose") return [action.dx, action.dy, action.dyaw];
  const waypoint = action.waypoints.at(-1);
  return waypoint ? [waypoint.dx, waypoint.dy, waypoint.dyaw] : [0, 0, 0];
}

function squaredDifference(a: readonly number[], b?: readonly number[]): number {
  if (!b) return 0;
  return a.reduce((total, value, index) => total + (value - (b[index] ?? 0)) ** 2, 0);
}

function pairForSeed(worldId: string, seed: number): NavigationPair {
  const pairs = NAVIGATION_PAIRS[worldId];
  if (!pairs || pairs.length === 0) throw new Error(`No authored navigation pairs for ${worldId}`);
  const index = deriveSeed(seed, "task_sampling") % pairs.length;
  const pair = pairs[index];
  if (!pair) throw new Error(`Navigation pair ${index} is unavailable`);
  return pair;
}

/** Vision-first reference task; metric goal state never enters the observation. */
export class NavigationTaskEnvironment {
  readonly config: NavigationTaskConfig;
  private distanceField?: GeodesicDistanceField;
  private goal?: Pose2;
  private previousDistance = Number.POSITIVE_INFINITY;
  private previousAction?: readonly number[];
  private requiresReset = true;

  constructor(readonly environment: KiwiVisualEnvironment, config: Partial<NavigationTaskConfig> = {}) {
    this.config = {
      ...DEFAULT_NAVIGATION_TASK_CONFIG,
      ...config,
      reward: { ...DEFAULT_REWARD_WEIGHTS, ...config.reward },
    };
  }

  reset(seed = 0): NavigationResetResult {
    if (!Number.isInteger(seed)) throw new Error("Task seed must be an integer");
    const world = this.environment.control.engine.world;
    const pair = pairForSeed(world.id, seed);
    const grid = new NavigationGrid(world, {
      robotRadiusM: this.environment.control.engine.config.robotConfig.radius,
    });
    const field = grid.createDistanceField(pair.goal);
    if (!grid.isFree(pair.start) || !grid.isFree(pair.goal)) {
      throw new Error(`Authored navigation pair ${pair.id} has an occupied endpoint`);
    }
    const distance = field.distance(pair.start);
    if (!Number.isFinite(distance)) {
      throw new Error(`Authored navigation pair ${pair.id} is not reachable`);
    }
    this.distanceField = field;
    this.goal = { ...pair.goal };
    this.previousDistance = distance;
    this.previousAction = undefined;
    this.requiresReset = false;
    const reset = this.environment.reset({ seed, pose: pair.start, goalPose: pair.goal });
    const info: NavigationResetInfo = {
      task_id: this.config.id,
      task_pair_id: pair.id,
      initial_geodesic_distance_m: distance,
      goal_rgb_is_policy_input: this.config.id === "image_goal_navigation_v1",
    };
    if (this.config.privilegedDebug) info.privileged = { start: { ...pair.start }, goal: { ...pair.goal } };
    return { ...reset, info };
  }

  step(action: EnvironmentAction): NavigationStepResult {
    if (this.requiresReset || !this.distanceField || !this.goal) {
      throw new Error("Navigation task must be reset before step");
    }
    const result = this.environment.step(action);
    const pose = result.current.robot.pose;
    const distance = this.distanceField.distance(pose);
    const reachedPosition = Math.hypot(pose.x - this.goal.x, pose.y - this.goal.y) <= this.config.successRadiusM;
    const reachedHeading = Math.abs(wrapAngle(pose.yaw - this.goal.yaw)) <= this.config.successYawToleranceRad;
    const success = reachedPosition && (!this.config.requireGoalHeading || reachedHeading);
    const currentAction = actionVector(action);
    const effort = controllerEffort(result.controllerCommands) / this.environment.control.config.controllerHz;
    const terms: RewardTerms = {
      progress: this.config.reward.progress * (this.previousDistance - distance),
      success: success ? this.config.reward.success : 0,
      collision: result.collisionTickCount > 0 ? -this.config.reward.collision : 0,
      time: -this.config.reward.time,
      controller: -this.config.reward.controller * effort,
      smoothness: -this.config.reward.smoothness * squaredDifference(currentAction, this.previousAction),
    };
    const reward = Object.values(terms).reduce((total, term) => total + term, 0);
    this.previousDistance = distance;
    this.previousAction = [...currentAction];
    const terminated = success;
    if (terminated || result.truncated) this.requiresReset = true;
    const info: NavigationStepInfo = {
      reward_terms: terms,
      geodesic_distance_m: distance,
      success,
      collision_tick_count: result.collisionTickCount,
    };
    if (success) info.termination_reason = "goal_reached";
    if (this.config.privilegedDebug) info.privileged = { pose: { ...pose }, goal: { ...this.goal } };
    return { ...result, reward, terminated, info };
  }
}

export function validateAuthoredNavigationPairs(): string[] {
  const failures: string[] = [];
  for (const world of WORLD_LIST) {
    for (const pair of NAVIGATION_PAIRS[world.id] ?? []) {
      const grid = new NavigationGrid(world);
      const distance = grid.createDistanceField(pair.goal).distance(pair.start);
      if (!grid.isFree(pair.start) || !grid.isFree(pair.goal) || !Number.isFinite(distance)) {
        failures.push(`${world.id}/${pair.id}`);
      }
    }
  }
  return failures;
}
