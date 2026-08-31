import { SimulatorScene } from "../render/scene";
import { KiwiSimEngine } from "../sim/engine";
import { IDEAL_SENSOR_PROFILE, RETAINED_ROBOT_PROFILE } from "../sim/hardware-profile";
import {
  KiwiRlEnvironment,
  type ActionMode,
  type EnvironmentAction,
  type RlEnvironmentEvent,
} from "../sim/rl-environment";
import type { PoseControllerConfig } from "../control/pose-controller";
import {
  CONTROLLER_REVISION,
  ENGINE_VERSION,
  PHYSICS_REVISION,
  PROTOCOL_VERSION,
  RENDERER_REVISION,
  worldRevision,
} from "../sim/revisions";
import { worldById, WORLDS } from "../sim/worlds";
import {
  NavigationTaskEnvironment,
  type NavigationTaskId,
  type RewardWeights,
} from "../tasks/navigation-v1";
import type { VisionObservation } from "../vision/temporal-context";
import { KiwiVisualEnvironment } from "../vision/visual-environment";
import { cameraCalibration } from "../vision/camera";
import { decodeWireMessage, encodeWireMessage, type WireArray } from "./protocol";

interface CreateRequest {
  world_id?: string;
  action_mode?: ActionMode;
  policy_hz?: number;
  controller_hz?: number;
  trajectory_lookahead_index?: number;
  max_episode_steps?: number;
  max_relative_translation_m?: number;
  max_relative_yaw_rad?: number;
  max_trajectory_waypoints?: number;
  controller?: Partial<PoseControllerConfig>;
  sensor_profile?: "ideal" | "retained-robot-maps-v1";
  context_length?: number;
  context_stride?: number;
  vision_width?: number;
  vision_height?: number;
  vertical_fov_deg?: number;
  observation_schema?: "vision_goal_v1" | "vision_v1";
  task?: NavigationTaskId | null;
  privileged_debug?: boolean;
  success_radius_m?: number;
  require_goal_heading?: boolean;
  success_yaw_tolerance_rad?: number;
  reward?: Partial<RewardWeights>;
}

interface RunnerRecord {
  scene: SimulatorScene;
  visual: KiwiVisualEnvironment;
  task?: NavigationTaskEnvironment;
  config: Required<Omit<CreateRequest, "task">> & { task: NavigationTaskId | null };
}

interface RequestDocument {
  payload?: unknown;
  env_id?: number;
}

const parameters = new URLSearchParams(window.location.search);
const socketUrl = parameters.get("ws");
if (!socketUrl) throw new Error("Visual runner requires a ws query parameter");
const rendererStack = parameters.get("renderer_stack") ?? "chromium-webgl";
const socket = new WebSocket(socketUrl);
socket.binaryType = "arraybuffer";
const environments = new Map<number, RunnerRecord>();
let nextEnvironmentId = 1;

function resolvedConfig(input: CreateRequest): RunnerRecord["config"] {
  const actionMode = input.action_mode ?? "relative_pose_v1";
  const policyHz = input.policy_hz ?? (actionMode === "twist_aligned_v1" ? 20 : 4);
  const observationSchema = input.observation_schema ?? "vision_goal_v1";
  const task = input.task === undefined ? "image_goal_navigation_v1" : input.task;
  if (observationSchema === "vision_goal_v1" && task !== "image_goal_navigation_v1") {
    throw new Error("vision_goal_v1 requires image_goal_navigation_v1");
  }
  if (observationSchema === "vision_v1" && task !== null) {
    throw new Error("vision_v1 is the task-free visual observation in protocol v1");
  }
  return {
    world_id: input.world_id ?? "home",
    action_mode: actionMode,
    policy_hz: policyHz,
    controller_hz: input.controller_hz ?? 20,
    trajectory_lookahead_index: input.trajectory_lookahead_index ?? 0,
    max_episode_steps: input.max_episode_steps ?? 400,
    max_relative_translation_m: input.max_relative_translation_m ?? 2,
    max_relative_yaw_rad: input.max_relative_yaw_rad ?? Math.PI,
    max_trajectory_waypoints: input.max_trajectory_waypoints ?? 32,
    controller: input.controller ?? {},
    sensor_profile: input.sensor_profile ?? "ideal",
    context_length: input.context_length ?? 6,
    context_stride: input.context_stride ?? 1,
    vision_width: input.vision_width ?? 320,
    vision_height: input.vision_height ?? 240,
    vertical_fov_deg: input.vertical_fov_deg ?? 72,
    observation_schema: observationSchema,
    task,
    privileged_debug: input.privileged_debug ?? false,
    success_radius_m: input.success_radius_m ?? 0.25,
    require_goal_heading: input.require_goal_heading ?? false,
    success_yaw_tolerance_rad: input.success_yaw_tolerance_rad ?? 0.25,
    reward: input.reward ?? {},
  };
}

function create(input: CreateRequest): { env_id: number; config: RunnerRecord["config"] } {
  const config = resolvedConfig(input);
  if (!WORLDS[config.world_id]) throw new Error(`Unknown world ${config.world_id}`);
  const world = worldById(config.world_id);
  const profile = config.sensor_profile === "ideal" ? IDEAL_SENSOR_PROFILE : RETAINED_ROBOT_PROFILE;
  const engine = new KiwiSimEngine(world, { hardwareProfile: profile });
  const control = new KiwiRlEnvironment(engine, {
    actionMode: config.action_mode,
    policyHz: config.policy_hz,
    controllerHz: config.controller_hz,
    trajectoryLookaheadIndex: config.trajectory_lookahead_index,
    maxEpisodeSteps: config.max_episode_steps,
    maxRelativeTranslationM: config.max_relative_translation_m,
    maxRelativeYawRad: config.max_relative_yaw_rad,
    maxTrajectoryWaypoints: config.max_trajectory_waypoints,
    controller: config.controller,
  });
  const canvas = document.createElement("canvas");
  canvas.dataset.kiwiEnvironment = String(nextEnvironmentId);
  document.body.append(canvas);
  const scene = new SimulatorScene(
    canvas,
    cameraCalibration(config.vision_width, config.vision_height, config.vertical_fov_deg),
  );
  const visual = new KiwiVisualEnvironment(control, scene, {
    contextLength: config.context_length,
    contextStride: config.context_stride,
  });
  const task = config.task
    ? new NavigationTaskEnvironment(visual, {
        id: config.task,
        privilegedDebug: config.privileged_debug,
        successRadiusM: config.success_radius_m,
        requireGoalHeading: config.require_goal_heading,
        successYawToleranceRad: config.success_yaw_tolerance_rad,
        reward: config.reward as RewardWeights,
      })
    : undefined;
  const envId = nextEnvironmentId;
  nextEnvironmentId += 1;
  environments.set(envId, { scene, visual, task, config });
  return { env_id: envId, config };
}

function observationResult(observation: VisionObservation, prefix = "observation"): {
  metadata: Record<string, unknown>;
  arrays: WireArray[];
} {
  const arrays: WireArray[] = [
    { name: `${prefix}.rgb`, dtype: "uint8", shape: observation.rgbShape, data: observation.rgb },
    { name: `${prefix}.rgb_valid`, dtype: "uint8", shape: [observation.rgbValid.length], data: observation.rgbValid },
    { name: `${prefix}.rgb_time_s`, dtype: "float64", shape: [observation.rgbTimeS.length], data: observation.rgbTimeS },
    { name: `${prefix}.rgb_sequence`, dtype: "uint32", shape: [observation.rgbSequence.length], data: observation.rgbSequence },
  ];
  if (observation.goalRgb && observation.goalRgbShape) {
    arrays.push({ name: `${prefix}.goal_rgb`, dtype: "uint8", shape: observation.goalRgbShape, data: observation.goalRgb });
  }
  return {
    metadata: {
      schema: observation.schema,
      rgb: `${prefix}.rgb`,
      rgb_valid: `${prefix}.rgb_valid`,
      rgb_time_s: `${prefix}.rgb_time_s`,
      rgb_sequence: `${prefix}.rgb_sequence`,
      goal_rgb: observation.goalRgb ? `${prefix}.goal_rgb` : null,
      goal_rgb_valid: observation.goalRgbValid,
      goal_rgb_sequence: observation.goalRgbSequence ?? null,
      calibration: observation.calibration,
    },
    arrays,
  };
}

function summarizeEvent(event: RlEnvironmentEvent): Record<string, unknown> {
  const summary: Record<string, unknown> = { type: event.type, simulation_time: event.simulationTime };
  if ("scheduledTime" in event) summary.scheduled_time = event.scheduledTime;
  if ("target" in event) summary.target = event.target;
  return summary;
}

function record(envId: number | undefined): RunnerRecord {
  if (envId === undefined) throw new Error("env_id is required");
  const value = environments.get(envId);
  if (!value) throw new Error(`Unknown environment ${envId}`);
  return value;
}

function reset(
  envId: number | undefined,
  payload: unknown,
  prefix = "observation",
): { result: unknown; arrays: WireArray[] } {
  const active = record(envId);
  const request = (payload ?? {}) as { seed?: number };
  const seed = request.seed ?? 0;
  const resetResult = active.task
    ? active.task.reset(seed)
    : { ...active.visual.reset({ seed }), info: { task_id: null } };
  const observation = observationResult(resetResult.observation, prefix);
  const world = active.visual.control.engine.world;
  return {
    result: {
      observation: observation.metadata,
      info: {
        ...resetResult.info,
        provenance: {
          protocol_version: PROTOCOL_VERSION,
          engine_version: ENGINE_VERSION,
          physics_revision: PHYSICS_REVISION,
          world_id: world.id,
          world_revision: worldRevision(world),
          observation_schema: active.config.observation_schema,
          action_mode: active.config.action_mode,
          policy_hz: active.config.policy_hz,
          controller_revision: CONTROLLER_REVISION,
          controller_config: active.visual.control.relativeController.controller.config,
          trajectory_lookahead_index: active.config.trajectory_lookahead_index,
          action_bounds: {
            max_relative_translation_m: active.config.max_relative_translation_m,
            max_relative_yaw_rad: active.config.max_relative_yaw_rad,
            max_trajectory_waypoints: active.config.max_trajectory_waypoints,
            max_linear_speed_mps: active.visual.control.engine.config.robotConfig.maxLinearSpeed,
            max_angular_speed_radps: active.visual.control.engine.config.robotConfig.maxAngularSpeed,
          },
          sensor_profile: active.config.sensor_profile,
          renderer_backend: rendererStack,
          renderer_revision: RENDERER_REVISION,
          camera_profile: active.scene.calibration.profile,
          seed,
          resolved_randomization: {},
        },
      },
    },
    arrays: observation.arrays,
  };
}

function step(
  envId: number | undefined,
  payload: unknown,
  prefix = "observation",
): { result: unknown; arrays: WireArray[] } {
  const active = record(envId);
  const request = payload as { action?: EnvironmentAction };
  if (!request?.action) throw new Error("step requires an action");
  const result = active.task
    ? active.task.step(request.action)
    : { ...active.visual.step(request.action), reward: 0, info: { reward_terms: {} } };
  const observation = observationResult(result.observation, prefix);
  return {
    result: {
      observation: observation.metadata,
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      info: {
        ...result.info,
        simulation_time_s: result.current.simulationTime,
        episode_step: active.visual.control.episodeSteps,
        events: result.events.map(summarizeEvent),
        controller_commands: result.controllerCommands,
      },
    },
    arrays: observation.arrays,
  };
}

function close(envId: number | undefined): { closed: number } {
  const active = record(envId);
  active.scene.dispose();
  active.scene.renderer.domElement.remove();
  environments.delete(envId as number);
  return { closed: envId as number };
}

function createMany(payload: unknown): { environments: ReturnType<typeof create>[] } {
  const request = payload as { configs?: CreateRequest[] };
  if (!Array.isArray(request?.configs)) throw new Error("create_many requires configs");
  return { environments: request.configs.map(create) };
}

function resetMany(payload: unknown): { result: unknown; arrays: WireArray[] } {
  const request = payload as { items?: { env_id?: number; seed?: number }[] };
  if (!Array.isArray(request?.items)) throw new Error("reset_many requires items");
  const arrays: WireArray[] = [];
  const items = request.items.map((item, index) => {
    const resetResult = reset(item.env_id, { seed: item.seed ?? 0 }, `items.${index}.observation`);
    arrays.push(...resetResult.arrays);
    return resetResult.result;
  });
  return { result: { items }, arrays };
}

function stepMany(payload: unknown): { result: unknown; arrays: WireArray[] } {
  const request = payload as { items?: { env_id?: number; action?: EnvironmentAction }[] };
  if (!Array.isArray(request?.items)) throw new Error("step_many requires items");
  const arrays: WireArray[] = [];
  const items = request.items.map((item, index) => {
    const stepResult = step(
      item.env_id,
      { action: item.action },
      `items.${index}.observation`,
    );
    arrays.push(...stepResult.arrays);
    return stepResult.result;
  });
  return { result: { items }, arrays };
}

function closeMany(payload: unknown): { closed: number[] } {
  const request = payload as { env_ids?: number[] };
  if (!Array.isArray(request?.env_ids)) throw new Error("close_many requires env_ids");
  return { closed: request.env_ids.map((envId) => close(envId).closed) };
}

async function handle(message: MessageEvent<ArrayBuffer>): Promise<void> {
  const decoded = decodeWireMessage(message.data);
  const { request_id: requestId, operation } = decoded.header;
  const request = (decoded.header.result ?? {}) as RequestDocument;
  try {
    let result: unknown;
    let arrays: WireArray[] = [];
    if (operation === "hello") {
      result = {
        protocol_version: PROTOCOL_VERSION,
        engine_version: ENGINE_VERSION,
        physics_revision: PHYSICS_REVISION,
        controller_revision: CONTROLLER_REVISION,
        renderer_revision: RENDERER_REVISION,
        renderer_backend: rendererStack,
        capabilities: {
          operations: [
            "hello", "create", "create_many", "reset", "reset_many",
            "step", "step_many", "close", "close_many",
          ],
          observation_schemas: ["vision_goal_v1", "vision_v1"],
          action_modes: ["relative_pose_v1", "relative_trajectory_v1", "twist_aligned_v1"],
          binary_arrays: true,
        },
      };
    } else if (operation === "create") {
      result = create((request.payload ?? {}) as CreateRequest);
    } else if (operation === "create_many") {
      result = createMany(request.payload);
    } else if (operation === "reset") {
      ({ result, arrays } = reset(request.env_id, request.payload));
    } else if (operation === "reset_many") {
      ({ result, arrays } = resetMany(request.payload));
    } else if (operation === "step") {
      ({ result, arrays } = step(request.env_id, request.payload));
    } else if (operation === "step_many") {
      ({ result, arrays } = stepMany(request.payload));
    } else if (operation === "close") {
      result = close(request.env_id);
    } else if (operation === "close_many") {
      result = closeMany(request.payload);
    } else {
      throw new Error(`Unknown operation ${String(operation)}`);
    }
    socket.send(encodeWireMessage({ request_id: requestId, ok: true, result }, arrays));
  } catch (error) {
    socket.send(
      encodeWireMessage({
        request_id: requestId,
        ok: false,
        error: {
          code: "request_failed",
          message: error instanceof Error ? error.message : String(error),
        },
      }),
    );
  }
}

socket.addEventListener("message", (message) => void handle(message as MessageEvent<ArrayBuffer>));
