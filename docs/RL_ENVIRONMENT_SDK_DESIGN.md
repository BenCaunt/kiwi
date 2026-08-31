# RL Environment and SDK Design

Status: the vision-first training slice across phases 0-4 is implemented;
existing simulator interfaces remain intact.

The delivered slice includes the deterministic engine, hierarchical actions,
scheduled visual observations and image goals, binary Chromium supervisor,
Gym-style Python and batch APIs, tunable collision-aware rewards, content
provenance, and trace replay. Scale optimizations and measured domain-
randomization profiles in phase 5 remain future fidelity work.

## Outcome

Give a vision-first RL training project a deterministic, Python-friendly
`reset`/`step` interface while keeping all current Kiwi interfaces intact:

- the browser simulator and its keyboard controls;
- the WebSocket bridge and its current messages;
- the `kiwi/sim` Zenoh namespace, topic suffixes, rates, and payloads;
- raw drivetrain and sensor-aligned frame transforms;
- the standalone Python simulator and its CLI.

The browser simulator's TypeScript simulation becomes the authoritative RL
backend. The current pure-Python simulator remains available for compatibility,
but is not the initial training backend because its worlds and dynamics already
differ from the richer browser simulator.

The first useful deliverable is goal-conditioned visual navigation with
temporal RGB context and robot-relative waypoint or pose-delta actions. A
robot-specific controller converts those medium-level actions to the existing
aligned velocity interface. Direct velocity actions and LiDAR/proprioception
remain available as explicit lower-level inputs rather than becoming the only
training surface.

## Current system and the missing boundary

The repository already has most of the hard pieces:

- `simulator-web/src/sim` is independent of Three.js and contains deterministic
  motion, collisions, frame transforms, LiDAR, pose history, hardware sensor
  modeling, and firmware payload generation.
- `FixedStepClock` runs physics at 120 Hz.
- `KiwiRobot` accepts raw or aligned velocity commands, models command timeout
  and drive response, and reports pose, velocity, command activity, and contact.
- `FirmwareContract` produces existing odometry, LD19, camera, and status wire
  payloads.
- `main.ts` currently assembles physics, sensor schedules, bridge publishing,
  keyboard priority, UI state, and rendering.
- `kiwi_sim_bridge.py` keeps simulator ground truth off Zenoh and exposes it only
  to a privileged loopback harness.

The missing piece is a single environment object that owns episode time and can
be advanced without wall-clock time or a browser animation frame.

## Architecture decision

Extract orchestration from `main.ts` into a renderer-independent
`KiwiSimEngine`, make visual rendering a first-class service beside it, and make
every external interface an adapter around those two components.

```text
                         +----------------------+
 Browser keyboard/UI --->|                      |---> state/sensor snapshots
 Zenoh/WebSocket cmd  --->|    KiwiSimEngine     |---> contact/events
 RL waypoint/twist    --->|  deterministic time  |---> firmware events
                         +----------+-----------+
                                    |
                         +----------+-----------+
                         | KiwiVisionRenderer   |---> current/context RGB
                         | camera + appearance  |---> goal RGB + metadata
                         +----------+-----------+
                                    |
             +----------------------+-----------------------+
             |                      |                       |
       Browser UI adapter     Zenoh adapter       Headless training host
       (existing view)        (existing wire)     (versioned local RPC)
                                                             |
                                                      Python Kiwi SDK
                                                             |
                                               optional Gymnasium adapter
```

This avoids three bad outcomes: driving RL through real-time Zenoh, duplicating
the advanced TypeScript simulation in Python, or making physics ownership
depend on Three.js. Vision still uses the same Three.js world as the interactive
simulator; it is attached through a narrow renderer contract so non-visual
workers and physics tests do not need a graphics context.

### Source-of-truth rules

1. TypeScript physics, collision geometry, sensor geometry, and advanced worlds
   are authoritative for browser and headless RL execution.
2. Existing Zenoh payload generators remain the authoritative compatibility
   path for the physical robot contract.
3. The same camera scene, camera calibration, and image orientation contract are
   used for interactive, headless-training, and firmware-compatible rendering.
4. Reward and episode termination are task concerns and do not enter robot
   physics.
5. The Python SDK owns the familiar training API, task rewards, wrappers, and
   vector-environment client, but does not reimplement physics.
6. Robot-relative pose/waypoint actions are the primary learned-control surface;
   the existing aligned body twist remains the controller output and an
   available low-level action mode.
7. The existing Python simulator is not removed or silently redirected in the
   first implementation. Its current CLI remains stable.

## Core engine contract

Suggested TypeScript surface:

```ts
interface EngineConfig {
  worldId: string;
  physicsHz: 120;
  sensorProfile: "ideal" | "retained-robot-maps-v1";
  lidarRays: number;             // default 180
}

interface VisionConfig {
  width: number;                 // native default 320
  height: number;                // native default 240
  verticalFovDeg: number;        // current Three.js render value is 72
  cameraHz: number;              // current hardware profile about 9.69
  contextLength: number;         // default 6: current plus five history frames
  contextStride: number;         // camera frames between context entries
  encoding: "rgb8" | "jpeg";
  includeGoalImage: boolean;
}

type ActionMode =
  | "relative_pose_v1"
  | "relative_trajectory_v1"
  | "twist_aligned_v1";

interface EnvironmentConfig {
  engine: EngineConfig;
  actionMode: ActionMode;
  policyHz: number;              // default 4 for waypoints, 20 for twist
  controllerHz: number;          // default 20; must divide physicsHz in v1
  observationSchema: "vision_goal_v1" | "vision_v1" | "sensors_v1" | "state_v1";
  vision?: VisionConfig;
  controller?: RelativePoseControllerConfig;
  maxEpisodeSteps: number;
}

interface ResetOptions {
  seed: number;
  worldId?: string;
  spawn?: Pose2;
  task?: TaskResetOptions;
  randomization?: DomainRandomizationOptions;
}

interface AlignedTwistAction {
  kind: "twist";
  vx: number;                    // m/s, sensor-aligned +X forward
  vy: number;                    // m/s, sensor-aligned +Y left
  omega: number;                 // rad/s, counter-clockwise positive
}

interface RelativePoseAction {
  kind: "relative_pose";
  dx: number;                    // m, +X camera-forward at action acceptance
  dy: number;                    // m, +Y left at action acceptance
  dyaw: number;                  // rad, counter-clockwise from acceptance yaw
}

interface RelativeTrajectoryAction {
  kind: "relative_trajectory";
  // All cumulative poses share the robot frame captured at action acceptance.
  waypoints: ReadonlyArray<Omit<RelativePoseAction, "kind">>;
}

type EnvironmentAction =
  | AlignedTwistAction
  | RelativePoseAction
  | RelativeTrajectoryAction;

interface EnvironmentStepResult {
  observation: Observation;
  transition: PrivilegedTransition;
  events: SimEvent[];
  terminated: boolean;
  truncated: boolean;
}

interface KiwiSimEngine {
  reset(options: ResetOptions): EngineResetResult;
  setAlignedTwist(action: AlignedTwistAction): void;
  advanceTicks(count: number): EngineAdvanceResult;
  snapshot(): Readonly<EngineSnapshot>;
  close(): void;
}

interface KiwiRlEnvironment {
  reset(options: ResetOptions): EnvironmentResetResult;
  step(action: EnvironmentAction): Promise<EnvironmentStepResult>;
  close(): Promise<void>;
}
```

`KiwiRlEnvironment.step(action)` is not a real-time operation. In waypoint mode
it anchors the relative target, runs the lower controller and physics for one
fixed policy interval, accumulates events, renders due camera observations, and
returns. With 4 Hz policy, 20 Hz controller, and 120 Hz physics defaults, that
is five controller updates and thirty physics ticks per policy action. The next
policy action replaces the previous target. It is not required to reach the
target before `step` returns.

In direct-twist mode, the default 20 Hz policy interval advances six physics
ticks and preserves the current command/watchdog behavior. There is no sleep,
animation callback, Zenoh hop, or wall-clock dependency in either mode.

The engine must also expose an integer-tick primitive for adapters. Browser
real-time accumulation can continue to use frame deltas, but RL stepping must
not depend on floating frame accumulation.

### Hierarchical action policy

All canonical actions use SI units in the existing sensor-aligned laptop frame:
+X is camera-forward, +Y is left, and positive yaw is counter-clockwise. This
matches `KiwiClient` and the frame already used by vision and LiDAR.

`relative_pose_v1` is the recommended first learned-control interface. A pose
delta is interpreted in the body frame captured when the action is accepted:

```text
target_world = action_start_world_pose compose (dx, dy, dyaw)
```

The target remains fixed in the world while a lower-level controller closes the
error. It must not be reinterpreted in the robot's newly rotated body frame on
every controller tick. A new policy action intentionally replaces the old
target, giving a fixed-rate receding-horizon interface.

`relative_trajectory_v1` accepts a short sequence of cumulative relative poses,
all expressed from that same action-start frame. A configurable, recorded
lookahead rule selects or interpolates the controller target. This supports any
visual-navigation policy that predicts a short waypoint horizon and leaves
velocity generation to a robot-specific controller.

`twist_aligned_v1` exposes the existing `[vx, vy, omega]` velocity control for
low-level policies, controller testing, teleoperation, and ablations.

The initial relative-pose controller should match the existing
`PoseStabilizingController`: independent planar/yaw feedback, rotation of map
error into the aligned body frame, linear/angular speed limits, and position/yaw
tolerances. Its implementation in the headless TypeScript host must pass shared
conformance fixtures against `scripts/kiwi_pose_controller.py`. Controller
configuration and the pose source (`ground_truth`, integrated noisy odometry,
or SLAM) are episode metadata. Ground truth is a debugging default only;
deployable evaluations use an estimator-backed source.

The Python SDK may also provide an `execute_until_reached` macro wrapper for
scripted evaluation. It is not the default RL step because variable-duration
actions make discounting and temporal credit assignment less clear.

The SDK may offer generic normalized waypoint or twist wrappers using declared
robot-specific bounds. Normalization is not part of the core engine contract:
the conversion, waypoint horizon, lookahead rule, and configured limits must be
visible in episode metadata. Out-of-range actions are rejected by default;
clipping is an explicit wrapper and is never silent. The Zenoh adapter continues
to accept raw-frame velocity commands and use the existing transform.

For NumPy/Gymnasium callers, the configured action mode fixes one numeric shape:

| Mode | Canonical SDK shape | Meaning |
|---|---:|---|
| `relative_pose_v1` | `float32[3]` | `[dx_m, dy_m, dyaw_rad]` |
| `relative_trajectory_v1` | `float32[horizon, 3]` | cumulative `[dx_m, dy_m, dyaw_rad]` from one action-start frame |
| `twist_aligned_v1` | `float32[3]` | `[vx_mps, vy_mps, omega_radps]` |

Model code is responsible for converting its own output tensor, normalization,
or angular representation into one of these canonical SI-unit actions. The SDK
does not carry model preprocessing or model-specific action adapters.

### Events, not final-tick flags

`RobotState.collided` describes the current physics tick. An RL action spans
several ticks, so the engine must accumulate events such as:

- `contact_started` and `contact_tick_count`;
- `command_timed_out`;
- `relative_target_replaced`, `relative_target_reached`, and
  `relative_target_timed_out`;
- `action_rejected` or `action_clipped`;
- `goal_reached`;
- `invalid_spawn` or `out_of_bounds` if those checks are added;
- sensor availability/fault events when fault injection exists.

Rewards should not infer an entire action's collision history from the final
tick's boolean.

## Python SDK contract

Place a normal Python package under `sdk/python`, for example:

```text
sdk/python/
  pyproject.toml
  src/kiwi_sim_sdk/
    __init__.py
    config.py
    env.py
    vector_env.py
    rewards.py
    tasks.py
    controllers.py
    vision.py
    preprocessing.py
    transport.py
    types.py
  examples/
    random_agent.py
    gymnasium_agent.py
```

The base dependency set should be small: Python, NumPy, and the bundled
headless simulator artifact. Gymnasium support should be an optional extra so
the core SDK is usable by custom trainers.

```py
from kiwi_sim_sdk import EnvConfig, KiwiEnv, VisionConfig

env = KiwiEnv(EnvConfig(
    world_id="home",
    observation_schema="vision_goal_v1",
    action_mode="relative_trajectory_v1",
    policy_hz=4,
    controller_hz=20,
    vision=VisionConfig(
        width=320,
        height=240,
        context_length=6,
        include_goal_image=True,
    ),
    sensor_profile="ideal",
    task="image_goal_navigation_v1",
))

observation, info = env.reset(seed=42)
while True:
    relative_waypoints = policy(observation)
    observation, reward, terminated, truncated, info = env.step(
        relative_waypoints
    )
    if terminated or truncated:
        break
env.close()
```

The return convention intentionally matches Gymnasium:

```text
reset(seed, options) -> observation, info
step(action)         -> observation, reward, terminated, truncated, info
```

`terminated` means the task reached a terminal state such as success.
`truncated` means an external limit such as maximum episode steps was reached.

### Headless transport

The initial local transport should use versioned JSON control envelopes plus a
length-framed binary payload path for arrays. RGB observations are too large to
base64 into newline-delimited JSON: one uncompressed 320 x 240 RGB frame is
230,400 bytes before temporal context or batched environments. The transport may
begin with local WebSocket binary frames or MessagePack and later add shared
memory, but image arrays must remain zero-copy-capable at the public boundary.

Required operations:

- `hello`: negotiate `protocol_version` and return engine/build metadata;
- `create`: construct one or more independent engines;
- `reset` / `reset_many`;
- `step` / `step_many`;
- `close`.

The Python public API must hide this transport. After profiling, arrays can move
to shared memory without changing user code. Do not make Zenoh the training
inner loop; it is asynchronous, real-time, JPEG-oriented, and valuable
specifically as the integration/hardware-compatibility interface.

### Version and provenance fields

Every reset must return enough information to reproduce the episode:

```json
{
  "protocol_version": 1,
  "engine_version": "0.1.0",
  "physics_revision": "...",
  "world_id": "home",
  "world_revision": "...",
  "observation_schema": "vision_goal_v1",
  "action_mode": "relative_trajectory_v1",
  "policy_hz": 4,
  "controller_revision": "...",
  "sensor_profile": "ideal",
  "renderer_backend": "chromium-webgl",
  "renderer_revision": "...",
  "camera_profile": "kiwi-front-render-v1",
  "seed": 42,
  "resolved_randomization": {}
}
```

World, physics, controller, renderer, and camera revisions should be content
hashes, not only package versions. A model trained against changed geometry,
control, or pixels must be identifiable.

## Observation design

Ship named, versioned observation presets rather than an open-ended dictionary
whose meaning can drift.

### `vision_goal_v1`

This is the primary goal-conditioned visual-navigation observation:

- `rgb`: `uint8[context, height, width, 3]`, chronological and upright;
- `rgb_valid`: `uint8[context]`, distinguishing real history from reset fill;
- `rgb_time_s`: `float64[context]` in simulation time;
- `goal_rgb`: `uint8[height, width, 3]`, upright and fixed for the active goal;
- `goal_rgb_valid`: scalar mask for tasks that can hide or drop goal context.

At reset, the first rendered image fills the fixed-size temporal tensor and only
the newest validity entry is true. A generic `repeat_first` history option may
mark the repeated frames valid when a policy requires a fully populated tensor;
the choice is named and recorded.

The base SDK returns the camera's native 320 x 240 RGB image without model-
specific resizing. Model code owns center cropping, resizing, tensor layout,
floating-point scaling, and normalization. This preserves one camera contract
while supporting different vision models.

The goal image is rendered from the task's goal pose or loaded from an authored
topological node. It uses the same camera calibration and scene renderer as the
current observation. Appearance matching is explicit: the default uses the same
episode appearance, while a seeded `goal_appearance_shift` profile can vary
lighting or texture to model a previously captured goal image.

### `vision_v1`

This is the exploration/non-image-goal variant. It contains `rgb`, `rgb_valid`,
and `rgb_time_s` with the same meanings and no goal image. Task commands or
language/route conditioning can be composed by a separately versioned wrapper;
they should not silently change this schema.

### `state_v1`

Fast privileged state for algorithm smoke tests and debugging:

- true pose as `[x, y, sin(yaw), cos(yaw)]`;
- true aligned velocity `[vx, vy, omega]`;
- goal in robot coordinates `[goal_x, goal_y]` and goal distance;
- previous applied action;
- contact indicator.

This mode is not a sim-to-real policy input and must be labeled privileged.

### `sensors_v1`

The non-visual sensor baseline and optional ablation observation:

- `lidar_m`: `float32[lidar_rays]` in meters;
- `lidar_hit`: `uint8[lidar_rays]`, keeping no-return distinct from a maximum
  range hit;
- measured aligned twist: `float32[3]`;
- IMU orientation represented as sine/cosine or quaternion components;
- previous applied action: `float32[3]`;
- task goal vector in the robot frame and remaining distance.

Arrays use fixed shapes and documented units. Normalization belongs in a
versioned wrapper. True pose, true velocity, and randomization parameters are
available to reward/evaluation code through an internal privileged transition,
not in the policy observation.

Whether a metric goal vector comes from ground truth or an estimator is a task
configuration field. Ground-truth goal vectors are useful for local-controller
training but must not be mistaken for a sensor available on the robot.

### Camera and renderer contract

Vision is required infrastructure, not a later optional renderer project. The
current simulator already defines a 320 x 240 Three.js camera at 0.22 m and
produces the physical camera's inverted JPEG payload. Refactor
that implementation behind a `KiwiVisionRenderer` with two outputs from the
same rendered buffer:

- canonical upright `rgb8` or JPEG for policy observations;
- the existing inverted JPEG plus `KVC1` header for Zenoh compatibility.

Every observation supplies or references camera metadata: width, height,
vertical and horizontal FOV, derived intrinsics, robot-to-camera extrinsics,
exposure/color profile, encoding, orientation, capture simulation time,
sequence, and renderer revision. Camera timing follows a deterministic
simulation-time schedule rather than rendering whenever Python happens to call
`step`.

There is a current calibration convention mismatch to resolve explicitly:
Three.js treats the configured `72` as vertical FOV, which at 4:3 implies about
88.18 degrees horizontal, while `kiwi_image_map.py` currently records 72 degrees
as horizontal. Phase 0 preserves today's rendered pixels and legacy metadata,
adds a calibration fixture, and makes the new `kiwi-front-render-v1` SDK profile
report the actual projection matrix/intrinsics. Any change to match a measured
physical horizontal FOV is a new camera profile and renderer revision, never a
silent reinterpretation.

The first headless visual backend should be a dedicated Chromium/WebGL training
page using the same Three.js scene and sensor camera as the interactive browser.
The page owns one or more `KiwiSimEngine` instances and renders immediately
after deterministic simulation steps; it never advances from animation-frame
wall time. A small supervisor exposes the versioned local RPC to Python. The
pure Node host remains available for non-visual benchmarks, but both import the
same engine.

This Chromium choice prioritizes visual parity and a working first release.
After measuring throughput, the `KiwiVisionRenderer` boundary allows a worker
pool, OffscreenCanvas, WebGPU, or another offscreen backend without changing the
Python environment API.

GPU pixels are not promised bit-for-bit identical across different browsers,
drivers, and hardware. Repeated runs on a pinned renderer stack should match;
cross-stack visual tests use calibration scenes, channel/error tolerances, and
perceptual hashes. Physics, action targets, task state, and reward remain exactly
deterministic independently of renderer differences.

### Vision randomization and fidelity

Seeded visual randomization is part of the episode manifest and uses random
streams independent from physics and sensor noise. Planned dimensions include:

- lighting direction, intensity, color temperature, and exposure;
- material hues, texture variants, clutter selection, and limited object poses;
- camera height, forward offset, yaw/pitch/roll, FOV, and lens distortion;
- image noise, blur, motion/latency effects, occlusion, and JPEG quality;
- controlled appearance mismatch between current and goal images.

The first release should implement a small, auditable subset and keep it off by
default. Procedural homes are useful for validating the interface, but visual
domain randomization does not by itself establish photorealism or sim-to-real
transfer. Each added effect needs a declared range and, where possible, a link
to measured camera behavior.

## Task and reward design

The engine reports transitions and events. A task converts them into reward,
termination, and episode metrics.

```py
class RewardFunction(Protocol):
    def reset(self, context: EpisodeContext) -> None: ...
    def evaluate(
        self,
        previous: PrivilegedState,
        action: EnvironmentAction,
        current: PrivilegedState,
        events: tuple[SimEvent, ...],
    ) -> RewardResult: ...
```

Provide `image_goal_navigation_v1` as the primary reference task. Reset chooses
an authored reachable start/goal pair, renders or loads the goal observation,
fills temporal current-image context, and exposes no metric goal coordinates to
the policy. Hidden task state retains the true goal pose for reward and
evaluation. `point_navigation_v1` remains a fast controller/debugging task.

The visual task's configurable reward terms can be:

```text
reward = progress_weight  * (previous_geodesic_distance - current_geodesic_distance)
       + success_bonus    * reached_goal
       - collision_cost   * contact_during_action
       - time_cost
       - controller_cost  * integrated_control_effort
       - smoothness_cost  * trajectory_or_twist_change
```

Return the terms separately in `info["reward_terms"]`. This makes reward tuning
observable and prevents a single scalar from hiding a broken component.

Recommended default episode rules:

- success when position is within a configured radius and, when requested,
  heading is within tolerance;
- collision is penalized but does not terminate by default;
- time limit produces `truncated=True`;
- unrecoverable invalid setup terminates with an explicit reason;
- optional stuck detection is a task rule and is disabled until validated.

Use collision-aware geodesic progress when a navigation graph is available.
Euclidean progress alone can reward driving toward a goal through a wall. Reward
evaluation consumes privileged state internally, while `info` exposes named
terms and aggregate metrics but not hidden goal coordinates unless privileged
debugging is explicitly enabled.

Task definitions should contain authored valid spawn/goal sets for each world in
the first version. Random free-space sampling can follow after it verifies robot
clearance and reachability; naive bounding-box samples will create goals inside
walls or disconnected rooms.

The recorder should optionally export upright frames, timestamps, odometry, and
future robot-relative pose labels in a documented generic trajectory dataset.
That is an auxiliary data-generation path, not a second action or camera
convention.

## Determinism and randomization

`reset(seed=N)` followed by the same action sequence must reproduce state,
events, observations, reward terms, and termination.

The current firmware contract uses random sensor effects, so the extraction
must introduce independent seed streams derived from the episode seed:

- dynamics;
- LiDAR;
- odometry/IMU;
- task sampling;
- controller perturbations and latency;
- current-image appearance;
- goal-image appearance;
- domain randomization.

This prevents enabling a Zenoh publisher, requesting an odometry report, or
rendering an extra evaluation frame from changing later LiDAR, appearance,
task, or controller randomness. Renderer randomness must also remain outside
the physics/sensor streams.

Domain randomization is off by default. When enabled, reset resolves declared
distributions to concrete values and records them in reset metadata. Useful
later dimensions include drive response, wheel/odometry scale, axis skew, IMU
drift, LiDAR dropout/noise, robot radius, action/controller latency, sensor
latency, camera calibration, and the visual dimensions listed above.
World geometry randomization should come only after baseline task learning is
stable.

## Record and replay

An episode trace should contain:

- reset metadata and resolved configuration;
- ordered SI-unit waypoint/pose-delta or twist actions;
- anchored world targets, selected lookahead targets, and generated twists;
- reward scalar and named terms;
- termination reason;
- RGB observations or frame hashes plus camera timestamps;
- final episode metrics.

The same trace must replay headlessly. A browser evaluation mode should also be
able to load seed/config/actions and visualize the episode. This is more useful
than trying to render every training worker live.

## Backward-compatibility plan

| Existing surface | Required behavior after extraction |
|---|---|
| Browser world selector, pause/reset, follow camera | Same visible behavior and controls |
| W/S/A/D/Q/E/Space keyboard control | Same speeds and keyboard-over-bridge priority |
| `ws://127.0.0.1:8767` bridge | Same hello, command, status, binary channel, and ground-truth messages |
| Zenoh `cmd_vel` | Same JSON, text, and 24-byte binary command handling |
| Zenoh odometry/LiDAR/camera/status | Same keys, payload shapes, camera inversion, frame transforms, and nominal rates |
| Default namespace safety | Remains `kiwi/sim` |
| `scripts/kiwi_simulator.py` | Unchanged in the initial SDK work |
| `scripts/kiwi_pose_controller.py` | Remains valid and becomes the conformance reference for medium-level control |
| Existing TypeScript and Python tests | Continue to pass; add parity tests rather than replace them |

`main.ts` becomes an adapter: it translates keyboard and bridge input into
engine commands, advances the engine from animation time, renders snapshots,
and publishes engine-generated firmware events. Existing priority logic stays
in this adapter.

The new relative-pose/trajectory action is additive at the SDK layer. In the
simulator its controller calls the engine's aligned-twist input. On a physical
robot an SDK/controller adapter emits the existing `cmd_vel` command through
`KiwiClient`; no firmware or Zenoh command contract has to change. If an
optional waypoint topic is added later, it is a laptop-side convenience and
must still terminate at the existing velocity interface.

## Proposed repository shape

```text
simulator-web/src/
  sim/
    engine.ts                 # reset, integer ticks, action step, snapshots
    scheduler.ts              # deterministic sensor/event deadlines
    events.ts
    observation.ts
    random.ts                 # named independent seed streams
    ...existing sim modules
  control/
    relative-pose.ts          # anchored SE(2) target + pose feedback
    trajectory-lookahead.ts   # model-agnostic waypoint horizon selection
    controller-contract.ts
  vision/
    renderer.ts               # KiwiVisionRenderer contract
    sensor-camera.ts          # calibration, orientation, RGB/JPEG outputs
    temporal-context.ts
    appearance.ts             # seeded visual profiles
  adapters/
    browser-runtime.ts        # keyboard/UI/render timing
    firmware-runtime.ts       # current bridge publication behavior
  headless/
    protocol.ts
    node-server.ts            # non-visual multi-engine host
    visual-runner.html        # deterministic Chromium/WebGL worker page
    supervisor.ts             # binary RPC and visual worker lifecycle

simulator-web/src/tasks/
  image-goal-navigation-v1.json
  point-navigation-v1.json

sdk/python/
  pyproject.toml
  src/kiwi_sim_sdk/
  examples/

docs/
  RL_ENVIRONMENT_SDK_DESIGN.md
  RL_SDK_PROTOCOL.md           # generated/precise wire schema when implemented
```

Keep `worlds.ts` authoritative in the first slice so browser and headless Node
import the exact same definitions. A later shared JSON schema can replace it if
external world authoring becomes important; do not solve that by making Python
parse TypeScript or maintaining another hand-copied world list.

## Implementation phases

### Phase 0: freeze current behavior

- Add golden tests for a fixed world, seed, and action sequence.
- Record browser/Zenoh frame-transform and payload compatibility fixtures.
- Add upright/raw camera orientation, effective vertical/horizontal FOV,
  intrinsics, extrinsics, and image fixtures; capture the legacy 72-degree
  metadata mismatch without silently changing it.
- Add shared fixtures for relative-pose composition and controller outputs.
- Add non-visual and visual throughput benchmarks before refactoring.
- Document which existing Python simulator is legacy versus authoritative for
  the new SDK.

Exit: current behavior is measurable and failures identify the changed surface.

### Phase 1: extract `KiwiSimEngine` and hierarchical control

- Move clock ownership, sensor deadlines, pose history, robot reset, and event
  accumulation out of `main.ts`.
- Add integer `advanceTicks`, policy-rate stepping, and controller-rate stepping.
- Split random streams and make reset seed all stochastic state.
- Return copied/read-only snapshots instead of leaking mutable robot state.
- Add anchored robot-relative pose and trajectory actions, lookahead selection,
  controller events, and generated-twist telemetry.
- Prove the TypeScript controller against the existing Python controller
  fixtures.
- Rewire `main.ts` to the engine without changing UI or transport behavior.

Exit: the browser looks and behaves the same, and the engine runs in a Vitest
test without DOM, Three.js, WebSocket, or wall-clock time. Relative actions are
frame-unambiguous and direct twist behavior remains unchanged.

### Phase 2: make vision a reusable simulator service

- Split the sensor camera from orbit/follow UI concerns behind
  `KiwiVisionRenderer`.
- Produce upright RGB for policies and the existing inverted firmware JPEG from
  the same render.
- Implement simulation-time camera scheduling and fixed-shape temporal context.
- Add goal-pose rendering and authored topological goal images.
- Build the deterministic Chromium/WebGL runner using the same scene and camera.
- Add seeded visual profiles and renderer/camera revision metadata.

Exit: an automated headless run returns time-aligned current/context/goal RGB,
while the interactive simulator and Zenoh camera payload remain visually and
contractually compatible.

### Phase 3: add the headless protocol and Python SDK

- Implement protocol negotiation, create/reset/step/close, binary array frames,
  structured errors, and build/world/physics/controller/renderer revisions.
- Support isolated environments and batched operations in both non-visual Node
  and visual Chromium hosts.
- Implement context-managed process lifetime and crash diagnostics.
- Define NumPy RGB/context observations, SI-unit waypoint/pose/twist actions,
  action validation, and seed flow.
- Add optional Gymnasium spaces/wrapper and `step_many` vector support.
- Add random-waypoint, random-twist, and image-goal examples plus package
  installation instructions.

Exit: the friend can install one editable Python package and run the standard
five-value visual step loop without reading simulator internals.

### Phase 4: visual task/reward/evaluation baseline

- Add `image_goal_navigation_v1` and `point_navigation_v1` with authored,
  reachable spawn/goal pairs.
- Add composable reward terms and explicit reward breakdowns.
- Add episode metrics: success, path length, shortest-path ratio, contact action
  count, controller tracking error/effort, elapsed simulated time, and final
  geodesic distance.
- Add trace record/replay and browser replay for qualitative evaluation.
- Add optional generic image/odometry/relative-waypoint dataset export.

Exit: a random visual-waypoint policy and a scripted/privileged waypoint policy
produce sensible, tested, different metrics and reproducible traces.

### Phase 5: scale and domain randomization

- Benchmark single and batched workers on the actual training machine.
- Profile rendering, image transfer, model inference, controller, and physics
  separately; optimize or add shared memory only where measurement requires it.
- Add declared randomization distributions and a curriculum API.
- Add worker-failure isolation and long-run determinism tests.

Exit: the agreed number of parallel environments runs stably, every episode
records resolved randomization, and throughput is documented.

### Later: richer visual fidelity and physics

- Evaluate faster offscreen backends behind `KiwiVisionRenderer`.
- Expand camera latency/exposure/noise and appearance randomization only from
  measured data or an explicit synthetic profile.
- Add multi-camera, depth, segmentation, or optical-flow schemas only when a
  task needs them; do not silently add channels to `vision_goal_v1`.
- Improve contact dynamics or actuator modeling only when a task demonstrates
  that the current kinematic model is the limiting sim-to-real gap.

## Acceptance tests

The SDK is ready for an RL handoff when all of the following are true:

1. Same seed + same actions gives identical physics, controller, task, and
   reward traces in repeated headless runs.
2. A pinned visual runner produces repeatable temporal/current/goal RGB; browser
   and headless calibration images match the defined pixel/perceptual tolerance.
3. Policy RGB is upright while the existing Zenoh `KVC1` JPEG remains in the
   physical camera's raw mounted orientation.
4. Every RGB frame has the correct simulation timestamp, camera metadata, and
   source pose; reset context fill and validity masks are deterministic.
5. A relative pose `(dx, dy, dyaw)` composes from the body pose at action
   acceptance, remains world-fixed while tracked, and is replaced only by a new
   policy action or explicit cancel.
6. Every trajectory waypoint is interpreted from the same action-start frame,
   and the recorded lookahead rule selects the expected controller target.
7. TypeScript and Python pose controllers match shared reference fixtures, and
   generated twists respect configured speed/tolerance limits.
8. A default 4 Hz waypoint action advances exactly five 20 Hz controller updates
   and thirty 120 Hz physics ticks; a default 20 Hz twist action advances six
   physics ticks.
9. Direct aligned-twist traces remain compatible with pre-SDK simulator traces.
10. Contact during any substep appears in the action's event set and reward
   inputs.
11. Reset clears clock, command watchdog, controller target, camera/context
   history, counters, task state, and every random stream.
12. Multiple environments in one host do not affect one another's random draws,
   appearance, renderer schedule, controller, or time.
13. Existing browser controls and existing TypeScript/Python simulator tests pass.
14. Existing Zenoh topics, payload fixtures, frame transforms, and safety
   namespace pass compatibility tests.
15. Policy observations contain no privileged pose or metric goal state unless
   an explicitly named privileged schema is selected.
16. Image-goal reward uses hidden collision-aware progress and returns a named
   term breakdown without leaking goal coordinates into the policy observation.
17. The non-visual headless benchmark has no real-time sleeps and demonstrates
   at least 10x real-time for one environment on the development machine. The
   visual benchmark separately reports rendered frames/s, transfer bandwidth,
   and batched scaling on the intended training machine rather than assuming
   non-visual throughput.
18. A saved trace replays to the same termination reason, reward terms,
   controller targets/twists, and final-state hash; images match the pinned
   renderer's defined tolerance.

## Handoff package for the RL collaborator

Provide these together:

- this repository at a pinned commit;
- `sdk/python` with a lockable version and editable-install instructions;
- a one-command visual headless build/start path, including the pinned browser;
- random waypoint/twist examples and one image-goal Gymnasium example;
- action and observation schemas with frames, units, shapes, and bounds;
- camera calibration/orientation metadata and model-preprocessing wrappers;
- relative-pose controller configuration and conformance fixtures;
- the reference task and reward-term documentation;
- a reproducible trace fixture and expected metrics;
- a list of known sim-to-real limitations.

The collaborator should be free to replace reward functions and training code
without forking robot physics or depending on the browser and Zenoh processes.

## Decisions to make with the RL collaborator

These are the useful early questions; none blocks the engine extraction:

1. Is the first policy image-goal navigation, goal-free visual exploration, or
   route/topological-image following?
2. What native image resolution, history length/stride, and preprocessing will
   the first policy use outside the SDK?
3. Does the policy output one relative pose or a fixed horizon of cumulative
   relative waypoints, and does it predict yaw?
4. What waypoint normalization, horizon, lookahead index/rule, and policy rate
   does that model expect?
5. Should the controller close targets from ground truth for algorithm bring-up,
   integrated noisy odometry for deployment realism, or the existing SLAM pose?
6. What visual curriculum is wanted: fixed appearance first, then camera/noise,
   lighting/material changes, and goal/current appearance mismatch?
7. Should LiDAR/proprioception be excluded for a vision-only baseline or exposed
   through a separately named multimodal schema?
8. What framework needs first-class validation: raw Python, Gymnasium,
   Stable-Baselines3, RLlib, or another trainer?
9. How many parallel visual environments and which CPU/GPU/browser stack define
   the throughput target?

The proposed defaults are `vision_goal_v1`, native upright 320 x 240 RGB,
current plus five history frames, an image-goal task, five cumulative
robot-relative pose waypoints with yaw, 4 Hz policy, 20 Hz pose controller,
120 Hz physics, fixed appearance for bring-up, later seeded visual
randomization, and an optional Gymnasium adapter. `relative_pose_v1` and
`twist_aligned_v1` stay equally well-defined for simpler policies and ablations.
