# Kiwi Sim SDK guide

The Kiwi Sim SDK is the Python interface for deterministic, vision-first robot
training. It exposes a Gym-style `reset`/`step` loop, starts the browser-based
simulator in headless Chromium, and returns NumPy observations directly to the
policy. It does not require Zenoh and does not run in real time.

Use this guide for the public Python API. For implementation details, see the
[environment design](RL_ENVIRONMENT_SDK_DESIGN.md) and the
[private transport protocol](RL_SDK_PROTOCOL.md).

## 1. Install from a repository checkout

The SDK currently expects the adjacent `simulator-web` project, so run these
commands from the repository root.

Prerequisites:

- Python 3.10 or newer
- Node.js and npm
- Chrome/Chromium, or permission to download the Playwright-pinned Chromium

Install the web runner and its pinned browser:

```sh
cd simulator-web
npm ci
npm run setup:headless
cd ..
```

Install the Python package in editable mode:

```sh
python3 -m pip install -e sdk/python
```

If Chrome or Chromium is already installed, `npm run build` is sufficient in
place of `npm run setup:headless`. Point the SDK at that browser if it is not in
a standard location:

```sh
export KIWI_CHROMIUM_EXECUTABLE=/absolute/path/to/chrome
```

## 2. Run one environment

This complete example moves 15 cm forward at every policy step:

```python
import numpy as np

from kiwi_sim_sdk import EnvConfig, KiwiEnv

config = EnvConfig(
    world_id="room",
    action_mode="relative_pose_v1",
    max_episode_steps=200,
)

with KiwiEnv(config) as env:
    observation, reset_info = env.reset(seed=42)

    while True:
        action = np.array([0.15, 0.0, 0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info.get("termination_reason", "time_limit"))
            break
```

Always close an environment. A `with` block is the easiest way to close its
Chromium worker even if the policy raises an exception. After `terminated` or
`truncated` becomes true, call `reset` before calling `step` again.

Creating `KiwiEnv` starts a local headless supervisor and may build the web
assets when they are missing. `reset(seed=...)` chooses an authored start/goal
pair deterministically. Repeating the same configuration, seed, action
sequence, simulator content, and renderer stack reproduces the episode.

## 3. Choose an action mode

All actions use SI units in the sensor-aligned robot frame:

- +X is camera-forward.
- +Y is left.
- Positive yaw or angular velocity is counter-clockwise.

The simulator rejects invalid or out-of-range actions; it does not silently
clip them.

| `action_mode` | NumPy action | Meaning | Default policy rate |
|---|---|---|---|
| `relative_pose_v1` | `(3,)` `[dx, dy, dyaw]` | A robot-relative target pose in metres and radians | 4 Hz |
| `relative_trajectory_v1` | `(H, 3)` rows of `[dx, dy, dyaw]` | Cumulative poses from one action-start frame | 4 Hz |
| `twist_aligned_v1` | `(3,)` `[vx, vy, omega]` | Body velocity in m/s and rad/s | 20 Hz |

### Relative pose

`relative_pose_v1` is the recommended starting point. The delta is transformed
to a fixed world-frame target when `step` accepts the action. The 20 Hz lower
controller tracks that target while the simulator advances one fixed policy
interval. A later policy step replaces the previous target.

```python
config = EnvConfig(action_mode="relative_pose_v1")
action = np.array([0.20, 0.05, 0.10], dtype=np.float32)
```

The default limits are 2 m of planar translation and pi radians of yaw per
action. Change them with `max_relative_translation_m` and
`max_relative_yaw_rad`.

### Relative trajectory

Every waypoint is cumulative from the robot frame captured at the start of the
step; the values are not offsets from the previous row. The configured
`trajectory_lookahead_index` selects the waypoint tracked by the lower
controller and must be less than `H`.

```python
config = EnvConfig(
    action_mode="relative_trajectory_v1",
    trajectory_lookahead_index=2,
)
action = np.array(
    [
        [0.10, 0.00, 0.00],
        [0.20, 0.02, 0.00],
        [0.30, 0.05, 0.05],
    ],
    dtype=np.float32,
)
```

The default maximum horizon is 32 waypoints. See
[`trajectory_agent.py`](../sdk/python/examples/trajectory_agent.py) for a
runnable example.

### Direct twist

Use `twist_aligned_v1` for low-level control, teleoperation, or controller
experiments:

```python
config = EnvConfig(action_mode="twist_aligned_v1")
action = np.array([0.20, 0.0, -0.25], dtype=np.float32)
```

The current robot limits are 0.8 m/s planar speed and 2.5 rad/s angular speed.
The resolved limits are also recorded in
`reset_info["provenance"]["action_bounds"]`.

Dictionary-form canonical actions are accepted too, but NumPy arrays are the
simplest public interface:

```python
env.step({"kind": "relative_pose", "dx": 0.2, "dy": 0.0, "dyaw": 0.0})
```

## 4. Read observations

The default `vision_goal_v1` observation is a dictionary:

| Key | Type and shape | Meaning |
|---|---|---|
| `schema` | `str` | `vision_goal_v1` or `vision_v1` |
| `rgb` | `uint8[C, H, W, 3]` | Chronological, upright RGB context; newest frame is `rgb[-1]` |
| `rgb_valid` | `uint8[C]` | 1 for captured history, 0 for reset-filled slots |
| `rgb_time_s` | `float64[C]` | Scheduled simulation capture times |
| `rgb_sequence` | `uint32[C]` | Deterministic camera sequence numbers |
| `goal_rgb` | `uint8[H, W, 3]` | Goal image; present in `vision_goal_v1` |
| `goal_rgb_valid` | scalar `uint8` | Whether a goal image is valid |
| `goal_rgb_sequence` | `int` or `None` | Goal capture sequence metadata |
| `calibration` | `dict` | Camera intrinsics, FOV, extrinsics, orientation, and clipping planes |

`C`, `H`, and `W` come from `VisionConfig`. At reset there is only one captured
current frame, so older context slots repeat that first frame and have a zero
mask. Use `rgb_valid` rather than treating repeated reset pixels as history.

```python
from kiwi_sim_sdk import EnvConfig, VisionConfig

config = EnvConfig(
    vision=VisionConfig(
        width=160,
        height=120,
        context_length=4,
        context_stride=2,
    )
)
```

The SDK deliberately leaves resizing, normalization, channel ordering for model
frameworks, and tensor conversion to policy code.

For task-free visual observations, use both settings below. Rewards are zero and
there is no `goal_rgb`:

```python
config = EnvConfig(observation_schema="vision_v1", task=None)
```

## 5. Read reset and step metadata

`reset` returns task identity and reproducibility metadata. Useful fields
include:

```python
observation, info = env.reset(seed=42)

print(info["task_pair_id"])
print(info["initial_geodesic_distance_m"])
print(info["provenance"]["world_revision"])
print(info["provenance"]["renderer_backend"])
print(info["provenance"]["action_bounds"])
```

`step` returns the standard five-tuple
`(observation, reward, terminated, truncated, info)`. The main `info` fields
are:

- `reward_terms`: named contribution from progress, success, collision, time,
  controller effort, and action smoothness.
- `geodesic_distance_m` and `success`: navigation task status.
- `collision_tick_count`: physics ticks with contact during this policy step.
- `simulation_time_s` and `episode_step`: deterministic episode time and index.
- `events`: camera, sensor, contact, timeout, and relative-target events.
- `controller_commands`: lower-controller targets and generated twist commands.
- `termination_reason`: `goal_reached` when the task terminates successfully.

`terminated` means the task goal was reached. `truncated` means
`max_episode_steps` was reached. The default image-goal task uses geodesic
progress plus success, collision, time, controller-effort, and smoothness
terms. Tune the weights without editing simulator code:

```python
from kiwi_sim_sdk import EnvConfig, RewardConfig, TaskConfig

config = EnvConfig(
    task_config=TaskConfig(
        success_radius_m=0.20,
        require_goal_heading=True,
        success_yaw_tolerance_rad=0.20,
        reward=RewardConfig(
            progress=2.0,
            success=10.0,
            collision=0.5,
            time=0.01,
            controller=0.001,
            smoothness=0.01,
        ),
    )
)
```

Set `privileged_debug=True` only for debugging, evaluation, or visualization.
It adds the metric robot/goal poses to `info`; those values should not be policy
inputs for the vision-first task.

## 6. Configure the environment

The built-in world IDs are:

- `home` (default)
- `home-machiya`
- `home-riad`
- `home-kerala`
- `room`
- `warehouse`
- `maze`

Common configuration objects:

```python
from kiwi_sim_sdk import (
    ControllerConfig,
    EnvConfig,
    RewardConfig,
    TaskConfig,
    VisionConfig,
)

config = EnvConfig(
    world_id="home",
    action_mode="relative_pose_v1",
    observation_schema="vision_goal_v1",
    policy_hz=4.0,
    controller_hz=20.0,
    max_episode_steps=400,
    sensor_profile="ideal",
    controller=ControllerConfig(
        kp_x=0.8,
        kp_y=0.8,
        kp_yaw=1.5,
        max_linear_speed=0.25,
        max_angular_speed=1.0,
    ),
    vision=VisionConfig(width=320, height=240, context_length=6),
)
```

Both `policy_hz` and `controller_hz` must divide the 120 Hz physics rate
exactly, and `controller_hz` must be at least `policy_hz`. If `policy_hz` is
omitted, it resolves to 4 Hz for relative actions and 20 Hz for direct twist.

`sensor_profile="ideal"` is the SDK default. The alternative
`retained-robot-maps-v1` applies the sensor profile calibrated from retained
physical-robot map runs. It affects simulator sensor behavior; visual domain
randomization is not yet part of protocol v1.

## 7. Run batched environments

`KiwiVectorEnv` performs synchronous batched calls through one Chromium worker.
Each environment can have a different configuration.

```python
import numpy as np

from kiwi_sim_sdk import EnvConfig, KiwiVectorEnv, VisionConfig

configs = [
    EnvConfig(world_id="room", vision=VisionConfig(width=160, height=120)),
    EnvConfig(world_id="warehouse", vision=VisionConfig(width=160, height=120)),
]

with KiwiVectorEnv(configs) as envs:
    observations, infos = envs.reset(seeds=[10, 11])
    actions = [
        np.array([0.15, 0.0, 0.0], dtype=np.float32),
        np.array([0.15, 0.0, 0.0], dtype=np.float32),
    ]
    observations, rewards, terminated, truncated, infos = envs.step(actions)
```

The number of seeds and actions must equal `envs.num_envs`.

## 8. Use the Gymnasium adapter

Install the optional dependency and import the adapter from its submodule:

```sh
python3 -m pip install -e 'sdk/python[gymnasium]'
```

```python
import numpy as np

from kiwi_sim_sdk import EnvConfig
from kiwi_sim_sdk.gymnasium import KiwiGymEnv

env = KiwiGymEnv(EnvConfig(world_id="room"))
try:
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.15, 0.0, 0.0], dtype=np.float32)
    )
finally:
    env.close()
```

The adapter supplies Gymnasium observation and action spaces. Its action space
is intentionally unbounded at the wrapper level, while the simulator still
enforces the limits recorded in reset provenance. A raw random sample can
therefore be rejected; clip or scale policy output to those limits.

## 9. Record and replay an episode

`EpisodeRecorder` saves the configuration, seed, canonical actions, named
rewards, events, controller output, termination state, provenance, and hashes
of every visual observation.

```python
import numpy as np

from kiwi_sim_sdk import EnvConfig, EpisodeRecorder, KiwiEnv, replay_trace

with KiwiEnv(EnvConfig(world_id="room")) as base_env:
    env = EpisodeRecorder(base_env)
    env.reset(seed=42)
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(
            np.array([0.1, 0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break
    env.save("episode.json")

report = replay_trace("episode.json")
print(report.matched, report.mismatches)
```

A replay mismatch can identify a changed reward, termination, or observation
hash. Exact pixels also depend on the recorded renderer stack remaining
compatible.

## 10. Inspect what the policy sees

Create a labeled goal/context montage:

```sh
python3 -m pip install -e 'sdk/python[preview]'
python3 sdk/python/examples/capture_preview.py --world room --output kiwi_rl_preview.png
```

Or stream current images, reward curves, and privileged robot/goal points to a
live Rerun viewer:

```sh
python3 -m pip install -e 'sdk/python[rerun]'
python3 sdk/python/examples/rerun_scope.py --world room
```

Other runnable examples:

- [`random_agent.py`](../sdk/python/examples/random_agent.py): basic episode
  loop.
- [`trajectory_agent.py`](../sdk/python/examples/trajectory_agent.py):
  cumulative trajectory actions.
- [`capture_preview.py`](../sdk/python/examples/capture_preview.py): static
  observation montage.
- [`rerun_scope.py`](../sdk/python/examples/rerun_scope.py): live episode
  inspection.

## Troubleshooting

### `No Chromium executable found`

Run `cd simulator-web && npm run install:browser`, install Chrome/Chromium, or
set `KIWI_CHROMIUM_EXECUTABLE` to the executable's absolute path.

### `Headless supervisor not found` or `Could not locate simulator-web`

Run from this repository checkout. For a nonstandard layout, set:

```sh
export KIWI_SIMULATOR_WEB_DIR=/absolute/path/to/kiwi-robot/simulator-web
```

The same paths can be supplied as `EnvConfig(simulator_web_dir=...,
chromium_executable=...)` using `pathlib.Path` values.

### Headless assets are missing or the build fails

Run:

```sh
cd simulator-web
npm ci
npm run build
```

`EnvConfig(build_if_missing=True)` is the default and builds missing assets
automatically, but the npm dependencies must already be installed. Set it to
`False` when training jobs should fail rather than build at startup.

### An action is rejected

Check its shape, finiteness, mode, and bounds. The complete resolved bounds are
in `reset_info["provenance"]["action_bounds"]`. For trajectories, also ensure
that the horizon contains `trajectory_lookahead_index + 1` rows.

### Do I need the Zenoh router or interactive simulator?

No. The SDK launches its own localhost Chromium/WebGL worker and communicates
over a private binary RPC. Zenoh is only needed for the interactive simulator's
hardware-compatible integration path.
