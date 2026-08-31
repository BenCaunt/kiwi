# Kiwi RL SDK Protocol v1

The public API is `kiwi_sim_sdk.KiwiEnv`; this document freezes the private
local transport so the Python and Chromium halves can be tested independently.

## Framing

Every request and response is one little-endian length-framed message:

```text
uint32 json_header_length
uint8  json_header[json_header_length]
uint8  binary_payload[header.binary_length]
```

The JSON header contains `protocol_version`, `request_id`, `arrays`, and
`binary_length`. Each array descriptor has a unique `name`, `dtype`, `shape`,
`offset`, and `byte_length` into the binary payload. V1 dtypes are `uint8`,
`uint32`, `float32`, and `float64`. RGB and temporal arrays are never base64
encoded.

Requests add `operation`; their arguments are in
`result: {"env_id": ..., "payload": ...}`. Successful responses set `ok: true`
and return a JSON result. Failures set `ok: false` and include
`error: {"code": ..., "message": ...}`. Requests are processed in order.

## Operations

- `hello`: protocol/build metadata and capabilities.
- `create`: create one independent environment from an environment config.
- `create_many`: create a batch from `payload.configs`.
- `reset`: reset `env_id` with a seed.
- `reset_many`: batch items `{env_id, seed}`.
- `step`: apply one canonical SI-unit action to `env_id`.
- `step_many`: batch items `{env_id, action}`.
- `close` / `close_many`: dispose WebGL and simulation resources.

The visual supervisor binds HTTP and WebSocket listeners to random localhost
ports, launches Chrome/Chromium, and bridges the same binary frames over stdio.
Zenoh is intentionally not in the training loop.

## Canonical actions

```json
{"kind":"relative_pose","dx":0.2,"dy":0.0,"dyaw":0.0}
{"kind":"relative_trajectory","waypoints":[{"dx":0.2,"dy":0.0,"dyaw":0.0}]}
{"kind":"twist","vx":0.1,"vy":0.0,"omega":0.0}
```

All values use the sensor-aligned robot frame: +X camera-forward, +Y left, and
positive yaw counter-clockwise. Relative trajectory points are cumulative from
one action-start frame. V1 rejects non-finite, mismatched, or out-of-range
actions; it never clips them silently.

`create` also carries the lower controller gains, speed/tolerance settings,
trajectory lookahead index, and explicit action bounds. Reset provenance echoes
the resolved values so controller tuning is reproducible.

## `vision_goal_v1`

The response observation metadata references these binary arrays:

- `rgb`: chronological upright `uint8[context,height,width,3]`;
- `rgb_valid`: `uint8[context]` reset/history mask;
- `rgb_time_s`: `float64[context]` scheduled simulation capture time;
- `rgb_sequence`: `uint32[context]` deterministic current-camera sequence;
- `goal_rgb`: upright `uint8[height,width,3]`;
- `goal_rgb_valid`: JSON scalar mask.

Calibration includes both vertical and derived horizontal FOV, intrinsics,
robot-to-camera extrinsics, orientation, near/far planes, and the named camera
profile. Goal metric coordinates remain inside task/reward code unless
`privileged_debug` is explicitly enabled.

## Reset provenance

Every reset returns protocol and engine versions plus content revisions for
physics, world geometry, controller, renderer, and camera. It also records the
action/observation modes, rates, sensor profile, seed, browser renderer stack,
and resolved randomization. These fields are part of episode identity.
