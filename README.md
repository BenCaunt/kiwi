# Kiwi Robot Firmware

Fresh PlatformIO project for a two-ESP32-S3 kiwi-drive robot.

## Architecture

- `env:master`: camera, FHL-LD19/LD19-style lidar serial input, Zenoh, and UART bridge to the follower.
- `env:follower`: three motor outputs, three AS5600 encoders through a TCA9548A/PCA9548A I2C mux, BNO08x IMU, kiwi kinematics, and UART telemetry back to the master.
- `src/common`: shared binary UART framing and payload definitions.

The master subscribes to `kiwi/xiao/cmd_vel` by default. It accepts one of:

- binary `VelocityCommandPayload`
- JSON: `{"vx":0.1,"vy":0.0,"omega":0.0}`
- text: `0.1 0.0 0.0`

The follower receives that command over UART, converts body twist into three wheel speeds, drives the motors, estimates the measured twist from encoder velocity, and sends timestamped `TwistReportPayload` packets back to the master.

## Build

```sh
pio run -e master
pio run -e follower
pio run -e follower_motor_ramp
pio run -e follower_motor_encoder_map
pio run -e follower_dominion_boot_cal
```

Upload each board with the matching environment:

```sh
pio run -e master -t upload
pio run -e follower -t upload
```

## Dominion ESC Notes (bench findings 2026-07-02)

- Each Dominion is a dual ESC and only arms when **both** channel inputs see a
  valid centered pulse. The unused fourth channel input is wired to
  `D0/GPIO1` (re-verified 2026-08-22), which every firmware must hold at
  `1500 us` (see `kEscAuxNeutralPin`). Leaving it floating produces the 5-blink
  signal-timeout failsafe and that ESC never drives its motor.
- A 5-blinking ESC needs an ESC power cycle after the signal is fixed.
- Motor deadbands vary per motor; the stiffest needed roughly `1850 us` (~70%)
  to start turning while the others move at `1650 us` (~30%). Account for this
  deadband in kinematics tuning.

## Motor -> Encoder Mapping (measured 2026-07-08, post chassis rebuild)

Measured with the `follower_motor_encoder_map` firmware and recorded in
`include/robot_config.h`. Re-run the mapping after any chassis or wiring
rework: the 2026-07-08 rebuild reshuffled all three channels.

| Firmware motor | Pin | Encoder mux channel | Positive command |
|---|---|---|---|
| motor 0 | `D0/GPIO1` | ch2 | counts decrease (polarity `-1`) |
| motor 1 | `D2/GPIO3` | ch1 | counts increase (polarity `+1`) |
| motor 2 | `D3/GPIO4` | ch0 | counts decrease (polarity `-1`) |
| aux neutral | `D1/GPIO2` | - | second Dominion arming input |

The second Dominion's *driven* input is on `D3`; the input the firmware pins at
neutral for arming is `D1`. `kEncoderPolarity` is chosen so a positive motor
command increases that motor's reported count.

The `follower_motor_encoder_map` environment is the bring-up/diagnostic tool.
It auto-runs a mapping pass (each motor +/- for 2 s while sampling all
encoders) 7 s after boot; send `s` over USB serial to cancel. Commands: `map`,
`power <pct>`, `watch <sec>` (hand-spin identification), `probe` (AS5600
presence + magnet status), `scan` (I2C bus scan), `pintest` (SDA/SCL
short/stuck detection), `aux <pct>` (drive the aux pin), `clock <khz>`, `s`.

## Follower Motor Ramp Test

The temporary `follower_motor_ramp` environment is for bench testing the wired
motors before the encoder mux arrives. It boots with all four outputs
(D0/D1/D2/D3) at RC/ESC neutral (`1500 us`) for 7 seconds and waits for a USB
serial command before sending any moving command. Channels are numbered
`1..4` = `D0..D3` in commands. Direct pulse holds revert to neutral after 8
seconds unless re-sent (runaway failsafe; the motor signal UI re-sends
automatically every 2 seconds).

```sh
pio run -e follower_motor_ramp -t upload --upload-port /dev/cu.usbmodem101
pio device monitor -p /dev/cu.usbmodem101 -b 115200
```

Useful serial commands:

- `status`: print current state and pulse widths.
- `neutral` or `n`: hold all outputs at `1500 us`.
- `low`: hold all outputs at `1000 us`.
- `high`: hold all outputs at `2000 us`.
- `pulse <us>`: hold all outputs at a pulse width, for example `pulse 1500`.
- `pulse <m> <us>`: hold motor `1`, `2`, or `3` at a pulse width while the others stay neutral.
- `pulses <d0> <d1> <d2>`: hold D0/D1/D2 at independent pulse widths.
- `a` or `ramp`: ramp all three motors through +/- the configured limit.
- `1`, `2`, `3`: ramp only D0/GPIO1, D1/GPIO2, or D2/GPIO3.
- `+`, `-`, or `limit <pct>`: adjust the ramp limit.
- `cal`: assisted Dominion calibration sequence: `2000 us`, then `1000 us`, then `1500 us`.
- `s` or `stop`: stop immediately and return all outputs to neutral.
- `?` or `help`: print help.

The default ramp limit is 20%. For Dominion calibration, send `cal`, then
immediately power-cycle ESC power while the ESP32 is holding `2000 us`.

If ESC power cannot be cycled separately from ESP32 power, use the dedicated
boot calibration firmware instead:

```sh
pio run -e follower_dominion_boot_cal -t upload --upload-port /dev/cu.usbmodem1101
```

After upload, power-cycle the shared ESP32/ESC supply. The firmware starts at
`2000 us` on boot, then switches to `1000 us`, then `1500 us`, and finally holds
neutral. Reflash `follower_motor_ramp` after calibration testing.

## Motor Signal UI

Run the local browser UI with:

```sh
~/.platformio/penv/bin/python scripts/motor_signal_server.py --connect
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The app talks to the
interactive `follower_motor_ramp` firmware over USB serial and sends
`pulses <d0> <d1> <d2>` commands from the sliders.

## Board Detection Notes

The currently connected follower/slave board is recorded in
[hardware/boards.json](hardware/boards.json). On 2026-07-02 UTC it was detected
on `/dev/cu.usbmodem101` with USB serial and ESP32-S3 MAC
`94:a9:90:d0:2e:f0`.

To check connected boards against the inventory:

```sh
~/.platformio/penv/bin/python scripts/detect_board.py
```

## Provisioning (runtime config, no reflash)

For the verified home/travel workflow, failure diagnosis, and LiDAR checks,
see [Kiwi Network, Zenoh, and LiDAR Runbook](docs/NETWORK_AND_ZENOH_RUNBOOK.md).

The master is intended to serve HTTP on `http://192.168.4.1/` from the
`KIWI-MASTER` soft-AP (password `seeedstudio`), and on its station IP once it
has joined a network. The HTTP/AP mode-transition correction and bounded
sensor-stream scheduler were flashed to the master on 2026-07-18; see the
runbook for recovery from older images. Settings persist in the master's NVS;
drive parameters are forwarded to the follower over UART (`DriveParams`
packet + ack) and persist in the follower's NVS. Compiled constants are
first-boot defaults only.

New network / travel workflow -- one command, run from the network the robot
should join (NOT from the robot AP):

```sh
python3 scripts/kiwi_provision.py --password <wifi-password>
```

The script detects your current SSID and laptop IP, switches this Mac onto
`KIWI-MASTER`, uploads the config, waits for the robot to join your network,
switches your Mac back, and verifies the robot is reachable at its new IP.
Pass `--ssid`/`--pc-ip` to override the detected values, then run
`./scripts/start_zenoh.sh`.

The robot's locator defaults to `udp/`: zenoh-pico's TCP transport on the
ESP32 starves under load (raw TCP measured 231 KB/s, but the zenoh session
managed ~15 KB/s and silently dropped nearly all payloads larger than a lidar
frame). Over UDP every topic streams at full rate. Laptop-side clients can
still connect to zenohd over TCP.

Calibration while the robot is on your network (applies live, no reboot):

```sh
python3 scripts/kiwi_provision.py --host <robot-sta-ip> --wheel-radius 0.024
python3 scripts/kiwi_provision.py --host <robot-sta-ip> --status
```

Runtime-tunable keys: `wifi_ssid`, `wifi_password`, `zenoh_connect`,
`zenoh_mode`, `wheel_radius_m`, `drive_base_radius_m`,
`max_wheel_surface_speed_mps`, `motor_polarity`, `velocity_command_timeout_ms`.

## Local Config (first-boot defaults, optional)

Compiled defaults can be baked in before the first flash so the robot comes up
on a known network without provisioning:

```sh
cp include/secrets.example.h include/secrets.h
cp include/local_zenoh.example.h include/local_zenoh.h
```

Then edit Wi-Fi, Zenoh connect string, and `ROBOT_NAMESPACE`. NVS-provisioned
values always win over these once set.

## Current Pin Assumptions

Edit [include/robot_config.h](include/robot_config.h) when wiring changes.

Master:

- follower UART: RX `D6`, TX `D7`, 460800 baud
- LD19 lidar data (lidar TX -> master RX): `D4`, 230400 baud; lidar has no serial RX
- LD19 lidar PWM speed control: `D3`, held low so the lidar self-regulates ~10 Hz (this unit does not spin with PWM floating). Raise `kLidarPwmDuty` for external RPM control.
- camera: Seeed XIAO ESP32S3 Sense camera connector

Follower:

- master UART: RX `D6`, TX `D7`, 460800 baud
- I2C mux and IMU root bus: SDA `D4`, SCL `D5`
- motor PWM: `D1`, `D2`, `D3` (aux neutral for the unused Dominion channel: `D0`)
- BNO08x: INT `D9`, RESET `D8`
- AS5600 mux channels: `2`, `1`, `0` for motors 0/1/2 (PCA9548A at `0x70`,
  A0/A1/A2 tied to GND)

Cross the UART wires between boards: master TX to follower RX, follower TX to master RX, and common ground.

## Zenoh Keys

Default namespace is `kiwi/xiao`.

- `kiwi/xiao/cmd_vel`: master subscriber for body velocity commands.
- `kiwi/xiao/cmd_vel/teleop`: manual command input used by `launch.py`.
- `kiwi/xiao/cmd_vel/navigation`: autonomous command input used by `launch.py`.
- `kiwi/xiao/cmd_vel/mux/status`: selected mux source and output command.
- `kiwi/xiao/odom/twist`: master publisher for follower-calculated twist JSON.
- `kiwi/xiao/camera/jpeg`: master publisher for JPEG frames with a 32-byte `KVC1` header.
- `kiwi/xiao/lidar/ld19/raw`: master publisher for raw 47-byte LD19-style lidar frames.
- `kiwi/xiao/status/master`: master status JSON.
- `kiwi/xiao/slam/pose`: laptop SLAM publisher for corrected map pose,
  map-to-odometry transform, and scan-match quality.
- `kiwi/xiao/slam/map`: compressed live occupancy grid published by SLAM for
  the dashboard and other visualization consumers.
- `kiwi/xiao/slam/image`: pose-correlated upright JPEG plus map pose and
  pinhole intrinsics, published when an image-map spacing threshold is met.
- `kiwi/xiao/navigation/trajectory`: current map-frame A* trajectory, planner
  metadata, and obstacle-inflation radius.
- `kiwi/xiao/navigation/state`: current controller pose, goal, pure-pursuit
  following point, progress, cross-track error, and command.

The LiDAR topic may contain one raw frame or a concatenated batch of raw
frames; every payload length is a positive multiple of 47 bytes. The bundled
LiDAR and dashboard scripts handle both forms.

## Laptop Control and Dashboard

The laptop client presents a sensor-aligned frame: +X points forward with the
LiDAR and camera, +Y points left, and positive angular velocity turns
counter-clockwise. The drivetrain's physical +X is currently 60 degrees
counter-clockwise from that direction, so `kiwi_client.py` rotates outgoing
linear commands by -60 degrees and incoming command/measured odometry by +60
degrees. Angular velocity is unchanged.

### One-command runtime

Install the agent runtime (which includes the SLAM requirements, MCP SDK, and
OpenCLIP), then run from the repository root:

```sh
python3 -m pip install -r requirements-agent.txt
python3 launch.py
```

The launcher gracefully replaces any existing Kiwi runtime after confirmation,
restarts Zenoh, and starts the command mux, headless SLAM, Rerun dashboard,
image-navigation gallery, agent MCP gateway, and keyboard teleop. It lists
complete saved maps to resume or prompts for a new map name. A resumed map
remains navigation-locked until LiDAR relocalization succeeds; its compatible
image capture manifest is selected automatically. MCP uses
`http://127.0.0.1:8766/mcp`; the gallery uses `http://127.0.0.1:8767/` because
port 8765 is used by the motor signal server.

Keyboard teleop has priority while a movement key is active. Press Space to
stop manual motion and release control back to navigation. If either input
stops publishing, the mux times it out and publishes zero. Ctrl-C in the
launcher stops navigation and teleop first, saves SLAM, closes the dashboard,
and publishes final zero commands. The healthy Zenoh router remains running so
the physical robot does not lose its UDP session between mapping runs.

For a non-interactive choice or to inspect the exact child commands:

```sh
python3 launch.py --map maps/kiwi_map
python3 launch.py --new-map maps/downstairs
python3 launch.py --map maps/kiwi_map --dry-run
```

Use `--resume-global` or `--resume-pose X Y YAW_DEG` when the robot was moved
after the saved map stopped. `--gamepad`, `--no-teleop`, `--no-dashboard`, and
`--no-image-navigation`, `--no-mcp`, `--mcp-port`, and
`--agent-max-travel-distance` are also available.

```sh
python3 scripts/kiwi_teleop.py --gamepad
python3 scripts/kiwi_dashboard.py
```

Gamepad teleop uses an 8% centered-stick deadzone. Centering the sticks releases
the command mux so MCP navigation can drive; Start/Menu (raw button 6 on the
DualSense/SDL mapping) toggles
teleop off completely for an explicit agent handoff, and toggles it back on for
manual control. Deliberate stick movement always reclaims teleop, so a stale or
unclear handoff state can never lock out manual driving. Disconnecting a
controller releases teleop immediately; reconnecting it is detected and opened
automatically without restarting the Kiwi runtime.

Both commands accept `--robot-yaw-deg` for recalibration (or `0` to inspect
the raw drivetrain frame). When `kiwi_slam.py` is running, the dashboard
consumes its live occupancy map, corrected pose, and pose-correlated camera
captures; before the first SLAM update it falls back to dead reckoning. The
SLAM map tile is a 3D scene with the occupancy grid on the floor and saved
images shown as pinhole camera frustums at their capture poses. The dashboard
also rotates camera frames 180 degrees on the laptop so the displayed image is
upright. LiDAR deskew is on
by default: the dashboard aligns the LD19 and follower timestamps, interpolates
the robot pose across each revolution, and logs motion-compensated points in
the world frame. Use `--no-lidar-deskew` for an uncompensated comparison, or
`--lidar-time-offset-ms N` to tune any residual LiDAR-to-IMU timing offset.

For a simple Python-only closed-loop pose move, run SLAM and then the pose
test in another terminal:

```sh
python3 scripts/kiwi_slam.py
python3 scripts/kiwi_pose_test.py
```

The test snapshots the current SLAM pose, composes a target 0.5 m forward and
0.5 m left of that pose with the same heading, and sends proportional
body-twist commands until it reaches the configured position/yaw tolerances.
The reusable controller is in `scripts/kiwi_pose_controller.py`.

## A* Navigation and Pure Pursuit

With SLAM and the dashboard running, send the robot to a goal expressed in the
SLAM map frame (meters):

```sh
python3 scripts/kiwi_slam.py
python3 scripts/kiwi_dashboard.py
python3 scripts/kiwi_navigation.py 1.5 -0.4 --goal-yaw-deg 90
```

`kiwi_navigation.py` plans an eight-connected A* trajectory on the live
occupancy grid and inflates occupied cells by 0.25 m by default. Unknown cells
are treated as blocked; `--allow-unknown` is available for deliberate
exploration. Diagonal corner cutting is disallowed. When a new SLAM map blocks
the remaining path, or cross-track error exceeds 0.35 m, the robot stops and
replans from its corrected pose.

The pure-pursuit layer selects a monotonic point 0.30 m ahead on the current
trajectory. That point is passed through the existing pose-stabilization
controller, retaining its map-axis proportional feedback, body-frame
conversion, speed limits, and final position/yaw settling behavior. Trajectory
following is capped at 0.12 m/s by default; use `--max-linear-speed` to tune the
cap. A stale SLAM pose always commands a stop. A* keeps the 0.25 m planning
inflation, while live pursuit uses a separate 0.18 m hard collision radius. The
0.07 m difference is a tracking/relocalization recovery buffer: a small pose
error can steer back to the already-approved route without making the replanner
treat the current pose as an impossible start. Both radii still block occupied
and unknown cells. Use `--runtime-collision-radius` to change the hard radius;
it cannot exceed `--inflation-radius`. Lookahead is shortened at corners until
the direct pursuit segment is clear at the hard radius; if no forward segment
is safe, the robot stops and replans. The dashboard plots the planned
trajectory, controller pose and heading, goal, and active following point under
`/map/navigation`.

For coordinated actions, `--max-travel-distance M` is a hard authorization
envelope over total translated distance. Every replan must fit inside the
remaining budget; exhausting it publishes a failed terminal state and stops.

Useful tuning options include `--inflation-radius`,
`--runtime-collision-radius`, `--lookahead`, `--kp-x`, `--kp-y`, `--kp-yaw`,
`--max-linear-speed`, `--max-angular-speed`, and `--replan-distance`. Use
`--namespace kiwi/sim` on SLAM, dashboard, and navigation to validate the full
loop in the simulator before driving hardware.

### Navigate by saved image

While the SLAM process that recorded the image map is still running, open the
local image gallery:

```sh
python3 scripts/kiwi_image_navigation.py
```

The gallery automatically selects the most recently updated session below
`maps/kiwi_map.images`, lets you browse or filter its camera frames, and shows
the saved map pose for each frame. Select a frame and press **Drive to this
pose** to launch the same A* and pure-pursuit navigator described above, using
the image's saved position and heading. **Stop robot** interrupts the active
navigator and sends a zero command.

By default, driving stays disabled until a fresh SLAM pose and occupancy map
arrive and the session ID on the live `slam/image` topic matches the selected
manifest. This matters because a SLAM map frame is local to one run. Using
`--resume` explicitly preserves that frame across restarted SLAM processes,
and it reloads and republishes the compatible prior image manifest after
LiDAR relocalization. The top-level launcher resolves the manifest once and
passes the same absolute path to SLAM and image navigation. Use
`--manifest PATH` to choose the current manifest, and use
`--skip-session-check` only when you have independently preserved and verified
map-frame alignment. The simulator variant is:

```sh
python3 scripts/kiwi_image_navigation.py --namespace kiwi/sim
```

### Agent MCP visual navigation

The image-navigation process also exposes one local Streamable HTTP MCP server
with these tools:

- `get_robot_status`
- `search_goal_images`
- `get_pose_on_map`
- `preview_image_goal`
- `navigate_to_image`
- `get_navigation_status`
- `get_navigation_report`
- `stop_navigation`

`preview_navigation_to_image` is a compatibility alias for
`preview_image_goal`.

The intended sequence is status, visual search or map inspection, preview,
explicitly approved motion, status monitoring, and a visual report or stop.
`get_navigation_report` returns an evenly sampled robot-camera contact sheet
and the actual live-SLAM trajectory over the occupancy map. It subscribes only
to `camera/jpeg`, `slam/pose`, and `slam/map`; simulator ground-truth position
is not used. Captures use stable
`<session_id>:<capture_id>` references. Preview uses the same live A* planner as
execution and expires after 30 seconds. Execution replans, rechecks the live
session, relocalization, map/pose freshness, command mux, and distance ceiling,
then starts exactly one action. Unknown cells remain blocked. Teleop preemption,
stale localization, a route that no longer fits the remaining distance budget,
or an explicit stop all terminate agent navigation.

Tools publish output schemas, successful results include `ok: true`, and safe
failures use a stable `{"ok": false, "error": {"code", "message",
"retryable", "suggested_tool", "details"}}` envelope. A session mismatch also
appears in `get_robot_status.recovery` with both session IDs and the selected
manifest path.

For diagnostic replays, `kiwi_slam.py --no-save` treats a resumed image map as
read-only: saved captures are republished for session identity, but neither the
graph nor the image manifest is modified. A new no-save run does not create an
image map because it could not persist the corresponding graph keyframes.

`search_goal_images` builds an exact NumPy cosine index using OpenCLIP
ViT-B/32. The first search may download model weights. Image embeddings are
stored beside the manifest in `clip-index-v1.json` and `clip-index-v1.npz` and
are reused by image checksum; pose-only manifest updates do not re-embed JPEGs.
Similarity is a ranking score, not calibrated confidence.

Add the running server to a project-scoped `.codex/config.toml` or to the Codex
desktop MCP settings:

```toml
[mcp_servers.kiwi]
url = "http://127.0.0.1:8766/mcp"
default_tools_approval_mode = "writes"
```

The endpoint binds only to loopback. For a static bearer token, set
`KIWI_MCP_TOKEN` before launching and add this line to the server table:

```toml
bearer_token_env_var = "KIWI_MCP_TOKEN"
```

`stop_navigation` stops this navigation action but is not a latched hardware
emergency stop. Keep a human at teleop during initial physical trials.

### Codex spatial-memory timeline

`scripts/kiwi_spatial_memory_timeline.py` extracts Kiwi spatial MCP calls from
a local Codex rollout and writes a self-contained HTML timeline. It accepts a
rollout path, thread UUID, or copied Codex thread URL:

```sh
python3 scripts/kiwi_spatial_memory_timeline.py \
  codex://threads/01a040eb-b34d-7db3-afd5-5cf8545994d8 \
  -o /tmp/kiwi-run.html
```

The page combines the time-ordered calls, visual-memory hits, planned routes,
executed goals, navigation reports, and embedded visual evidence. Pass
`--no-images` for a much smaller report.

## Zenoh-Compatible Simulator

A driveable 2D simulator publishes the same odometry, raw LD19, `KVC1` camera,
and status payloads as the robot, and consumes the same three `cmd_vel`
formats. Its safe default namespace is `kiwi/sim`, so the normal client tools
only need a namespace switch:

```sh
python3 -m pip install -r requirements-sim.txt
./scripts/start_zenoh.sh
python3 scripts/kiwi_simulator.py --environment room
python3 scripts/kiwi_teleop.py --namespace kiwi/sim
python3 scripts/kiwi_dashboard.py --namespace kiwi/sim
```

The simulator includes room, warehouse, and maze environments, an interactive
top-down viewer, headless mode, collisions, drive response and watchdog
behavior, encoder/IMU telemetry, and custom JSON worlds. See
[Kiwi Zenoh Simulator](docs/SIMULATOR.md) for controls, exact topic contracts,
custom environment format, and the physical-robot namespace safety note.

### Three.js simulator shell

The browser-based simulator under `simulator-web/` is the next-generation
visual frontend. Its deterministic motion, collision, frame-transform, and
LiDAR code is kept separate from Three.js. A local WebSocket bridge publishes
the physical robot's existing Zenoh command, odometry, raw LD19, `KVC1` camera,
and status contracts, so the dashboard, SLAM, navigation, and teleop clients do
not need a browser-simulator mode.

```sh
python3 -m pip install -r requirements-sim.txt
./scripts/start_zenoh.sh
python3 scripts/kiwi_sim_bridge.py --namespace kiwi/sim

# In another terminal:
cd simulator-web
npm install
npm run dev
```

See `simulator-web/README.md` for exact rates, controls, and remaining sensor-
fidelity and end-to-end work.

## 2D LiDAR SLAM

The native laptop-side graph-SLAM pipeline combines deskewed LD19 scans with
the aligned BNO08x/encoder odometry, performs correlative scan-to-submap
matching, verifies loop closures, optimizes an SE(2) pose graph, and exports a
ROS-compatible occupancy map. Its corrected pose is published back over
Zenoh; it does not require ROS at runtime.

The follower reports the BNO08x magnetometer-free Game Rotation Vector, and
the scan matcher uses RANSAC-detected long wall segments as a soft structural
prior. Clutter remains in the match, but receives less weight than stable room
geometry.

```sh
python3 -m pip install -r requirements-slam.txt
./scripts/start_zenoh.sh
python3 scripts/kiwi_slam.py --viewer --output maps/kiwi_map
```

Drive a closed path, revisit the starting area, and stop with Ctrl-C to write
`maps/kiwi_map.{pgm,yaml}`, the readable graph JSON, a compressed SLAM state,
and a timestamped `maps/kiwi_map.images/<session>/` image-map dataset. By
default an image is selected after 0.5 m of translation or 30 degrees of
rotation, with a 0.5 second minimum interval. The viewer places the upright
camera POV beside the live map and SLAM quality plots. See
[Kiwi 2D LiDAR SLAM](docs/LIDAR_SLAM.md) for the SOTA comparison, architecture,
tuning order, output contract, validation, and current limits.

Resume and extend that map from the saved final pose with:

```sh
python3 scripts/kiwi_slam.py --viewer --resume maps/kiwi_map
```

Keep the robot stopped until `relocalized` appears. Use `--resume-pose X Y
YAW_DEG` after moving it to a known map pose, or `--resume-global` to search
candidate places across the map. See the linked SLAM guide for the relocalizing
quality state and saved-session behavior. Prior image captures are resumed
automatically with their original session ID so the image navigator's safety
check continues to match the live capture stream.

To widen the search around the saved final pose or an explicit resume pose,
use `--resume-search-distance METERS`. For example,
`--resume-search-distance 3` searches a 3 m translation radius while retaining
the normal relocalization heading window.

## Calibration To Do

- Set `kWheelRadiusM`, `kDriveBaseRadiusM`, `kMaxWheelSurfaceSpeedMps`.
- Verify `kWheelAnglesRad` matches the physical wheel order.
- `kEncoderPolarity` is measured (2026-07-07); still verify `kMotorPolarity`
  against the kinematic wheel directions with a driving test.
- All three AS5600 magnets are detected (`md=1`) since the 2026-07-08 chassis
  rebuild but still read on the weak side (`ml=1`); nudging them closer to the
  sensors would add margin.
- Confirm the LD19 variant frame format and baud rate. The parser currently locks to `0x54 0x2c` 47-byte frames and publishes raw frames.
