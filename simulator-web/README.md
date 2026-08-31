# Kiwi Three.js Simulator

This is the browser-based simulator shell for Kiwi. The simulation core is
independent of Three.js: motion, collisions, frame transforms, the watchdog,
world geometry, and LiDAR all live in `src/sim`. Three.js renders that state but
does not own it.

## Run it locally

```sh
cd simulator-web
npm install
npm run dev
```

Open the URL printed by Vite. Drive with `W/S`, strafe with `A/D`, rotate with
`Q/E`, stop with Space, reset with `R`, and pause with `P`. The toolbar switches
between four furnished planar homes and the calibration room, warehouse, and
maze worlds, and toggles a following camera.

```sh
npm test
npm run build
```

## RL and vision SDK

The same simulation engine now powers a deterministic, vision-first training
environment without changing the interactive controls or Zenoh contracts.
Robot-relative pose deltas and cumulative short trajectories are the recommended
policy actions; a 20 Hz lower controller produces aligned velocity commands.
Direct aligned `[vx, vy, omega]` control remains available.

```sh
# Build the shared interactive and headless visual pages.
cd simulator-web
npm install
npm run setup:headless

# Repository root: install and run the Python SDK example.
python3 -m pip install -e sdk/python
python3 sdk/python/examples/random_agent.py
```

The Python API returns chronological upright RGB context, an optional image
goal, capture timestamps, masks, and full calibration. Rendering happens only
at deterministic simulation-time camera deadlines. The private local protocol
uses raw length-framed binary arrays, and batched environments share one
Chromium worker. See the [Sim SDK guide](../docs/SIM_SDK.md),
[design](../docs/RL_ENVIRONMENT_SDK_DESIGN.md),
[wire protocol](../docs/RL_SDK_PROTOCOL.md), and [SDK README](../sdk/python/README.md).

For a renderer-free throughput check (no real-time sleeps):

```sh
npm run benchmark:headless -- 10000
```

## Connect it to Zenoh

The browser uses a local WebSocket bridge so it can emit the exact same Zenoh
keys and payload shapes as the physical robot without embedding the Zenoh stack
in frontend code.

```sh
# Repository root: install Python simulator/bridge dependencies once.
python3 -m pip install -r requirements-sim.txt

# Terminal 1
./scripts/start_zenoh.sh

# Terminal 2
python3 scripts/kiwi_sim_bridge.py --namespace kiwi/sim

# Terminal 3
cd simulator-web
npm run dev
```

The status strip changes to `ZENOH kiwi/sim` when connected. Existing tools can
then use the browser simulator without a simulation-specific code path:

```sh
python3 scripts/kiwi_teleop.py --namespace kiwi/sim
python3 scripts/kiwi_dashboard.py --namespace kiwi/sim
python3 scripts/kiwi_slam.py --namespace kiwi/sim --viewer --output maps/web-sim
```

The bridge subscribes to `kiwi/sim/cmd_vel`. The browser publishes firmware-
compatible `odom/twist` JSON at 20 Hz, 20-frame CRC-valid raw LD19 batches at
20 Hz, an inverted `KVC1` forward-camera JPEG at 9.69 Hz, and `status/master` JSON
at 1 Hz. JSON, text, and 24-byte binary velocity commands are accepted. Local
keyboard input has priority while a drive key is held.

LD19 returns are acquired as 480 rolling rays per 10 Hz revolution from the
interpolated robot pose at each sample time. Packets retain the hardware's
12-point format, 20-packet network batching, CRC, speed field, and 30-second
wire timestamp rollover. The internal acquisition clock remains monotonic, as
the physical sensor motion does, and batches advance on an absolute cadence so
browser-step quantization cannot create artificial sensor/odometry drift.

The default `retained-robot-maps-v1` sensor profile is calibrated from the
accepted keyframes preserved in four physical-robot map bundles. It caps LiDAR
returns at 8 m, adds roughly 6% residual random loss plus a 5-12 degree
missing-return sector, and applies conservative encoder and IMU drift. The
home's doorway and obstacle geometry supplies the remaining no-returns; the
combined accepted-scan target is the physical run's roughly 234/298/353 points
at p10/median/p90. This is deliberately an accepted-scan model: raw intensity
noise, packet loss, CRC failures, and transport jitter were not retained and
are not invented here. The profile affects simulator telemetry only; the
physical robot and unprivileged SLAM code paths are unchanged.

## Planar home collection

The home collection targets contemporary upper- and middle-class furnishing
density while varying room topology, circulation, and visual landmarks:

| World id | Architectural direction | Rooms and navigation features |
|---|---|---|
| `home` | North American contemporary | Two bedrooms, open living/kitchen, hall, bath |
| `home-machiya` | Kyoto-inspired contemporary | Genkan, tatami room, garden court, study, engawa-like hall |
| `home-riad` | Marrakesh-inspired contemporary | Central fountain court, formal salon, library, three bedrooms |
| `home-kerala` | Kerala-inspired contemporary | Shaded veranda, garden court, study, family room, bedroom wings |

These are specific modern design studies, not attempts to represent whole
cultures. Each stays on one navigable level and uses meter-scale doorway gaps,
walls, and furniture as real collision and LiDAR geometry. Tables and desks use
height-aware component geometry: their visible legs remain solid while Kiwi can
drive and scan beneath a tabletop with sufficient vertical clearance. Procedural wood,
tile, terrazzo, tatami, stone, carpet, and mosaic textures require no network
assets and remain deterministic. Each home also defines its own sky/ground
light, sun color and direction, exposure, and practical interior fixtures.

The original `home` wall layout is intentionally preserved because the retained
sensor profile was calibrated against it. Furniture is resolved into the visible
components that intersect the robot or LiDAR height.

## Current boundary

The first vertical slice includes:

- deterministic 120 Hz fixed-step simulation;
- Kiwi-aligned holonomic motion and the physical 60-degree raw-frame transform;
- first-order drive response, command timeout, and circle/segment collisions;
- configurable 360-degree LiDAR ray casting with rolling-pose acquisition;
- procedural Three.js robot and environments, including four furnished planar
  homes with patterned room flooring, real doorway gaps, culturally varied
  layouts, layered lighting, and LiDAR-visible furniture;
- orbit and follow cameras, keyboard drive, pause/reset, and live telemetry;
- a WebSocket-to-Zenoh bridge with firmware-compatible command, odometry, LD19,
  camera, and status contracts;
- deterministic integer-tick RL stepping, anchored relative-pose/trajectory
  control, raw velocity control, named independent sensor random streams, and
  action-spanning events;
- first-class temporal RGB/image-goal observations, collision-aware navigation
  reward, content-hashed provenance, binary batched RPC, and trace replay.

## Next fidelity decisions

1. Measure visual throughput before selecting worker-pool or shared-memory
   optimizations; the Python API should stay unchanged.
2. Add measured, seeded visual and latency randomization profiles.
3. Add richer image-goal pair sets and physical-camera validation scenes.
