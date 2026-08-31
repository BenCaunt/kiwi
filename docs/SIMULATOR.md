# Kiwi Zenoh Simulator

The simulator provides a driveable 2D Kiwi robot without changing the laptop
control stack. It subscribes and publishes on the same Zenoh suffixes, uses
the same raw drivetrain frame, and emits the hardware's actual payload shapes.
The default namespace is `kiwi/sim` so a connected physical robot is not
commanded accidentally.

## Quick start

Install the simulator dependencies once:

```sh
python3 -m pip install -r requirements-sim.txt
```

Use three terminals:

```sh
# Terminal 1
./scripts/start_zenoh.sh

# Terminal 2: opens the interactive top-down world
python3 scripts/kiwi_simulator.py --environment room

# Terminal 3: use the existing controller
python3 scripts/kiwi_teleop.py --namespace kiwi/sim
```

The simulator window also has hold-to-drive controls:

- `W` / `S`: forward / backward
- `A` / `D`: strafe left / right
- `Q` / `E`: rotate counter-clockwise / clockwise
- `Space`: stop
- `R`: reset to the environment spawn
- `Escape`: quit

Run the existing sensor dashboard or SLAM pipeline against the simulated
namespace:

```sh
python3 scripts/kiwi_dashboard.py --namespace kiwi/sim
python3 scripts/kiwi_slam.py --namespace kiwi/sim --viewer --output maps/sim
```

For CI, remote execution, or SLAM data generation without a window:

```sh
python3 scripts/kiwi_simulator.py --headless --environment warehouse
```

## Built-in environments

| Name | Purpose |
|---|---|
| `room` | Furnished room with varied geometry; good first drive and scan test |
| `warehouse` | Long shelving aisles for deskew, odometry drift, and loop closures |
| `maze` | Tight orthogonal course for strafing and collision tests |

Select one with `--environment room`, `warehouse`, or `maze`. Override its
spawn with `--start x,y,yaw_degrees`.

## Zenoh compatibility

The default namespace differs for safety, but every suffix and payload matches
the physical robot:

| Key under the selected namespace | Direction | Simulator behavior |
|---|---|---|
| `cmd_vel` | subscribe | JSON, text, and 24-byte binary commands accepted |
| `odom/twist` | publish at 20 Hz | Firmware-shaped JSON in the raw drivetrain frame |
| `lidar/ld19/raw` | publish at 20 batches/s | 20 concatenated, CRC-valid 47-byte LD19 frames per sample |
| `camera/jpeg` | publish at 10 Hz | 32-byte `KVC1` header plus an inverted JPEG, like the mounted camera |
| `status/master` | publish at 1 Hz | Firmware status fields plus `simulator` and `environment` |

The real robot's drivetrain +X is 60 degrees counter-clockwise from
camera/LiDAR forward. The simulator preserves that relationship. Consequently
`kiwi_client.py` applies the same `--robot-yaw-deg` correction for real and
simulated runs, and positive aligned `vx` moves toward the camera view.

The binary `VelocityCommandPayload` command timeout is honored. JSON and text
commands use the firmware's 250 ms watchdog. Wheel speeds use the firmware's
three-wheel kinematics and are capped at the configured 3.23 m/s surface
speed. The world also provides first-order drive response, circular-body wall
collisions, encoder counts, fused-yaw IMU quaternions, and accelerometer data.

To use the literal hardware namespace for a fully drop-in test, first make
sure the physical robot is powered off or disconnected, then run:

```sh
python3 scripts/kiwi_simulator.py --namespace kiwi/xiao
```

Do not run a simulator and powered robot on the same namespace: subscribers
will receive interleaved sensor publishers and `cmd_vel` will reach both.

## Custom environments

Pass `--environment-file path/to/world.json`. Coordinates are meters and the
spawn yaw is degrees:

```json
{
  "name": "test-lab",
  "description": "A 6 m square with one central obstacle",
  "start": [-2.0, 0.0, 0.0],
  "walls": [
    {"from": [-3, -3], "to": [3, -3], "color": [80, 135, 185]},
    {"from": [3, -3], "to": [3, 3], "color": [80, 135, 185]},
    {"from": [3, 3], "to": [-3, 3], "color": [80, 135, 185]},
    {"from": [-3, 3], "to": [-3, -3], "color": [80, 135, 185]}
  ],
  "obstacles": [
    {"min": [-0.5, -0.5], "max": [0.5, 0.5], "color": [205, 145, 70]}
  ]
}
```

Walls are line segments. Obstacles are axis-aligned closed rectangles that
expand into four wall segments. Colors are optional RGB values used by the
viewer and simulated camera.

Useful tuning flags:

```sh
python3 scripts/kiwi_simulator.py --help
python3 scripts/kiwi_simulator.py \
  --environment warehouse \
  --lidar-noise-std-m 0.01 \
  --seed 42 \
  --speed 0.5 \
  --omega 1.5
```
