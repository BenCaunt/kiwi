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
  valid centered pulse. The second ESC's unused channel input is wired to
  `D1/GPIO2` (as of the 2026-07-07 rewiring), which every firmware must hold at
  `1500 us` (see `kEscAuxNeutralPin`). Leaving it floating produces the 5-blink
  signal-timeout failsafe and that ESC never drives its motor.
- A 5-blinking ESC needs an ESC power cycle after the signal is fixed.
- Motor deadbands vary per motor; the stiffest needed roughly `1850 us` (~70%)
  to start turning while the others move at `1650 us` (~30%). Account for this
  deadband in kinematics tuning.

## Motor -> Encoder Mapping (measured 2026-07-07)

Measured with the `follower_motor_encoder_map` firmware and recorded in
`include/robot_config.h`:

| Firmware motor | Pin | Encoder mux channel | Positive command |
|---|---|---|---|
| motor 0 | `D0/GPIO1` | ch1 | counts decrease (polarity `-1`) |
| motor 1 | `D2/GPIO3` | ch0 | counts increase (polarity `+1`) |
| motor 2 | `D3/GPIO4` | ch2 | counts decrease (polarity `-1`) |
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

## Local Config

Copy these files before flashing the master:

```sh
cp include/secrets.example.h include/secrets.h
cp include/local_zenoh.example.h include/local_zenoh.h
```

Then edit Wi-Fi, Zenoh connect string, and `ROBOT_NAMESPACE`.

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
- motor PWM: `D0`, `D2`, `D3` (aux neutral for the second Dominion: `D1`)
- BNO08x: INT `D9`, RESET `D8`
- AS5600 mux channels: `1`, `0`, `2` for motors 0/1/2 (PCA9548A at `0x70`,
  A0/A1/A2 tied to GND)

Cross the UART wires between boards: master TX to follower RX, follower TX to master RX, and common ground.

## Zenoh Keys

Default namespace is `kiwi/xiao`.

- `kiwi/xiao/cmd_vel`: master subscriber for body velocity commands.
- `kiwi/xiao/odom/twist`: master publisher for follower-calculated twist JSON.
- `kiwi/xiao/camera/jpeg`: master publisher for JPEG frames with a 32-byte `KVC1` header.
- `kiwi/xiao/lidar/ld19/raw`: master publisher for raw 47-byte LD19-style lidar frames.
- `kiwi/xiao/status/master`: master status JSON.

## Calibration To Do

- Set `kWheelRadiusM`, `kDriveBaseRadiusM`, `kMaxWheelSurfaceSpeedMps`.
- Verify `kWheelAnglesRad` matches the physical wheel order.
- `kEncoderPolarity` is measured (2026-07-07); still verify `kMotorPolarity`
  against the kinematic wheel directions with a driving test.
- Re-seat the ch0 AS5600 magnet: it reports magnet-not-detected (`md=0`) and
  drops counts under fast motion. All three magnets read on the weak side.
- Confirm the LD19 variant frame format and baud rate. The parser currently locks to `0x54 0x2c` 47-byte frames and publishes raw frames.
