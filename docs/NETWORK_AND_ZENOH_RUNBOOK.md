# Kiwi Network, Zenoh, and LiDAR Runbook

Use this runbook whenever Kiwi moves between the home network and the travel
router, Zenoh appears silent, or the LiDAR dashboard shows no useful data.

The important distinction is:

- Wi-Fi credentials tell the robot which access point to join.
- The Zenoh locator tells the robot the laptop IP to publish to.
- Reusing the same SSID and password does **not** make two networks use the
  same subnet or laptop IP.

Do not store the Wi-Fi password in this tracked file. The correct shared
`SpectrumSetup-86` credentials are already stored in the master ESP32's NVS.

for Ben:

I can just run:

python3 scripts/kiwi_provision.py \
  --host 192.168.4.1 \
  --zenoh-connect udp/192.168.1.250:7447 \
  --zenoh-mode client

  swap zenoh connect based on the table below for whatever the appropriate laptop ip is.

## Known-good configuration (verified 2026-07-17)

| Item | Home | Travel router |
|---|---:|---:|
| SSID | `SpectrumSetup-86` | `SpectrumSetup-86` |
| Router/LAN IP | `192.168.1.1` | `192.168.8.1` |
| Laptop IP | `192.168.1.250` | `192.168.8.109` |
| Robot IP seen during testing | `192.168.1.157` | `192.168.8.208` |
| Robot Zenoh locator | `udp/192.168.1.238:7447` | `udp/192.168.8.109:7447` |

Robot IPs are DHCP addresses and may change. Laptop IPs should be DHCP
reservations because the robot's Zenoh locator points at one explicitly.

After the scheduler fix was flashed on 2026-07-18, the home test received
2,080 LiDAR frames in 5 seconds (416 frames/s), with zero CRC failures.
Decoded revolutions contained roughly 489-503 non-zero ranges out of 492-504
points at about 10 revolutions/s. The same load delivered camera JPEGs at
9.7 FPS, odometry at 19.2 Hz, and status at 1 Hz.

## Two-minute golden path

Run all commands from the repository root:

```sh
cd /Users/bencaunt/Documents/kiwi-robot
```

1. Put the laptop on the same LAN the robot should use and confirm its IP:

   ```sh
   ipconfig getifaddr en0
   ```

2. Stop any old teleop before reconnecting the robot:

   ```sh
   pgrep -fl 'kiwi_teleop.py'
   ```

   Use `Ctrl-C` in that teleop terminal. If it was accidentally left in the
   background, stop the exact PID with `kill -INT <pid>`.

3. In a dedicated terminal, start Zenoh (or confirm that it is already
   listening):

   ```sh
   lsof -nP -iUDP:7447 -iTCP:7447
   ./scripts/start_zenoh.sh
   ```

   The start script safely detects an existing correctly configured router.
   If an old router is running with the wrong listeners, replace it with:

   ```sh
   ./scripts/start_zenoh.sh --restart
   ```

   Keep the `zenohd` terminal open while using the robot.

   To stop every local `zenohd` router before a clean restart:

   ```sh
   python3 scripts/stop_zenoh.py
   ```

   `./scripts/stop_zenoh.sh` is also available as a shortcut. Add `--dry-run`
   to either command to see what it would stop without changing any running
   processes.

4. Power-cycle the robot and wait 20-30 seconds.

5. Test the actual LiDAR ranges:

   ```sh
   python3 scripts/kiwi_lidar.py --check
   python3 scripts/kiwi_lidar.py
   ```

   `--check` should report a non-zero frame count and preferably zero CRC
   failures. The second command must print revolutions with a non-zero
   `valid` count and real `nearest` distances. Use `Ctrl-C` to stop it.

6. Start the dashboard:

   ```sh
   python3 scripts/kiwi_dashboard.py
   ```

   If Rerun opens with a black or frozen camera even though topic rates are
   healthy, refresh the Rerun view first. If necessary, close the old viewer
   and restart `kiwi_dashboard.py`; a stale Rerun viewport can remain black
   while the underlying JPEG topic is valid.

If step 5 receives real ranges, Wi-Fi, Zenoh, the LD19 byte stream, frame
parsing, and range decoding are all working. Do not reprovision anything.

## Updating only the Zenoh locator (preferred)

When the robot already joined Wi-Fi but points to the laptop IP from the other
network, update only the locator. This preserves Wi-Fi and all drive settings.

First prove that the robot's station HTTP endpoint is responding:

```sh
curl --connect-timeout 2 --max-time 4 http://<robot-ip>/status
```

Then update the locator directly without joining `KIWI-MASTER`:

```sh
python3 scripts/kiwi_provision.py \
  --host <robot-ip> \
  --zenoh-connect udp/<laptop-ip>:7447 \
  --zenoh-mode client
```

Home example:

```sh
python3 scripts/kiwi_provision.py \
  --host 192.168.1.157 \
  --zenoh-connect udp/192.168.1.238:7447 \
  --zenoh-mode client
```

Travel example (substitute the robot's current travel DHCP address):

```sh
python3 scripts/kiwi_provision.py \
  --host 192.168.8.208 \
  --zenoh-connect udp/192.168.8.109:7447 \
  --zenoh-mode client
```

The robot saves the value to NVS and reboots. Occasionally that reboot cuts
off the HTTP acknowledgment, so the client may time out even though the save
succeeded. Power-cycle once, wait 20-30 seconds, and use
`kiwi_lidar.py --check` plus `kiwi_lidar.py` as the authoritative verification.

### When station HTTP is unavailable

Joining `KIWI-MASTER` disconnects the laptop from the travel router and may
also disconnect Codex, SSH, remote shells, screen sharing, and any other
network-dependent assistant. A remote assistant must **not** switch the Mac to
`KIWI-MASTER` itself. It must print the complete recovery sequence first and
ask the person at the Mac to run it in a local terminal. Continue remote work
only after that person confirms the Mac is back on the travel LAN.

Use these commands when the robot answers ping on the travel LAN but its
station `/status` endpoint times out and the LiDAR check reports zero frames.
They update only the Zenoh locator; they do not overwrite the stored Wi-Fi
credentials or drive settings.

Before leaving the travel LAN, copy this whole block into a local terminal or
otherwise keep it available offline:

```sh
cd /Users/bencaunt/Documents/kiwi-robot

# Join the robot AP. This intentionally disconnects the Mac from the router.
networksetup -setairportnetwork en0 KIWI-MASTER seeedstudio
ipconfig getifaddr en0
curl --connect-timeout 2 --max-time 4 http://192.168.4.1/status

# Run this only if the AP /status request above succeeds.
python3 scripts/kiwi_provision.py \
  --host 192.168.4.1 \
  --zenoh-connect udp/192.168.8.109:7447 \
  --zenoh-mode client

# Always restore the travel LAN, including after a timeout or error above.
networksetup -setairportnetwork en0 SpectrumSetup-86
ipconfig getifaddr en0
```

The final IP should be `192.168.8.109`. If it is not, reconnect to
`SpectrumSetup-86` from the macOS Wi-Fi menu. Power-cycle the robot, wait
20-30 seconds, then verify from the restored travel LAN:

```sh
lsof -nP -iUDP:7447 -iTCP:7447
python3 scripts/kiwi_lidar.py --check
python3 scripts/kiwi_lidar.py
```

If `http://192.168.4.1/status` times out, do not run the provisioning command
and do not keep the Mac stranded on the robot AP. Restore `SpectrumSetup-86`
immediately and use the older-firmware/USB recovery guidance below.

## Home workflow

The desired home state is:

```text
laptop:        192.168.1.238
robot locator: udp/192.168.1.238:7447
zenohd:        listening on UDP and TCP port 7447
```

1. Confirm the laptop is `192.168.1.238`.
2. Confirm `zenohd` is listening.
3. Power-cycle the robot.
4. Run the two LiDAR checks from the golden path.
5. Only if they fail, find the robot's current IP and perform the direct
   locator update above.

The laptop may use 5 GHz while the ESP32 uses 2.4 GHz. They only need to be on
the same LAN. Do not disable 5 GHz.

## Travel workflow

### Recommended: make travel addressing match home

If practical, configure the travel router LAN as `192.168.1.1/24` and reserve
`192.168.1.238` for this laptop. With the same 2.4 GHz SSID/password, the robot
can then retain the home locator and no provisioning is required when moving
between networks.

Do not use this layout if the travel router's upstream/WAN network is also
`192.168.1.0/24`; overlapping WAN and LAN subnets can break routing. In that
case, retain the travel router's `192.168.8.0/24` LAN and use the next method.

### Existing travel addressing (`192.168.8.0/24`)

1. Reserve `192.168.8.109` for the laptop.
2. Join the travel LAN and confirm the laptop received that address.
3. Start `zenohd`.
4. Power-cycle the robot and find its `192.168.8.x` address.
5. Directly update the locator to `udp/192.168.8.109:7447`.
6. Power-cycle and run both LiDAR checks.

Switching back home requires changing the locator to
`udp/192.168.1.238:7447`. A single fixed locator cannot point at both laptop
addresses.

## Finding the robot IP

Preferred methods, in order:

1. Check the router's DHCP/client list for a new ESP32/unknown client.
2. Inspect the Mac's neighbor table after the robot boots:

   ```sh
   arp -a
   ```

3. Test a suspected address:

   ```sh
   ping -c 2 <robot-ip>
   curl --connect-timeout 2 --max-time 4 http://<robot-ip>/status
   ```

Ping proves the ESP32 is on the LAN. An HTTP timeout does not necessarily mean
the robot or Zenoh is offline; see the installed-firmware caveat below.

## Full Wi-Fi provisioning

Only use this when the robot's stored SSID/password genuinely need to change.
Run it while the laptop is on the target network, not while manually joined to
`KIWI-MASTER`:

```sh
python3 scripts/kiwi_provision.py --password '<target-wifi-password>'
```

Or provide everything explicitly:

```sh
python3 scripts/kiwi_provision.py \
  --ssid '<target-ssid>' \
  --password '<target-wifi-password>' \
  --pc-ip '<reserved-laptop-ip>' \
  --zenoh-mode client
```

The script uses a UDP locator by default. Do not substitute `tcp/` for the
ESP32 locator: Zenoh-pico TCP on this hardware dropped larger/high-rate
payloads under load, while UDP streamed every topic reliably.

If the target network is deliberately offline during recovery, the script now
supports saving and restoring the Mac without waiting for the target:

```sh
python3 scripts/kiwi_provision.py \
  --zenoh-connect udp/<laptop-ip>:7447 \
  --zenoh-mode client \
  --defer-network-verify
```

This is a fallback. Prefer the direct station-IP update whenever possible.

## Older-firmware HTTP caveat

Firmware installed before 2026-07-18 could invalidate its HTTP listener when
Wi-Fi changed from station-only mode to combined station/AP mode. Symptoms
included:

- `ping <robot-ip>` succeeds, but `http://<robot-ip>/status` times out;
- the Mac joins `KIWI-MASTER` at `192.168.4.2`, but
  `http://192.168.4.1/status` times out;
- DHCP and Zenoh may work even though HTTP does not.

The corrected firmware was flashed to master USB ID `E0:72:A1:FC:07:A0` on
2026-07-18. It now rebinds HTTP after the AP interface starts. If an older
master image is restored:

1. Prefer a direct station-IP update when HTTP happens to be available.
2. Treat `POST http://.../config` plus `Robot response: {"ok": true, ...}` as
   proof that configuration was accepted.
3. If a POST times out during reboot, verify behavior after a power-cycle
   rather than repeatedly overwriting settings.
4. Do not take down a shared network merely to make `KIWI-MASTER` available.

## What each diagnostic proves

| Observation | What it proves | What it does not prove |
|---|---|---|
| Robot answers ping | Robot joined the LAN | HTTP or Zenoh is healthy |
| `/status` shows `lidar_frames` increasing | Master receives LD19 frames over UART | Zenoh is delivering them to the laptop |
| `kiwi_lidar.py --check` reports frames | Zenoh delivers batched 47-byte frames and CRC can be checked | Distances inside valid frames are non-zero |
| `kiwi_lidar.py` reports valid points and nearest distance | Real, non-zero LiDAR ranges are arriving | Dashboard layout is correct |
| Teleop prints measured twist | Robot receives Zenoh commands and returns odometry | LiDAR is healthy |

## Symptom-to-action table

| Symptom | Likely cause | Action |
|---|---|---|
| `zenoh_ready: false` and locator contains the other subnet | Robot is targeting the old laptop IP | Directly update only `zenoh_connect` |
| Teleop commands appear but measured twist stays zero | Teleop reached local `zenohd`, but robot did not | Stop teleop; fix locator; verify LiDAR/odom before driving |
| `frames: 0 in 5 s` | Robot is not publishing through this Zenoh router | Check laptop IP, locator, `zenohd`, and LAN isolation |
| Frames are non-zero, but ranges are suspect | Frame transport alone is insufficient | Run `python3 scripts/kiwi_lidar.py` and inspect `valid`/`nearest` |
| Camera rate is healthy but Rerun remains black/frozen | Stale Rerun viewport or viewer process | Refresh Rerun; close old viewers and restart `kiwi_dashboard.py` |
| Joined `KIWI-MASTER`, but `/status` does not answer | Old firmware or a new HTTP regression | Restore the Mac; verify the 2026-07-18-or-newer master image |
| No `POST http://.../config` line appeared | Provisioning never wrote anything | Fix reachability; do not assume NVS changed |
| `networksetup` times out | macOS Wi-Fi command stalled, possibly after associating | Current script catches this and verifies actual SSID/IP before retrying |
| Same SSID/password, different router, no Zenoh | Same credentials but different subnet/laptop IP | Match LAN addressing or update the locator |

## Safety rules

- Stop all teleop/gamepad processes before repairing connectivity. A stale
  publisher can command motion immediately when the robot reconnects.
- Start driving only after odometry is visibly returning.
- Use `udp/<laptop-ip>:7447` for the robot and keep the laptop IP reserved.
- Updating only the locator does not alter Wi-Fi credentials, drive tuning,
  motor polarity, encoder polarity, or LiDAR configuration.
- A failed attempt before the `POST` line is non-destructive.

## Current master streaming architecture

The master firmware flashed on 2026-07-18 prevents high-rate sensor traffic
from starving the camera, odometry, status, HTTP, and network maintenance:

- Twenty fixed 47-byte LD19 frames are concatenated into each 940-byte Zenoh
  sample. `kiwi_lidar.py` and `kiwi_dashboard.py` unpack both legacy single
  frames and these batches transparently.
- The master drains only bounded UART work per loop.
- Only the newest follower report is published, at up to 20 Hz.
- Camera remains QVGA JPEG at 10 Hz and no longer competes with hundreds of
  tiny LiDAR transport operations each second.
- Status exposes loop latency, publish latency, UART high-water marks, and
  per-topic counters.

If custom code subscribes directly to `kiwi/xiao/lidar/ld19/raw`, treat every
sample as one or more concatenated 47-byte frames. Its length must be a
positive multiple of 47.

The verified steady-state values were approximately:

```text
camera       9.7 samples/s
lidar       20.8 batches/s = 416 decoded frames/s
odometry    19.2 samples/s
status       1.0 samples/s
loop gap    15-18 ms maximum
publish      6-7 ms maximum
free heap  ~227 KB, stable
```

## Reflashing the current master

The master is USB ID `E0:72:A1:FC:07:A0`. Confirm that identity before
flashing, because `/dev/cu.usbmodem101` is also reused by the follower at
other times.

```sh
python3 scripts/detect_board.py
~/.platformio/penv/bin/pio run -e master -t upload --upload-port <master-port>
```

Uploading the application does not erase NVS, but always verify the stored
SSID, Zenoh locator, and drive version afterward.
