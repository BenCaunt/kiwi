# Kiwi 2D LiDAR SLAM

## Design decision

There is no single universal "SOTA" package for 2D LiDAR SLAM. The useful
state of the art for this robot is a graph-based, multi-sensor pipeline, not a
learned end-to-end model:

| System | Relevant strength | Fit for Kiwi |
|---|---|---|
| [Cartographer](https://research.google/pubs/real-time-loop-closure-in-2d-lidar-slam/) | Correlative scan-to-submap matching, probability grids, and globally optimized loop constraints at 5 cm resolution | Strong algorithmic baseline, but its full ROS/C++ integration is heavy for this Zenoh-native project |
| [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox) | Mature Karto scan matcher, pose graph, map serialization, and long-term mapping | Strong ROS 2 production choice, but would add ROS solely as an adapter layer |
| [2DLIW-SLAM](https://arxiv.org/abs/2404.07644) | Joint 2D LiDAR, IMU, and wheel constraints; line-aware tracking; feature-based loop detection | Best research match for Kiwi's sensors, but the paper does not provide a drop-in Zenoh implementation |

A [2024 real-robot comparison](https://doi.org/10.20906/CBA2024/4198) of
GMapping, Hector, Cartographer, and Karto reported the best mean structural map
similarity for Cartographer. 2DLIW-SLAM reported lower relative pose error than
Cartographer in its four test scenes, especially under corridor degeneracy,
by including wheel/IMU constraints in the front end. Kiwi therefore combines
those ideas in a native implementation:

1. The BNO08x magnetometer-free Game Rotation Vector and encoder twist produce
   a timestamped odometry trajectory. Avoiding magnetic north prevents nearby
   motors, wiring, and furniture from imposing abrupt indoor yaw errors. The
   laptop client also detects nonphysical Game Rotation Vector origin changes
   against encoder angular velocity and rebases them without altering normal
   IMU motion.
2. The existing LD19 clock alignment and deskew transform each rolling scan
   into the final robot frame of that revolution.
3. Odometry predicts the next pose. A coarse correlative search against a
   rolling submap removes large error, followed by robust nonlinear
   distance-field refinement with an odometry prior. Deterministic RANSAC
   detects long scan-line segments and gives their points extra weight, so
   wall structure dominates clutter without assuming every return is a wall.
4. Motion-selected keyframes add distinct odometry and scan-match constraints
   to an SE(2) pose graph.
5. A rotation-invariant polar descriptor proposes old places. Candidates must
   be separated by actual traveled distance, not just elapsed keyframes. A
   wider scan-to-submap match must geometrically verify every proposed loop.
6. A verified loop remains pending until a second scan, after at least 30 cm of
   motion, independently produces a consistent global correction. Stationary
   timed keyframes cannot propose or confirm loops.
7. Robust sparse least-squares optimization distributes accepted loop error
   across the trajectory. Every front-end prediction is anchored to BNO08x
   heading relative to the start, and each keyframe carries the same heading
   as a graph factor, so weak corridor geometry cannot accumulate yaw drift.
8. Optimized keyframe scans are ray-traced into a probabilistic occupancy map.

The core is in `scripts/kiwi_slam_core.py`; `scripts/kiwi_slam.py` handles
Zenoh, clock alignment, the worker queue, pose publication, optional Rerun
visualization (camera POV, map, trajectory, and match quality), pose-correlated
image capture, and map saving. `scripts/kiwi_image_map.py` owns the image-map
wire format, spacing policy, manifest, and loop-closure reprojection.

## Install and run

Start the Zenoh router and verify odometry plus LiDAR first:

```sh
python3 -m pip install -r requirements-slam.txt
./scripts/start_zenoh.sh
python3 scripts/kiwi_lidar.py --check
python3 scripts/kiwi_slam.py --viewer --output maps/kiwi_map
```

Drive a closed path and revisit the starting area from roughly the same
location. Stop with Ctrl-C. The runner publishes corrected localization on
`kiwi/xiao/slam/pose`, a compressed live occupancy grid on
`kiwi/xiao/slam/map` for `kiwi_dashboard.py`, pose-correlated images on
`kiwi/xiao/slam/image`, and saves:

- `maps/kiwi_map.pgm`: ROS map-server-compatible occupancy image.
- `maps/kiwi_map.yaml`: resolution, origin, and occupancy thresholds.
- `maps/kiwi_map.graph.json`: readable optimized nodes and constraints.
- `maps/kiwi_map.slam.npz`: compact keyframe poses and scan points for replay.
- `maps/kiwi_map.images/<session>/manifest.json`: robot/map poses, raw odometry
  poses, acquisition times, pinhole intrinsics, and relative JPEG paths.
- `maps/kiwi_map.images/<session>/*.jpg`: upright camera frames ready for a
  later CLIP embedding/indexing pass.

## Resume a saved map

Restart against the saved graph and scan state with:

```sh
./scripts/start_zenoh.sh
python3 scripts/kiwi_slam.py --viewer --resume maps/kiwi_map
```

Leave the robot stopped near the pose where the saved run ended until the
console prints `relocalized`. The saved occupancy map is published immediately,
but pose quality reports `scan_matched: false` and `relocalizing: true` until a
live LiDAR scan is geometrically verified. No new keyframe or graph constraint
is added while verification is failing.

`--resume` accepts a prefix, `.graph.json` path, or `.slam.npz` path and needs
both the graph JSON and NPZ. It loads the saved tuning by default, preserves the
old map frame, and saves back to the same prefix on Ctrl-C unless `--output`
selects a new prefix. Version-one saved states load as a single historical
sensor session; subsequent saves use version two with per-session metadata.

If the robot was moved while SLAM was stopped, supply a known approximate pose
in the saved map frame:

```sh
python3 scripts/kiwi_slam.py --viewer --resume maps/kiwi_map \
  --resume-pose 1.25 -0.40 90
```

The default translation window is 1 m around that hint. Widen it when the
position is only approximately known, for example with
`--resume-search-distance 3`. Larger windows increase scan-matching work
quadratically.

If no pose estimate is available, `--resume-global` also tests
descriptor-selected locations throughout the map. It is slower and refuses to
choose when the best geometrically valid locations are too similar:

```sh
python3 scripts/kiwi_slam.py --viewer --resume maps/kiwi_map --resume-global
```

Every restart has a new wheel/IMU odometry origin. The first verified scan adds
an explicit `relocalization` graph constraint; it never creates an odometry edge
between the old shutdown pose and the new startup pose.

Image-map resume is automatic. SLAM searches the prefix's image manifests for
captures whose keyframe, acquisition time, raw pose, and map pose agree with the
loaded graph. It loads the largest compatible dataset, republishes its captures
after relocalization, appends new JPEGs to it, and reprojects each capture only
against keyframes from the sensor session that recorded it. An incompatible
newer manifest is ignored. Use `--resume-image-manifest PATH` to choose one
explicitly or `--no-resume-images` to start a new dataset.

The migrated manifest is committed only after LiDAR relocalization succeeds.
Existing JPEGs stay in place, the original image session ID is preserved for
the gallery's same-session safety check, and the image navigator remains
disabled until the live pose reports a verified scan match.
`--skip-session-check` is not needed for a normally resumed image map.

The pose payload includes `pose` in the global map frame, `map_to_odom`, scan
score, hit ratio, distance RMSE, keyframe count, loop count, and processing
latency. Consumers should reject or slow down autonomous motion when
`quality.scan_matched` is false for a sustained period.

The map is published after the first keyframe and every five keyframes by
default. Use `--map-publish-every N` to trade dashboard freshness against map
ray-tracing work and Zenoh bandwidth. The most recent compressed snapshot is
also retransmitted every five seconds while odometry is arriving, so a
dashboard started later can still join the live view.

Image-map recording starts automatically after the first corrected SLAM pose.
The default selector records the first available image, then another after at
least 0.5 m of translation or 30 degrees of rotation, subject to a 0.5 second
minimum interval. Camera acquisition time is aligned to the odometry clock
before interpolation. Each capture retains its raw odometry pose; after a loop
closure and again on shutdown, its map pose is recomputed relative to the
nearest optimized SLAM keyframe. The dashboard periodically receives saved
captures for late joining and displays them as upright image planes on pinhole
camera frustums in the 3D map tile.

Useful controls:

```sh
# Match the deskew timing calibration used by the dashboard.
python3 scripts/kiwi_slam.py --lidar-time-offset-ms 8 --viewer

# More detail, more compute and memory.
python3 scripts/kiwi_slam.py --map-resolution 0.03

# Denser image coverage for a future CLIP place index.
python3 scripts/kiwi_slam.py --image-distance-m 0.30 \
  --image-angle-deg 20 --image-min-interval-s 0.5

# Calibrate the 3D pinhole visualization for the physical camera.
python3 scripts/kiwi_slam.py --camera-horizontal-fov-deg 72 \
  --camera-height-m 0.10

# Run SLAM without collecting an image map.
python3 scripts/kiwi_slam.py --no-image-map

# Diagnose the front end without global loop corrections.
python3 scripts/kiwi_slam.py --no-loop-closure

# Strengthen BNO heading if scan matching still drifts in weak geometry.
python3 scripts/kiwi_slam.py --scan-yaw-prior-sigma-deg 2 \
  --heading-prior-sigma-deg 2
```

## Yaw and LiDAR calibration

Record exact robot topics while driving the calibration trajectory. Type phase
names into the recorder terminal to retain operator markers:

```sh
python3 scripts/kiwi_calibrate_slam.py record calibration/robot-run
python3 scripts/kiwi_calibrate_slam.py solve calibration/robot-run \
  --output calibration/robot.yaml
python3 scripts/kiwi_calibrate_slam.py validate calibration/robot-run \
  --calibration calibration/robot.yaml
```

The solver aggregates whole clockwise/counter-clockwise rotation segments,
then fits wheel scale and IMU scale/rate bias robustly. When stationary and
moving scans are both present it also grid-searches LiDAR time offset and
refines planar translation. The original odometry JSON and LD19 bytes remain
replayable; Rerun persistence is not required.

Use the result without changing firmware or Zenoh messages:

```sh
python3 scripts/kiwi_slam.py --yaw-estimator fused \
  --calibration calibration/robot.yaml
python3 scripts/kiwi_dashboard.py --calibration calibration/robot.yaml
```

CLI LiDAR flags override file values for controlled experiments. Keep
`--yaw-estimator legacy` as the default until the same physical recording has
been replayed through both paths and held-out map quality is neutral or better.

## Tuning order

Tune physical measurements before matcher thresholds:

1. Verify `wheel_radius_m`, drive-base radius, motor/encoder polarity, and the
   +60 degree drivetrain-to-sensor alignment.
2. Verify stationary BNO08x yaw and correct turns in the dashboard. The
   follower firmware uses the Game Rotation Vector: slow gyro drift is
   expected, but large jumps near metal are not.
3. Tune `--lidar-time-offset-ms` by driving past a straight wall and choosing
   the setting that makes the deskewed wall thinnest.
4. Confirm ordinary scans report `scan_matched`, score above 0.45, and hit
   ratio above roughly 30 percent. A sustained `scan_matched=false` means the
   LiDAR view no longer overlaps the trusted map (for example, the robot drove
   beneath furniture); those scans are deliberately quarantined.
5. Only then adjust keyframe distance, map resolution, or the thresholds in
   `SlamConfig`.

Default search assumes odometry error stays within 25 cm and 16 degrees over
one LiDAR revolution. The BNO08x supplies the initial turn estimate, but LiDAR
yaw corrections persist across scans; startup-relative IMU heading is only a
soft graph prior. This is deliberately tolerant of Game Rotation Vector drift
and of smaller magnetic corrections from older follower firmware. Increasing
the window raises runtime approximately with the number of x/y/yaw candidates.
The loop verifier uses a 75 cm translation window and runs every fifth
keyframe.

Rejected scans are never inserted into the local submap or pose graph. The
tracker dead-reckons through a temporarily occluded view (for example, under
furniture) and keeps matching against the last trusted submap until it
recovers. The local submap retains 48 keyframes so a full in-place turn cannot
evict every translated view of the room.

By default, loop candidates must be at least 2 m apart along the driven path,
and the robot must have translated at least 25 cm over the latest five
keyframes before a loop check runs. These guards prevent the one-second
maximum keyframe interval from turning stationary scans into repeated loop
closures. Two consistent verifications spanning at least 30 cm are required
before a loop constraint enters the graph.

The console and `kiwi/xiao/slam/pose` payload expose `loop_status` (for
example `no_candidate`, `geometry_rejected`, `correction_rejected`, `pending`,
or `accepted`). The saved graph JSON also records aggregate
`loop_diagnostics`, which separates candidate failure from correction gating.

The standalone dashboard has its own odometry origin, which may differ from
the SLAM process when the programs start at different times. It therefore
re-anchors its local odometry from each published map-frame robot pose; it does
not directly reuse SLAM's process-local `map_to_odom` transform.

## Validation and limits

`tests/test_kiwi_slam.py` covers SE(2) math, heading-invariant place
descriptors, scan recovery from drifting odometry, path/motion loop gating,
multi-scan loop confirmation, loop-error distribution, and map/state export.
The synthetic room scan currently processes in about
45 ms on the development arm64 Mac, below the LD19's approximately 100 ms
revolution period.

This is an implementation baseline, not a claim of benchmark parity with the
published 2DLIW-SLAM system. Kiwi supplies a fused, magnetometer-free BNO08x
Game Rotation Vector rather than raw timestamped IMU samples, so the front end
constrains scan matching with fused relative heading and wheel odometry instead
of performing raw IMU
preintegration. Loop proposals use a polar scan descriptor instead of
2DLIW-SLAM's point-line corner descriptor. No real-robot SLAM log or
ground-truth trajectory is checked into the repository yet, so map accuracy
and loop precision still require a closed-loop hardware run before autonomous
navigation should depend on this pose.
