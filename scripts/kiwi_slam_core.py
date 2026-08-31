"""Native 2D LiDAR/IMU/wheel graph-SLAM for the Kiwi robot.

The front end follows the scan-to-submap structure used by Cartographer: an
odometry prediction seeds a real-time correlative search, then a nonlinear
distance-field alignment refines the pose.  Keyframes form a pose graph with
separate wheel/IMU odometry, scan-matching, and verified loop-closure edges.

The module deliberately contains no Zenoh or viewer code so it can be tested
offline and reused by log replay tools.
"""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import ndimage, optimize, sparse


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def compose(a, b):
    """Compose SE(2) poses ``a * b``."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2(
        a.x + c * b.x - s * b.y,
        a.y + s * b.x + c * b.y,
        wrap_angle(a.yaw + b.yaw),
    )


def inverse(pose):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return Pose2(
        -c * pose.x - s * pose.y,
        s * pose.x - c * pose.y,
        wrap_angle(-pose.yaw),
    )


def between(a, b):
    """Return the pose of ``b`` expressed in ``a``."""
    return compose(inverse(a), b)


def transform_points(pose, points):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape((-1, 2))
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.array(((c, -s), (s, c)))
    return points @ rotation.T + np.array((pose.x, pose.y))


@dataclass
class SlamConfig:
    map_resolution_m: float = 0.05
    min_scan_points: int = 40
    max_scan_points: int = 480
    # Keep enough scan history that a fast in-place turn does not evict every
    # translated view of the room from the local submap. At the default
    # eight-degree rotation threshold a full turn alone can create 45 frames.
    local_keyframes: int = 48
    keyframe_translation_m: float = 0.12
    keyframe_rotation_rad: float = math.radians(8.0)
    keyframe_max_interval_s: float = 1.0
    # A time-only keyframe while stopped adds no information and eventually
    # makes map publication quadratic in the duration of a stationary run.
    keyframe_timed_min_translation_m: float = 0.02
    keyframe_timed_min_rotation_rad: float = math.radians(1.0)

    search_translation_m: float = 0.25
    # BNO08x heading predicts inter-scan rotation, so this is an error window,
    # not the maximum amount the robot may turn during one revolution.
    search_rotation_rad: float = math.radians(16.0)
    coarse_translation_step_m: float = 0.05
    coarse_rotation_step_rad: float = math.radians(2.0)
    distance_sigma_m: float = 0.08
    odom_prior_translation_sigma_m: float = 0.12
    odom_prior_rotation_sigma_rad: float = math.radians(8.0)
    # Below these values a scan is more likely to be a changed/occluded view
    # than a usable localization measurement. Keep it out of the trusted
    # submap and continue dead reckoning until overlap returns.
    min_match_score: float = 0.45
    min_hit_ratio: float = 0.28

    # Long contiguous lines are usually walls in an indoor 2D scan. Give
    # RANSAC-supported line points more influence without discarding clutter,
    # corners, or other non-wall geometry that resolves corridor degeneracy.
    wall_point_weight: float = 2.5
    wall_line_distance_m: float = 0.04
    wall_line_min_length_m: float = 0.45
    wall_line_min_points: int = 8
    wall_line_max_point_gap_m: float = 0.30
    wall_ransac_iterations: int = 40
    wall_max_lines: int = 8

    loop_check_every_keyframes: int = 5
    loop_min_separation_keyframes: int = 35
    # Keyframe separation alone is insufficient because the maximum keyframe
    # interval also creates stationary keyframes. Require actual wheel/IMU
    # translation both across the loop and between verification attempts.
    loop_min_path_separation_m: float = 2.0
    loop_min_recent_motion_m: float = 0.25
    loop_candidate_radius_m: float = 3.0
    loop_search_translation_m: float = 0.75
    loop_search_rotation_rad: float = math.radians(12.0)
    loop_min_descriptor_score: float = 0.56
    loop_min_match_score: float = 0.50
    loop_min_hit_ratio: float = 0.42
    loop_max_rmse_m: float = 0.16
    loop_max_correction_translation_m: float = 1.0
    loop_max_correction_rotation_rad: float = math.radians(35.0)
    # A single reflected scan can look geometrically convincing. Delay graph
    # insertion until the same map correction survives another moving scan.
    loop_confirmation_count: int = 2
    loop_confirmation_min_travel_m: float = 0.30
    loop_confirmation_candidate_window_keyframes: int = 12
    loop_confirmation_translation_tolerance_m: float = 0.20
    loop_confirmation_rotation_tolerance_rad: float = math.radians(8.0)
    descriptor_bins: int = 72

    odom_edge_translation_sigma_m: float = 0.08
    odom_edge_rotation_sigma_rad: float = math.radians(3.0)
    scan_edge_translation_sigma_m: float = 0.04
    scan_edge_rotation_sigma_rad: float = math.radians(2.5)
    loop_edge_translation_sigma_m: float = 0.035
    loop_edge_rotation_sigma_rad: float = math.radians(2.0)
    # Absolute (relative-to-start) BNO08x heading factor for every keyframe.
    # This prevents a run of locally plausible corridor matches from slowly
    # rotating the whole trajectory away from the fused IMU heading.
    heading_prior_rotation_sigma_rad: float = math.radians(15.0)
    # A heading prior is only useful while it agrees with the LiDAR-constrained
    # graph.  Game Rotation Vector faults can otherwise make this nominally
    # weak factor bend every wall in an otherwise excellent scan map.  Once
    # the session-relative disagreement exceeds this bound, LiDAR/scan edges
    # own the graph heading until the raw heading becomes consistent again.
    heading_prior_max_disagreement_rad: float = math.radians(12.0)
    # A magnetometer-free Game Rotation Vector is a useful relative-motion
    # seed, but it can drift; older follower firmware also reports the
    # magnetometer-fused vector. Preserve LiDAR yaw corrections across scans
    # instead of snapping every prediction back to the IMU's absolute yaw.
    absolute_imu_heading_prediction: bool = False

    # A resumed process has a new odometry origin.  Before extending the saved
    # graph, match a live scan into the old map and create an explicit session
    # anchor instead of connecting unrelated odometry coordinates.
    relocalization_search_translation_m: float = 1.0
    relocalization_search_rotation_rad: float = math.radians(45.0)
    relocalization_min_match_score: float = 0.52
    relocalization_min_hit_ratio: float = 0.34
    relocalization_max_rmse_m: float = 0.20
    relocalization_edge_translation_sigma_m: float = 0.035
    relocalization_edge_rotation_sigma_rad: float = math.radians(2.0)
    relocalization_global_candidates: int = 6
    relocalization_min_score_margin: float = 0.035


@dataclass
class MatchResult:
    pose: Pose2
    score: float
    hit_ratio: float
    rmse_m: float
    success: bool


@dataclass
class Keyframe:
    index: int
    time_s: float
    raw_pose: Pose2
    pose: Pose2
    points: np.ndarray
    descriptor: np.ndarray
    match_score: float
    travel_m: float = 0.0
    session_id: int = 0


@dataclass
class PendingLoop:
    candidate_index: int
    current_index: int
    correction: Pose2
    confirmations: int
    first_travel_m: float
    last_travel_m: float


@dataclass
class Constraint:
    i: int
    j: int
    relative_pose: Pose2
    translation_sigma_m: float
    rotation_sigma_rad: float
    kind: str
    score: float = 1.0


@dataclass
class SlamResult:
    pose: Pose2
    raw_pose: Pose2
    map_to_odom: Pose2
    match_score: float
    hit_ratio: float
    rmse_m: float
    scan_matched: bool
    keyframe_added: bool
    loop_closed: bool
    keyframes: int
    loop_closures: int
    heading_disagreement_rad: float
    processing_ms: float
    loop_status: str = "idle"
    wall_support_ratio: float = 0.0
    supported_line_length_m: float = 0.0
    wall_orientations_rad: tuple = ()


@dataclass(frozen=True)
class StructuralDiagnostics:
    support_ratio: float = 0.0
    supported_line_length_m: float = 0.0
    orientations_rad: tuple = ()


@dataclass
class OccupancyMap:
    data: np.ndarray  # int16: -1 unknown, 0 free, 100 occupied
    resolution_m: float
    origin_x: float
    origin_y: float


class DistanceField:
    """Euclidean distance-to-nearest-hit sampled in world coordinates."""

    def __init__(self, distances, resolution_m, origin_x, origin_y,
                 max_distance_m):
        self.distances = np.asarray(distances, dtype=np.float64)
        self.resolution_m = float(resolution_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.max_distance_m = float(max_distance_m)

    @classmethod
    def from_point_sets(cls, posed_point_sets, resolution_m=0.05,
                        padding_m=1.0, max_distance_m=1.0):
        world_sets = [transform_points(pose, points)
                      for pose, points in posed_point_sets if len(points)]
        if not world_sets:
            return None
        points = np.concatenate(world_sets, axis=0)
        low = np.floor((points.min(axis=0) - padding_m) / resolution_m) \
            * resolution_m
        high = np.ceil((points.max(axis=0) + padding_m) / resolution_m) \
            * resolution_m
        width, height = np.maximum(
            np.ceil((high - low) / resolution_m).astype(int) + 1, 3)
        occupied = np.zeros((height, width), dtype=bool)
        cells = np.rint((points - low) / resolution_m).astype(int)
        valid = ((cells[:, 0] >= 0) & (cells[:, 0] < width) &
                 (cells[:, 1] >= 0) & (cells[:, 1] < height))
        cells = cells[valid]
        occupied[cells[:, 1], cells[:, 0]] = True
        distances = ndimage.distance_transform_edt(~occupied) * resolution_m
        np.minimum(distances, max_distance_m, out=distances)
        return cls(distances, resolution_m, low[0], low[1], max_distance_m)

    def sample(self, world_points):
        points = np.asarray(world_points, dtype=np.float64)
        x = (points[:, 0] - self.origin_x) / self.resolution_m
        y = (points[:, 1] - self.origin_y) / self.resolution_m
        return ndimage.map_coordinates(
            self.distances,
            np.vstack((y, x)),
            order=1,
            mode="constant",
            cval=self.max_distance_m,
            prefilter=False,
        )


def _linspace_window(radius, step):
    count = max(1, int(math.ceil(2.0 * radius / step)))
    return np.linspace(-radius, radius, count + 1)


class CorrelativeScanMatcher:
    """Odometry-seeded correlative search plus nonlinear refinement."""

    def __init__(self, config):
        self.config = config
        self.last_structural_diagnostics = StructuralDiagnostics()

    def _sample_scan(self, points):
        points = np.asarray(points, dtype=np.float64)
        if len(points) <= self.config.max_scan_points:
            return points
        indices = np.linspace(0, len(points) - 1,
                              self.config.max_scan_points).astype(int)
        return points[indices]

    def _structural_weights(self, points):
        """Return robust per-point weights for long contiguous wall lines.

        RANSAC is deliberately applied only inside scan-order clusters. A
        global line fit can incorrectly join separate collinear chair or table
        edges and call the result a wall. Every point retains unit weight, so
        a non-Manhattan room falls back gracefully to ordinary scan matching.
        """
        points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
        weights = np.ones(len(points), dtype=np.float64)
        config = self.config
        if (len(points) < config.wall_line_min_points or
                config.wall_point_weight <= 1.0 or
                config.wall_ransac_iterations <= 0 or
                config.wall_max_lines <= 0):
            self.last_structural_diagnostics = StructuralDiagnostics()
            return weights

        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        boundaries = np.flatnonzero(
            gaps > config.wall_line_max_point_gap_m) + 1
        clusters = np.split(np.arange(len(points)), boundaries)
        rng = np.random.default_rng(0)
        lines_found = 0
        line_lengths = []
        orientations = []

        for cluster in clusters:
            remaining = np.asarray(cluster, dtype=np.int64)
            while (len(remaining) >= config.wall_line_min_points and
                   lines_found < config.wall_max_lines):
                best = None
                local = points[remaining]
                for _ in range(config.wall_ransac_iterations):
                    first, second = rng.choice(len(local), 2, replace=False)
                    baseline = local[second] - local[first]
                    baseline_length = float(np.linalg.norm(baseline))
                    if baseline_length < 0.5 * \
                            config.wall_line_min_length_m:
                        continue
                    normal = np.array((-baseline[1], baseline[0])) / \
                        baseline_length
                    distances = np.abs((local - local[first]) @ normal)
                    inliers = np.flatnonzero(
                        distances <= config.wall_line_distance_m)
                    if len(inliers) < config.wall_line_min_points:
                        continue

                    supported = local[inliers]
                    center = supported.mean(axis=0)
                    _u, _singular, axes = np.linalg.svd(
                        supported - center, full_matrices=False)
                    direction = axes[0]
                    refined_normal = np.array(
                        (-direction[1], direction[0]))
                    distances = np.abs((local - center) @ refined_normal)
                    inliers = np.flatnonzero(
                        distances <= config.wall_line_distance_m)
                    if len(inliers) < config.wall_line_min_points:
                        continue
                    projections = (local[inliers] - center) @ direction
                    span = float(projections.max() - projections.min())
                    if span < config.wall_line_min_length_m:
                        continue
                    score = len(inliers) * min(span, 4.0)
                    if best is None or score > best[0]:
                        orientation = math.atan2(direction[1], direction[0])
                        orientation = orientation % math.pi
                        best = (score, inliers, span, orientation)

                if best is None:
                    break
                inliers = best[1]
                wall_indices = remaining[inliers]
                weights[wall_indices] = config.wall_point_weight
                line_lengths.append(float(best[2]))
                orientations.append(float(best[3]))
                remaining = np.delete(remaining, inliers)
                lines_found += 1
            if lines_found >= config.wall_max_lines:
                break
        self.last_structural_diagnostics = StructuralDiagnostics(
            support_ratio=(float(np.mean(weights > 1.0))
                           if len(weights) else 0.0),
            supported_line_length_m=float(sum(line_lengths)),
            orientations_rad=tuple(orientations),
        )
        return weights

    def match(self, points, initial_pose, field, translation_window_m=None,
              rotation_window_rad=None, min_score=None, min_hit_ratio=None,
              use_odom_prior=True):
        points = self._sample_scan(points)
        if field is None or len(points) < self.config.min_scan_points:
            return MatchResult(initial_pose, 0.0, 0.0, math.inf, False)

        structural_weights = self._structural_weights(points)
        weight_sum = float(structural_weights.sum())

        translation_window = (self.config.search_translation_m
                              if translation_window_m is None
                              else float(translation_window_m))
        rotation_window = (self.config.search_rotation_rad
                           if rotation_window_rad is None
                           else float(rotation_window_rad))
        x_offsets = _linspace_window(
            translation_window, self.config.coarse_translation_step_m)
        yaw_offsets = _linspace_window(
            rotation_window, self.config.coarse_rotation_step_rad)

        best_score = -math.inf
        best_delta = np.zeros(3)
        initial_xy = np.array((initial_pose.x, initial_pose.y))
        for dyaw in yaw_offsets:
            yaw = initial_pose.yaw + dyaw
            c, s = math.cos(yaw), math.sin(yaw)
            rotated = points @ np.array(((c, s), (-s, c)))
            for dx in x_offsets:
                for dy in x_offsets:
                    world = rotated + initial_xy + np.array((dx, dy))
                    distances = field.sample(world)
                    likelihoods = np.exp(
                        -0.5 * np.square(distances /
                                         self.config.distance_sigma_m))
                    geometric = float(
                        np.dot(structural_weights, likelihoods) / weight_sum)
                    # A small prediction penalty resolves symmetric corridors
                    # without preventing the geometric matcher from correcting
                    # ordinary wheel slip.
                    penalty = 0.025 * (
                        (dx / max(translation_window, 1e-6)) ** 2 +
                        (dy / max(translation_window, 1e-6)) ** 2 +
                        (dyaw / max(rotation_window, 1e-6)) ** 2)
                    score = geometric - penalty
                    if score > best_score:
                        best_score = score
                        best_delta[:] = dx, dy, dyaw

        sqrt_weights = np.sqrt(structural_weights / weight_sum)

        def residual(delta):
            pose = Pose2(initial_pose.x + delta[0],
                         initial_pose.y + delta[1],
                         initial_pose.yaw + delta[2])
            distances = field.sample(transform_points(pose, points))
            values = [distances * sqrt_weights /
                      self.config.distance_sigma_m]
            if use_odom_prior:
                values.append(np.array((
                    delta[0] / self.config.odom_prior_translation_sigma_m,
                    delta[1] / self.config.odom_prior_translation_sigma_m,
                    delta[2] / self.config.odom_prior_rotation_sigma_rad,
                )))
            return np.concatenate(values)

        lower = np.array((-translation_window, -translation_window,
                          -rotation_window))
        upper = -lower
        refined = optimize.least_squares(
            residual,
            np.clip(best_delta, lower, upper),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=35,
        )
        delta = refined.x
        pose = Pose2(initial_pose.x + delta[0],
                     initial_pose.y + delta[1],
                     wrap_angle(initial_pose.yaw + delta[2]))
        distances = field.sample(transform_points(pose, points))
        likelihoods = np.exp(
            -0.5 * np.square(distances / self.config.distance_sigma_m))
        score = float(np.dot(structural_weights, likelihoods) / weight_sum)
        hits = distances < 2.0 * self.config.map_resolution_m
        hit_ratio = float(np.dot(structural_weights, hits) / weight_sum)
        rmse = float(np.sqrt(
            np.dot(structural_weights, np.square(distances)) / weight_sum))
        required_score = (self.config.min_match_score if min_score is None
                          else min_score)
        required_hits = (self.config.min_hit_ratio if min_hit_ratio is None
                         else min_hit_ratio)
        return MatchResult(pose, score, hit_ratio, rmse,
                           score >= required_score and
                           hit_ratio >= required_hits)


class PoseGraphSlam:
    """Incremental 2D pose-graph SLAM with geometrically verified loops."""

    def __init__(self, config=None):
        self.config = config or SlamConfig()
        self.matcher = CorrelativeScanMatcher(self.config)
        self.keyframes = []
        self.constraints = []
        self.last_raw_pose = None
        self.last_pose = None
        self.loop_closure_count = 0
        self._loop_pairs = set()
        self._pending_loop = None
        self.loop_diagnostics = {
            "checks": 0,
            "motion_gated": 0,
            "no_candidates": 0,
            "geometry_rejected": 0,
            "correction_rejected": 0,
            "proposals": 0,
            "accepted": 0,
        }
        self.last_loop_status = "waiting_for_separation"
        self.current_session_id = 0
        self.relocalization_required = False
        self.relocalization_global = False
        self.relocalization_hint = None
        self._relocalization_raw_origin = None
        self.runtime_metadata = {}

    @staticmethod
    def _paths(input_prefix):
        """Return graph/state paths from a prefix or either saved filename."""
        value = str(Path(input_prefix).expanduser())
        for suffix in (".slam.npz", ".graph.json", ".pgm", ".yaml"):
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                break
        prefix = Path(value)
        return (
            prefix,
            Path(f"{prefix}.graph.json"),
            Path(f"{prefix}.slam.npz"),
        )

    @staticmethod
    def _pose_from_values(values, label):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must contain three finite values")
        return Pose2(float(values[0]), float(values[1]),
                     wrap_angle(values[2]))

    @classmethod
    def load(cls, input_prefix, *, relocalization_hint=None,
             global_relocalization=False):
        """Load a saved graph and arm it for live-scan relocalization.

        Version-one files predate restartable sessions and are interpreted as
        one odometry session. Version two stores a session id per keyframe so
        heading priors never span independent follower/IMU startups.
        """
        prefix, graph_path, state_path = cls._paths(input_prefix)
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"saved SLAM graph not found: {graph_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid saved SLAM graph: {graph_path}") from exc
        if not isinstance(graph, dict):
            raise ValueError("saved SLAM graph must be a JSON object")
        graph_format = graph.get("format")
        if graph_format not in ("kiwi-pose-graph-v1", "kiwi-pose-graph-v2"):
            raise ValueError(f"unsupported saved SLAM graph format: {graph_format!r}")

        saved_config = graph.get("config", {})
        if not isinstance(saved_config, dict):
            raise ValueError("saved SLAM config must be a JSON object")
        known_config = {field.name for field in fields(SlamConfig)}
        config = SlamConfig(**{
            name: value for name, value in saved_config.items()
            if name in known_config
        })
        slam = cls(config)

        nodes = graph.get("nodes")
        constraints = graph.get("constraints")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("saved SLAM graph has no nodes")
        if not isinstance(constraints, list):
            raise ValueError("saved SLAM constraints must be a list")
        try:
            state_file = np.load(state_path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"saved SLAM state not found: {state_path}") from exc
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid saved SLAM state: {state_path}") from exc

        with state_file as state:
            required = {
                "points", "point_offsets", "poses", "raw_poses",
                "timestamps", "travel_m",
            }
            missing = sorted(required.difference(state.files))
            if missing:
                raise ValueError(
                    "saved SLAM state is missing " + ", ".join(missing))
            points = np.asarray(state["points"], dtype=np.float64)
            offsets = np.asarray(state["point_offsets"], dtype=np.int64)
            poses = np.asarray(state["poses"], dtype=np.float64)
            raw_poses = np.asarray(state["raw_poses"], dtype=np.float64)
            timestamps = np.asarray(state["timestamps"], dtype=np.float64)
            travel_m = np.asarray(state["travel_m"], dtype=np.float64)
            session_ids = (
                np.asarray(state["session_ids"], dtype=np.int64)
                if "session_ids" in state.files
                else np.zeros(len(nodes), dtype=np.int64)
            )

        count = len(nodes)
        if (points.ndim != 2 or points.shape[1:] != (2,) or
                not np.all(np.isfinite(points))):
            raise ValueError(
                "saved SLAM points must have shape (N, 2) and finite values")
        if (offsets.shape != (count + 1,) or offsets[0] != 0 or
                offsets[-1] != len(points) or np.any(np.diff(offsets) < 0)):
            raise ValueError("saved SLAM point offsets are inconsistent")
        for label, values, shape in (
            ("poses", poses, (count, 3)),
            ("raw poses", raw_poses, (count, 3)),
            ("timestamps", timestamps, (count,)),
            ("travel", travel_m, (count,)),
            ("session ids", session_ids, (count,)),
        ):
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"saved SLAM {label} must have shape {shape} and finite values")
        if np.any(session_ids < 0) or np.any(np.diff(session_ids) < 0):
            raise ValueError("saved SLAM session ids must be nonnegative and ordered")

        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or int(node.get("index", -1)) != index:
                raise ValueError("saved SLAM node indices must be contiguous")
            start, end = int(offsets[index]), int(offsets[index + 1])
            node_points = points[start:end]
            if int(node.get("points", end - start)) != end - start:
                raise ValueError(
                    f"node {index} point count disagrees with saved state")
            pose = cls._pose_from_values(poses[index], f"node {index} pose")
            raw_pose = cls._pose_from_values(
                raw_poses[index], f"node {index} raw pose")
            session_id = int(node.get("session_id", session_ids[index]))
            if session_id != int(session_ids[index]):
                raise ValueError(
                    f"node {index} session id disagrees with saved state")
            match_score = float(node.get("match_score", 1.0))
            if not math.isfinite(match_score):
                raise ValueError(f"node {index} match score must be finite")
            slam.keyframes.append(Keyframe(
                index=index,
                time_s=float(timestamps[index]),
                raw_pose=raw_pose,
                pose=pose,
                points=np.asarray(node_points, dtype=np.float32),
                descriptor=slam._descriptor(node_points),
                match_score=match_score,
                travel_m=float(travel_m[index]),
                session_id=session_id,
            ))

        for item in constraints:
            if not isinstance(item, dict):
                raise ValueError("saved SLAM constraint must be an object")
            try:
                i, j = int(item["i"]), int(item["j"])
                relative = item["relative_pose"]
                edge = Constraint(
                    i=i,
                    j=j,
                    relative_pose=Pose2(
                        float(relative["x"]), float(relative["y"]),
                        wrap_angle(float(relative["yaw"]))),
                    translation_sigma_m=float(item["translation_sigma_m"]),
                    rotation_sigma_rad=float(item["rotation_sigma_rad"]),
                    kind=str(item["kind"]),
                    score=float(item.get("score", 1.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid saved SLAM constraint") from exc
            edge_values = (
                edge.relative_pose.x, edge.relative_pose.y,
                edge.relative_pose.yaw, edge.translation_sigma_m,
                edge.rotation_sigma_rad, edge.score,
            )
            if (not 0 <= edge.i < edge.j < count or
                    not all(math.isfinite(value) for value in edge_values) or
                    edge.translation_sigma_m <= 0.0 or
                    edge.rotation_sigma_rad <= 0.0 or not edge.kind):
                raise ValueError("saved SLAM constraint is out of range")
            slam.constraints.append(edge)

        diagnostics = graph.get("loop_diagnostics")
        if isinstance(diagnostics, dict):
            for key in slam.loop_diagnostics:
                value = diagnostics.get(key)
                if isinstance(value, int) and value >= 0:
                    slam.loop_diagnostics[key] = value
        loop_edges = [edge for edge in slam.constraints
                      if edge.kind == "loop_closure"]
        slam.loop_closure_count = len(loop_edges)
        slam._loop_pairs = {(edge.i, edge.j) for edge in loop_edges}
        slam.current_session_id = int(session_ids[-1]) + 1
        slam.last_raw_pose = None
        slam.last_pose = slam.keyframes[-1].pose
        slam.relocalization_required = True
        slam.relocalization_global = bool(global_relocalization)
        slam.relocalization_hint = (
            slam.keyframes[-1].pose if relocalization_hint is None
            else cls._pose_from_values(
                (relocalization_hint.x, relocalization_hint.y,
                 relocalization_hint.yaw), "relocalization hint")
        )
        slam.last_loop_status = "relocalizing"
        slam.loaded_prefix = prefix
        return slam

    @staticmethod
    def _travel(pose):
        return math.hypot(pose.x, pose.y), abs(wrap_angle(pose.yaw))

    def _voxel_filter(self, points):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
        if not len(points):
            return points
        cells = np.floor(points / (self.config.map_resolution_m * 0.75)) \
            .astype(np.int64)
        _, indices = np.unique(cells, axis=0, return_index=True)
        return points[np.sort(indices)]

    def _descriptor(self, points):
        bins = self.config.descriptor_bins
        descriptor = np.full(bins, np.nan, dtype=np.float64)
        angles = np.arctan2(points[:, 1], points[:, 0])
        ranges = np.linalg.norm(points, axis=1)
        indices = np.floor((angles + math.pi) * bins /
                           (2.0 * math.pi)).astype(int) % bins
        for index, distance in zip(indices, ranges):
            if np.isnan(descriptor[index]) or distance < descriptor[index]:
                descriptor[index] = distance
        return descriptor

    @staticmethod
    def _descriptor_similarity(reference, query):
        best_score, best_shift = 0.0, 0
        for shift in range(len(reference)):
            rolled = np.roll(query, shift)
            valid = np.isfinite(reference) & np.isfinite(rolled)
            common = int(valid.sum())
            if common < max(10, len(reference) // 6):
                continue
            scale = np.maximum(np.maximum(reference[valid], rolled[valid]),
                               0.25)
            agreement = np.exp(-2.0 * np.abs(reference[valid] -
                                             rolled[valid]) / scale)
            coverage = min(1.0, common / (0.45 * len(reference)))
            score = float(np.mean(agreement) * coverage)
            if score > best_score:
                best_score, best_shift = score, shift
        return best_score, best_shift

    def _local_field(self, keyframes):
        return DistanceField.from_point_sets(
            [(keyframe.pose, keyframe.points) for keyframe in keyframes],
            resolution_m=self.config.map_resolution_m,
            padding_m=(self.config.search_translation_m + 0.6),
        )

    def _relocalization_neighborhood(self, anchor):
        """Select a bounded spatial submap around a relocalization anchor."""
        ranked = sorted(
            self.keyframes,
            key=lambda keyframe: (
                math.hypot(keyframe.pose.x - anchor.pose.x,
                           keyframe.pose.y - anchor.pose.y),
                abs(keyframe.index - anchor.index),
            ),
        )
        selected = ranked[:max(12, self.config.local_keyframes)]
        return sorted(selected, key=lambda keyframe: keyframe.index)

    def _relocalization_hypotheses(self, points):
        """Return (initial pose, anchor) hypotheses for one live scan."""
        hint = self.relocalization_hint or self.keyframes[-1].pose
        hint_anchor = min(
            self.keyframes,
            key=lambda keyframe: math.hypot(
                keyframe.pose.x - hint.x, keyframe.pose.y - hint.y),
        )
        hypotheses = [(hint, hint_anchor)]
        if not self.relocalization_global:
            return hypotheses

        descriptor = self._descriptor(points)
        ranked = []
        for candidate in self.keyframes:
            score, shift = self._descriptor_similarity(
                candidate.descriptor, descriptor)
            yaw_delta = shift * 2.0 * math.pi / self.config.descriptor_bins
            initial = Pose2(candidate.pose.x, candidate.pose.y,
                            wrap_angle(candidate.pose.yaw + yaw_delta))
            ranked.append((score, candidate, initial))
        ranked.sort(reverse=True, key=lambda item: item[0])
        selected = [hint_anchor]
        for _score, candidate, initial in ranked:
            if any(
                math.hypot(candidate.pose.x - other.pose.x,
                           candidate.pose.y - other.pose.y) < 0.75
                for other in selected
            ):
                continue
            hypotheses.append((initial, candidate))
            selected.append(candidate)
            if len(hypotheses) >= \
                    self.config.relocalization_global_candidates + 1:
                break
        return hypotheses

    def _relocalize(self, points, raw_pose, time_s, started):
        if self._relocalization_raw_origin is None:
            self._relocalization_raw_origin = raw_pose
        verified = []
        for initial, anchor in self._relocalization_hypotheses(points):
            neighborhood = self._relocalization_neighborhood(anchor)
            field = DistanceField.from_point_sets(
                [(keyframe.pose, keyframe.points)
                 for keyframe in neighborhood],
                resolution_m=self.config.map_resolution_m,
                padding_m=(
                    self.config.relocalization_search_translation_m + 0.6),
            )
            match = self.matcher.match(
                points,
                initial,
                field,
                translation_window_m=
                self.config.relocalization_search_translation_m,
                rotation_window_rad=
                self.config.relocalization_search_rotation_rad,
                min_score=self.config.relocalization_min_match_score,
                min_hit_ratio=self.config.relocalization_min_hit_ratio,
                use_odom_prior=False,
            )
            if match.success and \
                    match.rmse_m <= self.config.relocalization_max_rmse_m:
                verified.append((match, anchor))
        verified.sort(key=lambda item: (
            item[0].score, item[0].hit_ratio, -item[0].rmse_m),
            reverse=True)

        hint = self.relocalization_hint or self.keyframes[-1].pose
        if not verified:
            self.last_loop_status = "relocalizing:no_match"
            return self._result(
                raw_pose, hint, 0.0, 0.0, math.inf, False,
                False, False, started,
                heading_origin=(self._relocalization_raw_origin, hint),
            )
        if self.relocalization_global and len(verified) > 1:
            best, second = verified[0][0], verified[1][0]
            separation = between(best.pose, second.pose)
            separation_translation, separation_rotation = self._travel(
                separation)
            distinct_places = (
                separation_translation > 0.40 or
                separation_rotation > math.radians(15.0)
            )
            if distinct_places and best.score - second.score < \
                    self.config.relocalization_min_score_margin:
                self.last_loop_status = "relocalizing:ambiguous"
                return self._result(
                    raw_pose, best.pose, best.score, best.hit_ratio,
                    best.rmse_m, False, False, False, started,
                    heading_origin=(self._relocalization_raw_origin,
                                    best.pose),
                )

        match, anchor = verified[0]
        previous_index = len(self.keyframes)
        self._add_keyframe(
            points, raw_pose, match.pose, time_s, match.score,
            session_id=self.current_session_id,
        )
        current = self.keyframes[-1]
        self.constraints.append(Constraint(
            anchor.index,
            current.index,
            between(anchor.pose, current.pose),
            self.config.relocalization_edge_translation_sigma_m,
            self.config.relocalization_edge_rotation_sigma_rad,
            "relocalization",
            match.score,
        ))
        if current.index != previous_index:
            raise RuntimeError("relocalized keyframe index is inconsistent")
        self.last_raw_pose = raw_pose
        self.last_pose = match.pose
        self.relocalization_required = False
        self._relocalization_raw_origin = None
        self.last_loop_status = f"relocalized:{anchor.index}->{current.index}"
        return self._result(
            raw_pose, match.pose, match.score, match.hit_ratio, match.rmse_m,
            True, True, False, started)

    def process_scan(self, points, raw_pose, time_s=None):
        started = time.perf_counter()
        time_s = time.monotonic() if time_s is None else float(time_s)
        raw_pose = Pose2(float(raw_pose.x), float(raw_pose.y),
                         wrap_angle(raw_pose.yaw))
        points = self._voxel_filter(points)
        if len(points) < self.config.min_scan_points:
            raise ValueError(
                f"scan has {len(points)} usable points; "
                f"need at least {self.config.min_scan_points}")

        if self.relocalization_required:
            return self._relocalize(points, raw_pose, time_s, started)
        if self.last_loop_status.startswith("relocalized:"):
            self.last_loop_status = "idle"

        if not self.keyframes:
            self.matcher._structural_weights(points)
            pose = Pose2()
            self.last_raw_pose = raw_pose
            self.last_pose = pose
            self._add_keyframe(points, raw_pose, pose, time_s, 1.0)
            return self._result(raw_pose, pose, 1.0, 1.0, 0.0, True,
                                True, False, started)

        prediction = compose(
            self.last_pose,
            between(self.last_raw_pose, raw_pose),
        )
        if self.config.absolute_imu_heading_prediction:
            origin = self._session_origin(self.current_session_id)
            imu_heading = wrap_angle(
                origin.pose.yaw + raw_pose.yaw - origin.raw_pose.yaw)
            prediction = Pose2(prediction.x, prediction.y, imu_heading)
        local = self.keyframes[-self.config.local_keyframes:]
        match = self.matcher.match(
            points, prediction, self._local_field(local))
        pose = match.pose if match.success else prediction
        self.last_raw_pose = raw_pose
        self.last_pose = pose

        delta = between(self.keyframes[-1].pose, pose)
        translation, rotation = self._travel(delta)
        time_due = (time_s - self.keyframes[-1].time_s >=
                    self.config.keyframe_max_interval_s)
        timed_motion = (
            translation >= self.config.keyframe_timed_min_translation_m or
            rotation >= self.config.keyframe_timed_min_rotation_rad)
        keyframe_due = (
            translation >= self.config.keyframe_translation_m or
            rotation >= self.config.keyframe_rotation_rad or
            (time_due and timed_motion)
        )
        loop_closed = False
        # A rejected scan is not a measurement. Inserting it at the odometry
        # prediction would poison the local submap and create a fake
        # "scan_match" constraint that merely duplicates odometry. Continue
        # dead reckoning until a later scan can align with the last trusted
        # submap, then bridge the whole gap with one real keyframe.
        if keyframe_due and match.success:
            self._add_keyframe(points, raw_pose, pose, time_s,
                               match.score)
            loop_closed = self._try_loop_closure()
            if loop_closed:
                self.optimize()
                pose = self.keyframes[-1].pose
                self.last_pose = pose

        return self._result(
            raw_pose, pose, match.score, match.hit_ratio, match.rmse_m,
            match.success, keyframe_due and match.success,
            loop_closed, started)

    def _session_origin(self, session_id):
        for keyframe in self.keyframes:
            if keyframe.session_id == session_id:
                return keyframe
        return None

    def _result(self, raw_pose, pose, score, hit_ratio, rmse, matched,
                added, loop_closed, started, heading_origin=None):
        origin = self._session_origin(self.current_session_id)
        if heading_origin is None:
            if origin is None:
                heading_origin = (raw_pose, pose)
            else:
                heading_origin = (origin.raw_pose, origin.pose)
        origin_raw_pose, origin_map_pose = heading_origin
        expected_heading = wrap_angle(
            origin_map_pose.yaw + raw_pose.yaw - origin_raw_pose.yaw)
        heading_disagreement = wrap_angle(pose.yaw - expected_heading)
        structural = self.matcher.last_structural_diagnostics
        return SlamResult(
            pose=pose,
            raw_pose=raw_pose,
            map_to_odom=compose(pose, inverse(raw_pose)),
            match_score=float(score),
            hit_ratio=float(hit_ratio),
            rmse_m=float(rmse),
            scan_matched=bool(matched),
            keyframe_added=bool(added),
            loop_closed=bool(loop_closed),
            keyframes=len(self.keyframes),
            loop_closures=self.loop_closure_count,
            heading_disagreement_rad=heading_disagreement,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            loop_status=self.last_loop_status,
            wall_support_ratio=structural.support_ratio,
            supported_line_length_m=structural.supported_line_length_m,
            wall_orientations_rad=structural.orientations_rad,
        )

    def _add_keyframe(self, points, raw_pose, pose, time_s, match_score,
                      session_id=None):
        index = len(self.keyframes)
        previous = self.keyframes[-1] if self.keyframes else None
        session_id = (self.current_session_id if session_id is None
                      else int(session_id))
        same_session = (
            previous is not None and previous.session_id == session_id)
        odom_delta = (between(previous.raw_pose, raw_pose)
                      if same_session else Pose2())
        travel_m = ((previous.travel_m if previous is not None else 0.0) +
                    math.hypot(odom_delta.x, odom_delta.y))
        keyframe = Keyframe(
            index=index,
            time_s=time_s,
            raw_pose=raw_pose,
            pose=pose,
            points=np.asarray(points, dtype=np.float32),
            descriptor=self._descriptor(points),
            match_score=float(match_score),
            travel_m=travel_m,
            session_id=session_id,
        )
        if same_session:
            scan_delta = between(previous.pose, pose)
            self.constraints.append(Constraint(
                previous.index, index, odom_delta,
                self.config.odom_edge_translation_sigma_m,
                self.config.odom_edge_rotation_sigma_rad,
                "odometry",
            ))
            score_scale = 1.0 + max(0.0, 1.0 - match_score)
            self.constraints.append(Constraint(
                previous.index, index, scan_delta,
                self.config.scan_edge_translation_sigma_m * score_scale,
                self.config.scan_edge_rotation_sigma_rad * score_scale,
                "scan_match", match_score,
            ))
        self.keyframes.append(keyframe)

    def _loop_candidates(self, current):
        maximum = current.index - self.config.loop_min_separation_keyframes
        candidates = []
        for candidate in self.keyframes[:max(0, maximum + 1)]:
            score, shift = self._descriptor_similarity(
                candidate.descriptor, current.descriptor)
            distance = math.hypot(candidate.pose.x - current.pose.x,
                                  candidate.pose.y - current.pose.y)
            path_separation = current.travel_m - candidate.travel_m
            if (score >= self.config.loop_min_descriptor_score and
                    path_separation >=
                    self.config.loop_min_path_separation_m and
                    distance <= self.config.loop_candidate_radius_m):
                candidates.append((score, -distance, shift, candidate))
        candidates.sort(reverse=True, key=lambda item: item[:2])
        # Adjacent keyframes describe the same place. Verifying the best few
        # distinct candidates is enough and bounds callback latency.
        selected = []
        for item in candidates:
            if all(abs(item[3].index - other[3].index) > 8
                   for other in selected):
                selected.append(item)
            if len(selected) == 3:
                break
        return selected

    def _recent_loop_motion(self, current):
        previous_check = max(
            0, current.index - self.config.loop_check_every_keyframes)
        return current.travel_m - self.keyframes[previous_check].travel_m

    def _same_loop_hypothesis(self, candidate, current, correction):
        pending = self._pending_loop
        if pending is None or current.index <= pending.current_index:
            return False
        if (abs(candidate.index - pending.candidate_index) >
                self.config.loop_confirmation_candidate_window_keyframes):
            return False
        disagreement = between(pending.correction, correction)
        translation, rotation = self._travel(disagreement)
        return (
            translation <=
            self.config.loop_confirmation_translation_tolerance_m and
            rotation <= self.config.loop_confirmation_rotation_tolerance_rad
        )

    def _accept_loop(self, candidate, current, match, descriptor_score, pair):
        self.constraints.append(Constraint(
            candidate.index,
            current.index,
            between(candidate.pose, match.pose),
            self.config.loop_edge_translation_sigma_m,
            self.config.loop_edge_rotation_sigma_rad,
            "loop_closure",
            min(descriptor_score, match.score),
        ))
        self._loop_pairs.add(pair)
        self.loop_closure_count += 1
        self._pending_loop = None
        self.loop_diagnostics["accepted"] += 1
        self.last_loop_status = (
            f"accepted:{candidate.index}->{current.index}")
        return True

    def _try_loop_closure(self):
        current = self.keyframes[-1]
        if current.index < self.config.loop_min_separation_keyframes:
            self.last_loop_status = "waiting_for_separation"
            return False
        if current.index % self.config.loop_check_every_keyframes:
            return False
        self.loop_diagnostics["checks"] += 1
        if (self._recent_loop_motion(current) <
                self.config.loop_min_recent_motion_m):
            self.loop_diagnostics["motion_gated"] += 1
            self.last_loop_status = "waiting_for_motion"
            return False
        proposals = []
        candidates = self._loop_candidates(current)
        if not candidates:
            self.loop_diagnostics["no_candidates"] += 1
            self.last_loop_status = "no_candidate"
        geometry_rejected = False
        correction_rejected = False
        for descriptor_score, _neg_distance, shift, candidate in candidates:
            pair = (candidate.index, current.index)
            if pair in self._loop_pairs:
                continue
            neighborhood = self.keyframes[
                max(0, candidate.index - 5):candidate.index + 6]
            field = self._local_field(neighborhood)
            yaw_delta = shift * 2.0 * math.pi / self.config.descriptor_bins
            descriptor_yaw = wrap_angle(candidate.pose.yaw + yaw_delta)
            yaw_hypotheses = [descriptor_yaw]
            if abs(wrap_angle(current.pose.yaw - descriptor_yaw)) > \
                    self.config.coarse_rotation_step_rad:
                # Rotation-invariant descriptors are often 90/180-degree
                # ambiguous in Manhattan rooms. The magnetometer-free fused
                # heading supplies an independent branch hypothesis.
                yaw_hypotheses.append(current.pose.yaw)

            verified = []
            for initial_yaw in yaw_hypotheses:
                initial = Pose2(candidate.pose.x, candidate.pose.y,
                                initial_yaw)
                match = self.matcher.match(
                    current.points,
                    initial,
                    field,
                    translation_window_m=
                    self.config.loop_search_translation_m,
                    rotation_window_rad=self.config.loop_search_rotation_rad,
                    min_score=self.config.loop_min_match_score,
                    min_hit_ratio=self.config.loop_min_hit_ratio,
                    use_odom_prior=False,
                )
                if (not match.success or
                        match.rmse_m > self.config.loop_max_rmse_m):
                    geometry_rejected = True
                    continue
                correction = between(current.pose, match.pose)
                correction_translation, correction_rotation = \
                    self._travel(correction)
                # A 360-degree planar scan can be geometrically convincing at
                # a symmetric but incorrect place. Reject a correction that
                # conflicts strongly with the fused relative heading.
                if (correction_translation >
                        self.config.loop_max_correction_translation_m or
                        correction_rotation >
                        self.config.loop_max_correction_rotation_rad):
                    correction_rejected = True
                    continue
                verified.append(match)

            if not verified:
                continue
            match = max(verified, key=lambda item: (
                item.score, item.hit_ratio, -item.rmse_m))
            # Compare left-multiplicative map corrections, which remain in one
            # global frame as the robot changes position and heading.
            global_correction = compose(match.pose, inverse(current.pose))
            proposals.append((candidate, match, descriptor_score, pair,
                              global_correction))
            self.loop_diagnostics["proposals"] += 1
            if not self._same_loop_hypothesis(
                    candidate, current, global_correction):
                continue
            pending = self._pending_loop
            pending.candidate_index = candidate.index
            pending.current_index = current.index
            pending.correction = global_correction
            pending.confirmations += 1
            pending.last_travel_m = current.travel_m
            enough_confirmations = (
                pending.confirmations >= self.config.loop_confirmation_count)
            enough_travel = (
                current.travel_m - pending.first_travel_m >=
                self.config.loop_confirmation_min_travel_m)
            if enough_confirmations and enough_travel:
                return self._accept_loop(
                    candidate, current, match, descriptor_score, pair)
            self.last_loop_status = (
                f"pending:{pending.confirmations}/"
                f"{self.config.loop_confirmation_count}")
            return False

        if proposals:
            candidate, _match, _score, _pair, correction = proposals[0]
            if self.config.loop_confirmation_count <= 1:
                candidate, match, descriptor_score, pair, _correction = \
                    proposals[0]
                return self._accept_loop(
                    candidate, current, match, descriptor_score, pair)
            self._pending_loop = PendingLoop(
                candidate_index=candidate.index,
                current_index=current.index,
                correction=correction,
                confirmations=1,
                first_travel_m=current.travel_m,
                last_travel_m=current.travel_m,
            )
            self.last_loop_status = (
                f"pending:1/{self.config.loop_confirmation_count}")
        else:
            # Confirmation must survive consecutive moving checks; an isolated
            # reflected match must not remain armed indefinitely.
            self._pending_loop = None
            if candidates:
                if correction_rejected:
                    self.loop_diagnostics["correction_rejected"] += 1
                    self.last_loop_status = "correction_rejected"
                elif geometry_rejected:
                    self.loop_diagnostics["geometry_rejected"] += 1
                    self.last_loop_status = "geometry_rejected"
        return False

    @staticmethod
    def _edge_error(measured, predicted):
        error = between(measured, predicted)
        return np.array((error.x, error.y, wrap_angle(error.yaw)))

    def optimize(self, max_nfev=60):
        """Robustly optimize all graph poses while holding the origin fixed."""
        if len(self.keyframes) < 2 or not self.constraints:
            return
        initial = np.array([
            value
            for keyframe in self.keyframes[1:]
            for value in (keyframe.pose.x, keyframe.pose.y, keyframe.pose.yaw)
        ], dtype=np.float64)

        def unpack(values):
            poses = [self.keyframes[0].pose]
            poses.extend(Pose2(values[i], values[i + 1],
                               wrap_angle(values[i + 2]))
                         for i in range(0, len(values), 3))
            return poses

        session_origins = {}
        for index, keyframe in enumerate(self.keyframes):
            session_origins.setdefault(keyframe.session_id, index)
        heading_pairs = []
        if self.config.heading_prior_rotation_sigma_rad > 0.0:
            maximum = self.config.heading_prior_max_disagreement_rad
            for index, keyframe in enumerate(self.keyframes):
                origin_index = session_origins[keyframe.session_id]
                if index == origin_index:
                    continue
                origin = self.keyframes[origin_index]
                measured = wrap_angle(
                    keyframe.raw_pose.yaw - origin.raw_pose.yaw)
                lidar_relative = wrap_angle(
                    keyframe.pose.yaw - origin.pose.yaw)
                if (maximum > 0.0 and
                        abs(wrap_angle(lidar_relative - measured)) > maximum):
                    continue
                heading_pairs.append((index, origin_index))

        def residual(values):
            poses = unpack(values)
            errors = []
            for edge in self.constraints:
                predicted = between(poses[edge.i], poses[edge.j])
                error = self._edge_error(edge.relative_pose, predicted)
                errors.extend((
                    error[0] / edge.translation_sigma_m,
                    error[1] / edge.translation_sigma_m,
                    error[2] / edge.rotation_sigma_rad,
                ))
            heading_sigma = self.config.heading_prior_rotation_sigma_rad
            if heading_sigma > 0.0:
                for index, origin_index in heading_pairs:
                    keyframe = self.keyframes[index]
                    origin = self.keyframes[origin_index]
                    measured = wrap_angle(keyframe.raw_pose.yaw -
                                          origin.raw_pose.yaw)
                    predicted = wrap_angle(
                        poses[index].yaw - poses[origin_index].yaw)
                    errors.append(wrap_angle(predicted - measured) /
                                  heading_sigma)
            return np.asarray(errors)

        heading_factors = len(heading_pairs)
        jacobian = sparse.lil_matrix(
            (3 * len(self.constraints) + heading_factors, len(initial)),
            dtype=np.int8)
        for row, edge in enumerate(self.constraints):
            for node in (edge.i, edge.j):
                if node:
                    column = 3 * (node - 1)
                    jacobian[3 * row:3 * row + 3,
                             column:column + 3] = 1
        heading_row = 3 * len(self.constraints)
        for offset, (node, origin_node) in enumerate(heading_pairs):
            row = heading_row + offset
            if node:
                jacobian[row, 3 * (node - 1) + 2] = 1
            if origin_node:
                jacobian[row, 3 * (origin_node - 1) + 2] = 1
        solved = optimize.least_squares(
            residual,
            initial,
            jac_sparsity=jacobian.tocsr(),
            loss="huber",
            f_scale=2.0,
            max_nfev=max_nfev,
        )
        for keyframe, pose in zip(self.keyframes, unpack(solved.x)):
            keyframe.pose = pose
        self.last_pose = self.keyframes[-1].pose

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Integer grid cells along a ray, including both endpoints."""
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                return cells
            twice = 2 * error
            if twice > -dy:
                error -= dy
                x0 += sx
            if twice < dx:
                error += dx
                y0 += sy

    def build_occupancy_map(self, resolution_m=None, padding_m=0.5,
                            max_beams_per_scan=360):
        if not self.keyframes:
            raise RuntimeError("cannot build a map without keyframes")
        resolution = (self.config.map_resolution_m if resolution_m is None
                      else float(resolution_m))
        world_sets = [transform_points(k.pose, k.points)
                      for k in self.keyframes]
        all_points = np.concatenate(
            world_sets + [np.array([[k.pose.x, k.pose.y]])
                          for k in self.keyframes], axis=0)
        low = np.floor((all_points.min(axis=0) - padding_m) / resolution) \
            * resolution
        high = np.ceil((all_points.max(axis=0) + padding_m) / resolution) \
            * resolution
        width, height = np.maximum(
            np.ceil((high - low) / resolution).astype(int) + 1, 3)
        log_odds = np.zeros((height, width), dtype=np.float32)
        observed = np.zeros((height, width), dtype=bool)

        for keyframe, world_points in zip(self.keyframes, world_sets):
            start = np.rint((np.array((keyframe.pose.x, keyframe.pose.y)) -
                             low) / resolution).astype(int)
            stride = max(1, int(math.ceil(len(world_points) /
                                          max_beams_per_scan)))
            endpoints = np.rint((world_points[::stride] - low) /
                                resolution).astype(int)
            for end in endpoints:
                cells = self._bresenham(start[0], start[1], end[0], end[1])
                free = cells[:-1]
                if free:
                    xs, ys = zip(*free)
                    log_odds[ys, xs] -= 0.35
                    observed[ys, xs] = True
                x, y = cells[-1]
                if 0 <= x < width and 0 <= y < height:
                    log_odds[y, x] += 0.90
                    observed[y, x] = True
        np.clip(log_odds, -4.0, 4.0, out=log_odds)
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        data = np.rint(probability * 100.0).astype(np.int16)
        data[~observed] = -1
        return OccupancyMap(data, resolution, float(low[0]), float(low[1]))

    def save(self, output_prefix):
        """Save ROS-compatible PGM/YAML plus a replayable graph state NPZ."""
        prefix, _graph_path, _state_path = self._paths(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        occupancy = self.build_occupancy_map()
        pgm_path = Path(f"{prefix}.pgm")
        yaml_path = Path(f"{prefix}.yaml")
        graph_path = Path(f"{prefix}.graph.json")
        state_path = Path(f"{prefix}.slam.npz")

        image = np.full(occupancy.data.shape, 205, dtype=np.uint8)
        known = occupancy.data >= 0
        image[known] = np.rint(
            254.0 * (1.0 - occupancy.data[known] / 100.0)).astype(np.uint8)
        image = np.flipud(image)
        header = (f"P5\n{image.shape[1]} {image.shape[0]}\n255\n").encode()
        pgm_path.write_bytes(header + image.tobytes())
        yaml_path.write_text(
            f"image: {pgm_path.name}\n"
            f"resolution: {occupancy.resolution_m:.9g}\n"
            f"origin: [{occupancy.origin_x:.9g}, {occupancy.origin_y:.9g}, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n",
            encoding="utf-8",
        )
        graph = {
            "format": "kiwi-pose-graph-v2",
            "config": asdict(self.config),
            "runtime": dict(self.runtime_metadata),
            "loop_diagnostics": dict(self.loop_diagnostics),
            "nodes": [{
                "index": k.index,
                "time_s": k.time_s,
                "pose": asdict(k.pose),
                "raw_pose": asdict(k.raw_pose),
                "match_score": k.match_score,
                "travel_m": k.travel_m,
                "session_id": k.session_id,
                "points": len(k.points),
            } for k in self.keyframes],
            "constraints": [{
                **asdict(edge),
                "relative_pose": asdict(edge.relative_pose),
            } for edge in self.constraints],
        }
        graph_path.write_text(json.dumps(graph, indent=2) + "\n",
                              encoding="utf-8")
        offsets = np.zeros(len(self.keyframes) + 1, dtype=np.int64)
        for i, keyframe in enumerate(self.keyframes):
            offsets[i + 1] = offsets[i] + len(keyframe.points)
        points = np.concatenate([k.points for k in self.keyframes], axis=0)
        np.savez_compressed(
            state_path,
            points=points,
            point_offsets=offsets,
            poses=np.array([[k.pose.x, k.pose.y, k.pose.yaw]
                            for k in self.keyframes]),
            raw_poses=np.array([[k.raw_pose.x, k.raw_pose.y, k.raw_pose.yaw]
                                for k in self.keyframes]),
            timestamps=np.array([k.time_s for k in self.keyframes]),
            travel_m=np.array([k.travel_m for k in self.keyframes]),
            session_ids=np.array(
                [k.session_id for k in self.keyframes], dtype=np.int64),
        )
        return {
            "map": pgm_path,
            "metadata": yaml_path,
            "graph": graph_path,
            "state": state_path,
        }
