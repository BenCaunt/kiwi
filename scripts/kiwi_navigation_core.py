#!/usr/bin/env python3
"""Grid planning and trajectory following primitives for Kiwi navigation."""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from kiwi_pose_controller import Pose2, PoseStabilizingController, Twist2


DEFAULT_MAX_FOLLOWING_SPEED_MPS = 0.12
DEFAULT_RUNTIME_COLLISION_RADIUS_M = 0.18
DEFAULT_LIDAR_COLLISION_HORIZON_S = 0.80


class PathNotFound(RuntimeError):
    """Raised when a requested map-frame path cannot be planned safely."""


class WheelDistanceTracker:
    """Integrate encoder-reported planar speed without counting SLAM jumps."""

    _CLOCK_WRAP_US = 1 << 32

    def __init__(self, max_interval_s=0.5):
        self.max_interval_s = float(max_interval_s)
        self.distance_m = 0.0
        self._last_time_s = None
        self._last_clock_us = None
        self._clock_epoch_us = 0

    def update(self, report):
        raw_us = report.get("follower_time_us")
        if not isinstance(raw_us, (int, float)) or not math.isfinite(raw_us):
            return self.distance_m
        raw_us = int(raw_us) % self._CLOCK_WRAP_US
        if (self._last_clock_us is not None and
                raw_us < self._last_clock_us - self._CLOCK_WRAP_US // 2):
            self._clock_epoch_us += self._CLOCK_WRAP_US
        self._last_clock_us = raw_us
        time_s = (self._clock_epoch_us + raw_us) / 1_000_000.0
        measured = report.get("measured", {})
        try:
            speed = math.hypot(
                float(measured.get("vx", 0.0)),
                float(measured.get("vy", 0.0)),
            )
        except (AttributeError, TypeError, ValueError):
            speed = math.nan
        if self._last_time_s is not None:
            dt = time_s - self._last_time_s
            if (0.0 < dt <= self.max_interval_s and math.isfinite(speed)):
                self.distance_m += speed * dt
        self._last_time_s = time_s
        return self.distance_m


class SweptCircleCollisionGuard:
    """Check live body-frame LiDAR points against a translating robot disk."""

    def __init__(self, radius_m=DEFAULT_RUNTIME_COLLISION_RADIUS_M,
                 horizon_s=DEFAULT_LIDAR_COLLISION_HORIZON_S):
        self.radius_m = float(radius_m)
        self.horizon_s = float(horizon_s)
        if (not math.isfinite(self.radius_m) or self.radius_m < 0.0 or
                not math.isfinite(self.horizon_s) or self.horizon_s <= 0.0):
            raise ValueError("collision radius must be nonnegative and horizon positive")

    def time_to_collision(self, points, vx, vy):
        """Return the earliest disk/point collision time, or infinity."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,):
            raise ValueError("LiDAR points must have shape (N, 2)")
        if len(points) == 0:
            return math.inf
        velocity = np.array((float(vx), float(vy)), dtype=float)
        if not np.isfinite(points).all() or not np.isfinite(velocity).all():
            raise ValueError("LiDAR points and velocity must be finite")
        speed_squared = float(np.dot(velocity, velocity))
        if speed_squared <= 1e-12:
            return math.inf
        toward = points @ velocity
        distance_squared = np.einsum("ij,ij->i", points, points)
        radius_squared = self.radius_m ** 2
        # When already inside the conservative radius, only motion farther
        # into the obstacle is blocked; tangent/retreat motion remains usable.
        inside_closing = (distance_squared <= radius_squared) & (toward > 0.0)
        if np.any(inside_closing):
            return 0.0
        discriminant = toward * toward - speed_squared * (
            distance_squared - radius_squared)
        candidates = (toward > 0.0) & (discriminant >= 0.0) & \
            (distance_squared > radius_squared)
        if not np.any(candidates):
            return math.inf
        roots = (toward[candidates] - np.sqrt(discriminant[candidates])) / \
            speed_squared
        roots = roots[roots >= 0.0]
        return float(np.min(roots)) if len(roots) else math.inf

    def blocks(self, points, vx, vy):
        return self.time_to_collision(points, vx, vy) <= self.horizon_s


def stamp_lidar_obstacles(occupancy, pose, body_points, max_range_m=3.0):
    """Return a map copy with current LiDAR endpoints marked occupied."""
    points = np.asarray(body_points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("LiDAR points must have shape (N, 2)")
    if len(points) == 0:
        return occupancy
    ranges = np.linalg.norm(points, axis=1)
    points = points[(ranges > 0.02) & (ranges <= float(max_range_m))]
    if not len(points):
        return occupancy
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    world_x = pose.x + c * points[:, 0] - s * points[:, 1]
    world_y = pose.y + s * points[:, 0] + c * points[:, 1]
    columns = np.floor(
        (world_x - occupancy.origin_x) / occupancy.resolution_m + 0.5
    ).astype(int)
    rows = np.floor(
        (world_y - occupancy.origin_y) / occupancy.resolution_m + 0.5
    ).astype(int)
    valid = ((rows >= 0) & (rows < occupancy.data.shape[0]) &
             (columns >= 0) & (columns < occupancy.data.shape[1]))
    data = np.asarray(occupancy.data).copy()
    data[rows[valid], columns[valid]] = 100
    return type(occupancy)(
        data=data,
        resolution_m=occupancy.resolution_m,
        origin_x=occupancy.origin_x,
        origin_y=occupancy.origin_y,
        keyframes=occupancy.keyframes,
    )


def inflate_obstacles(obstacles, radius_m, resolution_m):
    """Inflate a boolean obstacle grid by a circular metric radius."""
    source = np.asarray(obstacles, dtype=bool)
    radius_m = float(radius_m)
    resolution_m = float(resolution_m)
    if source.ndim != 2 or 0 in source.shape:
        raise ValueError("obstacles must be a non-empty 2D array")
    if (not math.isfinite(radius_m) or not math.isfinite(resolution_m) or
            radius_m < 0.0 or resolution_m <= 0.0):
        raise ValueError("inflation radius must be non-negative and resolution positive")
    if radius_m == 0.0 or not source.any():
        return source.copy()

    radius_cells = int(math.ceil(radius_m / resolution_m))
    padded = np.pad(source, radius_cells, mode="constant", constant_values=False)
    inflated = np.zeros_like(source)
    height, width = source.shape
    epsilon = resolution_m * 1e-9
    for row_offset in range(-radius_cells, radius_cells + 1):
        for column_offset in range(-radius_cells, radius_cells + 1):
            distance = math.hypot(row_offset, column_offset) * resolution_m
            if distance > radius_m + epsilon:
                continue
            row_start = radius_cells + row_offset
            column_start = radius_cells + column_offset
            inflated |= padded[
                row_start:row_start + height,
                column_start:column_start + width,
            ]
    return inflated


class AStarPlanner:
    """Eight-connected A* over a live SLAM occupancy map."""

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(
        self,
        occupancy,
        inflation_radius_m=0.15,
        occupied_threshold=65,
        allow_unknown=False,
    ):
        self.data = np.asarray(occupancy.data)
        self.resolution_m = float(occupancy.resolution_m)
        self.origin_x = float(occupancy.origin_x)
        self.origin_y = float(occupancy.origin_y)
        self.inflation_radius_m = float(inflation_radius_m)
        self.occupied_threshold = int(occupied_threshold)
        self.allow_unknown = bool(allow_unknown)
        if self.data.ndim != 2 or 0 in self.data.shape:
            raise ValueError("occupancy map must be a non-empty 2D array")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("occupancy resolution must be positive")
        if not all(math.isfinite(value)
                   for value in (self.origin_x, self.origin_y)):
            raise ValueError("occupancy origin must be finite")
        if (not math.isfinite(self.inflation_radius_m) or
                self.inflation_radius_m < 0.0):
            raise ValueError("inflation radius must be non-negative")
        if not (0 <= self.occupied_threshold <= 100):
            raise ValueError("occupied threshold must be in [0, 100]")

        obstacles = self.data >= self.occupied_threshold
        if not self.allow_unknown:
            obstacles |= self.data < 0
        self.obstacles = obstacles
        self.blocked = inflate_obstacles(
            obstacles, self.inflation_radius_m, self.resolution_m)

    def world_to_cell(self, point):
        """Return ``(row, column)`` for a map-frame ``(x, y)`` point."""
        x, y = float(point[0]), float(point[1])
        column = int(math.floor(
            (x - self.origin_x) / self.resolution_m + 0.5))
        row = int(math.floor(
            (y - self.origin_y) / self.resolution_m + 0.5))
        return row, column

    def cell_to_world(self, cell):
        row, column = cell
        return np.array((
            self.origin_x + column * self.resolution_m,
            self.origin_y + row * self.resolution_m,
        ), dtype=float)

    def cell_is_free(self, cell):
        row, column = cell
        return (
            0 <= row < self.blocked.shape[0]
            and 0 <= column < self.blocked.shape[1]
            and not self.blocked[row, column]
        )

    @staticmethod
    def _heuristic(cell, goal):
        """Octile distance, matching the eight-connected movement costs."""
        dy = abs(cell[0] - goal[0])
        dx = abs(cell[1] - goal[1])
        diagonal = min(dx, dy)
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * diagonal

    def _neighbors(self, cell):
        row, column = cell
        for dy, dx, cost in self._NEIGHBORS:
            neighbor = row + dy, column + dx
            if not self.cell_is_free(neighbor):
                continue
            if dy and dx:
                # A diagonal cannot squeeze between two inflated obstacles.
                if (not self.cell_is_free((row + dy, column)) or
                        not self.cell_is_free((row, column + dx))):
                    continue
            yield neighbor, cost

    @staticmethod
    def _compress_cells(cells):
        if len(cells) <= 2:
            return cells
        compressed = [cells[0]]
        previous_direction = None
        for index in range(1, len(cells)):
            direction = (
                cells[index][0] - cells[index - 1][0],
                cells[index][1] - cells[index - 1][1],
            )
            if previous_direction is not None and direction != previous_direction:
                compressed.append(cells[index - 1])
            previous_direction = direction
        compressed.append(cells[-1])
        return compressed

    def plan(self, start_xy, goal_xy):
        """Plan a collision-free map-frame polyline from start to goal."""
        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        if (start_xy.shape != (2,) or goal_xy.shape != (2,) or
                not np.isfinite(start_xy).all() or not np.isfinite(goal_xy).all()):
            raise ValueError("start and goal must be finite 2D points")
        start = self.world_to_cell(start_xy)
        goal = self.world_to_cell(goal_xy)
        if not self.cell_is_free(start):
            raise PathNotFound("start is outside the map or inside the inflated obstacle map")
        if not self.cell_is_free(goal):
            raise PathNotFound("goal is outside the map or inside the inflated obstacle map")

        if start == goal:
            if np.linalg.norm(goal_xy - start_xy) <= 1e-12:
                return np.array([start_xy], dtype=float)
            return np.vstack((start_xy, goal_xy))

        frontier = []
        sequence = 0
        heapq.heappush(frontier, (self._heuristic(start, goal), 0.0, sequence, start))
        came_from = {}
        cost_to = {start: 0.0}
        reached = False
        while frontier:
            _priority, queued_cost, _sequence, current = heapq.heappop(frontier)
            if queued_cost > cost_to.get(current, math.inf) + 1e-12:
                continue
            if current == goal:
                reached = True
                break
            for neighbor, step_cost in self._neighbors(current):
                candidate = queued_cost + step_cost
                if candidate + 1e-12 >= cost_to.get(neighbor, math.inf):
                    continue
                cost_to[neighbor] = candidate
                came_from[neighbor] = current
                sequence += 1
                priority = candidate + self._heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, candidate, sequence, neighbor))
        if not reached:
            raise PathNotFound("no collision-free path connects start and goal")

        cells = [goal]
        while cells[-1] != start:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        cell_points = [self.cell_to_world(cell)
                       for cell in self._compress_cells(cells)]
        # Keep the start/goal cell centers as waypoints when the exact pose is
        # off-center. This guarantees the short endpoint segments stay inside
        # their already-validated free cells.
        points = [start_xy]
        for point in cell_points:
            if np.linalg.norm(point - points[-1]) > 1e-12:
                points.append(point)
        if np.linalg.norm(goal_xy - points[-1]) > 1e-12:
            points.append(goal_xy)
        return np.vstack(points)

    def plan_with_start_recovery(self, start_xy, goal_xy, recovery_planner):
        """Plan out of a soft buffer without weakening the rest of the route.

        ``self`` is the preferred-clearance planner and ``recovery_planner``
        is the hard collision envelope.  A robot that is hard-safe but starts
        inside only the preferred buffer may cross that soft-buffer region
        until it reaches preferred clearance.  Once it does, the search cannot
        re-enter the soft buffer.
        """
        if not isinstance(recovery_planner, AStarPlanner):
            raise TypeError("recovery_planner must be an AStarPlanner")
        if recovery_planner.inflation_radius_m > \
                self.inflation_radius_m + 1e-12:
            raise ValueError(
                "recovery inflation cannot exceed preferred inflation")
        if (recovery_planner.data.shape != self.data.shape or
                recovery_planner.resolution_m != self.resolution_m or
                recovery_planner.origin_x != self.origin_x or
                recovery_planner.origin_y != self.origin_y or
                not np.array_equal(recovery_planner.obstacles, self.obstacles)):
            raise ValueError(
                "preferred and recovery planners must use the same map")

        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        if (start_xy.shape != (2,) or goal_xy.shape != (2,) or
                not np.isfinite(start_xy).all() or
                not np.isfinite(goal_xy).all()):
            raise ValueError("start and goal must be finite 2D points")
        start = self.world_to_cell(start_xy)
        goal = self.world_to_cell(goal_xy)
        if self.cell_is_free(start):
            return self.plan(start_xy, goal_xy)
        if not recovery_planner.cell_is_free(start):
            raise PathNotFound(
                "start is outside the map or inside the hard collision envelope")
        if not self.cell_is_free(goal):
            raise PathNotFound(
                "goal is outside the map or inside the inflated obstacle map")

        # Prefer the shortest exposure to the soft-buffer region before route
        # length.  Twice the cell count is larger than any simple eight-way
        # grid path, making one additional soft cell more expensive than a
        # hard-safe detour across the map.
        soft_cell_penalty = 2.0 * self.blocked.size + 1.0
        frontier = []
        sequence = 0
        heapq.heappush(
            frontier, (self._heuristic(start, goal), 0.0, sequence, start))
        came_from = {}
        cost_to = {start: 0.0}
        reached = False
        while frontier:
            _priority, queued_cost, _sequence, current = heapq.heappop(frontier)
            if queued_cost > cost_to.get(current, math.inf) + 1e-12:
                continue
            if current == goal:
                reached = True
                break
            # Before egress, use the hard-clearance grid.  After the first
            # preferred-clearance cell, use preferred neighbors so the path
            # cannot dip back into the soft buffer or cut its corners.
            neighbor_source = (
                self if self.cell_is_free(current) else recovery_planner)
            for neighbor, step_cost in neighbor_source._neighbors(current):
                candidate = queued_cost + step_cost
                if not self.cell_is_free(neighbor):
                    candidate += soft_cell_penalty
                if candidate + 1e-12 >= cost_to.get(neighbor, math.inf):
                    continue
                cost_to[neighbor] = candidate
                came_from[neighbor] = current
                sequence += 1
                priority = candidate + self._heuristic(neighbor, goal)
                heapq.heappush(
                    frontier, (priority, candidate, sequence, neighbor))
        if not reached:
            raise PathNotFound(
                "no hard-safe egress reaches a preferred-clearance path")

        cells = [goal]
        while cells[-1] != start:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        egress_index = next(
            index for index, cell in enumerate(cells)
            if self.cell_is_free(cell))
        # Preserve the exact egress cell as a waypoint.  Runtime validation can
        # then check the prefix with the hard envelope and the suffix with the
        # preferred clearance without weakening later parts of the route.
        prefix = recovery_planner._compress_cells(cells[:egress_index + 1])
        suffix = self._compress_cells(cells[egress_index:])
        compressed = prefix + suffix[1:]
        cell_points = [self.cell_to_world(cell) for cell in compressed]
        points = [start_xy]
        for point in cell_points:
            if np.linalg.norm(point - points[-1]) > 1e-12:
                points.append(point)
        if np.linalg.norm(goal_xy - points[-1]) > 1e-12:
            points.append(goal_xy)
        return np.vstack(points)

    def path_is_free_with_start_recovery(self, points, recovery_planner):
        """Validate a hard-safe prefix followed by a preferred-safe suffix."""
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (2,) or
                len(points) == 0 or not np.isfinite(points).all()):
            return False
        start = self.world_to_cell(points[0])
        if self.cell_is_free(start):
            return self.path_is_free(points)
        if not recovery_planner.cell_is_free(start):
            return False
        for split in range(1, len(points)):
            if (recovery_planner.path_is_free(points[:split + 1]) and
                    self.path_is_free(points[split:])):
                return True
        return False

    def _segment_cells(self, start, end):
        """Yield every grid cell touched by a continuous line segment."""
        x0 = (float(start[0]) - self.origin_x) / self.resolution_m + 0.5
        y0 = (float(start[1]) - self.origin_y) / self.resolution_m + 0.5
        x1 = (float(end[0]) - self.origin_x) / self.resolution_m + 0.5
        y1 = (float(end[1]) - self.origin_y) / self.resolution_m + 0.5
        column, row = int(math.floor(x0)), int(math.floor(y0))
        end_column, end_row = int(math.floor(x1)), int(math.floor(y1))
        yield row, column
        if (row, column) == (end_row, end_column):
            return

        dx, dy = x1 - x0, y1 - y0
        step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
        step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
        t_delta_x = math.inf if step_x == 0 else 1.0 / abs(dx)
        t_delta_y = math.inf if step_y == 0 else 1.0 / abs(dy)
        next_x = column + 1 if step_x > 0 else column
        next_y = row + 1 if step_y > 0 else row
        t_max_x = math.inf if step_x == 0 else (next_x - x0) / dx
        t_max_y = math.inf if step_y == 0 else (next_y - y0) / dy

        while (row, column) != (end_row, end_column):
            if t_max_x < t_max_y - 1e-12:
                column += step_x
                t_max_x += t_delta_x
                yield row, column
            elif t_max_y < t_max_x - 1e-12:
                row += step_y
                t_max_y += t_delta_y
                yield row, column
            else:
                # At an exact corner, a finite-radius robot must clear both
                # side cells as well as the diagonally entered cell.
                yield row, column + step_x
                yield row + step_y, column
                column += step_x
                row += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                yield row, column

    def path_is_free(self, points):
        """Check every map cell touched by a world-frame polyline."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) == 0:
            return False
        if not np.isfinite(points).all():
            return False
        for start, end in zip(points[:-1], points[1:]):
            for cell in self._segment_cells(start, end):
                if not self.cell_is_free(cell):
                    return False
        return self.cell_is_free(self.world_to_cell(points[-1]))


@dataclass(frozen=True)
class FollowerOutput:
    command: Twist2
    following_pose: Pose2
    progress_m: float
    remaining_m: float
    cross_track_error_m: float
    pursuit_path_clear: bool
    complete: bool


class PurePursuitFollower:
    """Look ahead on a polyline and stabilize the Kiwi pose toward that point.

    The geometric pure-pursuit layer chooses a monotonic following point. The
    existing pose-stabilization controller supplies map-axis P feedback, body
    frame conversion, velocity limiting, and final pose tolerances.
    """

    def __init__(self, trajectory, controller, lookahead_m=0.30, goal_yaw=None,
                 goal_yaw_blend_distance_m=None):
        points = np.asarray(trajectory, dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) == 0 or
                not np.isfinite(points).all()):
            raise ValueError("trajectory must contain finite 2D points")
        if not isinstance(controller, PoseStabilizingController):
            raise TypeError("controller must be a PoseStabilizingController")
        self.points = points.copy()
        self.controller = controller
        self.lookahead_m = float(lookahead_m)
        if self.lookahead_m <= 0.0:
            raise ValueError("lookahead must be positive")

        if len(points) == 1:
            self.segment_vectors = np.empty((0, 2), dtype=float)
            self.segment_lengths = np.empty(0, dtype=float)
            self.cumulative_lengths = np.array([0.0])
            default_yaw = 0.0
        else:
            vectors = np.diff(points, axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            keep = np.concatenate(([True], lengths > 1e-12))
            self.points = points[keep]
            if len(self.points) == 1:
                self.segment_vectors = np.empty((0, 2), dtype=float)
                self.segment_lengths = np.empty(0, dtype=float)
                self.cumulative_lengths = np.array([0.0])
                default_yaw = 0.0
            else:
                self.segment_vectors = np.diff(self.points, axis=0)
                self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
                self.cumulative_lengths = np.concatenate((
                    [0.0], np.cumsum(self.segment_lengths)))
                final_vector = self.segment_vectors[-1]
                default_yaw = math.atan2(final_vector[1], final_vector[0])
        self.length_m = float(self.cumulative_lengths[-1])
        self.goal_yaw = default_yaw if goal_yaw is None else float(goal_yaw)
        if not math.isfinite(self.goal_yaw):
            raise ValueError("goal yaw must be finite")
        self.goal_yaw_blend_distance_m = float(
            self.lookahead_m if goal_yaw_blend_distance_m is None
            else goal_yaw_blend_distance_m)
        if (not math.isfinite(self.goal_yaw_blend_distance_m) or
                self.goal_yaw_blend_distance_m <= 0.0):
            raise ValueError("goal yaw blend distance must be positive")
        self.progress_m = 0.0

    def _point_and_heading_at(self, distance_m):
        if not len(self.segment_lengths):
            return self.points[0].copy(), self.goal_yaw
        distance_m = min(max(float(distance_m), 0.0), self.length_m)
        segment = min(
            int(np.searchsorted(self.cumulative_lengths, distance_m, side="right") - 1),
            len(self.segment_lengths) - 1,
        )
        offset = distance_m - self.cumulative_lengths[segment]
        fraction = offset / self.segment_lengths[segment]
        point = self.points[segment] + fraction * self.segment_vectors[segment]
        vector = self.segment_vectors[segment]
        heading = math.atan2(vector[1], vector[0])
        if distance_m >= self.length_m - 1e-12:
            heading = self.goal_yaw
        return point, heading

    def _nearest_progress(self, position):
        if not len(self.segment_lengths):
            return 0.0, float(np.linalg.norm(position - self.points[0]))
        best = (math.inf, math.inf)
        minimum_progress = max(0.0, self.progress_m - 1e-9)
        for index, (start, vector, length) in enumerate(zip(
                self.points[:-1], self.segment_vectors, self.segment_lengths)):
            projection = float(np.dot(position - start, vector) / (length * length))
            projection = min(max(projection, 0.0), 1.0)
            progress = self.cumulative_lengths[index] + projection * length
            if progress + 1e-9 < minimum_progress:
                continue
            projected = start + projection * vector
            distance = float(np.linalg.norm(position - projected))
            candidate = distance, progress
            if candidate < best:
                best = candidate
        if not math.isfinite(best[0]):
            point, _heading = self._point_and_heading_at(self.progress_m)
            return self.progress_m, float(np.linalg.norm(position - point))
        return best[1], best[0]

    def _tracking_heading(self, target_distance):
        """Return a continuous path heading through the pursuit window.

        Selecting the tangent of the segment containing the lookahead point
        makes the yaw setpoint jump at every A* corner.  The chord from current
        path progress to the lookahead point blends adjacent segment directions
        continuously while retaining the intended direction on straight runs.
        """
        progress_point, progress_heading = self._point_and_heading_at(
            self.progress_m)
        target_point, _target_heading = self._point_and_heading_at(
            target_distance)
        chord = target_point - progress_point
        if float(np.linalg.norm(chord)) <= 1e-12:
            return progress_heading
        return math.atan2(float(chord[1]), float(chord[0]))

    def _desired_heading(self, target_distance):
        path_heading = self._tracking_heading(target_distance)
        remaining = max(0.0, self.length_m - self.progress_m)
        blend = 1.0 - min(
            1.0, remaining / self.goal_yaw_blend_distance_m)
        # Smoothstep avoids an angular-rate step where final-yaw blending starts.
        blend = blend * blend * (3.0 - 2.0 * blend)
        yaw_delta = math.atan2(
            math.sin(self.goal_yaw - path_heading),
            math.cos(self.goal_yaw - path_heading),
        )
        return math.atan2(
            math.sin(path_heading + blend * yaw_delta),
            math.cos(path_heading + blend * yaw_delta),
        )

    def update(self, current, path_is_free=None):
        position = np.array((current.x, current.y), dtype=float)
        nearest_progress, cross_track_error = self._nearest_progress(position)
        self.progress_m = max(self.progress_m, nearest_progress)
        target_distance = min(self.length_m, self.progress_m + self.lookahead_m)
        pursuit_path_clear = True
        if path_is_free is not None:
            if not callable(path_is_free):
                raise TypeError("path_is_free must be callable")
            pursuit_path_clear = False
            # Pure pursuit naturally rounds corners. Shorten the lookahead to
            # the farthest directly visible point so that rounding never cuts
            # through the planner's inflated obstacle grid.
            for candidate_distance in np.linspace(
                    target_distance, self.progress_m, 17):
                candidate_point, _candidate_yaw = self._point_and_heading_at(
                    candidate_distance)
                segment = np.vstack((position, candidate_point))
                if (target_distance > self.progress_m + 1e-9 and
                        candidate_distance <= self.progress_m + 1e-9 and
                        np.linalg.norm(candidate_point - position) <= 1e-6):
                    continue
                if path_is_free(segment):
                    target_distance = float(candidate_distance)
                    pursuit_path_clear = True
                    break
        following_point, _following_tangent = self._point_and_heading_at(
            target_distance)
        following_yaw = self._desired_heading(target_distance)
        following_pose = Pose2(
            float(following_point[0]), float(following_point[1]), following_yaw)
        goal = Pose2(
            float(self.points[-1, 0]), float(self.points[-1, 1]), self.goal_yaw)
        complete = self.controller.at_target(current, goal)
        # At the final position, use the exact saved destination yaw while
        # translation is held.  Before then, approach it continuously.
        if self.controller.within_position_tolerance(current, goal):
            following_pose = Pose2(
                following_pose.x, following_pose.y, self.goal_yaw)
            target = goal
        else:
            target = following_pose
        command = (Twist2(0.0, 0.0, 0.0)
                   if complete or not pursuit_path_clear
                   else self.controller.command(current, target))
        return FollowerOutput(
            command=command,
            following_pose=following_pose,
            progress_m=self.progress_m,
            remaining_m=max(0.0, self.length_m - self.progress_m),
            cross_track_error_m=cross_track_error,
            pursuit_path_clear=pursuit_path_clear,
            complete=complete,
        )

    def remaining_trajectory(self):
        """Return the not-yet-followed portion for map collision checks."""
        current_point, _heading = self._point_and_heading_at(self.progress_m)
        if not len(self.segment_lengths):
            return np.array([current_point])
        next_index = int(np.searchsorted(
            self.cumulative_lengths, self.progress_m, side="right"))
        tail = self.points[next_index:]
        if len(tail) == 0:
            return np.array([current_point])
        return np.vstack((current_point, tail))
