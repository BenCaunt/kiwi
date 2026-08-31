import { distanceToSegment } from "../sim/math";
import type { Pose2, Vec2, WorldDefinition } from "../sim/types";
import {
  DEFAULT_ROBOT_COLLISION_HEIGHT_M,
  worldCollisionSegments,
} from "../sim/world-geometry";

export interface NavigationGridConfig {
  resolutionM: number;
  robotRadiusM: number;
  robotHeightM: number;
  clearanceM: number;
}

export const DEFAULT_NAVIGATION_GRID_CONFIG: Readonly<NavigationGridConfig> =
  Object.freeze({
    resolutionM: 0.1,
    robotRadiusM: 0.13,
    robotHeightM: DEFAULT_ROBOT_COLLISION_HEIGHT_M,
    clearanceM: 0.02,
  });

interface HeapEntry {
  index: number;
  distance: number;
}

class MinimumHeap {
  private values: HeapEntry[] = [];

  get length(): number {
    return this.values.length;
  }

  push(entry: HeapEntry): void {
    this.values.push(entry);
    let index = this.values.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      const parentValue = this.values[parent];
      if (!parentValue || parentValue.distance <= entry.distance) break;
      this.values[index] = parentValue;
      index = parent;
    }
    this.values[index] = entry;
  }

  pop(): HeapEntry | undefined {
    const first = this.values[0];
    const last = this.values.pop();
    if (!first || !last || this.values.length === 0) return first;
    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      if (left >= this.values.length) break;
      const leftValue = this.values[left];
      const rightValue = this.values[right];
      if (!leftValue) break;
      const child = rightValue && rightValue.distance < leftValue.distance ? right : left;
      const childValue = this.values[child];
      if (!childValue || childValue.distance >= last.distance) break;
      this.values[index] = childValue;
      index = child;
    }
    this.values[index] = last;
    return first;
  }
}

function finitePositive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be positive and finite`);
  }
}

/** A reusable collision-inflated occupancy grid for one immutable world. */
export class NavigationGrid {
  readonly config: NavigationGridConfig;
  readonly width: number;
  readonly height: number;
  readonly minX: number;
  readonly minY: number;
  readonly occupied: Uint8Array;
  private readonly collisionSegments;

  constructor(readonly world: WorldDefinition, config: Partial<NavigationGridConfig> = {}) {
    this.config = { ...DEFAULT_NAVIGATION_GRID_CONFIG, ...config };
    finitePositive(this.config.resolutionM, "resolutionM");
    finitePositive(this.config.robotRadiusM, "robotRadiusM");
    finitePositive(this.config.robotHeightM, "robotHeightM");
    if (!Number.isFinite(this.config.clearanceM) || this.config.clearanceM < 0) {
      throw new Error("clearanceM must be finite and non-negative");
    }
    const segments = worldCollisionSegments(world, this.config.robotHeightM);
    this.collisionSegments = segments;
    if (segments.length === 0) throw new Error("Navigation world has no collision geometry");
    const xs = segments.flatMap((segment) => [segment.start.x, segment.end.x]);
    const ys = segments.flatMap((segment) => [segment.start.y, segment.end.y]);
    const padding = this.config.resolutionM;
    this.minX = Math.min(...xs) - padding;
    this.minY = Math.min(...ys) - padding;
    this.width = Math.ceil((Math.max(...xs) + padding - this.minX) / this.config.resolutionM) + 1;
    this.height = Math.ceil((Math.max(...ys) + padding - this.minY) / this.config.resolutionM) + 1;
    this.occupied = new Uint8Array(this.width * this.height);
    const inflation = this.config.robotRadiusM + this.config.clearanceM;
    for (let index = 0; index < this.occupied.length; index += 1) {
      const point = this.point(index);
      if (segments.some((segment) => distanceToSegment(point, segment.start, segment.end) < inflation)) {
        this.occupied[index] = 1;
      }
    }
  }

  createDistanceField(goal: Vec2): GeodesicDistanceField {
    return new GeodesicDistanceField(this, goal);
  }

  isFree(point: Vec2): boolean {
    const inflation = this.config.robotRadiusM + this.config.clearanceM;
    return !this.collisionSegments.some(
      (segment) => distanceToSegment(point, segment.start, segment.end) < inflation,
    );
  }

  point(index: number): Vec2 {
    const xIndex = index % this.width;
    const yIndex = Math.floor(index / this.width);
    return {
      x: this.minX + xIndex * this.config.resolutionM,
      y: this.minY + yIndex * this.config.resolutionM,
    };
  }

  nearestFreeIndex(point: Vec2): number | undefined {
    let bestIndex: number | undefined;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let index = 0; index < this.occupied.length; index += 1) {
      if (this.occupied[index]) continue;
      const candidate = this.point(index);
      const distance = Math.hypot(candidate.x - point.x, candidate.y - point.y);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  nearbyFreeIndices(point: Vec2, radiusCells = 2): number[] {
    const centerX = Math.round((point.x - this.minX) / this.config.resolutionM);
    const centerY = Math.round((point.y - this.minY) / this.config.resolutionM);
    const indices: number[] = [];
    for (let dy = -radiusCells; dy <= radiusCells; dy += 1) {
      for (let dx = -radiusCells; dx <= radiusCells; dx += 1) {
        const x = centerX + dx;
        const y = centerY + dy;
        if (x < 0 || y < 0 || x >= this.width || y >= this.height) continue;
        const index = y * this.width + x;
        if (!this.occupied[index]) indices.push(index);
      }
    }
    return indices;
  }
}

/** Goal-specific shortest-path distance field used only by task/reward code. */
export class GeodesicDistanceField {
  readonly distances: Float64Array;
  readonly goalIndex: number;

  constructor(readonly grid: NavigationGrid, readonly goal: Vec2) {
    const goalIndex = grid.nearestFreeIndex(goal);
    if (goalIndex === undefined) throw new Error("Navigation grid contains no free cells");
    this.goalIndex = goalIndex;
    this.distances = new Float64Array(grid.occupied.length);
    this.distances.fill(Number.POSITIVE_INFINITY);
    this.distances[goalIndex] = 0;
    this.compute();
  }

  distance(point: Vec2 | Pose2): number {
    const nearby = this.grid.nearbyFreeIndices(point);
    const candidates = nearby.length > 0
      ? nearby
      : [this.grid.nearestFreeIndex(point)].filter(
          (index): index is number => index !== undefined,
        );
    let best = Number.POSITIVE_INFINITY;
    for (const index of candidates) {
      const cell = this.grid.point(index);
      best = Math.min(
        best,
        (this.distances[index] ?? Number.POSITIVE_INFINITY) +
          Math.hypot(point.x - cell.x, point.y - cell.y),
      );
    }
    return best;
  }

  private compute(): void {
    const { grid } = this;
    const heap = new MinimumHeap();
    heap.push({ index: this.goalIndex, distance: 0 });
    const directions = [
      [-1, 0], [1, 0], [0, -1], [0, 1],
      [-1, -1], [-1, 1], [1, -1], [1, 1],
    ] as const;
    while (heap.length > 0) {
      const current = heap.pop();
      if (!current || current.distance !== this.distances[current.index]) continue;
      const x = current.index % grid.width;
      const y = Math.floor(current.index / grid.width);
      for (const [dx, dy] of directions) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= grid.width || ny >= grid.height) continue;
        const neighbor = ny * grid.width + nx;
        if (grid.occupied[neighbor]) continue;
        if (dx !== 0 && dy !== 0) {
          const horizontal = y * grid.width + nx;
          const vertical = ny * grid.width + x;
          if (grid.occupied[horizontal] || grid.occupied[vertical]) continue;
        }
        const step = grid.config.resolutionM * (dx !== 0 && dy !== 0 ? Math.SQRT2 : 1);
        const candidate = current.distance + step;
        if (candidate >= (this.distances[neighbor] ?? Number.POSITIVE_INFINITY)) continue;
        this.distances[neighbor] = candidate;
        heap.push({ index: neighbor, distance: candidate });
      }
    }
  }
}
