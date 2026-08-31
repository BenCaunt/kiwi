import type { Vec2 } from "./types";

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

export function wrapAngle(angle: number): number {
  const twoPi = 2 * Math.PI;
  return ((angle + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
}

export function rotate(vector: Vec2, angle: number): Vec2 {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return {
    x: cosine * vector.x - sine * vector.y,
    y: sine * vector.x + cosine * vector.y,
  };
}

export function distanceToSegment(point: Vec2, start: Vec2, end: Vec2): number {
  const segmentX = end.x - start.x;
  const segmentY = end.y - start.y;
  const lengthSquared = segmentX * segmentX + segmentY * segmentY;
  if (lengthSquared < 1e-12) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }

  const projection = clamp(
    ((point.x - start.x) * segmentX + (point.y - start.y) * segmentY) /
      lengthSquared,
    0,
    1,
  );
  const closestX = start.x + projection * segmentX;
  const closestY = start.y + projection * segmentY;
  return Math.hypot(point.x - closestX, point.y - closestY);
}

export function raySegmentDistance(
  origin: Vec2,
  angle: number,
  start: Vec2,
  end: Vec2,
): number | null {
  const directionX = Math.cos(angle);
  const directionY = Math.sin(angle);
  const segmentX = end.x - start.x;
  const segmentY = end.y - start.y;
  const denominator = directionX * segmentY - directionY * segmentX;
  if (Math.abs(denominator) < 1e-12) return null;

  const offsetX = start.x - origin.x;
  const offsetY = start.y - origin.y;
  const distance = (offsetX * segmentY - offsetY * segmentX) / denominator;
  const alongSegment = (offsetX * directionY - offsetY * directionX) / denominator;
  if (distance < 0 || alongSegment < 0 || alongSegment > 1) return null;
  return distance;
}
