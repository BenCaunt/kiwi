import { rotate } from "./math";
import type { Vec2, WallSegment, WorldDefinition, WorldObject } from "./types";

export const DEFAULT_ROBOT_COLLISION_HEIGHT_M = 0.22;
export const TABLE_TOP_THICKNESS_M = 0.08;
export const TABLE_LEG_SIZE_M = 0.07;
export const TABLE_LEG_INSET_M = 0.12;
export const DESK_DRAWER_HEIGHT_M = 0.18;
export const DESK_DRAWER_WIDTH_RATIO = 0.42;
export const DESK_DRAWER_DEPTH_RATIO = 0.72;
export const DESK_DRAWER_X_RATIO = 0.23;
export const DESK_DRAWER_CENTER_BELOW_TOP_M = 0.2;

interface ObjectPart {
  position: Vec2;
  size: Vec2;
  minHeight: number;
  maxHeight: number;
}

function rectangleSegments(
  position: Vec2,
  size: Vec2,
  yaw: number,
): WallSegment[] {
  const halfX = size.x / 2;
  const halfY = size.y / 2;
  const corners = [
    { x: -halfX, y: -halfY },
    { x: halfX, y: -halfY },
    { x: halfX, y: halfY },
    { x: -halfX, y: halfY },
  ].map((corner) => {
    const rotated = rotate(corner, yaw);
    return {
      x: position.x + rotated.x,
      y: position.y + rotated.y,
    };
  });

  return corners.map((start, index) => ({
    start,
    end: corners[(index + 1) % corners.length] ?? start,
  }));
}

function tableParts(object: WorldObject): ObjectPart[] {
  const topThickness = Math.min(TABLE_TOP_THICKNESS_M, object.height);
  const topBottom = Math.max(0, object.height - topThickness);
  const legHeight = topBottom;
  const legX = object.size.x / 2 - TABLE_LEG_INSET_M;
  const legY = object.size.y / 2 - TABLE_LEG_INSET_M;
  const parts: ObjectPart[] = [
    {
      position: { x: 0, y: 0 },
      size: object.size,
      minHeight: topBottom,
      maxHeight: object.height,
    },
  ];

  for (const x of [-legX, legX]) {
    for (const y of [-legY, legY]) {
      parts.push({
        position: { x, y },
        size: { x: TABLE_LEG_SIZE_M, y: TABLE_LEG_SIZE_M },
        minHeight: 0,
        maxHeight: legHeight,
      });
    }
  }

  if (object.kind === "desk") {
    const centerHeight = object.height - DESK_DRAWER_CENTER_BELOW_TOP_M;
    parts.push({
      position: { x: object.size.x * DESK_DRAWER_X_RATIO, y: 0 },
      size: {
        x: object.size.x * DESK_DRAWER_WIDTH_RATIO,
        y: object.size.y * DESK_DRAWER_DEPTH_RATIO,
      },
      minHeight: Math.max(0, centerHeight - DESK_DRAWER_HEIGHT_M / 2),
      maxHeight: centerHeight + DESK_DRAWER_HEIGHT_M / 2,
    });
  }

  return parts;
}

function objectParts(object: WorldObject): ObjectPart[] {
  if (
    object.kind === "table" ||
    object.kind === "low-table" ||
    object.kind === "desk"
  ) {
    return tableParts(object);
  }
  return [{
    position: { x: 0, y: 0 },
    size: object.size,
    minHeight: 0,
    maxHeight: object.height,
  }];
}

function partSegments(object: WorldObject, part: ObjectPart): WallSegment[] {
  const yaw = object.yaw ?? 0;
  const offset = rotate(part.position, yaw);
  return rectangleSegments(
    {
      x: object.position.x + offset.x,
      y: object.position.y + offset.y,
    },
    part.size,
    yaw,
  );
}

export function objectSegments(object: WorldObject): WallSegment[] {
  return rectangleSegments(object.position, object.size, object.yaw ?? 0);
}

export function objectCollisionSegments(
  object: WorldObject,
  robotHeight = DEFAULT_ROBOT_COLLISION_HEIGHT_M,
): WallSegment[] {
  return objectParts(object)
    .filter(
      (part) => part.maxHeight > 0 && part.minHeight < robotHeight,
    )
    .flatMap((part) => partSegments(object, part));
}

export function objectLidarSegments(
  object: WorldObject,
  sensorHeight: number,
): WallSegment[] {
  return objectParts(object)
    .filter(
      (part) =>
        part.minHeight <= sensorHeight && part.maxHeight >= sensorHeight,
    )
    .flatMap((part) => partSegments(object, part));
}

export function worldCollisionSegments(
  world: WorldDefinition,
  robotHeight = DEFAULT_ROBOT_COLLISION_HEIGHT_M,
): WallSegment[] {
  return [
    ...world.walls,
    ...(world.objects ?? [])
      .filter((object) => object.collidable !== false)
      .flatMap((object) => objectCollisionSegments(object, robotHeight)),
  ];
}

export function worldLidarSegments(
  world: WorldDefinition,
  sensorHeight = 0.205,
): WallSegment[] {
  return [
    ...world.walls,
    ...(world.objects ?? [])
      .filter((object) => object.lidarVisible !== false)
      .flatMap((object) => objectLidarSegments(object, sensorHeight)),
  ];
}
