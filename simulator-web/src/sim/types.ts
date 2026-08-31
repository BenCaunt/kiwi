export interface Vec2 {
  x: number;
  y: number;
}

export interface Pose2 extends Vec2 {
  yaw: number;
}

export interface Twist2 {
  vx: number;
  vy: number;
  omega: number;
}

export type SurfacePattern =
  | "carpet"
  | "concrete"
  | "mosaic"
  | "stone"
  | "tatami"
  | "terrazzo"
  | "tile"
  | "wood";

export type WallMaterial =
  | "glass"
  | "plaster"
  | "stone"
  | "tile"
  | "wood";

export interface WallSegment {
  start: Vec2;
  end: Vec2;
  color?: number;
  height?: number;
  thickness?: number;
  material?: WallMaterial;
}

export interface FloorZone {
  id: string;
  label: string;
  min: Vec2;
  max: Vec2;
  color: number;
  pattern?: SurfacePattern;
  accentColor?: number;
  patternScale?: number;
  patternRotation?: number;
}

export type WorldObjectKind =
  | "bed"
  | "bench"
  | "bookshelf"
  | "chair"
  | "counter"
  | "cushion"
  | "desk"
  | "fountain"
  | "island"
  | "lamp"
  | "low-table"
  | "ottoman"
  | "plant"
  | "rug"
  | "screen"
  | "sofa"
  | "stool"
  | "table"
  | "tub"
  | "vanity"
  | "wardrobe"
  | "toilet";

export interface WorldObject {
  id: string;
  kind: WorldObjectKind;
  position: Vec2;
  size: Vec2;
  height: number;
  yaw?: number;
  color?: number;
  accentColor?: number;
  pattern?: SurfacePattern;
  collidable?: boolean;
  lidarVisible?: boolean;
}

export type LightFixtureKind = "lantern" | "pendant" | "recessed";

export interface WorldLight {
  id: string;
  position: Vec2;
  height: number;
  color: number;
  intensity: number;
  distance: number;
  fixture?: LightFixtureKind;
}

export interface WorldAmbience {
  background: number;
  fogColor: number;
  fogDensity: number;
  skyColor: number;
  groundColor: number;
  hemisphereIntensity: number;
  sunColor: number;
  sunIntensity: number;
  sunPosition: { x: number; y: number; z: number };
  exposure: number;
}

export interface WorldDefinition {
  id: string;
  name: string;
  description: string;
  category?: "home" | "test";
  style?: string;
  tags?: string[];
  spawn: Pose2;
  walls: WallSegment[];
  floorColor?: number;
  floorZones?: FloorZone[];
  objects?: WorldObject[];
  lights?: WorldLight[];
  ambience?: WorldAmbience;
}

export interface LidarSample {
  angle: number;
  distance: number;
  hit: boolean;
}

export const ZERO_TWIST: Readonly<Twist2> = Object.freeze({
  vx: 0,
  vy: 0,
  omega: 0,
});
