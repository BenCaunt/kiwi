import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  DRACOLoader,
  DRACO_GLTF_CONFIG,
} from "three/addons/loaders/DRACOLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import type {
  FloorZone,
  LidarSample,
  Pose2,
  SurfacePattern,
  WallSegment,
  WorldDefinition,
  WorldLight,
  WorldObject,
} from "../sim/types";
import {
  DESK_DRAWER_CENTER_BELOW_TOP_M,
  DESK_DRAWER_DEPTH_RATIO,
  DESK_DRAWER_HEIGHT_M,
  DESK_DRAWER_WIDTH_RATIO,
  DESK_DRAWER_X_RATIO,
  TABLE_LEG_INSET_M,
  TABLE_LEG_SIZE_M,
  TABLE_TOP_THICKNESS_M,
} from "../sim/world-geometry";
import {
  KIWI_FRONT_RENDER_CAMERA,
  rgbaBottomUpToUprightRgb,
  rotateRgb180,
  type CameraCalibration,
  type KiwiVisionRenderer,
} from "../vision/camera";
import { renderedMeshBounds } from "./model-bounds";

const ROBOT_RADIUS = 0.13;
const ROBOT_MODEL_URL = "/assets/kiwi/kiwi-robot.glb";
// The Onshape assembly's camera points along +Z. The simulator's aligned
// forward axis is +X, so rotate the exported Y-up model by 90 degrees.
const ROBOT_MODEL_YAW = Math.PI / 2;

function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh || child instanceof THREE.LineSegments)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => {
      const mappedMaterial = material as THREE.Material & { map?: THREE.Texture | null };
      mappedMaterial.map?.dispose();
      material.dispose();
    });
  });
}

function createProceduralRobot(): THREE.Group {
  const robot = new THREE.Group();

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0xe8f0f2,
    metalness: 0.55,
    roughness: 0.34,
  });
  const edgeMaterial = new THREE.MeshStandardMaterial({
    color: 0x172127,
    metalness: 0.25,
    roughness: 0.48,
  });
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(ROBOT_RADIUS, ROBOT_RADIUS, 0.09, 48),
    bodyMaterial,
  );
  body.position.y = 0.095;
  body.castShadow = true;
  robot.add(body);

  const top = new THREE.Mesh(
    new THREE.CylinderGeometry(0.102, 0.108, 0.04, 48),
    edgeMaterial,
  );
  top.position.y = 0.16;
  top.castShadow = true;
  robot.add(top);

  const sensorMaterial = new THREE.MeshStandardMaterial({
    color: 0x72e5cc,
    emissive: 0x173c35,
    roughness: 0.28,
  });
  const lidar = new THREE.Mesh(
    new THREE.CylinderGeometry(0.035, 0.035, 0.03, 32),
    sensorMaterial,
  );
  lidar.position.set(0, 0.198, 0);
  robot.add(lidar);

  const forward = new THREE.Mesh(
    new THREE.ConeGeometry(0.025, 0.085, 18),
    sensorMaterial,
  );
  forward.rotation.z = -Math.PI / 2;
  forward.position.set(0.145, 0.12, 0);
  robot.add(forward);

  const wheelMaterial = new THREE.MeshStandardMaterial({
    color: 0x101418,
    roughness: 0.8,
  });
  for (let index = 0; index < 3; index += 1) {
    const angle = (index / 3) * Math.PI * 2;
    const wheel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.032, 0.032, 0.025, 20),
      wheelMaterial,
    );
    wheel.rotation.set(Math.PI / 2, angle, 0);
    wheel.position.set(
      Math.cos(angle) * ROBOT_RADIUS,
      0.055,
      -Math.sin(angle) * ROBOT_RADIUS,
    );
    wheel.castShadow = true;
    robot.add(wheel);
  }
  return robot;
}

function configureRobotModel(model: THREE.Object3D): void {
  model.name = "kiwi-onshape-model";
  model.rotation.y = ROBOT_MODEL_YAW;
  model.updateMatrixWorld(true);

  // Onshape exports around the assembly origin rather than the contact plane.
  // Ground the rendered mesh from its actual bounds so future CAD revisions do
  // not require a hand-tuned vertical offset.
  // The optimized CAD mesh can retain unreferenced vertices outside the
  // rendered triangles. Three.js' default object bounds include those stale
  // vertices, which lifted this model well above the floor. Ground only from
  // vertices actually referenced by the mesh indices.
  const bounds = renderedMeshBounds(model);
  if (!bounds.isEmpty()) model.position.y = -bounds.min.y;
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.castShadow = true;
    child.receiveShadow = true;
  });
}

function createRobot(): THREE.Group {
  const robot = new THREE.Group();
  robot.name = "kiwi-robot";

  // Keep a lightweight fallback visible while the CAD mesh is decoded, and if
  // an asset deployment is incomplete. Successful loading swaps it atomically.
  const fallback = createProceduralRobot();
  fallback.name = "kiwi-procedural-fallback";
  robot.userData.modelSource = "procedural-loading";
  robot.add(fallback);

  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath(DRACO_GLTF_CONFIG);
  const loader = new GLTFLoader();
  loader.setDRACOLoader(dracoLoader);
  loader.load(
    ROBOT_MODEL_URL,
    ({ scene }) => {
      configureRobotModel(scene);
      dracoLoader.dispose();
      if (robot.userData.disposed) {
        disposeObject(scene);
        return;
      }
      robot.remove(fallback);
      disposeObject(fallback);
      robot.add(scene);
      robot.userData.modelSource = "onshape-glb";
    },
    undefined,
    (error) => {
      dracoLoader.dispose();
      robot.userData.modelSource = "procedural-fallback";
      console.warn("Unable to load the Kiwi Onshape model; using fallback geometry", error);
    },
  );

  return robot;
}

function boxMesh(
  width: number,
  height: number,
  depth: number,
  color: number,
  roughness = 0.68,
): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.04 }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function colorCss(color: number): string {
  return `#${new THREE.Color(color).getHexString()}`;
}

function seededRandom(seedText: string): () => number {
  let seed = 2166136261;
  for (const character of seedText) {
    seed ^= character.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }
  return () => {
    seed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    seed ^= seed + Math.imul(seed ^ (seed >>> 7), 61 | seed);
    return ((seed ^ (seed >>> 14)) >>> 0) / 4294967296;
  };
}

function createSurfaceTexture(
  pattern: SurfacePattern,
  color: number,
  accentColor: number,
  seedText: string,
): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  if (!context) return new THREE.CanvasTexture(canvas);
  const random = seededRandom(seedText);
  context.fillStyle = colorCss(color);
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = colorCss(accentColor);
  context.fillStyle = colorCss(accentColor);

  if (pattern === "wood") {
    context.globalAlpha = 0.34;
    for (let y = 0; y <= 512; y += 64) {
      context.lineWidth = 3;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(512, y);
      context.stroke();
      const offset = (Math.floor(y / 64) % 2) * 112;
      for (let x = offset; x <= 512; x += 224) {
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(x, y + 64);
        context.stroke();
      }
    }
    context.globalAlpha = 0.12;
    for (let y = 18; y < 512; y += 32) {
      context.beginPath();
      context.moveTo(0, y);
      context.bezierCurveTo(150, y - 7, 360, y + 8, 512, y - 2);
      context.stroke();
    }
  } else if (pattern === "tile") {
    context.globalAlpha = 0.5;
    context.lineWidth = 5;
    for (let value = 0; value <= 512; value += 96) {
      context.beginPath();
      context.moveTo(value, 0);
      context.lineTo(value, 512);
      context.moveTo(0, value);
      context.lineTo(512, value);
      context.stroke();
    }
  } else if (pattern === "mosaic") {
    context.globalAlpha = 0.72;
    for (let y = 0; y < 512; y += 64) {
      for (let x = 0; x < 512; x += 64) {
        const inset = 9;
        context.beginPath();
        context.moveTo(x + 32, y + inset);
        context.lineTo(x + 64 - inset, y + 32);
        context.lineTo(x + 32, y + 64 - inset);
        context.lineTo(x + inset, y + 32);
        context.closePath();
        if ((x + y) % 128 === 0) context.fill();
        else context.stroke();
      }
    }
  } else if (pattern === "tatami") {
    context.globalAlpha = 0.32;
    context.lineWidth = 2;
    for (let x = 0; x <= 512; x += 8) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x + 22, 512);
      context.stroke();
    }
    context.globalAlpha = 0.75;
    context.lineWidth = 12;
    context.strokeRect(6, 6, 500, 500);
    context.beginPath();
    context.moveTo(256, 0);
    context.lineTo(256, 512);
    context.stroke();
  } else if (pattern === "stone") {
    context.globalAlpha = 0.24;
    context.lineWidth = 5;
    for (let y = 0; y < 512; y += 86) {
      const offset = (Math.floor(y / 86) % 2) * 46;
      for (let x = -offset; x < 512; x += 112) {
        context.strokeRect(x + 3, y + 3, 104, 78);
      }
    }
  } else {
    const count = pattern === "terrazzo" ? 230 : pattern === "carpet" ? 620 : 320;
    context.globalAlpha = pattern === "carpet" ? 0.16 : 0.28;
    for (let index = 0; index < count; index += 1) {
      const x = random() * 512;
      const y = random() * 512;
      const radius = pattern === "terrazzo" ? 2 + random() * 7 : 0.5 + random() * 2;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
  }

  context.globalAlpha = 1;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 4;
  return texture;
}

function createSurfaceMaterial(
  pattern: SurfacePattern,
  color: number,
  accentColor: number,
  width: number,
  depth: number,
  seedText: string,
  scale = 1,
  rotation = 0,
): THREE.MeshStandardMaterial {
  const texture = createSurfaceTexture(pattern, color, accentColor, seedText);
  const repeatScale = Math.max(0.28, scale);
  texture.repeat.set(
    Math.max(1, width / (1.45 * repeatScale)),
    Math.max(1, depth / (1.45 * repeatScale)),
  );
  texture.center.set(0.5, 0.5);
  texture.rotation = rotation;
  return new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: texture,
    roughness: pattern === "tile" || pattern === "mosaic" ? 0.72 : 0.9,
    metalness: 0.01,
  });
}

function createWallMaterial(segment: WallSegment, length: number): THREE.Material {
  const color = segment.color ?? 0x476f8a;
  if (segment.material === "glass") {
    return new THREE.MeshPhysicalMaterial({
      color,
      transparent: true,
      opacity: 0.42,
      roughness: 0.18,
      metalness: 0.08,
      transmission: 0.24,
      depthWrite: false,
    });
  }
  const pattern = segment.material === "tile"
    ? "tile"
    : segment.material === "stone"
      ? "stone"
      : segment.material === "wood"
        ? "wood"
        : "concrete";
  const accent = new THREE.Color(color).offsetHSL(0, -0.02, -0.12).getHex();
  return createSurfaceMaterial(
    pattern,
    color,
    accent,
    length,
    segment.height ?? 1.2,
    `${segment.start.x}:${segment.start.y}:${segment.end.x}:${segment.end.y}`,
    segment.material === "plaster" ? 2.4 : 0.8,
  );
}

function createLightFixture(definition: WorldLight): THREE.Group {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: definition.fixture === "lantern" ? 0x48352a : 0x8a7258,
    roughness: 0.5,
    metalness: 0.35,
  });
  const glow = new THREE.MeshStandardMaterial({
    color: definition.color,
    emissive: definition.color,
    emissiveIntensity: 2.6,
    roughness: 0.25,
  });

  if (definition.fixture === "pendant") {
    const cord = boxMesh(0.018, 0.34, 0.018, 0x3b342f, 0.6);
    cord.position.y = 0.17;
    const shade = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.18, 24, 1, true), material);
    shade.position.y = -0.07;
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 12), glow);
    bulb.position.y = -0.15;
    group.add(cord, shade, bulb);
  } else if (definition.fixture === "lantern") {
    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.3, 0.22),
      new THREE.MeshStandardMaterial({
        color: 0x49362b,
        wireframe: true,
        roughness: 0.55,
      }),
    );
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), glow);
    group.add(frame, bulb);
  } else {
    const disk = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.025, 20), glow);
    group.add(disk);
  }

  const point = new THREE.PointLight(
    definition.color,
    definition.intensity,
    definition.distance,
    1.65,
  );
  point.position.y = -0.08;
  group.add(point);
  group.position.set(definition.position.x, definition.height, -definition.position.y);
  group.userData.worldLightId = definition.id;
  return group;
}

function createFloorLabel(zone: FloorZone): THREE.Mesh | null {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = "600 32px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillStyle = "rgba(234, 232, 222, 0.34)";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(zone.label, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const width = Math.min(2.4, Math.max(1.2, zone.max.x - zone.min.x - 0.5));
  const label = new THREE.Mesh(
    new THREE.PlaneGeometry(width, width * (canvas.height / canvas.width)),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  label.rotation.x = -Math.PI / 2;
  label.position.set(
    (zone.min.x + zone.max.x) / 2,
    0.014,
    -(zone.min.y + zone.max.y) / 2,
  );
  return label;
}

function createFurniture(object: WorldObject): THREE.Group {
  const group = new THREE.Group();
  const width = object.size.x;
  const depth = object.size.y;
  const height = object.height;
  const color = object.color ?? 0x6d675f;
  const accent = object.accentColor ?? new THREE.Color(color).offsetHSL(0, -0.03, 0.11).getHex();
  const dark = new THREE.Color(color).offsetHSL(0, 0, -0.13).getHex();

  if (object.kind === "rug") {
    const rug = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.014, depth),
      createSurfaceMaterial(
        object.pattern ?? "carpet",
        color,
        accent,
        width,
        depth,
        object.id,
        0.55,
      ),
    );
    rug.position.y = 0.009;
    rug.castShadow = false;
    rug.receiveShadow = true;
    group.add(rug);
  } else if (object.kind === "bed") {
    const base = boxMesh(width, height * 0.42, depth, dark);
    base.position.y = height * 0.21;
    const mattress = boxMesh(width * 0.96, height * 0.46, depth * 0.91, color, 0.9);
    mattress.position.set(0, height * 0.62, -depth * 0.02);
    const pillow = boxMesh(width * 0.7, 0.1, depth * 0.22, accent, 0.96);
    pillow.position.set(0, height * 0.91, depth * 0.3);
    const headboard = boxMesh(width, height * 1.25, 0.09, dark);
    headboard.position.set(0, height * 0.62, depth / 2 - 0.045);
    group.add(base, mattress, pillow, headboard);
  } else if (object.kind === "sofa") {
    const seat = boxMesh(width, height * 0.42, depth, color, 0.88);
    seat.position.y = height * 0.21;
    const back = boxMesh(width, height, 0.16, dark, 0.82);
    back.position.set(0, height / 2, depth / 2 - 0.08);
    const armWidth = 0.16;
    for (const side of [-1, 1]) {
      const arm = boxMesh(armWidth, height * 0.78, depth, dark, 0.82);
      arm.position.set(side * (width / 2 - armWidth / 2), height * 0.39, 0);
      group.add(arm);
    }
    const cushionCount = Math.max(1, Math.round(width / 0.75));
    for (let index = 0; index < cushionCount; index += 1) {
      const cushion = boxMesh(
        width / cushionCount - 0.055,
        0.09,
        depth * 0.58,
        accent,
        0.92,
      );
      cushion.position.set(
        -width / 2 + (index + 0.5) * (width / cushionCount),
        height * 0.48,
        -depth * 0.06,
      );
      group.add(cushion);
    }
    group.add(seat, back);
  } else if (object.kind === "chair") {
    const seat = boxMesh(width * 0.82, 0.09, depth * 0.78, color, 0.86);
    seat.position.y = height * 0.52;
    const back = boxMesh(width * 0.82, height * 0.5, 0.09, accent, 0.82);
    back.position.set(0, height * 0.73, depth * 0.34);
    for (const x of [-width * 0.32, width * 0.32]) {
      for (const z of [-depth * 0.28, depth * 0.28]) {
        const leg = boxMesh(0.045, height * 0.5, 0.045, dark, 0.7);
        leg.position.set(x, height * 0.25, z);
        group.add(leg);
      }
    }
    group.add(seat, back);
  } else if (
    object.kind === "table" ||
    object.kind === "low-table" ||
    object.kind === "desk"
  ) {
    const top = boxMesh(width, TABLE_TOP_THICKNESS_M, depth, color, 0.54);
    top.position.y = height - TABLE_TOP_THICKNESS_M / 2;
    const inset = TABLE_LEG_INSET_M;
    for (const x of [-width / 2 + inset, width / 2 - inset]) {
      for (const z of [-depth / 2 + inset, depth / 2 - inset]) {
        const leg = boxMesh(
          TABLE_LEG_SIZE_M,
          height - TABLE_TOP_THICKNESS_M,
          TABLE_LEG_SIZE_M,
          dark,
          0.62,
        );
        leg.position.set(x, (height - TABLE_TOP_THICKNESS_M) / 2, z);
        group.add(leg);
      }
    }
    group.add(top);
    if (object.kind === "desk") {
      const drawer = boxMesh(
        width * DESK_DRAWER_WIDTH_RATIO,
        DESK_DRAWER_HEIGHT_M,
        depth * DESK_DRAWER_DEPTH_RATIO,
        accent,
        0.62,
      );
      drawer.position.set(
        width * DESK_DRAWER_X_RATIO,
        height - DESK_DRAWER_CENTER_BELOW_TOP_M,
        0,
      );
      group.add(drawer);
    }
  } else if (
    object.kind === "counter" ||
    object.kind === "island" ||
    object.kind === "vanity"
  ) {
    const cabinet = boxMesh(width * 0.96, height * 0.91, depth * 0.92, color);
    cabinet.position.y = height * 0.455;
    const top = boxMesh(width, height * 0.09, depth, accent, 0.42);
    top.position.y = height * 0.955;
    group.add(cabinet, top);
    if (object.kind === "vanity") {
      const basin = new THREE.Mesh(
        new THREE.CylinderGeometry(depth * 0.26, depth * 0.3, 0.05, 24),
        new THREE.MeshStandardMaterial({ color: 0xd8d9d5, roughness: 0.35 }),
      );
      basin.position.y = height + 0.015;
      group.add(basin);
    }
  } else if (object.kind === "wardrobe") {
    const cabinet = boxMesh(width, height, depth, color, 0.72);
    cabinet.position.y = height / 2;
    const seam = boxMesh(width + 0.004, height * 0.82, 0.018, dark, 0.62);
    seam.position.set(0, height * 0.52, -depth / 2 - 0.01);
    const handleA = boxMesh(0.025, 0.16, 0.025, accent, 0.35);
    const handleB = handleA.clone();
    handleA.position.set(-0.06, height * 0.53, -depth / 2 - 0.035);
    handleB.position.set(0.06, height * 0.53, -depth / 2 - 0.035);
    group.add(cabinet, seam, handleA, handleB);
  } else if (object.kind === "bookshelf") {
    const back = boxMesh(width, height, depth * 0.16, dark, 0.74);
    back.position.set(0, height / 2, depth * 0.42);
    const sideWidth = Math.min(0.07, width * 0.16);
    for (const side of [-1, 1]) {
      const panel = boxMesh(sideWidth, height, depth, color, 0.7);
      panel.position.set(side * (width / 2 - sideWidth / 2), height / 2, 0);
      group.add(panel);
    }
    const shelfCount = Math.max(2, Math.round(height / 0.34));
    for (let index = 0; index <= shelfCount; index += 1) {
      const shelf = boxMesh(width, 0.045, depth, color, 0.66);
      shelf.position.y = (index / shelfCount) * height;
      group.add(shelf);
      if (index < shelfCount) {
        const bookCount = 3 + (index % 2);
        for (let book = 0; book < bookCount; book += 1) {
          const bookWidth = (width - sideWidth * 3) / (bookCount + 1);
          const bookHeight = height / shelfCount * (0.55 + ((book + index) % 3) * 0.1);
          const bookMesh = boxMesh(
            bookWidth * 0.72,
            bookHeight,
            depth * 0.52,
            [0x7b473d, 0x445e5a, 0xb1844d, 0x6c6275][(book + index) % 4] ?? accent,
            0.82,
          );
          bookMesh.position.set(
            -width / 2 + sideWidth * 1.6 + (book + 0.5) * bookWidth,
            (index / shelfCount) * height + bookHeight / 2,
            -depth * 0.12,
          );
          group.add(bookMesh);
        }
      }
    }
    group.add(back);
  } else if (object.kind === "bench" || object.kind === "stool") {
    const seat = boxMesh(width, 0.1, depth, color, 0.66);
    seat.position.y = height - 0.05;
    const inset = Math.min(0.13, width * 0.22);
    for (const x of [-width / 2 + inset, width / 2 - inset]) {
      const leg = boxMesh(0.07, height - 0.1, depth * 0.65, dark, 0.7);
      leg.position.set(x, (height - 0.1) / 2, 0);
      group.add(leg);
    }
    group.add(seat);
  } else if (object.kind === "cushion" || object.kind === "ottoman") {
    const cushion = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.48, width * 0.5, height, 24),
      new THREE.MeshStandardMaterial({ color, roughness: 0.94 }),
    );
    cushion.scale.z = depth / width;
    cushion.position.y = height / 2;
    cushion.castShadow = true;
    cushion.receiveShadow = true;
    group.add(cushion);
  } else if (object.kind === "tub") {
    const rim = 0.1;
    const endA = boxMesh(rim, height, depth, color, 0.38);
    const endB = boxMesh(rim, height, depth, color, 0.38);
    endA.position.set(-width / 2 + rim / 2, height / 2, 0);
    endB.position.set(width / 2 - rim / 2, height / 2, 0);
    const sideA = boxMesh(width - rim * 2, height, rim, color, 0.38);
    const sideB = boxMesh(width - rim * 2, height, rim, color, 0.38);
    sideA.position.set(0, height / 2, -depth / 2 + rim / 2);
    sideB.position.set(0, height / 2, depth / 2 - rim / 2);
    const water = boxMesh(width - rim * 2.2, 0.02, depth - rim * 2.2, 0x6f9da7, 0.18);
    water.position.y = height * 0.46;
    water.castShadow = false;
    group.add(endA, endB, sideA, sideB, water);
  } else if (object.kind === "toilet") {
    const bowl = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.42, width * 0.35, height * 0.5, 28),
      new THREE.MeshStandardMaterial({ color, roughness: 0.32 }),
    );
    bowl.scale.z = 1.25;
    bowl.position.set(0, height * 0.25, -depth * 0.06);
    bowl.castShadow = true;
    const tank = boxMesh(width * 0.84, height * 0.7, depth * 0.3, color, 0.34);
    tank.position.set(0, height * 0.42, depth * 0.35);
    group.add(bowl, tank);
  } else if (object.kind === "fountain") {
    const basin = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.5, width * 0.48, height * 0.62, 40),
      new THREE.MeshStandardMaterial({ color: accent, roughness: 0.45 }),
    );
    basin.scale.z = depth / width;
    basin.position.y = height * 0.31;
    basin.castShadow = true;
    const water = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.41, width * 0.41, 0.025, 40),
      new THREE.MeshPhysicalMaterial({
        color,
        roughness: 0.12,
        metalness: 0.08,
        transparent: true,
        opacity: 0.82,
      }),
    );
    water.scale.z = depth / width;
    water.position.y = height * 0.63;
    const center = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.1, width * 0.15, height * 0.72, 20),
      new THREE.MeshStandardMaterial({ color: accent, roughness: 0.48 }),
    );
    center.position.y = height * 0.62;
    group.add(basin, water, center);
  } else if (object.kind === "screen") {
    const panelMaterial = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: 0.76,
      roughness: 0.86,
      side: THREE.DoubleSide,
    });
    const panel = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), panelMaterial);
    panel.position.y = height / 2;
    const frameThickness = Math.min(0.055, width * 0.25);
    for (const x of [-width / 2, 0, width / 2]) {
      const upright = boxMesh(frameThickness, height, depth * 1.5, dark, 0.62);
      upright.position.set(x, height / 2, 0);
      group.add(upright);
    }
    for (const y of [0, height / 2, height]) {
      const rail = boxMesh(width, frameThickness, depth * 1.5, dark, 0.62);
      rail.position.set(0, y, 0);
      group.add(rail);
    }
    group.add(panel);
  } else if (object.kind === "lamp") {
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.045, width * 0.055, height * 0.75, 16),
      new THREE.MeshStandardMaterial({ color: dark, metalness: 0.45, roughness: 0.46 }),
    );
    stem.position.y = height * 0.38;
    const shade = new THREE.Mesh(
      new THREE.ConeGeometry(width * 0.42, height * 0.28, 24, 1, true),
      new THREE.MeshStandardMaterial({ color: accent, roughness: 0.75, side: THREE.DoubleSide }),
    );
    shade.position.y = height * 0.82;
    group.add(stem, shade);
  } else if (object.kind === "plant") {
    const pot = new THREE.Mesh(
      new THREE.CylinderGeometry(width * 0.32, width * 0.42, height * 0.42, 20),
      new THREE.MeshStandardMaterial({ color: 0x765746, roughness: 0.86 }),
    );
    pot.position.y = height * 0.21;
    pot.castShadow = true;
    const leaves = new THREE.Mesh(
      new THREE.IcosahedronGeometry(width * 0.54, 1),
      new THREE.MeshStandardMaterial({ color, roughness: 0.92 }),
    );
    leaves.scale.y = 1.25;
    leaves.position.y = height * 0.72;
    leaves.castShadow = true;
    group.add(pot, leaves);
  }

  group.position.set(object.position.x, 0, -object.position.y);
  group.rotation.y = object.yaw ?? 0;
  group.userData.worldObjectId = object.id;
  return group;
}

export class SimulatorScene implements KiwiVisionRenderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly calibration: Readonly<CameraCalibration>;
  private readonly hemisphere = new THREE.HemisphereLight(0xbcd9ea, 0x152127, 1.8);
  private readonly sun = new THREE.DirectionalLight(0xffffff, 2.2);
  private readonly sensorCamera: THREE.PerspectiveCamera;
  private readonly sensorTarget: THREE.WebGLRenderTarget;
  private readonly sensorCanvas = document.createElement("canvas");
  private readonly worldRoot = new THREE.Group();
  private readonly robot = createRobot();
  private readonly lidarGeometry = new THREE.BufferGeometry();
  private readonly lidarLines: THREE.LineSegments;
  private followRobot = false;

  constructor(
    canvas: HTMLCanvasElement,
    calibration: Readonly<CameraCalibration> = KIWI_FRONT_RENDER_CAMERA,
  ) {
    this.calibration = calibration;
    this.sensorCamera = new THREE.PerspectiveCamera(
      calibration.verticalFovDeg,
      calibration.width / calibration.height,
      calibration.nearM,
      calibration.farM,
    );
    this.sensorTarget = new THREE.WebGLRenderTarget(
      calibration.width,
      calibration.height,
    );
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.sensorCanvas.width = this.calibration.width;
    this.sensorCanvas.height = this.calibration.height;

    this.scene.background = new THREE.Color(0x091016);
    this.scene.fog = new THREE.FogExp2(0x091016, 0.026);
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.025, 80);
    this.camera.position.set(4.4, 5.4, 4.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 18;
    this.controls.target.set(0, 0, 0);

    this.scene.add(this.hemisphere);
    this.sun.position.set(-3.5, 7, 4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.camera.left = -8;
    this.sun.shadow.camera.right = 8;
    this.sun.shadow.camera.top = 8;
    this.sun.shadow.camera.bottom = -8;
    this.sun.shadow.bias = -0.00015;
    this.scene.add(this.sun);

    this.scene.add(this.worldRoot, this.robot);
    this.lidarLines = new THREE.LineSegments(
      this.lidarGeometry,
      new THREE.LineBasicMaterial({
        color: 0x71f2d0,
        transparent: true,
        opacity: 0.34,
        depthWrite: false,
      }),
    );
    this.lidarLines.frustumCulled = false;
    this.scene.add(this.lidarLines);
    window.addEventListener("resize", this.resize);
    this.resize();
  }

  loadWorld(world: WorldDefinition): void {
    while (this.worldRoot.children.length > 0) {
      const child = this.worldRoot.children.pop();
      if (child) disposeObject(child);
    }

    if (world.ambience) {
      const ambience = world.ambience;
      this.scene.background = new THREE.Color(ambience.background);
      this.scene.fog = new THREE.FogExp2(ambience.fogColor, ambience.fogDensity);
      this.hemisphere.color.setHex(ambience.skyColor);
      this.hemisphere.groundColor.setHex(ambience.groundColor);
      this.hemisphere.intensity = ambience.hemisphereIntensity;
      this.sun.color.setHex(ambience.sunColor);
      this.sun.intensity = ambience.sunIntensity;
      this.sun.position.set(
        ambience.sunPosition.x,
        ambience.sunPosition.y,
        ambience.sunPosition.z,
      );
      this.renderer.toneMappingExposure = ambience.exposure;
    } else {
      this.scene.background = new THREE.Color(0x091016);
      this.scene.fog = new THREE.FogExp2(0x091016, 0.026);
      this.hemisphere.color.setHex(0xbcd9ea);
      this.hemisphere.groundColor.setHex(0x152127);
      this.hemisphere.intensity = 1.8;
      this.sun.color.setHex(0xffffff);
      this.sun.intensity = 2.2;
      this.sun.position.set(-3.5, 7, 4);
      this.renderer.toneMappingExposure = 1.1;
    }

    const xs = world.walls.flatMap((segment) => [segment.start.x, segment.end.x]);
    const ys = world.walls.flatMap((segment) => [segment.start.y, segment.end.y]);
    const minX = Math.min(...xs) - 0.5;
    const maxX = Math.max(...xs) + 0.5;
    const minY = Math.min(...ys) - 0.5;
    const maxY = Math.max(...ys) + 0.5;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(maxX - minX, maxY - minY),
      new THREE.MeshStandardMaterial({
        color: world.floorColor ?? 0x182127,
        roughness: 0.94,
        metalness: 0.02,
      }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set((minX + maxX) / 2, 0, -(minY + maxY) / 2);
    floor.receiveShadow = true;
    this.worldRoot.add(floor);

    for (const zone of world.floorZones ?? []) {
      const width = zone.max.x - zone.min.x;
      const depth = zone.max.y - zone.min.y;
      const zoneFloor = new THREE.Mesh(
        new THREE.PlaneGeometry(width - 0.03, depth - 0.03),
        createSurfaceMaterial(
          zone.pattern ?? "concrete",
          zone.color,
          zone.accentColor ?? new THREE.Color(zone.color).offsetHSL(0, 0, -0.14).getHex(),
          width,
          depth,
          `${world.id}:${zone.id}`,
          zone.patternScale,
          zone.patternRotation,
        ),
      );
      zoneFloor.rotation.x = -Math.PI / 2;
      zoneFloor.position.set(
        (zone.min.x + zone.max.x) / 2,
        0.004,
        -(zone.min.y + zone.max.y) / 2,
      );
      zoneFloor.receiveShadow = true;
      this.worldRoot.add(zoneFloor);
      const label = createFloorLabel(zone);
      if (label) this.worldRoot.add(label);
    }

    const grid = new THREE.GridHelper(
      Math.max(maxX - minX, maxY - minY),
      Math.ceil(Math.max(maxX - minX, maxY - minY) * 2),
      0x36505c,
      0x263740,
    );
    grid.position.set((minX + maxX) / 2, 0.008, -(minY + maxY) / 2);
    const gridMaterial = grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = world.category === "home" ? 0.07 : 0.42;
    this.worldRoot.add(grid);

    for (const segment of world.walls) {
      const dx = segment.end.x - segment.start.x;
      const dy = segment.end.y - segment.start.y;
      const length = Math.hypot(dx, dy);
      const height = segment.height ?? 1.2;
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(length, height, segment.thickness ?? 0.055),
        createWallMaterial(segment, length),
      );
      mesh.position.set(
        (segment.start.x + segment.end.x) / 2,
        height / 2,
        -(segment.start.y + segment.end.y) / 2,
      );
      mesh.rotation.y = Math.atan2(dy, dx);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.worldRoot.add(mesh);
      if (segment.material !== "glass") {
        const capColor = new THREE.Color(segment.color ?? 0x476f8a)
          .offsetHSL(0, -0.02, 0.09)
          .getHex();
        const cap = boxMesh(
          length + 0.015,
          0.028,
          (segment.thickness ?? 0.055) + 0.025,
          capColor,
          0.52,
        );
        cap.position.copy(mesh.position);
        cap.position.y = height + 0.014;
        cap.rotation.y = mesh.rotation.y;
        this.worldRoot.add(cap);
      }
    }

    for (const object of world.objects ?? []) {
      this.worldRoot.add(createFurniture(object));
    }

    for (const fixture of world.lights ?? []) {
      this.worldRoot.add(createLightFixture(fixture));
    }

    if (!this.followRobot) {
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;
      const span = Math.max(maxX - minX, maxY - minY);
      this.camera.position.set(
        centerX + span * 0.58,
        span * 0.66,
        -centerY + span * 0.58,
      );
      this.controls.target.set(centerX, 0, -centerY);
      this.controls.update();
    }
  }

  setFollowRobot(enabled: boolean): void {
    this.followRobot = enabled;
  }

  updateRobot(pose: Pose2): void {
    this.robot.position.set(pose.x, 0, -pose.y);
    this.robot.rotation.y = pose.yaw;
    if (this.followRobot) {
      const desired = new THREE.Vector3(pose.x - 1.8, 1.5, -pose.y + 1.8);
      this.camera.position.lerp(desired, 0.055);
      this.controls.target.lerp(new THREE.Vector3(pose.x, 0.1, -pose.y), 0.09);
    }
  }

  updateLidar(pose: Pose2, samples: readonly LidarSample[]): void {
    const positions = new Float32Array(samples.length * 6);
    samples.forEach((sample, index) => {
      const angle = pose.yaw + sample.angle;
      const offset = index * 6;
      positions[offset] = pose.x;
      positions[offset + 1] = 0.205;
      positions[offset + 2] = -pose.y;
      positions[offset + 3] = pose.x + Math.cos(angle) * sample.distance;
      positions[offset + 4] = 0.205;
      positions[offset + 5] = -(pose.y + Math.sin(angle) * sample.distance);
    });
    this.lidarGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.lidarGeometry.computeBoundingSphere();
  }

  private readSensorRgba(pose: Pose2): Uint8Array {
    const { width, height, robotToCamera } = this.calibration;
    this.sensorCamera.position.set(
      pose.x + Math.cos(pose.yaw) * robotToCamera.forwardM -
        Math.sin(pose.yaw) * robotToCamera.leftM,
      robotToCamera.heightM,
      -(
        pose.y +
        Math.sin(pose.yaw) * robotToCamera.forwardM +
        Math.cos(pose.yaw) * robotToCamera.leftM
      ),
    );
    this.sensorCamera.lookAt(
      this.sensorCamera.position.x + Math.cos(pose.yaw + robotToCamera.yawRad),
      this.sensorCamera.position.y + Math.sin(robotToCamera.pitchRad),
      this.sensorCamera.position.z - Math.sin(pose.yaw + robotToCamera.yawRad),
    );

    const previousTarget = this.renderer.getRenderTarget();
    const robotVisible = this.robot.visible;
    const lidarVisible = this.lidarLines.visible;
    this.robot.visible = false;
    this.lidarLines.visible = false;
    this.renderer.setRenderTarget(this.sensorTarget);
    this.renderer.render(this.scene, this.sensorCamera);
    const pixels = new Uint8Array(width * height * 4);
    this.renderer.readRenderTargetPixels(
      this.sensorTarget,
      0,
      0,
      width,
      height,
      pixels,
    );
    this.renderer.setRenderTarget(previousTarget);
    this.robot.visible = robotVisible;
    this.lidarLines.visible = lidarVisible;
    return pixels;
  }

  captureRgb(pose: Pose2): Uint8Array {
    const { width, height } = this.calibration;
    const pixels = this.readSensorRgba(pose);
    // WebGL rows begin at the bottom. Policy images are canonical upright RGB.
    return rgbaBottomUpToUprightRgb(pixels, width, height);
  }

  captureCameraJpeg(pose: Pose2): Uint8Array {
    const { width, height } = this.calibration;
    const rgb = this.captureRgb(pose);
    const context = this.sensorCanvas.getContext("2d");
    if (!context) return new Uint8Array();
    const image = context.createImageData(width, height);
    // The physical camera is mounted 180 degrees from the canonical policy view.
    const rotated = rotateRgb180(rgb, width, height);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      image.data[pixel * 4] = rotated[pixel * 3] ?? 0;
      image.data[pixel * 4 + 1] = rotated[pixel * 3 + 1] ?? 0;
      image.data[pixel * 4 + 2] = rotated[pixel * 3 + 2] ?? 0;
      image.data[pixel * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    const encoded = this.sensorCanvas.toDataURL("image/jpeg", 0.72);
    const base64 = encoded.slice(encoded.indexOf(",") + 1);
    const binary = atob(base64);
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    window.removeEventListener("resize", this.resize);
    this.controls.dispose();
    disposeObject(this.worldRoot);
    this.robot.userData.disposed = true;
    disposeObject(this.robot);
    this.lidarGeometry.dispose();
    this.sensorTarget.dispose();
    this.renderer.dispose();
  }

  private resize = (): void => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  };
}
