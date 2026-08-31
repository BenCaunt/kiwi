import "./styles.css";

import { KeyboardDrive } from "./input";
import { SimulatorScene } from "./render/scene";
import { KiwiSimEngine, type CameraDueEvent } from "./sim/engine";
import { RETAINED_ROBOT_PROFILE } from "./sim/hardware-profile";
import { BRIDGE_CHANNEL } from "./sim/zenoh-contract";
import { WORLD_LIST, worldById } from "./sim/worlds";
import type { WorldDefinition } from "./sim/types";
import { ZenohBridgeClient } from "./transport/zenoh-bridge";

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as T;
}

const canvas = requiredElement<HTMLCanvasElement>("viewport");
const worldSelect = requiredElement<HTMLSelectElement>("world-select");
const pauseButton = requiredElement<HTMLButtonElement>("pause-button");
const resetButton = requiredElement<HTMLButtonElement>("reset-button");
const viewButton = requiredElement<HTMLButtonElement>("view-button");
const worldInformation = {
  name: requiredElement<HTMLElement>("world-name"),
  style: requiredElement<HTMLElement>("world-style"),
  description: requiredElement<HTMLElement>("world-description"),
  tags: requiredElement<HTMLElement>("world-tags"),
};
const telemetry = {
  runState: requiredElement<HTMLElement>("run-state"),
  transportState: requiredElement<HTMLElement>("transport-state"),
  simTime: requiredElement<HTMLElement>("sim-time"),
  x: requiredElement<HTMLElement>("pose-x"),
  y: requiredElement<HTMLElement>("pose-y"),
  yaw: requiredElement<HTMLElement>("pose-yaw"),
  speed: requiredElement<HTMLElement>("speed"),
  lidar: requiredElement<HTMLElement>("lidar"),
  collision: requiredElement<HTMLElement>("collision"),
};

function populateWorldSelect(): void {
  worldSelect.replaceChildren();
  for (const category of ["home", "test"] as const) {
    const group = document.createElement("optgroup");
    group.label = category === "home" ? "Planar homes" : "Test environments";
    for (const definition of WORLD_LIST.filter((candidate) => candidate.category === category)) {
      const option = document.createElement("option");
      option.value = definition.id;
      option.textContent = definition.name;
      group.append(option);
    }
    worldSelect.append(group);
  }
  worldSelect.value = "home";
}

function updateWorldInformation(definition: WorldDefinition): void {
  worldInformation.name.textContent = definition.name;
  worldInformation.style.textContent = definition.style ?? "Simulation environment";
  worldInformation.description.textContent = definition.description;
  worldInformation.tags.replaceChildren();
  for (const tag of definition.tags ?? []) {
    const item = document.createElement("li");
    item.textContent = tag;
    worldInformation.tags.append(item);
  }
}

populateWorldSelect();

const view = new SimulatorScene(canvas);
const keyboard = new KeyboardDrive();
const hardwareProfile = RETAINED_ROBOT_PROFILE;
const engine = new KiwiSimEngine(worldById(worldSelect.value), {
  hardwareProfile,
});
let paused = false;
let following = false;
let keyboardWasActive = false;
let previousFrame = performance.now();

const bridge = new ZenohBridgeClient({
  onCommand: ({ twist, timeoutSeconds }) => {
    engine.recordCommand();
    if (!keyboard.isActive()) {
      engine.setRawCommand(twist, timeoutSeconds);
    }
  },
  onState: (state, namespace) => {
    telemetry.transportState.textContent =
      state === "online"
        ? `ZENOH ${namespace ?? "ONLINE"}`
        : state === "connecting"
          ? "ZENOH CONNECTING"
          : "ZENOH OFFLINE";
    telemetry.transportState.classList.toggle("online", state === "online");
    telemetry.transportState.classList.toggle("connecting", state === "connecting");
  },
});

function reset(nextWorld = engine.world): void {
  const snapshot = engine.reset(nextWorld);
  keyboardWasActive = false;
  view.loadWorld(nextWorld);
  updateWorldInformation(nextWorld);
  view.updateRobot(snapshot.robot.pose);
  view.updateLidar(snapshot.robot.pose, snapshot.lidar);
}

function togglePaused(): void {
  paused = !paused;
  pauseButton.textContent = paused ? "Resume" : "Pause";
}

worldSelect.addEventListener("change", () => reset(worldById(worldSelect.value)));
pauseButton.addEventListener("click", togglePaused);
resetButton.addEventListener("click", () => reset());
viewButton.addEventListener("click", () => {
  following = !following;
  view.setFollowRobot(following);
  viewButton.textContent = following ? "Free camera" : "Follow robot";
});

function updateTelemetry(): void {
  const { pose, velocityAligned, collided } = engine.robot.state;
  const hits = engine.lidar.filter((sample) => sample.hit);
  const closest = hits.reduce(
    (minimum, sample) => Math.min(minimum, sample.distance),
    Number.POSITIVE_INFINITY,
  );
  telemetry.runState.textContent = paused ? "PAUSED" : "RUNNING";
  telemetry.simTime.textContent = `${engine.simulationTime.toFixed(1)} s`;
  telemetry.x.textContent = `${pose.x.toFixed(2)} m`;
  telemetry.y.textContent = `${pose.y.toFixed(2)} m`;
  telemetry.yaw.textContent = `${((pose.yaw * 180) / Math.PI).toFixed(1)}°`;
  telemetry.speed.textContent = `${Math.hypot(
    velocityAligned.vx,
    velocityAligned.vy,
  ).toFixed(2)} m/s`;
  telemetry.lidar.textContent = Number.isFinite(closest)
    ? `${closest.toFixed(2)} m`
    : "NO RETURN";
  telemetry.collision.textContent = collided ? "CONTACT" : "CLEAR";
  telemetry.collision.classList.toggle("warning", collided);
}

function animate(nowMs: number): void {
  requestAnimationFrame(animate);
  const frameSeconds = (nowMs - previousFrame) / 1000;
  previousFrame = nowMs;

  if (keyboard.consumePause()) togglePaused();
  if (keyboard.consumeReset()) reset();

  if (!paused) {
    const result = engine.advanceFrame(frameSeconds, (activeEngine) => {
      const keyboardActive = keyboard.isActive();
      if (keyboardActive) {
        activeEngine.setAlignedCommand(keyboard.command());
        if (!keyboardWasActive) activeEngine.recordCommand();
      } else if (keyboardWasActive) {
        activeEngine.stop();
      }
      keyboardWasActive = keyboardActive;
    });

    let latestCamera: CameraDueEvent | undefined;
    for (const event of result.events) {
      if (event.type === "camera_due") {
        latestCamera = event;
      } else if (!bridge.connected) {
        continue;
      } else if (event.type === "odometry") {
        bridge.publish(BRIDGE_CHANNEL.ODOMETRY, event.payload);
      } else if (event.type === "lidar") {
        bridge.publish(BRIDGE_CHANNEL.LIDAR, event.payload);
      } else if (event.type === "status") {
        bridge.publish(BRIDGE_CHANNEL.STATUS, event.payload);
      } else if (event.type === "ground_truth") {
        bridge.publishGroundTruth(
          event.worldId,
          event.pose,
          event.simulationTime,
        );
      }
    }
    if (latestCamera && bridge.connected) {
      const jpeg = view.captureCameraJpeg(latestCamera.pose);
      bridge.publish(
        BRIDGE_CHANNEL.CAMERA,
        engine.cameraPayload(jpeg, latestCamera.scheduledTime),
      );
    }
  }

  view.updateRobot(engine.robot.state.pose);
  view.updateLidar(engine.robot.state.pose, engine.lidar);
  updateTelemetry();
  view.render();
}

reset();
bridge.start();
window.addEventListener("beforeunload", () => bridge.stop());
requestAnimationFrame(animate);
