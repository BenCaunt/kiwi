import { KiwiSimEngine } from "../sim/engine";
import { IDEAL_SENSOR_PROFILE } from "../sim/hardware-profile";
import { KiwiRlEnvironment } from "../sim/rl-environment";
import { worldById } from "../sim/worlds";

const requestedSteps = Number(process.argv[2] ?? 10_000);
if (!Number.isInteger(requestedSteps) || requestedSteps <= 0) {
  throw new Error("Benchmark step count must be a positive integer");
}

const engine = new KiwiSimEngine(worldById("warehouse"), {
  hardwareProfile: IDEAL_SENSOR_PROFILE,
});
const environment = new KiwiRlEnvironment(engine, {
  actionMode: "twist_aligned_v1",
  policyHz: 20,
  maxEpisodeSteps: requestedSteps + 1,
});
environment.reset({ seed: 42 });
const startedAt = performance.now();
for (let index = 0; index < requestedSteps; index += 1) {
  const direction = Math.floor(index / 200) % 2 === 0 ? 1 : -1;
  environment.step({ kind: "twist", vx: 0.1 * direction, vy: 0, omega: 0.1 });
}
const elapsedSeconds = (performance.now() - startedAt) / 1000;
const simulationSeconds = engine.simulationTime;
console.log(JSON.stringify({
  steps: requestedSteps,
  physics_ticks: requestedSteps * environment.physicsTicksPerPolicyStep,
  simulation_seconds: simulationSeconds,
  wall_seconds: elapsedSeconds,
  simulation_realtime_factor: simulationSeconds / elapsedSeconds,
  policy_steps_per_second: requestedSteps / elapsedSeconds,
  sleeps: 0,
}));
