import { describe, expect, it } from "vitest";

import {
  composeRelativePose,
  PoseStabilizingController,
  RelativePoseController,
} from "./pose-controller";
import fixtures from "./pose-controller-fixtures.json";

describe("PoseStabilizingController", () => {
  it.each(fixtures)("matches shared Python fixture $name", ({ config, current, target, expected }) => {
    const command = new PoseStabilizingController(config).command(current, target);
    expect(command.vx).toBeCloseTo(expected.vx, 12);
    expect(command.vy).toBeCloseTo(expected.vy, 12);
    expect(command.omega).toBeCloseTo(expected.omega, 12);
  });

  it("matches the Python controller's map-to-body command convention", () => {
    const controller = new PoseStabilizingController({
      kpX: 1,
      kpY: 1,
      maxLinearSpeed: 10,
    });
    const command = controller.command(
      { x: 0, y: 0, yaw: Math.PI / 2 },
      { x: 1, y: 0, yaw: Math.PI / 2 },
    );

    expect(command.vx).toBeCloseTo(0, 7);
    expect(command.vy).toBeCloseTo(-1, 7);
    expect(command.omega).toBeCloseTo(0, 7);
  });

  it("limits planar speed and wraps yaw error along the shortest path", () => {
    const controller = new PoseStabilizingController({
      kpX: 2,
      kpY: 1,
      kpYaw: 2,
      maxLinearSpeed: 1,
      maxAngularSpeed: 0.5,
    });
    const command = controller.command(
      { x: 0, y: 0, yaw: (179 * Math.PI) / 180 },
      { x: 1, y: 1, yaw: (-179 * Math.PI) / 180 },
    );

    expect(Math.hypot(command.vx, command.vy)).toBeCloseTo(1);
    expect(command.omega).toBeCloseTo((4 * Math.PI) / 180);
  });
});

describe("RelativePoseController", () => {
  it("composes a body-relative target in the action-start frame", () => {
    const target = composeRelativePose(
      { x: 1, y: 2, yaw: Math.PI / 2 },
      { dx: 0.5, dy: 0.5, dyaw: 0 },
    );

    expect(target.x).toBeCloseTo(0.5);
    expect(target.y).toBeCloseTo(2.5);
    expect(target.yaw).toBeCloseTo(Math.PI / 2);
  });

  it("keeps the accepted target fixed as the robot moves", () => {
    const controller = new RelativePoseController();
    controller.setRelativeTarget(
      { x: 0, y: 0, yaw: 0 },
      { dx: 1, dy: 0, dyaw: Math.PI / 2 },
    );

    const before = controller.target;
    controller.command({ x: 0.3, y: 0.1, yaw: Math.PI / 4 });

    expect(controller.target).toEqual(before);
  });

  it("anchors every trajectory point to the same origin", () => {
    const controller = new RelativePoseController();
    const selected = controller.setRelativeTrajectory(
      { x: 1, y: 2, yaw: Math.PI / 2 },
      [
        { dx: 1, dy: 0, dyaw: 0 },
        { dx: 2, dy: 0.5, dyaw: Math.PI / 2 },
      ],
      1,
    );

    expect(controller.trajectory[0]?.x).toBeCloseTo(1);
    expect(controller.trajectory[0]?.y).toBeCloseTo(3);
    expect(selected.x).toBeCloseTo(0.5);
    expect(selected.y).toBeCloseTo(4);
    expect(selected.yaw).toBeCloseTo(-Math.PI);
  });
});
