import { wrapAngle } from "./math";
import type { Pose2 } from "./types";

interface TimedPose {
  time: number;
  pose: Pose2;
}

function copyPose(pose: Pose2): Pose2 {
  return { x: pose.x, y: pose.y, yaw: pose.yaw };
}

/** Short pose history used to reconstruct the pose of each rolling LiDAR ray. */
export class TimedPoseHistory {
  private readonly maximumAge: number;
  private samples: TimedPose[] = [];

  constructor(maximumAge = 0.25) {
    this.maximumAge = maximumAge;
  }

  reset(time: number, pose: Pose2): void {
    this.samples = [{ time, pose: copyPose(pose) }];
  }

  append(time: number, pose: Pose2): void {
    const previous = this.samples.at(-1);
    if (previous && time <= previous.time) return;
    this.samples.push({ time, pose: copyPose(pose) });
    const cutoff = time - this.maximumAge;
    while (this.samples.length > 2 && (this.samples[1]?.time ?? time) < cutoff) {
      this.samples.shift();
    }
  }

  interpolate(time: number): Pose2 | undefined {
    const first = this.samples[0];
    const last = this.samples.at(-1);
    if (!first || !last || time < first.time - 1e-9 || time > last.time + 1e-9) {
      return undefined;
    }
    if (time <= first.time) return copyPose(first.pose);
    if (time >= last.time) return copyPose(last.pose);

    let low = 0;
    let high = this.samples.length - 1;
    while (high - low > 1) {
      const middle = Math.floor((low + high) / 2);
      if ((this.samples[middle]?.time ?? time) <= time) low = middle;
      else high = middle;
    }
    const before = this.samples[low];
    const after = this.samples[high];
    if (!before || !after) return undefined;
    const fraction = (time - before.time) / (after.time - before.time);
    return {
      x: before.pose.x + fraction * (after.pose.x - before.pose.x),
      y: before.pose.y + fraction * (after.pose.y - before.pose.y),
      yaw: wrapAngle(
        before.pose.yaw + fraction * wrapAngle(after.pose.yaw - before.pose.yaw),
      ),
    };
  }
}
