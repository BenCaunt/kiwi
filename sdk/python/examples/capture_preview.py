from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kiwi_sim_sdk import EnvConfig, KiwiEnv, VisionConfig


def labeled(image: np.ndarray, label: str, scale: int = 2) -> Image.Image:
    panel = Image.fromarray(image, mode="RGB")
    panel = panel.resize((panel.width * scale, panel.height * scale))
    framed = Image.new("RGB", (panel.width, panel.height + 28), "#0c141b")
    framed.paste(panel, (0, 28))
    ImageDraw.Draw(framed).text((10, 8), label, fill="#a8f3df")
    return framed


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a Kiwi RL vision preview")
    parser.add_argument("--world", default="room")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("kiwi_rl_preview.png"))
    args = parser.parse_args()

    config = EnvConfig(
        world_id=args.world,
        action_mode="relative_pose_v1",
        vision=VisionConfig(width=320, height=240, context_length=4),
    )
    with KiwiEnv(config) as environment:
        observation, reset_info = environment.reset(seed=args.seed)
        for _ in range(args.steps):
            observation, _, terminated, truncated, _ = environment.step(
                np.array([0.12, 0.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break

    panels = [labeled(observation["goal_rgb"], "IMAGE GOAL")]
    for index, (frame, valid, capture_time) in enumerate(
        zip(
            observation["rgb"],
            observation["rgb_valid"],
            observation["rgb_time_s"],
            strict=True,
        )
    ):
        state = "valid" if valid else "reset fill"
        panels.append(labeled(frame, f"CONTEXT {index} · t={capture_time:.2f}s · {state}"))

    columns = 3
    rows = (len(panels) + columns - 1) // columns
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    montage = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height + 42),
        "#071016",
    )
    title = (
        f"Kiwi vision_goal_v1 · {args.world} · seed {args.seed} · "
        f"pair {reset_info['task_pair_id']}"
    )
    ImageDraw.Draw(montage).text((12, 14), title, fill="#eef8f5")
    for index, panel in enumerate(panels):
        x = (index % columns) * cell_width
        y = 42 + (index // columns) * cell_height
        montage.paste(panel, (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
