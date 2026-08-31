# Kiwi simulator SDK

For installation, environment semantics, action and observation reference,
batching, Gymnasium, replay, and troubleshooting, see the
[complete Sim SDK guide](../../docs/SIM_SDK.md).

Install from the repository:

```bash
python -m pip install -e sdk/python
```

The SDK starts the deterministic Chromium/WebGL runner. Run
`cd simulator-web && npm run install:browser` for the package-lock-pinned
Chromium, or set `KIWI_CHROMIUM_EXECUTABLE` to an existing Chrome/Chromium.

```python
import numpy as np
from kiwi_sim_sdk import EnvConfig, KiwiEnv

with KiwiEnv(EnvConfig(world_id="room")) as env:
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.2, 0.0, 0.0], dtype=np.float32)
    )
```

`rgb` is chronological upright `uint8[context,height,width,3]`; `rgb_time_s`
and `rgb_sequence` identify each scheduled capture, and `goal_rgb` is the image
goal. Model-specific resizing, normalization, and tensor layout stay outside
the SDK.

Use `KiwiVectorEnv` for synchronous `reset_many`/`step_many`. Wrap an environment
with `EpisodeRecorder`, then call `replay_trace`, to reproduce actions, named
reward terms, termination, controller output, events, and frame hashes. Install
the optional `gymnasium` extra and import `KiwiGymEnv` from
`kiwi_sim_sdk.gymnasium` when a Gymnasium base class and spaces are useful.

## See the observations

Write a labeled current/context/goal montage without opening a live viewer:

```bash
python -m pip install -e 'sdk/python[preview]'
python sdk/python/examples/capture_preview.py --world room
```

Or inspect the episode as a live Rerun scope with camera images, reward curves,
and privileged robot/goal points:

```bash
python -m pip install -e 'sdk/python[rerun]'
python sdk/python/examples/rerun_scope.py --world room
```
