from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import EnvConfig
from .transport import HeadlessTransport, WireResponse

Observation = dict[str, Any]

_DTYPES: dict[str, np.dtype[Any]] = {
    "uint8": np.dtype(np.uint8),
    "uint32": np.dtype("<u4"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


def _arrays(response: WireResponse) -> dict[str, npt.NDArray[Any]]:
    arrays: dict[str, npt.NDArray[Any]] = {}
    for descriptor in response.header.get("arrays", []):
        name = str(descriptor["name"])
        dtype = _DTYPES[str(descriptor["dtype"])]
        shape = tuple(int(value) for value in descriptor["shape"])
        offset = int(descriptor["offset"])
        byte_length = int(descriptor["byte_length"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != byte_length:
            raise ValueError(f"Array {name} shape and byte length disagree")
        arrays[name] = np.frombuffer(
            response.binary,
            dtype=dtype,
            count=expected // dtype.itemsize,
            offset=offset,
        ).reshape(shape)
    return arrays


def decode_result_document(
    response: WireResponse, result: Mapping[str, Any]
) -> tuple[Observation, dict[str, Any]]:
    metadata = dict(result["observation"])
    arrays = _arrays(response)
    observation: Observation = {
        "schema": metadata["schema"],
        "rgb": arrays[str(metadata["rgb"])],
        "rgb_valid": arrays[str(metadata["rgb_valid"])],
        "rgb_time_s": arrays[str(metadata["rgb_time_s"])],
        "rgb_sequence": arrays[str(metadata["rgb_sequence"])],
        "goal_rgb_valid": np.uint8(metadata["goal_rgb_valid"]),
        "goal_rgb_sequence": metadata["goal_rgb_sequence"],
        "calibration": metadata["calibration"],
    }
    goal_name = metadata.get("goal_rgb")
    if goal_name is not None:
        observation["goal_rgb"] = arrays[str(goal_name)]
    return observation, dict(result.get("info", {}))


def _decode_result(response: WireResponse) -> tuple[Observation, dict[str, Any]]:
    return decode_result_document(response, response.header["result"])


def _triple(values: object, name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} action must have shape (3,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} action must contain finite values")
    return [float(value) for value in array]


def encode_action(config: EnvConfig, action: object) -> dict[str, object]:
    if isinstance(action, Mapping) and "kind" in action:
        return dict(action)
    if config.action_mode == "relative_trajectory_v1":
        array = np.asarray(action, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
            raise ValueError(
                f"relative trajectory action must have shape (horizon, 3), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("relative trajectory action must contain finite values")
        return {
            "kind": "relative_trajectory",
            "waypoints": [
                {"dx": float(row[0]), "dy": float(row[1]), "dyaw": float(row[2])}
                for row in array
            ],
        }
    values = _triple(action, config.action_mode)
    if config.action_mode == "relative_pose_v1":
        return {"kind": "relative_pose", "dx": values[0], "dy": values[1], "dyaw": values[2]}
    return {"kind": "twist", "vx": values[0], "vy": values[1], "omega": values[2]}


class KiwiEnv:
    """Gym-style visual environment with fixed-duration deterministic actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        _transport: HeadlessTransport | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self._owns_transport = _transport is None
        self._transport = _transport or HeadlessTransport(
            self.config.simulator_web_dir,
            self.config.chromium_executable,
            self.config.build_if_missing,
        )
        self.hello = dict(self._transport.call("hello").header["result"])
        created = self._transport.call("create", payload=self.config.protocol_document())
        self._env_id = int(created.header["result"]["env_id"])
        self._closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        request: dict[str, object] = {"seed": int(seed or 0)}
        if options:
            request["options"] = dict(options)
        return _decode_result(
            self._transport.call("reset", env_id=self._env_id, payload=request)
        )

    def step(
        self, action: object
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        encoded = encode_action(self.config, action)
        response = self._transport.call(
            "step", env_id=self._env_id, payload={"action": encoded}
        )
        observation, info = _decode_result(response)
        result = response.header["result"]
        return (
            observation,
            float(result["reward"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            info,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._transport.call("close", env_id=self._env_id)
        finally:
            if self._owns_transport:
                self._transport.close()
            self._closed = True

    def __enter__(self) -> KiwiEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
