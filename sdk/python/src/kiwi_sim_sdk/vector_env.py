from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .config import EnvConfig
from .env import Observation, decode_result_document, encode_action
from .transport import HeadlessTransport


class KiwiVectorEnv:
    """Synchronous batch API using one Chromium worker and one binary RPC."""

    def __init__(self, configs: Sequence[EnvConfig]) -> None:
        if not configs:
            raise ValueError("KiwiVectorEnv requires at least one configuration")
        first = configs[0]
        if first is None:
            raise ValueError("Configuration is unavailable")
        self.configs = list(configs)
        self._transport = HeadlessTransport(
            first.simulator_web_dir,
            first.chromium_executable,
            first.build_if_missing,
        )
        self.hello = dict(self._transport.call("hello").header["result"])
        response = self._transport.call(
            "create_many",
            payload={"configs": [config.protocol_document() for config in self.configs]},
        )
        environments = response.header["result"]["environments"]
        self._env_ids = [int(item["env_id"]) for item in environments]
        self._closed = False

    @property
    def num_envs(self) -> int:
        return len(self._env_ids)

    def reset(
        self, seeds: Sequence[int] | None = None
    ) -> tuple[list[Observation], list[dict[str, Any]]]:
        resolved = list(seeds) if seeds is not None else [0] * self.num_envs
        if len(resolved) != self.num_envs:
            raise ValueError("seeds must have one item per environment")
        response = self._transport.call(
            "reset_many",
            payload={
                "items": [
                    {"env_id": env_id, "seed": int(seed)}
                    for env_id, seed in zip(self._env_ids, resolved, strict=True)
                ]
            },
        )
        decoded = [
            decode_result_document(response, item)
            for item in response.header["result"]["items"]
        ]
        return [item[0] for item in decoded], [item[1] for item in decoded]

    def step(
        self, actions: Sequence[object]
    ) -> tuple[
        list[Observation], list[float], list[bool], list[bool], list[dict[str, Any]]
    ]:
        if len(actions) != self.num_envs:
            raise ValueError("actions must have one item per environment")
        response = self._transport.call(
            "step_many",
            payload={
                "items": [
                    {
                        "env_id": env_id,
                        "action": encode_action(config, action),
                    }
                    for env_id, config, action in zip(
                        self._env_ids, self.configs, actions, strict=True
                    )
                ]
            },
        )
        observations: list[Observation] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []
        for item in response.header["result"]["items"]:
            observation, info = decode_result_document(response, item)
            observations.append(observation)
            rewards.append(float(item["reward"]))
            terminated.append(bool(item["terminated"]))
            truncated.append(bool(item["truncated"]))
            infos.append(info)
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._transport.call("close_many", payload={"env_ids": self._env_ids})
        finally:
            self._transport.close()
            self._closed = True

    def __enter__(self) -> KiwiVectorEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
