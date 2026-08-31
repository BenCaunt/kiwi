#!/usr/bin/env python3
"""Bridge the browser simulator's WebSocket transport to Kiwi's Zenoh keys.

The browser publishes firmware-compatible payloads as binary WebSocket frames.
This process forwards them unchanged to Zenoh and translates all supported
``cmd_vel`` payload forms into raw-frame velocity commands for the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import signal
import sys
from typing import Final


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kiwi_sim_core import parse_velocity_payload  # noqa: E402


CHANNEL_SUFFIXES: Final = {
    1: "odom/twist",
    2: "lidar/ld19/raw",
    3: "camera/jpeg",
    4: "status/master",
}


class KiwiBrowserBridge:
    def __init__(
        self,
        connect: str,
        namespace: str,
        host: str,
        port: int,
    ):
        import zenoh

        self.connect = connect
        self.namespace = namespace.rstrip("/")
        self.host = host
        self.port = port
        self.clients: set[object] = set()
        self.client_roles: dict[object, str] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event = asyncio.Event()
        self.stats = {
            "browser_connections": 0,
            "commands_forwarded": 0,
            "invalid_commands": 0,
            "samples_forwarded": 0,
        }

        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([connect]))
        self.session = zenoh.open(config)
        self.publishers = {
            channel: self.session.declare_publisher(
                f"{self.namespace}/{suffix}"
            )
            for channel, suffix in CHANNEL_SUFFIXES.items()
        }
        self.command_subscriber = self.session.declare_subscriber(
            f"{self.namespace}/cmd_vel", self._on_command
        )

    def _on_command(self, sample) -> None:
        try:
            twist, timeout_s = parse_velocity_payload(bytes(sample.payload))
            message = json.dumps(
                {
                    "type": "command",
                    "twist": {
                        "vx": twist.vx,
                        "vy": twist.vy,
                        "omega": twist.omega,
                    },
                    "timeout_s": timeout_s,
                },
                separators=(",", ":"),
                allow_nan=False,
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            self.stats["invalid_commands"] += 1
            print(f"ignored invalid cmd_vel: {exc}", file=sys.stderr)
            return

        self.stats["commands_forwarded"] += 1
        if self.loop is not None and self.clients:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_to_role(message, "simulator"), self.loop
            )

    async def _broadcast_to_role(self, message: str, role: str) -> None:
        disconnected = []
        for client in tuple(self.clients):
            if self.client_roles.get(client) != role:
                continue
            try:
                await client.send(message)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.clients.discard(client)
            self.client_roles.pop(client, None)

    async def _handle_client(self, websocket) -> None:
        self.clients.add(websocket)
        self.stats["browser_connections"] += 1
        await websocket.send(
            json.dumps(
                {
                    "type": "bridge-status",
                    "connected": True,
                    "namespace": self.namespace,
                    "channels": CHANNEL_SUFFIXES,
                },
                separators=(",", ":"),
            )
        )
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_control(websocket, message)
                    continue
                if not message:
                    continue
                channel = message[0]
                publisher = self.publishers.get(channel)
                if publisher is None:
                    print(
                        f"ignored browser frame with unknown channel {channel}",
                        file=sys.stderr,
                    )
                    continue
                publisher.put(message[1:])
                self.stats["samples_forwarded"] += 1
        finally:
            self.clients.discard(websocket)
            self.client_roles.pop(websocket, None)

    async def _handle_control(self, websocket, message: str) -> None:
        try:
            document = json.loads(message)
        except json.JSONDecodeError:
            return
        if document.get("type") == "hello":
            role = document.get("role")
            if role in {"simulator", "privileged-harness"}:
                self.client_roles[websocket] = role
            await websocket.send(
                json.dumps(
                    {
                        "type": "bridge-status",
                        "connected": True,
                        "namespace": self.namespace,
                    },
                    separators=(",", ":"),
                )
            )
        elif (
            document.get("type") == "ground-truth"
            and self.client_roles.get(websocket) == "simulator"
        ):
            # This deliberately stays on the loopback WebSocket. It must never
            # become a Zenoh topic visible to the unprivileged robot stack.
            await self._broadcast_to_role(message, "privileged-harness")
        elif document.get("type") == "ping":
            await websocket.send('{"type":"pong"}')

    async def run(self) -> None:
        from websockets.asyncio.server import serve

        self.loop = asyncio.get_running_loop()
        async with serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=2 * 1024 * 1024,
            compression=None,
        ):
            print(
                f"Kiwi browser bridge ready: ws://{self.host}:{self.port} "
                f"-> {self.namespace} via {self.connect}",
                flush=True,
            )
            await self.stop_event.wait()

    def request_stop(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        self.session.close()


async def async_main(args) -> None:
    bridge = KiwiBrowserBridge(
        args.connect,
        args.namespace,
        args.host,
        args.port,
    )
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            try:
                loop.add_signal_handler(signum, bridge.request_stop)
            except NotImplementedError:
                pass
    try:
        await bridge.run()
    finally:
        bridge.close()
        print("Kiwi browser bridge stopped", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--namespace", default="kiwi/sim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
