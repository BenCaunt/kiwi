from __future__ import annotations

import json
import os
import struct
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

PROTOCOL_VERSION = 1


class KiwiProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class WireResponse:
    header: dict[str, object]
    binary: bytes


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Headless supervisor closed with {remaining} bytes unread")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def default_simulator_web_dir() -> Path:
    configured = os.environ.get("KIWI_SIMULATOR_WEB_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    source = Path(__file__).resolve()
    repository_candidate = source.parents[4] / "simulator-web"
    if repository_candidate.is_dir():
        return repository_candidate
    raise FileNotFoundError(
        "Could not locate simulator-web. Set EnvConfig.simulator_web_dir or "
        "KIWI_SIMULATOR_WEB_DIR."
    )


class HeadlessTransport:
    """Private length-framed stdio transport; public callers never handle it."""

    def __init__(
        self,
        simulator_web_dir: Path | None = None,
        chromium_executable: Path | None = None,
        build_if_missing: bool = True,
    ) -> None:
        self.simulator_web_dir = (simulator_web_dir or default_simulator_web_dir()).resolve()
        supervisor = self.simulator_web_dir / "headless" / "supervisor.mjs"
        assets = self.simulator_web_dir / "dist" / "visual-runner.html"
        if not supervisor.is_file():
            raise FileNotFoundError(f"Headless supervisor not found: {supervisor}")
        if not assets.is_file():
            if not build_if_missing:
                raise FileNotFoundError(
                    f"Headless assets not found: {assets}; run npm run build"
                )
            subprocess.run(
                ["npm", "run", "build"],
                cwd=self.simulator_web_dir,
                check=True,
            )
        environment = os.environ.copy()
        if chromium_executable is not None:
            environment["KIWI_CHROMIUM_EXECUTABLE"] = str(chromium_executable)
        self._process = subprocess.Popen(
            ["node", str(supervisor)],
            cwd=self.simulator_web_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Supervisor stdio was not created")
        self._request_id = 0
        self._lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=80)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for line in iter(self._process.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    @property
    def recent_stderr(self) -> tuple[str, ...]:
        return tuple(self._stderr)

    def call(
        self,
        operation: str,
        *,
        env_id: int | None = None,
        payload: object | None = None,
    ) -> WireResponse:
        with self._lock:
            if self._process.poll() is not None:
                detail = "\n".join(self.recent_stderr)
                raise KiwiProtocolError(
                    f"Headless supervisor exited with {self._process.returncode}\n{detail}"
                )
            self._request_id += 1
            request: dict[str, object] = {"payload": payload or {}}
            if env_id is not None:
                request["env_id"] = env_id
            header = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": self._request_id,
                "operation": operation,
                "result": request,
                "arrays": [],
                "binary_length": 0,
            }
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            assert self._process.stdin is not None
            self._process.stdin.write(struct.pack("<I", len(encoded)))
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
            assert self._process.stdout is not None
            header_length = struct.unpack("<I", _read_exact(self._process.stdout, 4))[0]
            response_header = json.loads(
                _read_exact(self._process.stdout, header_length).decode("utf-8")
            )
            binary_length = int(response_header.get("binary_length", 0))
            binary = _read_exact(self._process.stdout, binary_length)
            if response_header.get("request_id") != self._request_id:
                raise KiwiProtocolError("Headless response request_id mismatch")
            if response_header.get("protocol_version") != PROTOCOL_VERSION:
                raise KiwiProtocolError("Headless protocol version mismatch")
            if not response_header.get("ok"):
                error = response_header.get("error", {})
                message = error.get("message", "Unknown headless request failure")
                raise KiwiProtocolError(str(message))
            return WireResponse(response_header, binary)

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)

    def __enter__(self) -> HeadlessTransport:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
