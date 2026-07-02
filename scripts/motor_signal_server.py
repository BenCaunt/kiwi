#!/usr/bin/env python3
"""Local browser UI for sending RC pulse widths to the follower over USB serial."""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import serial
from serial.tools import list_ports


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "hardware" / "boards.json"
DEFAULT_BAUD = 115200
TARGET_SERIAL = "94A990D02EF0"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kiwi Motor Signals</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d252d;
      --muted: #64717d;
      --line: #d8dee6;
      --accent: #0b6bcb;
      --danger: #b42318;
      --ok: #157f3b;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(980px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 16px;
    }
    header, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
    }
    .dot.ok { background: var(--ok); }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    button, select, input[type="number"] {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 14px;
    }
    button {
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    button.danger {
      color: var(--danger);
      border-color: #f2b8b5;
    }
    select {
      min-width: 260px;
      padding: 0 8px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .motor {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      display: grid;
      gap: 12px;
      min-width: 0;
    }
    .motor-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
    }
    .motor-title {
      font-weight: 650;
      font-size: 14px;
    }
    .motor-value {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 14px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    input[type="number"] {
      width: 100%;
      padding: 0 8px;
      font-variant-numeric: tabular-nums;
    }
    .presets {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .log {
      min-height: 170px;
      max-height: 260px;
      overflow: auto;
      background: #101820;
      color: #d7e4ef;
      border-radius: 8px;
      padding: 12px;
      font: 12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      white-space: pre-wrap;
    }
    @media (max-width: 760px) {
      main { width: min(100vw - 20px, 980px); margin: 10px auto; }
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .presets { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      select { min-width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Kiwi Motor Signals</h1>
      <div class="status"><span id="dot" class="dot"></span><span id="statusText">Disconnected</span></div>
    </header>

    <section>
      <h2>Connection</h2>
      <div class="toolbar">
        <select id="portSelect"></select>
        <button id="refreshBtn">Refresh</button>
        <button id="connectBtn" class="primary">Connect</button>
        <button id="disconnectBtn">Disconnect</button>
        <button id="statusBtn">Status</button>
      </div>
    </section>

    <section>
      <h2>Pulse Widths</h2>
      <div class="grid" id="motorGrid"></div>
    </section>

    <section>
      <h2>Presets</h2>
      <div class="presets">
        <button data-preset="1000">1000 us</button>
        <button data-preset="1500">1500 us</button>
        <button data-preset="2000">2000 us</button>
        <button id="calBtn">Cal Sequence</button>
        <button id="bootCalBtn">Flash Boot Cal</button>
        <button id="rampBtn">Ramp</button>
        <button id="stopBtn" class="danger">Stop</button>
        <button id="clearLogBtn">Clear Log</button>
      </div>
    </section>

    <section>
      <h2>Serial Log</h2>
      <div id="log" class="log"></div>
    </section>
  </main>

  <script>
    const motors = [
      { key: "d0", label: "D0 / GPIO1", value: 1500 },
      { key: "d1", label: "D1 / GPIO2", value: 1500 },
      { key: "d2", label: "D2 / GPIO3", value: 1500 },
    ];
    const state = { connected: false, sendTimer: null, lastSent: "" };
    const grid = document.getElementById("motorGrid");
    const logEl = document.getElementById("log");
    const dot = document.getElementById("dot");
    const statusText = document.getElementById("statusText");
    const portSelect = document.getElementById("portSelect");

    function clampPulse(v) {
      return Math.max(1000, Math.min(2000, Math.round(Number(v) || 1500)));
    }

    function renderMotors() {
      grid.innerHTML = "";
      for (const motor of motors) {
        const wrap = document.createElement("div");
        wrap.className = "motor";
        wrap.innerHTML = `
          <div class="motor-head">
            <div class="motor-title">${motor.label}</div>
            <div class="motor-value" id="${motor.key}-value">${motor.value} us</div>
          </div>
          <input id="${motor.key}-range" type="range" min="1000" max="2000" step="1" value="${motor.value}">
          <input id="${motor.key}-number" type="number" min="1000" max="2000" step="1" value="${motor.value}">
        `;
        grid.appendChild(wrap);
        const range = wrap.querySelector(`#${motor.key}-range`);
        const number = wrap.querySelector(`#${motor.key}-number`);
        const value = wrap.querySelector(`#${motor.key}-value`);
        function update(v) {
          motor.value = clampPulse(v);
          range.value = motor.value;
          number.value = motor.value;
          value.textContent = `${motor.value} us`;
          scheduleSend();
        }
        range.addEventListener("input", () => update(range.value));
        number.addEventListener("change", () => update(number.value));
      }
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function setStatus(connected, text) {
      state.connected = connected;
      dot.classList.toggle("ok", connected);
      statusText.textContent = text;
    }

    function appendLog(lines) {
      if (!lines || !lines.length) return;
      logEl.textContent += lines.join("");
      if (logEl.textContent.length > 12000) {
        logEl.textContent = logEl.textContent.slice(-10000);
      }
      logEl.scrollTop = logEl.scrollHeight;
    }

    async function refreshPorts() {
      const data = await api("/api/ports");
      portSelect.innerHTML = "";
      for (const port of data.ports) {
        const opt = document.createElement("option");
        opt.value = port.device;
        opt.textContent = `${port.device}${port.role ? " - " + port.role : ""}`;
        portSelect.appendChild(opt);
      }
      if (data.preferred_port) portSelect.value = data.preferred_port;
    }

    function pulsePayload() {
      return {
        d0: motors[0].value,
        d1: motors[1].value,
        d2: motors[2].value,
      };
    }

    function scheduleSend() {
      clearTimeout(state.sendTimer);
      state.sendTimer = setTimeout(sendPulses, 80);
    }

    async function sendPulses() {
      const payload = pulsePayload();
      const signature = `${payload.d0},${payload.d1},${payload.d2}`;
      if (signature === state.lastSent) return;
      state.lastSent = signature;
      try {
        await api("/api/pulses", { method: "POST", body: JSON.stringify(payload) });
      } catch (err) {
        setStatus(false, err.message);
      }
    }

    function setAll(value) {
      const pulse = clampPulse(value);
      for (const motor of motors) motor.value = pulse;
      renderMotors();
      state.lastSent = "";
      sendPulses();
    }

    async function command(commandText) {
      try {
        await api("/api/command", { method: "POST", body: JSON.stringify({ command: commandText }) });
      } catch (err) {
        setStatus(false, err.message);
      }
    }

    async function pollState() {
      try {
        const data = await api("/api/state");
        setStatus(data.connected, data.connected ? `Connected ${data.port}` : "Disconnected");
        appendLog(data.lines);
      } catch (err) {
        setStatus(false, err.message);
      } finally {
        setTimeout(pollState, 500);
      }
    }

    document.getElementById("refreshBtn").addEventListener("click", refreshPorts);
    document.getElementById("connectBtn").addEventListener("click", async () => {
      const data = await api("/api/connect", { method: "POST", body: JSON.stringify({ port: portSelect.value }) });
      setStatus(data.connected, data.connected ? `Connected ${data.port}` : "Disconnected");
    });
    document.getElementById("disconnectBtn").addEventListener("click", async () => {
      const data = await api("/api/disconnect", { method: "POST", body: "{}" });
      setStatus(data.connected, "Disconnected");
    });
    document.getElementById("statusBtn").addEventListener("click", () => command("status"));
    document.getElementById("calBtn").addEventListener("click", () => command("cal"));
    document.getElementById("bootCalBtn").addEventListener("click", async () => {
      await api("/api/flash_boot_cal", { method: "POST", body: "{}" });
    });
    document.getElementById("rampBtn").addEventListener("click", () => command("ramp"));
    document.getElementById("stopBtn").addEventListener("click", async () => { setAll(1500); await command("stop"); });
    document.getElementById("clearLogBtn").addEventListener("click", () => { logEl.textContent = ""; });
    for (const btn of document.querySelectorAll("[data-preset]")) {
      btn.addEventListener("click", () => setAll(btn.dataset.preset));
    }

    renderMotors();
    refreshPorts().catch(err => setStatus(false, err.message));
    pollState();
  </script>
</body>
</html>
"""


def normalize(value: str | None) -> str:
    return (value or "").replace(":", "").replace("-", "").upper()


def load_target_serial() -> str:
    if not INVENTORY.exists():
        return TARGET_SERIAL
    try:
        inventory = json.loads(INVENTORY.read_text())
    except json.JSONDecodeError:
        return TARGET_SERIAL
    for board in inventory.get("boards", []):
        if board.get("role") == "follower":
            return normalize(board.get("usb_serial") or board.get("mac")) or TARGET_SERIAL
    return TARGET_SERIAL


# The firmware reverts direct pulse holds to neutral after 8 s (runaway
# failsafe). Re-send the active pulses well inside that window so slider
# positions hold while the bridge is alive.
HEARTBEAT_PERIOD_S = 2.0


class SerialBridge:
    def __init__(self, baud: int) -> None:
        self.baud = baud
        self.lock = threading.Lock()
        self.ser: serial.Serial | None = None
        self.reader: threading.Thread | None = None
        self.heartbeat: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lines: queue.Queue[str] = queue.Queue(maxsize=300)
        self.port: str | None = None
        self.last_pulses: str | None = None

    def connect(self, port: str) -> None:
        with self.lock:
            self._close_locked()
            self.ser = serial.Serial(port, self.baud, timeout=0.1)
            self.ser.dtr = True
            self.ser.rts = False
            self.port = port
            self.stop_event.clear()
            self.reader = threading.Thread(target=self._read_loop, daemon=True)
            self.reader.start()
            self.heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat.start()
            self._log(f"connected {port}\n")

    def disconnect(self) -> None:
        with self.lock:
            self._close_locked()

    def connected(self) -> bool:
        with self.lock:
            return self.ser is not None and self.ser.is_open

    def send(self, command: str, quiet: bool = False) -> None:
        command = command.strip()
        if not command:
            return
        with self.lock:
            if self.ser is None or not self.ser.is_open:
                raise RuntimeError("serial port is not connected")
            self.ser.write((command + "\n").encode("ascii"))
            self.ser.flush()
        if command.startswith("pulses "):
            self.last_pulses = command
        elif command in ("s", "stop", "n", "neutral", "cal", "ramp", "low", "high", "min", "max"):
            self.last_pulses = None
        if not quiet:
            self._log(f">>> {command}\n")

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(HEARTBEAT_PERIOD_S):
            command = self.last_pulses
            if command is None:
                continue
            try:
                self.send(command, quiet=True)
            except RuntimeError:
                return

    def drain_lines(self) -> list[str]:
        drained: list[str] = []
        while True:
            try:
                drained.append(self.lines.get_nowait())
            except queue.Empty:
                return drained

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                ser = self.ser
            if ser is None:
                return
            try:
                data = ser.read(4096)
            except serial.SerialException as exc:
                self._log(f"serial read error: {exc}\n")
                return
            if data:
                self._log(data.decode("utf-8", errors="replace"))
            else:
                time.sleep(0.02)

    def _log(self, text: str) -> None:
        try:
            self.lines.put_nowait(text)
        except queue.Full:
            try:
                self.lines.get_nowait()
            except queue.Empty:
                pass
            self.lines.put_nowait(text)

    def _close_locked(self) -> None:
        self.stop_event.set()
        self.last_pulses = None
        if self.ser is not None:
            try:
                self.ser.close()
            except serial.SerialException:
                pass
        self.ser = None
        self.port = None


def list_serial_ports() -> tuple[list[dict[str, Any]], str | None]:
    target_serial = load_target_serial()
    ports: list[dict[str, Any]] = []
    preferred: str | None = None
    for port in list_ports.comports():
        vid_pid = ""
        if port.vid is not None and port.pid is not None:
            vid_pid = f"{port.vid:04X}:{port.pid:04X}"
        serial_number = normalize(port.serial_number)
        role = "follower" if serial_number == target_serial else ""
        if role and preferred is None:
            preferred = port.device
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "serial": port.serial_number or "",
                "vid_pid": vid_pid,
                "role": role,
            }
        )
    return ports, preferred


def validate_pulse(value: Any) -> int:
    pulse = int(value)
    if pulse < 1000 or pulse > 2000:
        raise ValueError("pulse must be between 1000 and 2000")
    return pulse


def flash_boot_cal(bridge: SerialBridge) -> None:
    bridge.disconnect()
    import subprocess

    result = subprocess.run(
        [
            str(Path.home() / ".platformio" / "penv" / "bin" / "pio"),
            "run",
            "-e",
            "follower_dominion_boot_cal",
            "-t",
            "upload",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    bridge._log(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("boot-cal firmware upload failed")


class Handler(BaseHTTPRequestHandler):
    bridge: SerialBridge

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(INDEX_HTML)
        elif self.path == "/api/ports":
            ports, preferred = list_serial_ports()
            self._send_json({"ports": ports, "preferred_port": preferred})
        elif self.path == "/api/state":
            self._send_json(
                {
                    "connected": self.bridge.connected(),
                    "port": self.bridge.port,
                    "lines": self.bridge.drain_lines(),
                }
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/connect":
                port = payload.get("port") or list_serial_ports()[1]
                if not port:
                    raise RuntimeError("no follower serial port found")
                self.bridge.connect(str(port))
                self._send_json({"connected": True, "port": self.bridge.port})
            elif self.path == "/api/disconnect":
                self.bridge.disconnect()
                self._send_json({"connected": False})
            elif self.path == "/api/pulses":
                d0 = validate_pulse(payload.get("d0"))
                d1 = validate_pulse(payload.get("d1"))
                d2 = validate_pulse(payload.get("d2"))
                self.bridge.send(f"pulses {d0} {d1} {d2}")
                self._send_json({"ok": True})
            elif self.path == "/api/command":
                command = str(payload.get("command", "")).strip()
                if not re.fullmatch(r"[-A-Za-z0-9_ +.?]+", command):
                    raise ValueError("invalid command")
                self.bridge.send(command)
                self._send_json({"ok": True})
            elif self.path == "/api/flash_boot_cal":
                flash_boot_cal(self.bridge)
                self._send_json({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--connect", action="store_true", help="Connect to the recorded follower board at startup.")
    args = parser.parse_args()

    bridge = SerialBridge(args.baud)
    Handler.bridge = bridge

    if args.connect:
        _, preferred = list_serial_ports()
        if preferred:
            bridge.connect(preferred)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Motor signal UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.disconnect()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
