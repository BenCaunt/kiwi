import type { Pose2, Twist2 } from "../sim/types";

export interface BridgeCommand {
  twist: Twist2;
  timeoutSeconds?: number;
}

export type BridgeState = "connecting" | "offline" | "online";

export interface BridgeCallbacks {
  onCommand(command: BridgeCommand): void;
  onState(state: BridgeState, namespace?: string): void;
}

interface CommandDocument {
  type: "command";
  twist: Twist2;
  timeout_s?: number | null;
}

interface StatusDocument {
  type: "bridge-status";
  connected: boolean;
  namespace?: string;
}

function isTwist(value: unknown): value is Twist2 {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<Twist2>;
  return [candidate.vx, candidate.vy, candidate.omega].every(
    (component) => typeof component === "number" && Number.isFinite(component),
  );
}

export class ZenohBridgeClient {
  readonly url: string;
  state: BridgeState = "offline";
  namespace?: string;
  commandsReceived = 0;
  samplesSent = 0;
  private socket?: WebSocket;
  private reconnectTimer?: number;
  private stopped = true;
  private readonly callbacks: BridgeCallbacks;

  constructor(callbacks: BridgeCallbacks, url = "ws://127.0.0.1:8767") {
    this.callbacks = callbacks;
    this.url = url;
  }

  get connected(): boolean {
    return this.state === "online" && this.socket?.readyState === WebSocket.OPEN;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer !== undefined) window.clearTimeout(this.reconnectTimer);
    this.socket?.close();
    this.setState("offline");
  }

  publish(channel: number, payload: Uint8Array): boolean {
    if (!this.connected || !this.socket) return false;
    const frame = new Uint8Array(payload.length + 1);
    frame[0] = channel;
    frame.set(payload, 1);
    this.socket.send(frame);
    this.samplesSent += 1;
    return true;
  }

  publishGroundTruth(world: string, pose: Pose2, simulationTime: number): boolean {
    if (!this.connected || !this.socket) return false;
    this.socket.send(
      JSON.stringify({
        type: "ground-truth",
        world,
        pose,
        simulation_time_s: simulationTime,
      }),
    );
    return true;
  }

  private setState(state: BridgeState, namespace = this.namespace): void {
    this.state = state;
    this.namespace = namespace;
    this.callbacks.onState(state, namespace);
  }

  private connect(): void {
    if (this.stopped) return;
    this.setState("connecting");
    const socket = new WebSocket(this.url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.addEventListener("open", () => {
      socket.send(
        JSON.stringify({
          type: "hello",
          client: "kiwi-threejs",
          role: "simulator",
        }),
      );
    });
    socket.addEventListener("message", (event) => this.handleMessage(event.data));
    socket.addEventListener("close", () => {
      if (this.socket === socket) this.socket = undefined;
      this.setState("offline");
      if (!this.stopped) {
        this.reconnectTimer = window.setTimeout(() => this.connect(), 1500);
      }
    });
    socket.addEventListener("error", () => socket.close());
  }

  private handleMessage(message: unknown): void {
    if (typeof message !== "string") return;
    let document: CommandDocument | StatusDocument;
    try {
      document = JSON.parse(message) as CommandDocument | StatusDocument;
    } catch {
      return;
    }

    if (document.type === "bridge-status") {
      this.setState(document.connected ? "online" : "offline", document.namespace);
      return;
    }
    if (document.type !== "command" || !isTwist(document.twist)) return;
    const timeoutSeconds =
      typeof document.timeout_s === "number" && Number.isFinite(document.timeout_s)
        ? document.timeout_s
        : undefined;
    this.commandsReceived += 1;
    this.callbacks.onCommand({ twist: document.twist, timeoutSeconds });
  }
}
