import type { Twist2 } from "./sim/types";

export class KeyboardDrive {
  readonly linearSpeed: number;
  readonly angularSpeed: number;
  private readonly pressed = new Set<string>();
  resetRequested = false;
  pauseRequested = false;

  constructor(linearSpeed = 0.5, angularSpeed = 1.4) {
    this.linearSpeed = linearSpeed;
    this.angularSpeed = angularSpeed;
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.clear);
  }

  private onKeyDown = (event: KeyboardEvent): void => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes((event.target as HTMLElement).tagName)) {
      return;
    }
    const key = event.key.toLowerCase();
    if (["w", "a", "s", "d", "q", "e", "r", "p", " "].includes(key)) {
      event.preventDefault();
    }
    if (!event.repeat && key === "r") this.resetRequested = true;
    if (!event.repeat && key === "p") this.pauseRequested = true;
    this.pressed.add(key);
  };

  private onKeyUp = (event: KeyboardEvent): void => {
    this.pressed.delete(event.key.toLowerCase());
  };

  private clear = (): void => {
    this.pressed.clear();
  };

  consumeReset(): boolean {
    const requested = this.resetRequested;
    this.resetRequested = false;
    return requested;
  }

  consumePause(): boolean {
    const requested = this.pauseRequested;
    this.pauseRequested = false;
    return requested;
  }

  isActive(): boolean {
    return ["w", "a", "s", "d", "q", "e", " "].some((key) =>
      this.pressed.has(key),
    );
  }

  command(): Twist2 {
    if (this.pressed.has(" ")) return { vx: 0, vy: 0, omega: 0 };
    return {
      vx:
        this.linearSpeed *
        (Number(this.pressed.has("w")) - Number(this.pressed.has("s"))),
      vy:
        this.linearSpeed *
        (Number(this.pressed.has("a")) - Number(this.pressed.has("d"))),
      omega:
        this.angularSpeed *
        (Number(this.pressed.has("q")) - Number(this.pressed.has("e"))),
    };
  }
}
