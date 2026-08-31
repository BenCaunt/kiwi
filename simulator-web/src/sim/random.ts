/** Stable named seed derivation so enabling one subsystem cannot shift another. */
export function deriveSeed(episodeSeed: number, streamName: string): number {
  let hash = (episodeSeed >>> 0) ^ 0x811c9dc5;
  for (let index = 0; index < streamName.length; index += 1) {
    hash ^= streamName.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export class SeededRandom {
  private state: number;
  private spareNormal: number | null = null;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  }

  normal(): number {
    if (this.spareNormal !== null) {
      const value = this.spareNormal;
      this.spareNormal = null;
      return value;
    }
    const u = Math.max(this.uniform(), Number.EPSILON);
    const v = this.uniform();
    const magnitude = Math.sqrt(-2 * Math.log(u));
    const phase = 2 * Math.PI * v;
    this.spareNormal = magnitude * Math.sin(phase);
    return magnitude * Math.cos(phase);
  }
}
