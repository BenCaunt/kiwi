import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const GLB_JSON_CHUNK_TYPE = 0x4e4f534a;

describe("Kiwi CAD model", () => {
  it("ships a compact Draco-compressed GLB with the simulator", () => {
    const model = readFileSync(
      new URL("../../public/assets/kiwi/kiwi-robot.glb", import.meta.url),
    );

    expect(model.subarray(0, 4).toString("ascii")).toBe("glTF");
    expect(model.readUInt32LE(4)).toBe(2);
    expect(model.readUInt32LE(8)).toBe(model.length);
    expect(model.length).toBeLessThan(2 * 1024 * 1024);

    const jsonLength = model.readUInt32LE(12);
    expect(model.readUInt32LE(16)).toBe(GLB_JSON_CHUNK_TYPE);
    const document = JSON.parse(
      model.subarray(20, 20 + jsonLength).toString("utf8"),
    ) as { extensionsRequired?: string[] };
    expect(document.extensionsRequired).toContain("KHR_draco_mesh_compression");
  });
});
