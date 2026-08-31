import { describe, expect, it } from "vitest";

import { decodeWireMessage, encodeWireMessage } from "./protocol";

describe("headless wire protocol", () => {
  it("length-frames JSON metadata and raw typed arrays without base64", () => {
    const message = encodeWireMessage(
      { request_id: 7, ok: true, result: { schema: "vision_v1" } },
      [{ name: "rgb", dtype: "uint8", shape: [1, 2, 1, 3], data: new Uint8Array([1, 2, 3, 4, 5, 6]) }],
    );
    const decoded = decodeWireMessage(message);
    expect(decoded.header.request_id).toBe(7);
    expect(decoded.header.arrays[0]).toMatchObject({ name: "rgb", byte_length: 6 });
    expect(Array.from(decoded.binary)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("rejects truncated messages", () => {
    const message = new Uint8Array(encodeWireMessage({ request_id: 1, operation: "hello" }));
    expect(() => decodeWireMessage(message.subarray(0, -1))).toThrow(/length/);
  });
});
