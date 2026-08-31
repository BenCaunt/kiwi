import { describe, expect, it } from "vitest";

import { contentRevision } from "./revisions";

describe("contentRevision", () => {
  it("is insensitive to object key order but sensitive to values", () => {
    expect(contentRevision({ a: 1, b: 2 })).toBe(contentRevision({ b: 2, a: 1 }));
    expect(contentRevision({ a: 1, b: 2 })).not.toBe(contentRevision({ a: 1, b: 3 }));
  });
});
