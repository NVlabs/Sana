import assert from "node:assert/strict";
import test from "node:test";
import { sourceFile, validateSelection } from "../src/inputs.js";

test("requires all inputs and bounded integer settings", () => {
  const selected = {
    video: {},
    image: {},
    prompt: "original prompt",
    seed: 42,
    limit: 1450,
  };
  assert.doesNotThrow(() => validateSelection(selected));
  for (const change of [
    { video: null },
    { image: null },
    { prompt: " " },
    { seed: NaN },
    { seed: 1.5 },
    { limit: -1 },
    { limit: 14401 },
  ]) {
    assert.throws(() => validateSelection({ ...selected, ...change }));
  }
});
test("custom file bytes are preserved and empty/oversized inputs refused", async () => {
  const file = new File(["semantic video"], "input.mp4");
  assert.equal(await sourceFile(file, "video"), file);
  await assert.rejects(sourceFile(new File([], "empty.mp4"), "video"));
  await assert.rejects(sourceFile({ size: 257 * 1048576 }, "video"));
  await assert.rejects(sourceFile({ size: 17 * 1048576 }, "image"));
  await assert.rejects(
    sourceFile("https://untrusted.invalid/input.mp4", "video"),
  );
});
