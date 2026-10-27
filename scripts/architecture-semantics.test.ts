import { expect, test } from "bun:test";
import { architectureGlyph, concatSegments, hasLearnedBias, straightChains } from "../src/components/architectureSemantics";
import { makeNode } from "../src/model";

test("operation glyphs stay distinct regardless of tensor rank", () => {
  expect(["concat", "add", "residual", "gather", "dot", "dense"].map(k => architectureGlyph(k as any)))
    .toEqual(["concat", "add", "add", "gather", "dot", "dense"]);
});
test("concatenation partitions the bar in source order and in proportion to widths", () => {
  expect(concatSegments([2, 1, 1])).toEqual([[0, 0.5], [0.5, 0.75], [0.75, 1]]);
  expect(concatSegments([0, NaN])).toEqual([[0, 0.5], [0.5, 1]]);
});
test("bias is enabled for older layers and can be disabled", () => {
  const n = makeNode("dense", 0, 0);
  expect(hasLearnedBias(n)).toBe(true);
  delete n.data.params.bias;
  expect(hasLearnedBias(n)).toBe(true);
  expect(hasLearnedBias(n, true)).toBe(false);
  n.data.params.bias = 0;
  expect(hasLearnedBias(n)).toBe(false);
});
test("only unbranched paths of the same family form straight chains", () => {
  const edges = [{source:"a",target:"b"},{source:"b",target:"c"},{source:"c",target:"d"},{source:"c",target:"e"},{source:"e",target:"f"}];
  expect(straightChains(["a","b","c","d","e","f"], edges, id => id === "f" ? "spatial" : "vector"))
    .toEqual([["a","b","c"]]);
});
