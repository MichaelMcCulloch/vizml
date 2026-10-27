import { expect, test } from "bun:test";
import { architectureGlyph, concatSegments, denseRunBoundaries, hasLearnedBias, straightChains } from "../src/components/architectureSemantics";
import { makeNode } from "../src/model";

test("operation glyphs stay distinct regardless of tensor rank", () => {
  expect(["concat", "add", "residual", "gather", "dot", "dense"].map(k => architectureGlyph(k as any)))
    .toEqual(["concat", "add", "add", "gather", "dot", "dense"]);
});
test("concatenation allocates one equal bar segment per incoming edge", () => {
  expect(concatSegments(5)).toEqual([[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1]]);
  expect(concatSegments(0)).toEqual([]);
});
test("dense chains have one input and one output, with forks starting new runs", () => {
  const edges = [{source:"i",target:"a"},{source:"a",target:"b"},{source:"b",target:"c"},{source:"c",target:"x"},{source:"c",target:"d"}];
  const roles = denseRunBoundaries(["i","a","b","c","d","x"], edges, id => ["a","b","c","d"].includes(id));
  expect([...roles.starts]).toEqual(["a", "d"]);
  expect([...roles.ends]).toEqual(["c", "d"]);
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
