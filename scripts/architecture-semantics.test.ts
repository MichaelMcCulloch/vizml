import { expect, test } from "bun:test";
import { architectureGlyph, concatInputs, concatSegments, denseContinuesThroughActivation, denseRunBoundaries, hasLearnedBias, straightChains, sharesDenseInputColumn } from "../src/components/architectureSemantics";

test("a dense layer followed by activation and another dense layer is hidden", () => {
  const glyphs = { a: "dense", act: "activation", b: "dense", out: "other" } as const;
  const edges = [{ source: "a", target: "act" }, { source: "act", target: "b" }, { source: "b", target: "out" }];
  const kind = (id: string) => glyphs[id as keyof typeof glyphs];
  expect(denseContinuesThroughActivation("a", edges, kind)).toBe(true);
  expect(denseContinuesThroughActivation("b", edges, kind)).toBe(false);
  expect(denseContinuesThroughActivation("b", edges.map(edge => ({ source: edge.target, target: edge.source })), kind)).toBe(true);
  expect(denseContinuesThroughActivation("a", [...edges, { source: "act", target: "out" }], kind)).toBe(false);
});
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


test("dense input columns reflect the declared feature slice", () => {
  expect(sharesDenseInputColumn(true, [13708], 79)).toBe(false);
  expect(sharesDenseInputColumn(true, [79], 79)).toBe(true);
  expect(sharesDenseInputColumn(true, [79], undefined)).toBe(true);
  expect(sharesDenseInputColumn(false, [79], 79)).toBe(false);
  expect(sharesDenseInputColumn(true, [12, 79], 79)).toBe(false);
});


test("focused concatenations preserve hidden inputs and original segment order", () => {
  const edges = ["a", "b", "c", "d", "e"].map(source => ({source, target:"join"}));
  const inputs = concatInputs("join", edges, new Set(["a", "b", "c", "d", "e"]), new Set(["a", "c", "e"]));
  expect(inputs.map(input => [input.edge.source, input.visible])).toEqual([["a", true], ["b", false], ["c", true], ["d", false], ["e", true]]);
  expect(concatSegments(inputs.length)[2]).toEqual([0.4, 0.6]);
});

test("schematic neuron bias uses matching declared tensor ownership", () => {
  const n = makeNode("dense", 0, 0);
  delete n.data.params.bias;
  n.data.module_paths = ["encoder.condition"];
  n.data.parameter_tensors = [{name:"encoder.condition.weight"}, {name:"encoder.condition.bias"}];
  expect(hasLearnedBias(n, true)).toBe(true);
  n.data.params.bias = 0;
  expect(hasLearnedBias(n, true)).toBe(false);
  delete n.data.params.bias;
  n.data.parameter_tensors = [{name:"encoder.condition.weight"}, {name:"encoder.other.bias"}];
  expect(hasLearnedBias(n, true)).toBe(false);
  n.data.parameter_tensors = [{name:"encoder.condition.bias"}];
  expect(hasLearnedBias(n, true)).toBe(false);
});
