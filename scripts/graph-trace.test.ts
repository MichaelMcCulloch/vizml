import { expect, test } from "bun:test";
import { traceNodes } from "../src/graphTrace";
import { makeNode, type ModelGraph } from "../src/model";

const graph: ModelGraph = { version: 1, name: "Branched graph", nodes: ["input", "a", "b", "join", "out", "unused"].map(id => ({ ...makeNode("dense", 0, 0), id })),
  edges: [["input", "a"], ["input", "b"], ["a", "join"], ["b", "join"], ["join", "out"]].map(([source, target]) => ({ id: source + target, source, target })) };

test("dependency traces include every ancestor without including sibling consumers", () => {
  expect([...traceNodes(graph, "a", "upstream")!].sort()).toEqual(["a", "input"]);
  expect([...traceNodes(graph, "join", "upstream")!].sort()).toEqual(["a", "b", "input", "join"]);
});
test("a path through a layer excludes side paths that merely share a parent or join", () => {
  expect([...traceNodes(graph, "a", "path")!].sort()).toEqual(["a", "input", "join", "out"]);
  expect([...traceNodes(graph, "a", "downstream")!].sort()).toEqual(["a", "join", "out"]);
});
test("all and missing selections do not filter; invalid cycles terminate", () => {
  expect(traceNodes(graph, "a", "all")).toBeNull();
  expect(traceNodes(graph, null, "path")).toBeNull();
  expect(traceNodes(graph, "deleted", "path")).toBeNull();
  expect([...traceNodes({ ...graph, edges: [...graph.edges, {id:"cycle",source:"out",target:"a"}, {id:"missing",source:"missing",target:"a"}] }, "a", "path")!].sort()).toEqual(["a", "b", "input", "join", "out"]);
});
