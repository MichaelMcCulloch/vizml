import { expect, test } from "bun:test";
import { autoLayout, autoLayoutSubset, canConnect, cleanGraph, editedSchematic, insertDenseAfter, parseGraph, type Analysis, type ModelGraph } from "../src/model";
import { preferredSnapshotScope } from "../src/snapshotView";

const diagram = (): ModelGraph => parseGraph(JSON.stringify({ version: 1, name: "Schematic", mode: "schematic", metadata: { weights: { id: "old" }, parameter_counts: { total: 4 } }, nodes: [
  { id: "a", position: { x: 0, y: 0 }, data: { kind: "feature_input", label: "A", params: {}, output_shape: ["B", 4], parameter_count: 0 } },
  { id: "b", position: { x: 200, y: 0 }, data: { kind: "classification", label: "B", params: {}, output_shape: ["B", 1], parameter_count: 4 } },
  { id: "c", position: { x: 400, y: 0 }, data: { kind: "dense", label: "C", params: {}, output_shape: ["B", 1], parameter_count: 0 } },
], edges: [{ id: "ab", source: "a", target: "b" }] }));

test("schematic editing permits declared connections but prevents cycles and duplicates", () => {
  const graph = diagram();
  expect(canConnect(graph, "b", "c")).toBe(true);
  expect(canConnect(graph, "b", "a")).toBe(false);
  expect(canConnect(graph, "a", "b")).toBe(false);
});
test("structural schematic edits detach weights without mutating source provenance", () => {
  const original = diagram();
  const edited = editedSchematic(original);
  expect(edited.metadata?.weights).toBeUndefined();
  expect(edited.metadata?.source_edited).toBe(true);
  expect(edited.metadata?.parameter_counts).toEqual({ total: 4 });
  expect(original.metadata?.weights).toEqual({ id: "old" });
});

test("edited schematics reopen with all newly added nodes visible", () => {
  const graph = diagram();
  graph.metadata = { ...graph.metadata, default_scope: "narrow", view_scopes: [{ id: "narrow", label: "Narrow", ids: ["a", "b"] }] };
  expect(preferredSnapshotScope(graph)).toBe("narrow");
  expect(preferredSnapshotScope(editedSchematic(graph))).toBe("all");
});

test("inserting a dense layer routes all outgoing branches through it", () => {
  const graph = diagram();
  graph.nodes[2].data.params = { units: 8, bias: 0, activation: "relu" };
  graph.edges.push({ id: "ca", source: "c", target: "a" }, { id: "cb", source: "c", target: "b" });
  const result = insertDenseAfter(graph, "c")!;
  const inserted = result.graph.nodes.find(node => node.id === result.id)!;
  expect(inserted.data.params.units).toBe(8);
  expect(inserted.data.params.bias).toBe(0);
  expect(inserted.data.declaration_pending).toBe(true);
  expect(result.graph.edges.filter(edge => edge.source === "c").map(edge => edge.target)).toEqual([result.id]);
  expect(result.graph.edges.filter(edge => edge.source === result.id).map(edge => edge.target)).toEqual(["a", "b"]);
  expect(result.graph.metadata?.weights).toBeUndefined();
  expect(graph.edges.filter(edge => edge.source === "c")).toHaveLength(2);
});

test("inserting into a linear MLP makes room without moving unrelated nodes", () => {
  const graph = diagram();
  graph.nodes[0].data.kind = "dense";
  graph.nodes[0].data.params = {units: 4, bias: 1};
  graph.nodes[2].position.y = 400;
  const result = insertDenseAfter(graph, "a")!;
  expect(result.graph.nodes.find(n => n.id === result.id)!.position).toEqual({x: 230, y: 0});
  expect(result.graph.nodes.find(n => n.id === "b")!.position.x).toBe(430);
  expect(result.graph.nodes.find(n => n.id === "c")!.position.x).toBe(400);
});

test("insertion finds a free row when an unrelated branch occupies the new column", () => {
  const graph = diagram();
  graph.nodes[0].data.kind = "dense";
  graph.nodes[2].position = { x: 230, y: 0 };
  const result = insertDenseAfter(graph, "a")!;
  expect(result.graph.nodes.find(n => n.id === result.id)!.position).toEqual({ x: 230, y: 140 });
  expect(result.graph.nodes.find(n => n.id === "c")!.position).toEqual({ x: 230, y: 0 });
});

test("branch layout keeps hidden nodes and connections intact and avoids their cards", () => {
  const graph = diagram();
  graph.nodes[1].position = { x: 1000, y: 0 };
  graph.nodes[2].position = { x: 250, y: 0 };
  const result = autoLayoutSubset(graph, new Set(["a", "b"]));
  expect(result.nodes.find(n => n.id === "c")!.position).toEqual({ x: 250, y: 0 });
  expect(result.nodes.find(n => n.id === "a")!.position).toEqual({ x: 0, y: 200 });
  expect(result.nodes.find(n => n.id === "b")!.position).toEqual({ x: 250, y: 200 });
  expect(result.edges).toEqual(graph.edges);
  expect(graph.nodes[1].position).toEqual({ x: 1000, y: 0 });
});

test("full layout keeps an unbranched continuation aligned", () => {
  const graph = diagram();
  graph.edges.push({ id: "bc", source: "b", target: "c" });
  const result = autoLayout(graph, null);
  expect(result.nodes.map(n => n.position.y)).toEqual([40, 40, 40]);
  expect(result.nodes.map(n => n.position.x)).toEqual([0, 250, 500]);
  expect(result.edges).toEqual(graph.edges);
});

test("layout follows current connections even when analysis has stale depths", () => {
  const graph = diagram();
  graph.edges.push({ id: "bc", source: "b", target: "c" });
  const info = (depth: number) => ({ depth, shape: [], input_shape: [], parameters: 0, elements: 0, merge: "", dependencies: 0 });
  const stale: Analysis = { valid: true, nodes: { a: info(2), b: info(0), c: info(0) }, order: ["a", "b", "c"], parameters: 0, activation_bytes: 0, depth: 2, errors: [], warnings: [], outputs: [] };
  const result = autoLayout(graph, stale);
  expect(result.nodes.map(node => node.position.x)).toEqual([0, 250, 500]);
});

test("3D layer positions survive a portable graph round trip", () => {
  const graph = diagram();
  graph.metadata = { ...graph.metadata, architecture_layout: { c: [2, 0, -4] } };
  const reopened = parseGraph(JSON.stringify(cleanGraph(graph)));
  expect(reopened.metadata?.architecture_layout).toEqual({ c: [2, 0, -4] });
});
