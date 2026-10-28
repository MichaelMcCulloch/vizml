import { expect, test } from "bun:test";
import { canConnect, editedSchematic, insertDenseAfter, parseGraph, type ModelGraph } from "../src/model";
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
