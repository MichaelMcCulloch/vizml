import { expect, test } from "bun:test";
import { canConnect, editedSchematic, parseGraph, type ModelGraph } from "../src/model";
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
