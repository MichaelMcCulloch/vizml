import { expect, test } from "bun:test";
import { adoptUserNodes, ConnectionMode, getEdgePosition, Position, type InternalNodeBase } from "@xyflow/system";
import { cleanGraph, makeNode, type ModelGraph } from "../src/model";
import { withNodeMeasurements } from "../src/graphView";

function fixture() {
  const graph: ModelGraph = {
    version: 1,
    name: "Graph replacement",
    nodes: [makeNode("dense", 0, 0, "a"), makeNode("dense", 250, 0, "b")].map(node => ({
      ...node, measured: { width: 185, height: 112 },
    })),
    edges: [{ id: "a-b", source: "a", target: "b" }],
  };
  const lookup = new Map<string, InternalNodeBase>();
  const parents = new Map<string, Map<string, InternalNodeBase>>();
  adoptUserNodes(graph.nodes, lookup, parents);
  for (const node of lookup.values()) {
    node.internals.handleBounds = {
      source: [{ id: "out", nodeId: node.id, type: "source", position: Position.Right, x: 181, y: 52, width: 8, height: 8 }],
      target: [{ id: "in", nodeId: node.id, type: "target", position: Position.Left, x: -4, y: 52, width: 8, height: 8 }],
    };
  }
  const position = () => getEdgePosition({
    id: "a-b", sourceNode: lookup.get("a")!, targetNode: lookup.get("b")!,
    sourceHandle: "out", targetHandle: "in", connectionMode: ConnectionMode.Strict,
  });
  return { graph, lookup, parents, position };
}

test("history/model replacements keep existing edges renderable during adoption", () => {
  // Reproduce the library failure with the same cleaned nodes used by history.
  const broken = fixture();
  expect(broken.position()).not.toBeNull();
  adoptUserNodes(cleanGraph(broken.graph).nodes, broken.lookup, broken.parents);
  expect(broken.position()).toBeNull();

  const fixed = fixture();
  const replacement = cleanGraph(fixed.graph);
  replacement.nodes[1].position = { x: 500, y: 200 };
  const next = withNodeMeasurements(replacement, fixed.graph);
  adoptUserNodes(next.nodes, fixed.lookup, fixed.parents);
  const position = fixed.position();
  expect(position).not.toBeNull();
  expect(position!.targetX).toBe(496);
  expect(position!.targetY).toBe(256);
  expect(next.edges).toBe(replacement.edges);
  expect(replacement.nodes[0].measured).toBeUndefined();
});

test("editing labels and parameters retains bounds until the new glyph is measured", () => {
  const { graph, lookup, parents, position } = fixture();
  const replacement = cleanGraph(graph);
  replacement.nodes[0].data.label = "Updated dense layer";
  replacement.nodes[0].data.params.units = 256;
  const next = withNodeMeasurements(replacement, graph);
  adoptUserNodes(next.nodes, lookup, parents);
  expect(position()).not.toBeNull();
  expect(next.nodes[0].data.label).toBe("Updated dense layer");
  expect(next.nodes[0].data.params.units).toBe(256);
});

test("reused IDs with different ports discard stale measurements", () => {
  const { graph, lookup, parents, position } = fixture();
  const replacement: ModelGraph = {
    ...graph,
    nodes: graph.nodes.map(node => node.id === "a"
      ? { ...node, data: { ...node.data, kind: "feature_input" } }
      : node),
  };
  const next = withNodeMeasurements(replacement, graph);
  expect(next.nodes[0].measured).toBeUndefined();
  expect(next.nodes[1]).toBe(graph.nodes[1]);
  adoptUserNodes(next.nodes, lookup, parents);
  expect(lookup.get("a")!.internals.handleBounds).toBeUndefined();
  expect(position()).toBeNull(); // Wait for the new handles, never reuse the old ones.
});

test("switching a reused node between builder and schematic refreshes its ports", () => {
  const { graph } = fixture();
  const next = withNodeMeasurements({
    ...graph,
    nodes: graph.nodes.map(node => ({ ...node, data: { ...node.data, output_shape: [128] } })),
  }, graph);
  expect(next.nodes.every(node => !node.measured)).toBe(true);
});

test("new measurements win and genuinely new nodes still initialize normally", () => {
  const { graph } = fixture();
  const replacement = {
    ...graph,
    nodes: [...graph.nodes.map(node => ({ ...node, measured: { width: 185, height: 140 } })), makeNode("dense", 500, 0, "c")],
  };
  expect(withNodeMeasurements(replacement, graph)).toBe(replacement);
  expect(replacement.nodes[2].measured).toBeUndefined();
  const unmeasured = cleanGraph(graph);
  expect(withNodeMeasurements(cleanGraph(graph), unmeasured).nodes[0].measured).toBeUndefined();
});

test("retained display measurements never leak into portable graphs", () => {
  const { graph } = fixture();
  const portable = cleanGraph(graph);
  const restored = withNodeMeasurements(portable, graph);
  expect(cleanGraph(restored)).toEqual(portable);
  expect(restored.nodes[0].measured).not.toBe(graph.nodes[0].measured);
});
