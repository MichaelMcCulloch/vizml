import type { LayerNode, ModelGraph } from "./model";

function hasMeasurements(node: LayerNode) {
  return Number.isFinite(node.measured?.width) && Number.isFinite(node.measured?.height)
    && (node.measured?.width ?? 0) > 0 && (node.measured?.height ?? 0) > 0;
}

/** History and imports omit view state. Keep valid sizes for unchanged glyphs
 * so React Flow retains their port bounds while it adopts the replacement. */
export function withNodeMeasurements(next: ModelGraph, previous: ModelGraph): ModelGraph {
  const previousNodes = new Map(previous.nodes.map(node => [node.id, node]));
  let changed = false;
  const nodes = next.nodes.map(node => {
    const old = previousNodes.get(node.id);
    const sameGlyph = old && old.type === node.type && old.data.kind === node.data.kind
      && Array.isArray(old.data.output_shape) === Array.isArray(node.data.output_shape);
    if (old && !sameGlyph && node.measured) {
      changed = true;
      const { measured: _measured, ...unmeasured } = node;
      return unmeasured;
    }
    if (!hasMeasurements(node) && sameGlyph && hasMeasurements(old)) {
      changed = true;
      return { ...node, measured: { ...old.measured } };
    }
    return node;
  });
  return changed ? { ...next, nodes } : next;
}
