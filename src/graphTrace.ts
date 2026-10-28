import type { ModelGraph } from "./model";

export type TraceMode = "all" | "upstream" | "downstream" | "path";

/** Trace ancestors and descendants separately: sibling branches are not paths through a node. */
export function traceNodes(graph: ModelGraph, selected: string | null, mode: TraceMode): Set<string> | null {
  if (!selected || mode === "all" || !graph.nodes.some(node => node.id === selected)) return null;
  const known = new Set(graph.nodes.map(node => node.id));
  const parents = new Map<string, string[]>(), children = new Map<string, string[]>();
  for (const edge of graph.edges) {
    if (!known.has(edge.source) || !known.has(edge.target)) continue;
    parents.set(edge.target, [...(parents.get(edge.target) ?? []), edge.source]);
    children.set(edge.source, [...(children.get(edge.source) ?? []), edge.target]);
  }
  const result = new Set([selected]);
  for (const links of mode === "upstream" ? [parents] : mode === "downstream" ? [children] : [parents, children]) {
    const visited = new Set([selected]), queue = [selected];
    while (queue.length) for (const id of links.get(queue.pop()!) ?? []) {
      if (visited.has(id)) continue;
      visited.add(id);
      result.add(id);
      queue.push(id);
    }
  }
  return result;
}
