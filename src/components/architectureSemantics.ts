import type { Kind, LayerNode } from "../model";

export type ArchitectureGlyph = "concat" | "add" | "gather" | "dot" | "dense" | "normalize" | "pool" | "activation" | "embedding" | "other";

/** Operation identity must take precedence over the shape of its output tensor. */
export function architectureGlyph(kind: Kind): ArchitectureGlyph {
  if (kind === "concat") return "concat";
  if (kind === "add" || kind === "residual") return "add";
  if (kind === "gather") return "gather";
  if (kind === "dot") return "dot";
  if (kind === "dense" || kind === "classification" || kind === "regression") return "dense";
  if (kind === "layernorm" || kind === "standardize") return "normalize";
  if (kind === "global_avgpool" || kind === "globalavgpool2d" || kind === "globalmaxpool2d" || kind === "maxpool2d") return "pool";
  if (kind === "activation") return "activation";
  if (kind === "embedding") return "embedding";
  return "other";
}

export function hasLearnedBias(node: LayerNode, schematic = false): boolean {
  // An explicit declaration wins, including an explicit false. Older builder
  // graphs omitted bias because Linear defaulted to true.
  const declared = node.data.params.bias;
  if (declared !== undefined) {
    if (typeof declared === "boolean") return declared;
    if (typeof declared === "number") return declared === 1;
    if (typeof declared === "string") return ["1", "true"].includes(declared.trim().toLowerCase());
    return false;
  }
  if (!schematic) return true;
  // A schematic may instead declare its owned tensors. Only infer a bias for
  // an explicitly named module, not an arbitrary bias in a composite block.
  const paths = node.data.module_paths;
  const tensors = node.data.parameter_tensors;
  if (!Array.isArray(paths) || !Array.isArray(tensors)) return false;
  const names = new Set(tensors.flatMap(tensor => tensor && typeof tensor === "object" && typeof tensor.name === "string" ? [tensor.name] : []));
  return paths.some(path => typeof path === "string" && names.has(`${path}.weight`) && names.has(`${path}.bias`));
}

/** Assign equal, non-overlapping segments in the order of incoming edges. */
export function concatSegments(count: number): [number, number][] {
  if (!Number.isInteger(count) || count <= 0) return [];
  return Array.from({ length: count }, (_, i) => [i / count, (i + 1) / count]);
}

/** Preserve the full input order when a focused view hides some branches. */
export function concatInputs<T extends { source: string; target: string }>(
  target: string, edges: T[], knownIds: Set<string>, visibleIds: Set<string>,
): { edge: T; visible: boolean }[] {
  return edges.filter(edge => edge.target === target && knownIds.has(edge.source))
    .map(edge => ({ edge, visible: visibleIds.has(edge.source) }));
}

/** Dense runs are display groups only; no operators or dependencies are changed. */
export function denseRunBoundaries(
  ids: string[], edges: { source: string; target: string }[], isDense: (id: string) => boolean,
): { starts: Set<string>; ends: Set<string> } {
  const valid = new Set(ids);
  const incoming = new Map<string, string[]>(), outgoing = new Map<string, string[]>();
  for (const { source, target } of edges) {
    if (!valid.has(source) || !valid.has(target)) continue;
    incoming.set(target, [...(incoming.get(target) ?? []), source]);
    outgoing.set(source, [...(outgoing.get(source) ?? []), target]);
  }
  const connected = (a: string, b: string) => isDense(a) && isDense(b)
    && outgoing.get(a)?.length === 1 && incoming.get(b)?.length === 1;
  const starts = new Set<string>(), ends = new Set<string>();
  for (const id of ids.filter(isDense)) {
    const parents = incoming.get(id) ?? [], children = outgoing.get(id) ?? [];
    if (parents.length !== 1 || !connected(parents[0], id)) starts.add(id);
    if (children.length !== 1 || !connected(id, children[0])) ends.add(id);
  }
  return { starts, ends };
}

/** A pointwise activation between dense layers does not make the first an output. */
export function denseContinuesThroughActivation(
  id: string, edges: { source: string; target: string }[],
  kind: (id: string) => ArchitectureGlyph | undefined,
): boolean {
  let current = id;
  const visited = new Set([id]);
  for (;;) {
    const next = edges.filter(edge => edge.source === current);
    if (next.length !== 1) return false;
    current = next[0].target;
    if (visited.has(current) || edges.filter(edge => edge.target === current).length !== 1) return false;
    visited.add(current);
    const glyph = kind(current);
    if (glyph === "dense") return true;
    if (glyph !== "activation") return false;
  }
}

/** Align unbranched paths, without forcing joins or forks into a chain. */
export function straightChains(
  ids: string[], edges: { source: string; target: string }[],
  family: (id: string) => string,
): string[][] {
  const valid = new Set(ids);
  const incoming = new Map<string, string[]>(), outgoing = new Map<string, string[]>();
  for (const {source, target} of edges) {
    if (!valid.has(source) || !valid.has(target)) continue;
    outgoing.set(source, [...(outgoing.get(source) ?? []), target]);
    incoming.set(target, [...(incoming.get(target) ?? []), source]);
  }
  const next = (id: string) => {
    const children = outgoing.get(id) ?? [];
    if (children.length !== 1) return undefined;
    const child = children[0];
    return (incoming.get(child)?.length === 1 && family(id) === family(child) && family(id) !== "other") ? child : undefined;
  };
  const predecessor = new Map<string, string>();
  for (const id of ids) { const child = next(id); if (child) predecessor.set(child, id); }
  const chains: string[][] = [];
  for (const id of ids) {
    if (predecessor.has(id)) continue;
    const chain = [id];
    const seen = new Set(chain);
    let child = next(id);
    while (child && !seen.has(child)) { chain.push(child); seen.add(child); child = next(child); }
    if (chain.length > 1) chains.push(chain);
  }
  return chains;
}

/** A dense input column can be shared only with the vector it actually consumes. */
export function sharesDenseInputColumn(isInputNode: boolean, shape: number[] | undefined, inputFeatures: unknown): boolean {
  const declared = Number(inputFeatures);
  return isInputNode && shape?.length === 1 && (!declared || shape[0] === declared);
}
