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
  // Older builder graphs omit bias because Linear historically defaulted to true.
  // Schematics may carry an explicit boolean from the imported source.
  return schematic && node.data.params.bias === undefined ? false : Number(node.data.params.bias ?? 1) !== 0;
}

/** Assign each input a non-overlapping segment of the bar in edge order. */
export function concatSegments(weights: number[]): [number, number][] {
  const safe = weights.map(n => Number.isFinite(n) && n > 0 ? n : 1);
  const total = safe.reduce((a, b) => a + b, 0);
  let cursor = 0;
  return safe.map(n => {
    const start = cursor / total;
    cursor += n;
    return [start, cursor / total];
  });
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
