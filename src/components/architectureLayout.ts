/** Place paths on a compact 3D grid, keeping unbranched paths straight. */
export function separateAlignedPaths(
  ids: string[], chains: string[][],
  positions: Map<string, { x: number; y: number; z: number }>,
  radius: (id: string) => number,
  excluded: Set<string> = new Set(),
  grid = 2,
): void {
  const grouped = new Set(chains.flat());
  const groups = [
    ...chains.map(chain => chain.filter(id => positions.has(id) && !excluded.has(id))).filter(chain => chain.length),
    ...ids.filter(id => !grouped.has(id) && positions.has(id) && !excluded.has(id)).map(id => [id]),
  ];
  // Larger paths get their preferred lane first; individual joins can then
  // occupy the nearest available grid cell without bending the path.
  groups.sort((a, b) => b.length - a.length);
  const placed: string[][] = [];
  for (const group of groups) {
    const avg = (axis: "y" | "z") => group.reduce((sum, id) => sum + positions.get(id)![axis], 0) / group.length;
    // Pull the old flow lanes closer before finding the nearest free cell.
    const preferredY = Math.round(avg("y") * 0.3 / grid);
    const preferredZ = Math.round(avg("z") * 0.3 / grid);
    const available = (y: number, z: number) => placed.every(other => group.every(id => other.every(otherId => {
      const p = positions.get(id)!, q = positions.get(otherId)!;
      if (Math.abs(p.x - q.x) > grid * 0.5) return true;
      const clearance = Math.max(3.5, radius(id) + radius(otherId) + 1.5);
      return Math.hypot(y - q.y, z - q.z) >= clearance;
    })));
    let chosen: [number, number] | undefined;
    for (let ring = 0; !chosen; ring++) {
      const candidates: [number, number][] = [];
      for (let dy = -ring; dy <= ring; dy++) for (let dz = -ring; dz <= ring; dz++) {
        if (Math.max(Math.abs(dy), Math.abs(dz)) === ring) candidates.push([dy, dz]);
      }
      candidates.sort((a, b) => a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2);
      for (const [dy, dz] of candidates) {
        const y = (preferredY + dy) * grid, z = (preferredZ + dz) * grid;
        if (available(y, z)) { chosen = [y, z]; break; }
      }
    }
    for (const id of group) { const point = positions.get(id)!; point.y = chosen[0]; point.z = chosen[1]; }
    placed.push(group);
  }
}

/** Pack each dependency as close as its glyph footprints permit on the grid. */
export function compactGridColumns(
  ids: string[], edges: { source: string; target: string }[],
  footprint: (id: string) => { left: number; right: number }, grid = 2,
): Map<string, number> {
  const valid = new Set(ids);
  const incoming = new Map<string, string[]>(), outgoing = new Map<string, string[]>();
  for (const {source, target} of edges) {
    if (!valid.has(source) || !valid.has(target)) continue;
    incoming.set(target, [...(incoming.get(target) ?? []), source]);
    outgoing.set(source, [...(outgoing.get(source) ?? []), target]);
  }
  const gap = (a: string, b: string) => Math.max(grid,
    Math.ceil((footprint(a).right + footprint(b).left + 0.35) / grid) * grid);
  const x = new Map<string, number>();
  for (const id of ids) {
    const parents = (incoming.get(id) ?? []).filter(parent => x.has(parent));
    x.set(id, parents.length ? Math.max(...parents.map(parent => x.get(parent)! + gap(parent, id))) : 0);
  }
  // Move shorter branches toward their first join, without changing their
  // dependency order. Long skip connections remain long by necessity.
  for (const id of [...ids].reverse()) {
    const children = outgoing.get(id) ?? [];
    if (children.length) x.set(id, Math.max(x.get(id)!, Math.min(...children.map(child => x.get(child)! - gap(id, child)))));
  }
  return x;
}
