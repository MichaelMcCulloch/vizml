/** Resolve cross-channel overlaps while keeping each aligned path rigid. */
export function separateAlignedPaths(
  ids: string[], chains: string[][],
  positions: Map<string, { x: number; y: number; z: number }>,
  radius: (id: string) => number,
  excluded: Set<string> = new Set(),
): void {
  const grouped = new Set(chains.flat());
  const groups = [
    ...chains.map(chain => chain.filter(id => positions.has(id) && !excluded.has(id))).filter(chain => chain.length),
    ...ids.filter(id => !grouped.has(id) && positions.has(id) && !excluded.has(id)).map(id => [id]),
  ];
  for (const group of groups) {
    if (group.length < 2) continue;
    const y = group.reduce((sum, id) => sum + positions.get(id)!.y, 0) / group.length;
    const z = group.reduce((sum, id) => sum + positions.get(id)!.z, 0) / group.length;
    for (const id of group) { const point = positions.get(id)!; point.y = y; point.z = z; }
  }
  // Compare only nodes at similar depths; distant parts of a path need not
  // repel each other. Move whole paths to preserve their centerlines.
  for (let pass = 0; pass < 24; pass++) {
    let moved = false;
    for (let a = 0; a < groups.length; a++) for (let b = a + 1; b < groups.length; b++) {
      let needed = 0;
      for (const left of groups[a]) for (const right of groups[b]) {
        const p = positions.get(left)!, q = positions.get(right)!;
        if (Math.abs(p.x - q.x) > 3.5) continue;
        const clearance = Math.max(3.5, radius(left) + radius(right) + 1.5);
        needed = Math.max(needed, clearance - Math.hypot(p.y - q.y, p.z - q.z));
      }
      if (needed <= 0.01) continue;
      const p = positions.get(groups[a][0])!, q = positions.get(groups[b][0])!;
      let dy = p.y - q.y, dz = p.z - q.z;
      const distance = Math.hypot(dy, dz);
      if (distance < 0.001) { dy = 1; dz = 0; } else { dy /= distance; dz /= distance; }
      const step = needed * 0.51;
      for (const id of groups[a]) { const point = positions.get(id)!; point.y += dy * step; point.z += dz * step; }
      for (const id of groups[b]) { const point = positions.get(id)!; point.y -= dy * step; point.z -= dz * step; }
      moved = true;
    }
    if (!moved) break;
  }
}
