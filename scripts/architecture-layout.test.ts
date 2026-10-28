import { expect, test } from "bun:test";
import { compactGridColumns, placeSideInputs, separateAlignedPaths } from "../src/components/architectureLayout";

test("separates overlapping branches without breaking their centerlines", () => {
  const positions = new Map([
    ["a", {x: 0, y: 0, z: 0}], ["b", {x: 4, y: 1, z: 0}],
    ["c", {x: 0, y: 0, z: 0}], ["d", {x: 4, y: -1, z: 0}],
  ]);
  separateAlignedPaths(["a", "b", "c", "d"], [["a", "b"], ["c", "d"]], positions, () => 1);
  const [a, b, c, d] = ["a", "b", "c", "d"].map(id => positions.get(id)!);
  expect([a.y, a.z]).toEqual([b.y, b.z]);
  expect([c.y, c.z]).toEqual([d.y, d.z]);
  expect(Math.hypot(a.y - c.y, a.z - c.z)).toBeGreaterThanOrEqual(3.5);
});

test("leaves parked attachments alone and does not separate distant paths", () => {
  const positions = new Map([
    ["a", {x: 0, y: 0, z: 0}], ["b", {x: 10, y: 0, z: 0}],
    ["bias", {x: 0, y: 0, z: 0}],
  ]);
  separateAlignedPaths(["a", "b", "bias"], [], positions, () => 1, new Set(["bias"]));
  expect([...positions.values()].map(p => [p.y, p.z])).toEqual([[0, 0], [0, 0], [0, 0]]);
});

test("automatic placement uses the manual two-unit grid and compacts sparse lanes", () => {
  const positions = new Map([
    ["a", {x: 0, y: 0.3, z: 8}], ["b", {x: 2, y: 0.3, z: 8}],
    ["c", {x: 0, y: -8, z: -8}],
  ]);
  separateAlignedPaths(["a", "b", "c"], [["a", "b"]], positions, () => 0.25);
  for (const point of positions.values()) {
    expect(point.x % 2).toBe(0);
    expect(Math.abs(point.y % 2)).toBe(0);
    expect(Math.abs(point.z % 2)).toBe(0);
  }
  expect(Math.abs(positions.get("a")!.z)).toBeLessThan(8);
  expect(positions.get("a")!.z).toBe(positions.get("b")!.z);
});

test("packs independent branches without widening every depth for one large glyph", () => {
  const edges = [
    {source:"a", target:"b"}, {source:"b", target:"c"}, {source:"c", target:"d"},
    {source:"x", target:"y"}, {source:"y", target:"z"},
  ];
  const columns = compactGridColumns(["a", "x", "b", "y", "c", "z", "d"], edges,
    id => ({left: id === "y" ? 1.85 : 0.15, right: 0.15}));
  expect(columns.get("b")! - columns.get("a")!).toBe(2);
  expect(columns.get("d")! - columns.get("a")!).toBe(6);
  expect(columns.get("y")! - columns.get("x")!).toBe(4);
  for (const edge of edges) expect(columns.get(edge.target)!).toBeGreaterThan(columns.get(edge.source)!);
  for (const x of columns.values()) expect(x % 2).toBe(0);
});

test("bias side inputs sit immediately behind their dependent on distinct grid cells", () => {
  const points = new Map([
    ["add", {x: 8, y: 0, z: 0}], ["bias-a", {x: 0, y: 0, z: 0}],
    ["bias-b", {x: 0, y: 0, z: 0}], ["other", {x: 6, y: -4, z: -4}],
  ]);
  placeSideInputs([{id:"bias-a",target:"add"}, {id:"bias-b",target:"add"}], points, () => 1);
  const a = points.get("bias-a")!, b = points.get("bias-b")!;
  expect(a.x).toBe(6);
  expect(a.z).toBe(-4);
  expect(a.y).toBe(0);
  expect(Math.abs(a.y - b.y)).toBeGreaterThanOrEqual(4);
  for (const point of [a, b]) for (const value of [point.x, point.y, point.z]) expect(Math.abs(value % 2)).toBe(0);
});
