import { expect, test } from "bun:test";
import { separateAlignedPaths } from "../src/components/architectureLayout";

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
