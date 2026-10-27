import { expect, test } from "bun:test";
import { formatDeclaredShape, parseDeclaredShape } from "../src/schematicEditing";

test("shape fields accept normal notation, commas, and existing JSON", () => {
  expect(parseDeclaredShape("B × L × 64")).toEqual(["B", "L", 64]);
  expect(parseDeclaredShape("B, 3, 45, 49")).toEqual(["B", 3, 45, 49]);
  expect(parseDeclaredShape("B x 2*L x 64")).toEqual(["B", "2*L", 64]);
  expect(parseDeclaredShape('["B", "L", 64]')).toEqual(["B", "L", 64]);
});
test("shape fields reject invalid or empty dimensions and keep unusual symbols lossless", () => {
  for (const input of ["", "B", "B × -3", "B × 1.5", "B,,3", "[]", "[true]", "[1,2,3,4,5,6,7]"])
    expect(() => parseDeclaredShape(input)).toThrow();
  const unusual = ["B", "2×L", "32", "1.5", "A x B"];
  expect(parseDeclaredShape(formatDeclaredShape(unusual))).toEqual(unusual);
});
