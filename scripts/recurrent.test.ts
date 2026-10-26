import { expect, test } from "bun:test";
import * as THREE from "three";
import { catalog, cleanGraph, examples, isInput, makeNode, parseGraph, recurrentKinds, type NodeAnalysis } from "../src/model";
import { recurrentGeometry } from "../src/components/recurrentGeometry";
import { pairConnectionPorts, makeConnectionCurve } from "../src/components/connectionGeometry";
import { pathProgress } from "../src/components/connectionFlow";

function glyph(kind: typeof recurrentKinds[number], hidden: number, layers: number, bidirectional = 0, size = [16, 24]) {
  const node = makeNode(kind);
  node.data.params = { ...node.data.params, hidden_size: hidden, layers, bidirectional };
  const spatial = kind.startsWith("conv_");
  const shape = spatial ? [1048576, hidden, ...size] : [1048576, hidden * (1 + bidirectional)];
  const info: NodeAnalysis = { shape, input_shape: [], parameters: 0, elements: 1, merge: "single", dependencies: 1, depth: 1 };
  const group = new THREE.Group(), meshes: THREE.Object3D[] = [];
  const result = recurrentGeometry(group, node, info, catalog[kind].color, false, meshes);
  return { group, meshes, result };
}

test("all recurrent starter graphs round-trip through portable JSON", async () => {
  for (const kind of recurrentKinds) {
    const expected = cleanGraph(examples[kind]());
    expect(parseGraph(await Bun.file(new URL(`../templates/${kind}.vizml.json`, import.meta.url)).text())).toEqual(expected);
    expect(expected.nodes.some(n => n.data.kind === kind)).toBe(true);
    expect(isInput(expected.nodes[0].data.kind)).toBe(true);
    expect(expected.edges).toHaveLength(expected.nodes.length - 1);
    expect(catalog[kind].category).toBe("Recurrent");
  }
});

test("all vector cells share circular ports without twisting their connections", () => {
  for (const hidden of [1, 2, 3, 12, 64, 65535]) {
    const all = ["rnn", "gru", "lstm"].map(kind => glyph(kind as "rnn", hidden, 2, 1).result);
    expect(all[0].ports).toEqual(all[1].ports);
    expect(all[1].ports).toEqual(all[2].ports);
    for (const { ports, inputPorts, ringRadius } of all) {
      expect(ports).toHaveLength(Math.min(hidden * 2, 24));
      for (const p of [...ports!, ...inputPorts!]) {
        expect(p.x).toBe(0);
        expect(Math.hypot(p.y, p.z)).toBeCloseTo(ringRadius);
      }
    }
    const from = all[0].ports!, to = all[2].ports!.map(p => p.clone().add(new THREE.Vector3(5, 0, 0)));
    for (const [a, b] of pairConnectionPorts(from, to)) {
      expect(a.y).toBeCloseTo(b.y);
      expect(a.z).toBeCloseTo(b.z);
      const mid = makeConnectionCurve(a, b).getPoint(0.5);
      expect(mid.y).toBeCloseTo(a.y);
      expect(mid.z).toBeCloseTo(a.z);
    }
  }
});

test("closed recurrence threads have continuous distance coordinates for shared flow", () => {
  for (const kind of recurrentKinds) {
    const { result } = glyph(kind, 32, 2);
    expect(result.loops).toHaveLength(kind.endsWith("lstm") ? 2 : 1);
    for (const loop of result.loops) {
      expect(loop[0].distanceTo(loop.at(-1)!)).toBeLessThan(1e-10);
      const { progress, length } = pathProgress(loop);
      expect(length).toBeGreaterThan(1);
      expect(progress[0]).toBe(0);
      expect(progress.at(-1)).toBeCloseTo(1);
      expect([...progress].every(Number.isFinite)).toBe(true);
    }
  }
});

test("spatial ports match the translucent volume and all glyphs stay bounded", () => {
  for (const kind of recurrentKinds) {
    const { group, meshes, result } = glyph(kind, 65535, 16, kind.startsWith("conv_") ? 0 : 1, [1, 256]);
    expect(meshes.length).toBeLessThan(20);
    expect(meshes.every(m => typeof m.userData.nodeId === "string")).toBe(true);
    group.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(group);
    expect(Math.max(Math.abs(bounds.min.x), Math.abs(bounds.max.x))).toBeLessThanOrEqual(result.halfDepth);
    expect(Math.max(Math.abs(bounds.min.y), Math.abs(bounds.max.y), Math.abs(bounds.min.z), Math.abs(bounds.max.z))).toBeLessThanOrEqual(result.radius);
    let vertices = 0;
    group.traverse(object => {
      const geometry = (object as THREE.Mesh).geometry;
      if (!geometry) return;
      vertices += geometry.attributes.position.count;
      expect(Array.from(geometry.attributes.position.array).every(Number.isFinite)).toBe(true);
    });
    expect(vertices).toBeLessThan(10000);
    if (result.spatial) {
      expect(result.ports).toBeUndefined();
      expect(result.width).toBeGreaterThan(result.height);
      for (const face of [result.spatialFaces!.input, result.spatialFaces!.output]) {
        expect(face.dimensions).toEqual([result.height, result.width]);
        expect(face.center).toEqual(new THREE.Vector3());
      }
    }
  }
});

test("open woven cells can be selected through the center of their weave", () => {
  for (const kind of recurrentKinds) {
    const { group, meshes } = glyph(kind, 32, 2);
    group.updateMatrixWorld(true);
    const ray = new THREE.Raycaster(new THREE.Vector3(3, 0, 0), new THREE.Vector3(-1, 0, 0));
    expect(ray.intersectObjects(meshes).length).toBeGreaterThan(0);
  }
});
