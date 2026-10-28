import { expect, test } from "bun:test";
import * as THREE from "three";
import { connectionExtent, makeConnectionCurve, pairConnectionPorts, spatialConnection } from "../src/components/connectionGeometry";
import { applyFlowAttributes, flowProfile, pathProgress } from "../src/components/connectionFlow";

const row = (x: number, count: number, descending = false) =>
  Array.from({ length: count }, (_, i) => new THREE.Vector3(
    x, count === 1 ? 0 : (i / (count - 1) - 0.5) * (descending ? -3 : 3), 0,
  ));

test("sequence-to-vector bundles preserve spatial order instead of twisting", () => {
  for (const [source, target] of [[row(0, 12, true), row(6, 13)], [row(0, 13), row(6, 12, true)]]) {
    const originalSource = source.map(point => point.clone());
    const originalTarget = target.map(point => point.clone());
    const pairs = pairConnectionPorts(source, target);
    expect(pairs[0].map(point => point.y)).toEqual([1.5, 1.5]);
    expect(pairs.at(-1)!.map(point => point.y)).toEqual([-1.5, -1.5]);
    const curves = pairs.map(([start, end]) => makeConnectionCurve(start, end, 0.18));
    for (let sample = 0; sample <= 40; sample++)
      for (let i = 1; i < curves.length; i++)
        expect(curves[i - 1].getPoint(sample / 40).y).toBeGreaterThan(curves[i].getPoint(sample / 40).y);
    expect(source).toEqual(originalSource);
    expect(target).toEqual(originalTarget);
  }
});

test("unequal bundles span both ends and single ports connect centrally", () => {
  const pairs = pairConnectionPorts(row(0, 13), row(3, 4, true));
  expect(pairs).toHaveLength(4);
  [1.5, 0.5, -0.5, -1.5].forEach((y, i) => expect(pairs[i][0].y).toBeCloseTo(y));
  expect(pairConnectionPorts(row(0, 13), row(3, 1))[0][0].y).toBe(0);
  expect(pairConnectionPorts([], row(3, 4))).toEqual([]);
});

test("symmetric circular ports do not exchange sides when resized or moved in 3D", () => {
  for (const count of [6, 8, 12, 16, 24]) {
    const circle = (x: number, y: number, z: number, radius: number) => Array.from({ length: count }, (_, i) =>
      new THREE.Vector3(x, y + Math.cos(i / count * Math.PI * 2) * radius, z + Math.sin(i / count * Math.PI * 2) * radius));
    const source = circle(0, 0, 0, 1.02);
    const target = circle(5, 100, 7, 0.72).reverse();
    for (const [a, b] of pairConnectionPorts(source, target)) {
      expect(a.y / 1.02).toBeCloseTo((b.y - 100) / 0.72);
      expect(a.z / 1.02).toBeCloseTo((b.z - 7) / 0.72);
    }
  }
});

test("ribbon boundaries never invert, including reversed and manually offset layers", () => {
  for (const x of [8, 0, -4]) {
    const source = connectionExtent(row(0, 13));
    const target = connectionExtent(row(x, 12, true).map(point => point.add(new THREE.Vector3(0, 8, -6))));
    const top = makeConnectionCurve(source.top, target.top, 0.18);
    const bottom = makeConnectionCurve(source.bottom, target.bottom, 0.18);
    for (let sample = 0; sample <= 40; sample++) {
      const a = top.getPoint(sample / 40), b = bottom.getPoint(sample / 40);
      expect(a.y - b.y).toBeGreaterThan(0);
      expect(a.x).toBeCloseTo(b.x);
      expect(a.z).toBeCloseTo(b.z);
    }
  }
  const single = connectionExtent(row(0, 1));
  expect(single.top.y - single.bottom.y).toBeCloseTo(0.24);
});

test("spatial-grid corners retain their orientation through a downsampling connection", () => {
  const volume = spatialConnection(new THREE.Vector3(0, 0, 0), new THREE.Vector3(6, 4, -5), [6, 4], [2, 1]);
  const { curves, geometry } = volume;
  for (let sample = 0; sample < curves[0].length; sample++) {
    const [a, b, c, d] = curves.map(points => points[sample]);
    expect(c.y - a.y).toBeGreaterThan(0);
    expect(b.z - a.z).toBeGreaterThan(0);
    expect(a.y).toBeCloseTo(b.y);
    expect(c.y).toBeCloseTo(d.y);
    expect(a.z).toBeCloseTo(d.z);
    expect(b.z).toBeCloseTo(c.z);
    expect(volume.centerline[sample]).toEqual(a.clone().add(c).multiplyScalar(0.5));
  }
  const position = geometry.attributes.position;
  expect(volume.stations).toHaveLength(position.count);
  expect(volume.across).toHaveLength(position.count);
  expect(volume.faceOpacity).toHaveLength(position.count);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let i = 0; i < geometry.index!.count; i += 3) {
    a.fromBufferAttribute(position, geometry.index!.getX(i));
    b.fromBufferAttribute(position, geometry.index!.getX(i + 1));
    c.fromBufferAttribute(position, geometry.index!.getX(i + 2));
    expect(b.sub(a).cross(c.sub(a)).lengthSq()).toBeGreaterThan(0);
  }
  geometry.dispose();
});

test("flow travels by distance even with uneven curve samples", () => {
  const points = [0, 1, 4, 10].map(x => new THREE.Vector3(x, 0, 0));
  const { progress, length } = pathProgress(points);
  expect(length).toBe(10);
  expect(progress[0]).toBe(0);
  expect(progress[1]).toBeCloseTo(0.1);
  expect(progress[2]).toBeCloseTo(0.4);
  expect(progress[3]).toBe(1);
  expect([...pathProgress([points[0], points[0]]).progress]).toEqual([0, 0]);
  expect(pathProgress([]).length).toBe(0);
});

test("throughput has a restrained pace and long connections do not race", () => {
  const slow = flowProfile(0.2, "edge", 10);
  const fast = flowProfile(1, "edge", 10);
  const long = flowProfile(1, "edge", 100);
  expect(fast.rate).toBeGreaterThan(slow.rate);
  expect(fast.rate / slow.rate).toBeLessThan(1.4);
  expect(long.rate).toBeLessThan(fast.rate);
  expect(fast.strength).toBeLessThanOrEqual(1);
  expect(flowProfile(0, "edge", 10).strength).toBe(0);
  for (const length of [0, 1, 100, Number.NaN]) {
    const profile = flowProfile(1, "edge", length);
    expect(Object.values(profile).every(Number.isFinite)).toBe(true);
    expect(1 / (profile.rate * (1 + 2 * profile.width))).toBeGreaterThanOrEqual(6);
  }
});

test("ribbon sides and line strands share the same flow without mutating geometry", () => {
  const geometry = new THREE.PlaneGeometry(4, 2, 2, 1);
  const before = geometry.attributes.position.array.slice();
  const progress = [0, 0.5, 1, 0, 0.5, 1];
  const profile = flowProfile(0.8, "bundle", 4);
  applyFlowAttributes(geometry, progress, { profile, color: "#7acfd6", endColor: "#baabec", opacity: 0.06, surface: true, flushEnd: true }, [0, 0, 0, 1, 1, 1]);
  expect(geometry.attributes.position.array).toEqual(before);
  expect(geometry.attributes.aFlow.count).toBe(geometry.attributes.position.count);
  for (let i = 0; i < progress.length; i++) {
    expect(geometry.attributes.aFlow.getX(i)).toBeCloseTo(profile.phase);
    expect(geometry.attributes.aFlow.getY(i)).toBeCloseTo(profile.rate);
    expect(geometry.attributes.aProgress.getX(i)).toBe(progress[i]);
    expect(geometry.attributes.aEnds.getX(i)).toBe(0);
    expect(geometry.attributes.aEnds.getY(i)).toBe(1);
  }
  expect(geometry.attributes.aAcross.getX(0)).toBe(0);
  expect(geometry.attributes.aAcross.getX(3)).toBe(1);
  geometry.dispose();
});
