import * as THREE from "three";

class QuinticBezierCurve3 extends THREE.Curve<THREE.Vector3> {
  constructor(
    private v0: THREE.Vector3,
    private v1: THREE.Vector3,
    private v2: THREE.Vector3,
    private v3: THREE.Vector3,
    private v4: THREE.Vector3,
    private v5: THREE.Vector3,
  ) {
    super();
  }

  getPoint(t: number, target = new THREE.Vector3()) {
    const m = 1 - t;
    const b0 = m ** 5;
    const b1 = 5 * m ** 4 * t;
    const b2 = 10 * m ** 3 * t ** 2;
    const b3 = 10 * m ** 2 * t ** 3;
    const b4 = 5 * m * t ** 4;
    const b5 = t ** 5;
    return target
      .set(0, 0, 0)
      .addScaledVector(this.v0, b0)
      .addScaledVector(this.v1, b1)
      .addScaledVector(this.v2, b2)
      .addScaledVector(this.v3, b3)
      .addScaledVector(this.v4, b4)
      .addScaledVector(this.v5, b5);
  }
}

class ArchCurve3 extends THREE.Curve<THREE.Vector3> {
  constructor(
    private start: THREE.Vector3,
    private end: THREE.Vector3,
    private bow: number,
  ) {
    super();
  }

  getPoint(t: number, target = new THREE.Vector3()) {
    target.copy(this.start).lerp(this.end, t);
    target.y += this.bow * Math.sin(Math.PI * t) ** 2;
    return target;
  }
}

export const makeConnectionCurve = (
  start: THREE.Vector3,
  end: THREE.Vector3,
  bow = 0,
) => {
  const forward = end.x - start.x;
  if (forward <= 0.25)
    return bow
      ? new ArchCurve3(start, end, bow)
      : new THREE.LineCurve3(start, end);
  const out = Math.min(3, forward * 0.3);
  const control = start.clone().add(new THREE.Vector3(out, 0, 0));
  const enter = end.clone().sub(new THREE.Vector3(out, 0, 0));
  if (bow === 0)
    return new THREE.CubicBezierCurve3(start, control, enter, end);
  const liftA = start.clone().lerp(end, 0.35);
  const liftB = start.clone().lerp(end, 0.65);
  liftA.y += bow;
  liftB.y += bow;
  return new QuinticBezierCurve3(
    start,
    control,
    liftA,
    liftB,
    enter,
    end,
  );
};

// Glyphs may enumerate their samples in either direction. Connections always
// use the same spatial orientation, independent of tensor/token index order.
function orderedPorts(points: THREE.Vector3[]) {
  const minY = Math.min(...points.map(point => point.y));
  const height = Math.max(...points.map(point => point.y)) - minY;
  // Symmetric points on a circle can differ by a few floating-point bits.
  // World-space translation changes those bits; quantized local rows keep
  // left/right pairs in the same order after moving or resizing a glyph.
  const row = (point: THREE.Vector3) => height > 0 ? Math.round((point.y - minY) / height * 1e6) : 0;
  return [...points].sort((a, b) => row(b) - row(a) || a.z - b.z || a.x - b.x);
}

export function pairConnectionPorts(source: THREE.Vector3[], target: THREE.Vector3[]) {
  const from = orderedPorts(source);
  const to = orderedPorts(target);
  const count = Math.min(from.length, to.length);
  const sample = (points: THREE.Vector3[], index: number) =>
    points[Math.round((count === 1 ? 0.5 : index / (count - 1)) * (points.length - 1))];
  return Array.from({ length: count }, (_, index): [THREE.Vector3, THREE.Vector3] =>
    [sample(from, index), sample(to, index)],
  );
}

export function connectionExtent(points: THREE.Vector3[]) {
  const ordered = orderedPorts(points);
  const top = (ordered[0] ?? new THREE.Vector3()).clone();
  const bottom = (ordered.at(-1) ?? top).clone();
  if (top.y - bottom.y < 0.24) {
    const middle = (top.y + bottom.y) / 2;
    top.y = middle + 0.12;
    bottom.y = middle - 0.12;
  }
  return { top, bottom };
}

/** A loft with a fixed corner correspondence, so shrinking grids never twist. */
export function spatialConnection(
  source: THREE.Vector3,
  target: THREE.Vector3,
  sourceSize: [height: number, width: number],
  targetSize: [height: number, width: number],
) {
  // Perimeter order matters: joining corners in row order would cross two walls.
  const corners = [[-1, -1], [-1, 1], [1, 1], [1, -1]];
  const curves = corners.map(([y, z]) => makeConnectionCurve(
    source.clone().add(new THREE.Vector3(0, y * sourceSize[0] / 2, z * sourceSize[1] / 2)),
    target.clone().add(new THREE.Vector3(0, y * targetSize[0] / 2, z * targetSize[1] / 2)),
    0.15,
  ).getPoints(32));
  const positions: number[] = [];
  const indices: number[] = [];
  const across: number[] = [];
  const stations: number[] = [];
  const faceOpacity: number[] = [];
  for (let face = 0; face < 4; face++) {
    const first = curves[face], second = curves[(face + 1) % 4];
    const base = positions.length / 3;
    for (let i = 0; i < first.length; i++) {
      positions.push(...first[i].toArray(), ...second[i].toArray());
      across.push(0, 1);
      stations.push(i, i);
      // A small difference between horizontal and vertical faces gives the
      // volume definition without introducing light sources or extra motion.
      const opacity = face % 2 === 0 ? 0.72 : 1;
      faceOpacity.push(opacity, opacity);
      if (i < first.length - 1) {
        const a = base + i * 2;
        indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  const centerline = curves[0].map((point, i) => point.clone().add(curves[2][i]).multiplyScalar(0.5));
  return { geometry, curves, centerline, stations, across, faceOpacity };
}
