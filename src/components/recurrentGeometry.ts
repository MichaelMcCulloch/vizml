import * as THREE from "three";
import type { LayerNode, NodeAnalysis } from "../model";

export const spatialScale = (value: number) =>
  Math.max(0.65, Math.min(5.3, Math.sqrt(value / 32) * 3.4));

export function recurrentMetrics(node: LayerNode, info: NodeAnalysis) {
  const spatial = node.data.kind.startsWith("conv_");
  const layers = Math.max(1, Math.min(4, Number(node.data.params.layers ?? 1)));
  const directions = spatial ? 1 : Number(node.data.params.bidirectional ?? 0) ? 2 : 1;
  const h = spatialScale(info.shape.at(-2) ?? 1), w = spatialScale(info.shape.at(-1) ?? 1);
  return { spatial, cell: node.data.kind.replace("conv_", ""), layers, directions,
    height: h / Math.max(h, w) * 2.05, width: w / Math.max(h, w) * 2.05,
    ringRadius: 1.55, radius: 1.62, halfDepth: (layers - 1) * 0.07 + (directions - 1) * 0.04 + 0.045 };
}

export type SpatialFace = { center: THREE.Vector3; dimensions: [number, number] };

/** A woven recurrence glyph. Fine threads and a shared perimeter dominate;
 * gate accents and LSTM's inner memory thread remain deliberately small.
 * Spatial cells keep a faint lattice and volume inside the same silhouette.
 * These are bounded architectural samples, never learned activity or weights.
 */
export function recurrentGeometry(
  group: THREE.Group, node: LayerNode, info: NodeAnalysis, color: string,
  focused: boolean, meshes: THREE.Object3D[], opacity = 1,
) {
  const metrics = recurrentMetrics(node, info);
  const { spatial, cell, layers, directions, height, width, ringRadius, halfDepth } = metrics;
  const front = halfDepth - 0.045;
  const count = Math.max(1, Math.min(24, Number(node.data.params.hidden_size ?? (spatial ? 32 : 64)) * directions));
  const inputCount = Math.max(1, Math.min(24, info.input_shape[1] ?? 16));
  const point = (x: number, radius: number, angle: number) => new THREE.Vector3(x, Math.cos(angle) * radius, Math.sin(angle) * radius);
  const circle = (x: number, radius: number, count = 96, reverse = false) =>
    Array.from({ length: count + 1 }, (_, i) => point(x, radius, (reverse ? -1 : 1) * i / count * Math.PI * 2));
  const ports = (size: number) => Array.from({ length: size }, (_, i) => point(0, ringRadius, i / size * Math.PI * 2));
  const weave: number[] = [], lattice: number[] = [];
  const line = (a: THREE.Vector3, b: THREE.Vector3, target = weave) => target.push(...a.toArray(), ...b.toArray());
  const meshMaterial = (alpha: number) => new THREE.MeshBasicMaterial({ color, transparent: true,
    opacity: alpha * opacity, depthWrite: false, side: THREE.DoubleSide });
  const rim = meshMaterial(focused ? 0.5 : 0.31), echo = meshMaterial(0.12), dotsMaterial = meshMaterial(0.67);
  const pickable = (mesh: THREE.Mesh) => { mesh.userData.nodeId = node.id; meshes.push(mesh); group.add(mesh); };
  const hoop = (x: number, radius: number, material: THREE.Material, tube = 0.011) => {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 6, 96), material);
    ring.rotation.y = Math.PI / 2; ring.position.x = x; pickable(ring);
  };
  for (let layer = 0; layer < layers; layer++) for (let direction = 0; direction < directions; direction++) {
    const x = (layer - (layers - 1) / 2) * 0.14 + (direction - (directions - 1) / 2) * 0.08;
    hoop(x, ringRadius, layer === layers - 1 && direction === directions - 1 ? rim : echo);
  }
  // Two crossing thread families leave a quiet opening at the center.
  const threads = Math.max(12, count);
  for (let i = 0; i < threads; i++) for (const skip of [Math.floor(threads * 0.29), Math.floor(threads * 0.38)]) {
    const a = point(front, ringRadius, i / threads * Math.PI * 2);
    const b = point(front, ringRadius, (i + skip) / threads * Math.PI * 2);
    line(a, b);
  }
  const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.036, 8, 6), dotsMaterial, count);
  const transform = new THREE.Matrix4();
  for (let i = 0; i < count; i++) dots.setMatrixAt(i, transform.makeTranslation(point(front, ringRadius, i / count * Math.PI * 2)));
  dots.instanceMatrix.needsUpdate = true; pickable(dots);

  if (spatial) {
    // One translucent body, with a few sampled grid sheets rather than separate machines.
    const volume = new THREE.Mesh(new THREE.BoxGeometry(halfDepth * 2, height, width), meshMaterial(focused ? 0.065 : 0.035));
    pickable(volume);
    for (const x of layers > 1 ? [-front, front] : [0]) {
      const rows = Math.min(8, info.shape.at(-2) ?? 1), cols = Math.min(8, info.shape.at(-1) ?? 1);
      for (let row = 0; row <= rows; row++) {
        const y = (row / rows - 0.5) * height;
        line(new THREE.Vector3(x, y, -width / 2), new THREE.Vector3(x, y, width / 2), lattice);
      }
      for (let col = 0; col <= cols; col++) {
        const z = (col / cols - 0.5) * width;
        line(new THREE.Vector3(x, -height / 2, z), new THREE.Vector3(x, height / 2, z), lattice);
      }
    }
  } else {
    // The open weave remains easy to select without adding a visible surface.
    const hitArea = new THREE.Mesh(new THREE.CircleGeometry(ringRadius, 48), meshMaterial(0));
    hitArea.rotation.y = Math.PI / 2; pickable(hitArea);
  }

  const loops = [circle(front, ringRadius)];
  if (directions === 2) loops.push(circle(-front, ringRadius, 96, true));
  if (cell === "lstm") {
    hoop(front, 0.47, echo, 0.008);
    loops.push(circle(front, 0.47));
    if (directions === 2) loops.push(circle(-front, 0.47, 96, true));
  }
  // A pair of soft stitches suggests GRU gating. LSTM's three tiny stitches
  // sit on its memory thread. The inspector carries the actual mechanism.
  const gates = cell === "gru" ? 2 : cell === "lstm" ? 3 : 0;
  for (let i = 0; i < gates; i++) {
    const start = i / gates * Math.PI * 2 + 0.3;
    const radius = cell === "lstm" ? 0.47 : 0.57;
    const arc = Array.from({ length: 13 }, (_, step) => point(front + 0.004, radius, start + step / 12 * 0.32));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(arc),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.28 * opacity, depthWrite: false })));
  }
  for (const [positions, alpha] of [[weave, 0.13], [lattice, 0.11]] as const) if (positions.length) group.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(positions, 3)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: alpha * opacity * (focused ? 1.25 : 1), depthWrite: false }),
  ));
  const face = (): SpatialFace => ({ center: new THREE.Vector3(), dimensions: [height, width] });
  return { ...metrics, ports: spatial ? undefined : ports(count), inputPorts: spatial ? undefined : ports(inputCount),
    spatialFaces: spatial ? { input: face(), output: face() } : undefined, loops };
}
