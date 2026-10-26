import * as THREE from "three";
import type { LayerNode, NodeAnalysis } from "../model";

// These are sampled connectivity glyphs, never trained attention weights.
// Geometry is bounded independently of sequence length, heads, and expert count.
export function sequenceGeometry(
  group: THREE.Group,
  node: LayerNode,
  info: NodeAnalysis,
  color: string,
  focused: boolean,
  meshes: THREE.Object3D[],
  badge: (text: string, color: string, y: number) => void,
): THREE.Vector3[] {
  const p = node.data.params;
  const streams = info.layout === "streams";
  const tokens = info.shape[0];
  const dim = info.shape.at(-1) ?? 1;
  const rows = Math.min(12, tokens);
  const width =
    node.data.kind === "token_input"
      ? 0.5
      : Math.min(2.5, Math.max(1.1, Math.log2(dim + 1) * 0.22));
  const height = 3;
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: true,
    opacity: focused ? 0.22 : 0.12,
    roughness: 0.5,
    metalness: 0.12,
    depthWrite: false,
  });
  const lineMaterial = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: focused ? 0.6 : 0.32,
  });
  const lineBatches = new Map<THREE.LineBasicMaterial, number[]>();
  const line = (points: THREE.Vector3[], mat = lineMaterial) => {
    const positions = lineBatches.get(mat) ?? [];
    for (let i = 1; i < points.length; i++)
      positions.push(
        ...points[i - 1].toArray(),
        ...points[i].toArray(),
      );
    lineBatches.set(mat, positions);
  };
  const box = (
    geometry: THREE.BoxGeometry,
    x: number,
    z: number,
    mat = material,
  ) => {
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.position.set(x, 0, z);
    mesh.userData.nodeId = node.id;
    meshes.push(mesh);
    group.add(mesh);
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      lineMaterial,
    );
    outline.position.copy(mesh.position);
    group.add(outline);
  };
  const sampled = Array.from({ length: rows }, (_, i) =>
    rows === 1 ? 0 : (0.5 - i / (rows - 1)) * height,
  );
  if (streams) {
    const lanes = 4;
    const positions = Array.from({ length: lanes }, (_, i) => (i - 1.5) * 0.9);
    const streamSkipMaterial = new THREE.LineBasicMaterial({
      color: "#e8a286",
      transparent: true,
      opacity: 0.55,
    });
    for (const z of positions) {
      box(new THREE.BoxGeometry(0.12, height, 0.65), 0, z);
      for (const y of sampled)
        line([
          new THREE.Vector3(-0.05, y, z - 0.3),
          new THREE.Vector3(0.1, y, z + 0.3),
        ]);
      const skip = new THREE.CubicBezierCurve3(
        new THREE.Vector3(-0.7, 0.6, z),
        new THREE.Vector3(-0.7, 2.5, z),
        new THREE.Vector3(0.7, 2.5, z),
        new THREE.Vector3(0.7, 0.6, z),
      );
      line(skip.getPoints(24), streamSkipMaterial);
    }
    if (node.data.kind === "deepseek_stage") {
      const mixing = new THREE.LineBasicMaterial({
        color: "#e8a286",
        transparent: true,
        opacity: 0.16,
      });
      for (const a of positions)
        for (const b of positions)
          line(
            [
              new THREE.Vector3(-0.65, -0.3, a),
              new THREE.Vector3(0.65, 0.3, b),
            ],
            mixing,
          );
      const bank = new THREE.InstancedMesh(
        new THREE.BoxGeometry(0.16, 0.17, 0.19),
        new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.1 }),
        16,
      );
      const transform = new THREE.Object3D();
      for (let i = 0; i < 16; i++) {
        transform.position.set(
          (i % 2) * 0.25,
          -1.9,
          (Math.floor(i / 2) - 3.5) * 0.34,
        );
        transform.updateMatrix();
        bank.setMatrixAt(i, transform.matrix);
        bank.setColorAt(
          i,
          new THREE.Color(i < 6 ? "#8ecdf4" : i === 15 ? "#7acfd6" : "#2c3743"),
        );
      }
      bank.userData.nodeId = node.id;
      meshes.push(bank);
      group.add(bank);
      badge("6 / 384 EXPERTS + SHARED", "#9fb0c3", 2.7);
    } else badge("4 RESIDUAL STREAMS", "#e8a286", 2.7);
  } else {
    const repeated = node.data.kind === "transformer_block";
    const attention =
      repeated || ["attention", "cross_attention"].includes(node.data.kind);
    const depth = repeated ? Math.min(6, Number(p.blocks)) : 1;
    if (!attention) {
      box(new THREE.BoxGeometry(0.1, height + 0.16, width), 0, 0);
      for (let i = 0; i < sampled.length; i++) {
        const prefix = info.prefix_tokens ?? 0;
        const rowMaterial =
          i < Math.min(prefix, 5)
            ? new THREE.LineBasicMaterial({
                color: i === 0 ? "#8ecdf4" : "#d6bf83",
                transparent: true,
                opacity: 0.8,
              })
            : lineMaterial;
        line(
          [
            new THREE.Vector3(0.08, sampled[i], -width / 2),
            new THREE.Vector3(0.08, sampled[i], width / 2),
          ],
          rowMaterial,
        );
      }
      for (let i = 1; i < 7; i++)
        line([
          new THREE.Vector3(0.07, -height / 2, -width / 2 + (width * i) / 7),
          new THREE.Vector3(0.07, height / 2, -width / 2 + (width * i) / 7),
        ]);
    } else {
      const heads = Math.min(8, Number(p.heads ?? 8));
      const cells = 6;
      const causal = repeated ? p.mode !== "encoder" : !!Number(p.causal);
      for (let layer = 0; layer < depth; layer++) {
        const x = (layer - (depth - 1) / 2) * 0.2;
        box(new THREE.BoxGeometry(0.06, height + 0.16, width), x, 0);
      }
      const count = heads * cells * cells;
      const tiles = new THREE.InstancedMesh(
        new THREE.BoxGeometry(
          0.045,
          (height / cells) * 0.83,
          (width / cells) * 0.83,
        ),
        new THREE.MeshStandardMaterial({
          transparent: true,
          opacity: focused ? 0.7 : 0.43,
          roughness: 0.48,
          depthWrite: false,
        }),
        count,
      );
      const transform = new THREE.Object3D();
      let index = 0;
      for (let h = 0; h < heads; h++)
        for (let q = 0; q < cells; q++)
          for (let k = 0; k < cells; k++) {
            transform.position.set(
              (h - (heads - 1) / 2) * 0.14,
              (0.5 - (q + 0.5) / cells) * height,
              ((k + 0.5) / cells - 0.5) * width,
            );
            transform.updateMatrix();
            tiles.setMatrixAt(index, transform.matrix);
            tiles.setColorAt(
              index,
              new THREE.Color(
                causal && k > q ? "#26303b" : color,
              ).multiplyScalar(
                causal && k > q ? 0.45 : 0.6 + ((q + k + h) % 3) * 0.18,
              ),
            );
            index++;
          }
      tiles.userData.nodeId = node.id;
      meshes.push(tiles);
      group.add(tiles);
      if (repeated) {
        const blockSkipMaterial = new THREE.LineBasicMaterial({
          color: "#e8a286",
          transparent: true,
          opacity: 0.7,
        });
        for (const [z, lift] of [
          [-width * 0.3, 2.3],
          [width * 0.3, 2.7],
        ]) {
          const skip = new THREE.CubicBezierCurve3(
            new THREE.Vector3(-0.8, 0.6, z),
            new THREE.Vector3(-0.8, lift, z),
            new THREE.Vector3(0.8, lift, z),
            new THREE.Vector3(0.8, 0.6, z),
          );
          line(skip.getPoints(28), blockSkipMaterial);
        }
      }
      badge(
        `${p.heads} HEADS${repeated ? ` · ${p.blocks} BLOCKS` : ""}${causal ? " · CAUSAL" : ""}`,
        color,
        3.1,
      );
    }
  }
  for (const [mat, positions] of lineBatches)
    group.add(
      new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute(
          "position",
          new THREE.Float32BufferAttribute(positions, 3),
        ),
        mat,
      ),
    );
  // Local to the layer group so manual 3D placement can translate the whole
  // glyph and its connectivity together.
  return sampled.map((y) => new THREE.Vector3(0, y, 0));
}
