import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  Box,
  RotateCcw,
  Rotate3D,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Undo2,
} from "lucide-react";
import {
  catalog,
  isInput,
  isOutput,
  shapeText,
  type Analysis,
  type ModelGraph,
} from "../model";
import { sequenceGeometry } from "./sequenceGeometry";

type Props = {
  graph: ModelGraph;
  analysis: Analysis | null;
  theme: "light" | "dark";
  selected: string | null;
  onSelect: (id: string) => void;
  expanded: boolean;
  onExpand: () => void;
};
type SceneAPI = {
  reset: () => void;
  resetLayout: () => void;
  setRotate: (enabled: boolean) => void;
  setLabels: (enabled: boolean) => void;
};

type Offset = [number, number, number];

const LAYOUT_KEY_PREFIX = "vizml:architecture:manual:";
const SNAP_STEP = 2;

function readManualLayout(name: string): Record<string, Offset> {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY_PREFIX + name);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const offsets: Record<string, Offset> = {};
    for (const [id, value] of Object.entries(parsed)) {
      if (
        Array.isArray(value) &&
        value.length === 3 &&
        value.every((item) => typeof item === "number" && Number.isFinite(item))
      )
        offsets[id] = [value[0], value[1], value[2]];
    }
    return offsets;
  } catch {
    return {};
  }
}

function storeManualLayout(name: string, offsets: Record<string, Offset>) {
  try {
    localStorage.setItem(LAYOUT_KEY_PREFIX + name, JSON.stringify(offsets));
  } catch {
    // Layout is a convenience; a full or disabled storage must not break the view.
  }
}

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

function displayColor(hex: string, theme: "light" | "dark") {
  if (theme === "dark") return hex;
  const color = new THREE.Color(hex);
  color.multiplyScalar(0.74);
  return `#${color.getHexString()}`;
}

function labelSprite(text: string, color: string, size = 0.34) {
  const canvas = document.createElement("canvas");
  canvas.width = 768;
  canvas.height = 80;
  const context = canvas.getContext("2d")!;
  context.font = "500 30px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = color;
  context.fillText(text.slice(0, 40), 384, 40);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      depthTest: false,
      transparent: true,
    }),
  );
  sprite.scale.set(size * 9.6, size, 1);
  sprite.userData.isLabel = true;
  return sprite;
}

function labelPairSprite(
  name: string,
  nameColor: string,
  detail: string,
  detailColor: string,
  size = 0.68,
) {
  const canvas = document.createElement("canvas");
  canvas.width = 768;
  canvas.height = 150;
  const context = canvas.getContext("2d")!;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = "500 32px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillStyle = nameColor;
  context.fillText(name.slice(0, 40), 384, 46);
  context.font = "500 27px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillStyle = detailColor;
  context.fillText(detail.slice(0, 40), 384, 108);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      depthTest: false,
      transparent: true,
    }),
  );
  sprite.scale.set(size * 9.6, size * 2.15, 1);
  sprite.userData.isLabel = true;
  return sprite;
}

export default function Architecture({
  graph,
  analysis,
  theme,
  selected,
  onSelect,
  expanded,
  onExpand,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<SceneAPI | null>(null);
  const [rotating, setRotating] = useState(false);
  const [labels, setLabels] = useState(true);
  const [failed, setFailed] = useState(false);
  const select = useRef(onSelect);
  select.current = onSelect;
  const rotateValue = useRef(rotating);
  rotateValue.current = rotating;
  const labelsValue = useRef(labels);
  labelsValue.current = labels;
  const cameraView = useRef<{
    position: number[];
    target: number[];
    zoom: number;
    signature: string;
  } | null>(null);
  // Graph positions deliberately do not influence the generated architecture.
  const signature = JSON.stringify({
    name: graph.name,
    theme,
    nodes: graph.nodes.map((n) => ({ id: n.id, data: n.data })),
    edges: graph.edges.map((e) => ({ source: e.source, target: e.target })),
    shapes: analysis?.nodes,
    selected,
  });

  useEffect(() => {
    const element = host.current;
    if (!element || !analysis || !graph.nodes.length) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      setFailed(true);
      return;
    }
    setFailed(false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setClearColor(0x101610, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive 3D model architecture. Drag to orbit, scroll to zoom, click a layer to inspect.",
    );
    renderer.domElement.setAttribute("role", "img");
    element.appendChild(renderer.domElement);
    const readout = document.createElement("div");
    readout.className = "snap-readout";
    readout.hidden = true;
    element.appendChild(readout);
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-15, 15, 7, -7, 0.1, 300);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.autoRotate = rotateValue.current;
    controls.autoRotateSpeed = 0.45;
    controls.minZoom = 0.3;
    controls.maxZoom = 8;
    controls.maxPolarAngle = Math.PI * 0.88;
    scene.add(
      new THREE.AmbientLight(theme === "light" ? 0xffffff : 0xdde6f0, 1.7),
    );
    const keyLight = new THREE.DirectionalLight(0xffffff, 3);
    keyLight.position.set(-6, 12, 8);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(
      theme === "light" ? 0xc9d7ff : 0x8ab4ff,
      2.5,
    );
    rimLight.position.set(12, 5, -8);
    scene.add(rimLight);
    const labelInk = theme === "light" ? "#34404e" : "#b8c2ce";
    const labelStrong = theme === "light" ? "#111923" : "#f1f5fa";
    const badgeInk = theme === "light" ? "#65717f" : "#8b98a6";
    const outputInk = theme === "light" ? "#4055c7" : "#aebcff";

    const byId = new Map(graph.nodes.map((n) => [n.id, n]));
    const validNodes = analysis.order.filter(
      (id) => analysis.nodes[id] && byId.has(id),
    );
    const depths = new Map(
      validNodes.map((id) => [id, analysis.nodes[id].depth]),
    );
    if (graph.mode === "schematic") {
      for (const id of validNodes) {
        const parents = graph.edges.filter(e => e.target === id && byId.has(e.source));
        depths.set(id, parents.length ? Math.max(...parents.map(e => depths.get(e.source)! + 1)) : 0);
      }
    }
    // Right-align short input branches at the union while preserving topological order.
    for (const id of [...validNodes].reverse()) {
      const children = graph.edges
        .filter((e) => e.source === id)
        .map((e) => depths.get(e.target))
        .filter((d): d is number => d !== undefined);
      if (children.length)
        depths.set(id, Math.max(depths.get(id)!, Math.min(...children) - 1));
    }
    const firstDepth = Math.min(0, ...depths.values());
    const viewStart = graph.mode === "schematic" ? Math.min(...depths.values()) : firstDepth;
    if (Number.isFinite(viewStart)) for (const [id, depth] of depths) depths.set(id, depth - viewStart);
    const maxDepth = Math.max(1, ...depths.values());
    const spacing = 2.55;
    const depthX = new Map<number, number>();
    if (graph.template?.startsWith("dinov3_convnext_")) {
      let cursor = 0;
      for (const depth of [...new Set(depths.values())].sort((a, b) => a - b)) {
        depthX.set(depth, cursor);
        const kinds = validNodes
          .filter((id) => depths.get(id) === depth)
          .map((id) => byId.get(id)!.data.kind);
        cursor += kinds.includes("convnext_stage")
          ? 3
          : kinds.every((kind) => kind === "layernorm" || kind === "embedding")
            ? 0.9
            : 2.1;
      }
    }
    const xForDepth = (depth: number) => depthX.get(depth) ?? depth * spacing;
    const maxX = Math.max(1, ...[...depths.values()].map(xForDepth));
    const pointMap = new Map<string, THREE.Vector3>();
    const neurons = new Map<string, THREE.Vector3[]>();
    const halfDepth = new Map<string, number>();
    const meshes: THREE.Object3D[] = [];

    // Dominant-flow hierarchy. Each node keeps one parent as its current; the
    // other parent edges become deltas across the model. The largest child
    // continues a flow, substantial forks open a new vertical tier, and small
    // forks stay on their parent's tier and fan out laterally. The result is a
    // river trunk with parallel channels: X is progress, Y is flow hierarchy,
    // Z is the sibling channel.
    const flowHeight = new Map<string, number>();
    for (const id of [...validNodes].reverse()) {
      let longest = 0;
      for (const edge of graph.edges) {
        if (edge.source !== id) continue;
        const child = flowHeight.get(edge.target);
        if (child !== undefined) longest = Math.max(longest, child + 1);
      }
      flowHeight.set(id, longest);
    }
    // "Like favors like": spatial stages stay with spatial stages, and dense
    // head stacks stay with dense head stacks instead of dragging the CNN
    // trunk into an MLP or cutting the board projection out of the trunk.
    const layoutFamily = (id: string) => {
      const info = analysis.nodes[id];
      if (!info) return "other";
      if (info.layout === "streams" || info.shape.length === 2)
        return "sequence";
      if (info.shape.length === 3) return "spatial";
      if (info.shape.length === 1) return "vector";
      return "other";
    };
    const isSkipEdge = (edge: (typeof graph.edges)[number]) => {
      const role = typeof edge.data?.role === "string" ? edge.data.role : "";
      if (role.includes("skip")) return true;
      const target = byId.get(edge.target);
      return (
        target?.data.kind === "residual" &&
        (depths.get(edge.target)! - depths.get(edge.source)! > 1 ||
          byId.get(edge.source)!.data.params.role === "skip_projection")
      );
    };
    const parentOf = new Map<string, string>();
    for (const id of validNodes) {
      const family = layoutFamily(id);
      const candidates = graph.edges
        .filter(
          (edge) =>
            edge.target === id &&
            byId.has(edge.source) &&
            depths.has(edge.source),
        )
        .map((edge) => ({
          id: edge.source,
          skip: isSkipEdge(edge),
          family: layoutFamily(edge.source),
          height: flowHeight.get(edge.source) ?? 0,
          depth: depths.get(edge.source) ?? 0,
        }));
      if (!candidates.length) continue;
      candidates.sort(
        (a, b) =>
          Number(a.family !== family) - Number(b.family !== family) ||
          Number(a.skip) - Number(b.skip) ||
          b.height - a.height ||
          b.depth - a.depth ||
          a.id.localeCompare(b.id),
      );
      parentOf.set(id, candidates[0].id);
    }

    const flowChildren = new Map<string, string[]>();
    const flowRoots: string[] = [];
    for (const id of validNodes) {
      const parent = parentOf.get(id);
      if (!parent) {
        flowRoots.push(id);
        continue;
      }
      const children = flowChildren.get(parent) ?? [];
      children.push(id);
      flowChildren.set(parent, children);
    }
    const flowSize = new Map<string, number>();
    const flowDepth = new Map<string, number>();
    const measureFlow = (id: string): { size: number; height: number } => {
      let size = 1;
      let height = 0;
      for (const child of flowChildren.get(id) ?? []) {
        const measured = measureFlow(child);
        size += measured.size;
        height = Math.max(height, measured.height + 1);
      }
      flowSize.set(id, size);
      flowDepth.set(id, height);
      return { size, height };
    };
    for (const root of flowRoots) measureFlow(root);
    flowRoots.sort(
      (a, b) =>
        (flowDepth.get(b) ?? 0) - (flowDepth.get(a) ?? 0) ||
        (flowSize.get(b) ?? 0) - (flowSize.get(a) ?? 0) ||
        a.localeCompare(b),
    );

    type FlowLane = { id: number; level: number; size: number; z: number };
    const lanes: FlowLane[] = [];
    const laneOfNode = new Map<string, number>();
    const lateralOfNode = new Map<string, number>();
    const forkCenters = new Map<string, number[]>();
    const majorThreshold = Math.max(4, Math.round(validNodes.length * 0.06));
    const walkFlow = (
      node: string,
      lane: number,
      lateral: number,
    ): void => {
      lanes[lane].size += 1;
      laneOfNode.set(node, lane);
      lateralOfNode.set(node, lateral);
      const family = layoutFamily(node);
      const children = [...(flowChildren.get(node) ?? [])].sort(
        (a, b) =>
          (flowSize.get(b) ?? 1) - (flowSize.get(a) ?? 1) ||
          (flowDepth.get(b) ?? 0) - (flowDepth.get(a) ?? 0) ||
          a.localeCompare(b),
      );
      const input = isInput(byId.get(node)!.data.kind);
      const sameFamilyChildren = children.filter(
        (child) => layoutFamily(child) === family,
      );
      const sameFamilyMajor = sameFamilyChildren.filter(
        (child) => (flowSize.get(child) ?? 1) >= majorThreshold,
      );
      // A shared input that opens two substantial same-kind branches is a
      // river fork: keep the fork node in its own lane and send both branches
      // into sibling lanes instead of letting one absorb the fork.
      const forkHeader = sameFamilyMajor.length >= 2;
      const inheritor = forkHeader
        ? input
          ? children.find((child) => layoutFamily(child) !== family)
          : undefined
        : input
          ? children[0]
          : sameFamilyChildren[0];
      let minor = 0;
      children.forEach((child) => {
        if (child === inheritor) {
          walkFlow(child, lane, lateral);
          return;
        }
        const changesFamily = layoutFamily(child) !== family;
        if (
          changesFamily ||
          (flowSize.get(child) ?? 1) >= majorThreshold
        ) {
          const next = lanes.length;
          lanes.push({
            id: next,
            level: lanes[lane].level + 1,
            size: 0,
            z: 0,
          });
          if (forkHeader && !inheritor && sameFamilyMajor.includes(child)) {
            const centers = forkCenters.get(node) ?? [];
            centers.push(next);
            forkCenters.set(node, centers);
          }
          walkFlow(child, next, 0);
          return;
        }
        minor += 1;
        const distance = Math.ceil(minor / 2) * 5;
        const direction = minor % 2 === 1 ? 1 : -1;
        walkFlow(child, lane, lateral + direction * distance);
      });
    };
    const createFlowLane = (root: string, level: number) => {
      const next = lanes.length;
      lanes.push({ id: next, level, size: 0, z: 0 });
      walkFlow(root, next, 0);
    };
    flowRoots.forEach((root, index) => {
      if (index === 0) {
        createFlowLane(root, 0);
      } else if ((flowSize.get(root) ?? 1) >= majorThreshold) {
        createFlowLane(root, 1);
      } else {
        const distance = Math.ceil(index / 2) * 5;
        const direction = index % 2 === 1 ? 1 : -1;
        walkFlow(root, 0, direction * distance);
      }
    });

    const flowLaneSpacing = 8;
    const flowLevelSpacing = 8;
    const lanesByLevel = new Map<number, FlowLane[]>();
    for (const lane of lanes) {
      const list = lanesByLevel.get(lane.level) ?? [];
      list.push(lane);
      lanesByLevel.set(lane.level, list);
    }
    for (const list of lanesByLevel.values()) {
      list.sort(
        (a, b) =>
          (a.id === 0 ? -1 : b.id === 0 ? 1 : 0) ||
          b.size - a.size ||
          a.id - b.id,
      );
      list.forEach((lane, index) => {
        const slot =
          index === 0
            ? 0
            : (index % 2 === 1 ? 1 : -1) * Math.ceil(index / 2);
        lane.z = slot * flowLaneSpacing;
      });
    }
    // A pure fork keeps its own short lane in the flow tree, but visually it
    // belongs behind the midpoint of the branches it opens. This centers a
    // shared input such as the board planes between the actor and value stems.
    const displayLevel = new Map<string, number>();
    const displayLateral = new Map<string, number>();
    for (const [id, childLanes] of forkCenters) {
      if (!childLanes.length) continue;
      displayLevel.set(
        id,
        childLanes.reduce((total, laneId) => total + lanes[laneId].level, 0) /
          childLanes.length,
      );
      displayLateral.set(
        id,
        childLanes.reduce((total, laneId) => total + lanes[laneId].z, 0) /
          childLanes.length,
      );
    }
    const spatialScale = (v: number) =>
      Math.max(0.65, Math.min(5.3, Math.sqrt(v / 32) * 3.4));
    for (const id of validNodes) {
      const node = byId.get(id)!;
      const info = analysis.nodes[id];
      if (info.shape.length === 3) {
        const count =
          node.data.kind === "convnext_stage"
            ? Math.min(12, Number(node.data.params.blocks))
            : node.data.kind === "layernorm"
              ? 1
              : Math.max(1, Math.min(7, Math.ceil(Math.log2(info.shape[0] + 1))));
        halfDepth.set(id, ((count - 1) * 0.13 + 0.07) / 2);
      } else if (
        info.shape.length === 2 ||
        info.layout === "streams" ||
        node.data.kind === "token_input"
      ) {
        if (node.data.kind === "transformer_block") {
          const layers = Math.min(6, Number(node.data.params.blocks));
          halfDepth.set(id, (layers - 1) * 0.1 + 0.03);
        } else if (
          node.data.kind === "attention" ||
          node.data.kind === "cross_attention"
        ) {
          halfDepth.set(id, 0.03);
        } else if (info.layout === "streams") {
          halfDepth.set(id, 0.06);
        } else {
          halfDepth.set(id, 0.05);
        }
      } else if (
        node.data.kind === "concat" ||
        node.data.kind === "residual" ||
        node.data.kind === "flatten"
      ) {
        halfDepth.set(id, 0.4);
      } else {
        halfDepth.set(id, 0.15);
      }
    }
    const pointFor = (id: string) => {
      const depth = depths.get(id)!;
      const lane = lanes[laneOfNode.get(id) ?? 0];
      return new THREE.Vector3(
        xForDepth(depth),
        0.3 -
          (displayLevel.get(id) ?? lane?.level ?? 0) * flowLevelSpacing,
        displayLateral.get(id) ??
          (lane?.z ?? 0) + (lateralOfNode.get(id) ?? 0),
      );
    };
    const halfWidthOf = (id: string) => {
      const shape = analysis.nodes[id]?.shape ?? [];
      if (shape.length === 3) return spatialScale(shape[2]) / 2;
      if (shape.length === 2) return 0.9;
      return 0.25;
    };
    // Bias leaves are side inputs to an add/residual. Park them immediately
    // behind the layer they bias, centered on it, instead of leaving them at
    // the tail of a distant context lane.
    const parkedAt = new Map<string, THREE.Vector3>();
    for (const targetId of validNodes) {
      const target = byId.get(targetId);
      if (!target || !["add", "residual"].includes(target.data.kind)) continue;
      const incoming = graph.edges.filter(
        (edge) =>
          edge.target === targetId &&
          byId.has(edge.source) &&
          depths.has(edge.source),
      );
      if (incoming.length < 2) continue;
      const dominant = parentOf.get(targetId);
      let parked = 0;
      for (const edge of incoming) {
        if (edge.source === dominant) continue;
        const source = byId.get(edge.source);
        if (!source || source.data.kind !== "dense") continue;
        if ((flowChildren.get(edge.source)?.length ?? 0) > 0) continue;
        if (parkedAt.has(edge.source)) continue;
        const targetPoint = pointFor(targetId);
        const point = new THREE.Vector3(
          targetPoint.x - (halfDepth.get(targetId) ?? 0.15) - 0.8,
          targetPoint.y,
          targetPoint.z -
            (halfWidthOf(targetId) + 1.5 + parked * 1.6),
        );
        parkedAt.set(edge.source, point);
        parked += 1;
      }
    }
    let maxLevel = 0;
    let maxAbsZ = 0;
    for (const lane of lanes) maxLevel = Math.max(maxLevel, lane.level);
    for (const id of validNodes) {
      const point = parkedAt.get(id) ?? pointFor(id);
      maxAbsZ = Math.max(maxAbsZ, Math.abs(point.z));
    }
    const frameMinX = -4;
    const frameMaxX = maxX + 4;
    const frameMinY = 0.3 - maxLevel * flowLevelSpacing - 4.2;
    const frameMaxY = 0.3 + 4.8;
    const frameMinZ = -(maxAbsZ + 4);
    const frameMaxZ = maxAbsZ + 4;
    const frameSizeX = frameMaxX - frameMinX;
    const frameSizeY = frameMaxY - frameMinY;
    const frameSizeZ = frameMaxZ - frameMinZ;
    const sceneExtent = Math.max(frameSizeX, frameSizeY, frameSizeZ, 12);
    camera.far = Math.max(300, sceneExtent * 4);
    const centerX = (frameMinX + frameMaxX) / 2;
    const centerY = (frameMinY + frameMaxY) / 2;
    const centerZ = (frameMinZ + frameMaxZ) / 2;

    const manual = readManualLayout(graph.name);
    const manualOffset = (id: string) => {
      const offset = manual[id] ?? [0, 0, 0];
      return new THREE.Vector3(offset[0], offset[1], offset[2]);
    };
    const layerGroups = new Map<string, THREE.Group>();

    for (const id of validNodes) {
      const node = byId.get(id)!;
      const info = analysis.nodes[id];
      const point = parkedAt.get(id) ?? pointFor(id);
      pointMap.set(id, point);
      const group = new THREE.Group();
      group.position.copy(point).add(manualOffset(id));
      layerGroups.set(id, group);
      const color = displayColor(catalog[node.data.kind].color, theme);
      const focused = selected === id;
      const opacity = selected && !focused ? 0.72 : 1;
      const material = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: (theme === "light" ? 0.42 : 0.26) * opacity,
        roughness: 0.38,
        metalness: 0.16,
        emissive: color,
        emissiveIntensity: focused ? 0.2 : theme === "light" ? 0.08 : 0.045,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      if (
        info.shape.length === 2 ||
        info.layout === "streams" ||
        node.data.kind === "token_input"
      ) {
        const positions = sequenceGeometry(
          group,
          node,
          info,
          color,
          focused,
          meshes,
          (text, tint, y) => {
            const label = labelSprite(text, tint, 0.42);
            label.position.set(0, y, 0);
            group.add(label);
          },
        );
        neurons.set(id, positions);
        if (node.data.kind === "transformer_block") {
          const layers = Math.min(6, Number(node.data.params.blocks));
          halfDepth.set(id, (layers - 1) * 0.1 + 0.03);
        } else if (
          node.data.kind === "attention" ||
          node.data.kind === "cross_attention"
        ) {
          halfDepth.set(id, 0.03);
        } else if (info.layout === "streams") {
          halfDepth.set(id, 0.06);
        } else {
          halfDepth.set(id, 0.05);
        }
      } else if (info.shape.length === 3) {
        const [channels, h, w] = info.shape;
        const height = spatialScale(h),
          width = spatialScale(w);
        const count =
          node.data.kind === "convnext_stage"
            ? Math.min(12, Number(node.data.params.blocks))
            : node.data.kind === "layernorm"
              ? 1
              : Math.max(1, Math.min(7, Math.ceil(Math.log2(channels + 1))));
        halfDepth.set(id, ((count - 1) * 0.13 + 0.07) / 2);
        const boxGeometry = new THREE.BoxGeometry(0.07, height, width);
        const boxes = new THREE.InstancedMesh(boxGeometry, material, count);
        boxes.userData.nodeId = id;
        const outlineGeometry = new THREE.EdgesGeometry(boxGeometry);
        const outlinePositions: number[] = [];
        const boxTransform = new THREE.Object3D();
        for (let i = 0; i < count; i++) {
          const x = (i - (count - 1) / 2) * 0.13;
          boxTransform.position.set(x, 0, 0);
          boxTransform.updateMatrix();
          boxes.setMatrixAt(i, boxTransform.matrix);
          const edgePosition = outlineGeometry.attributes.position;
          for (let j = 0; j < edgePosition.count; j++)
            outlinePositions.push(
              edgePosition.getX(j) + x,
              edgePosition.getY(j),
              edgePosition.getZ(j),
            );
        }
        boxes.instanceMatrix.needsUpdate = true;
        meshes.push(boxes);
        group.add(boxes);
        group.add(
          new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute(
              "position",
              new THREE.Float32BufferAttribute(outlinePositions, 3),
            ),
            new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: focused
                ? 0.95
                : (theme === "light" ? 0.9 : 0.62) * opacity,
            }),
          ),
        );
        outlineGeometry.dispose();
        // Fine grid on the front feature map conveys spatial structure without per-cell geometry.
        const divisions = Math.min(8, h, w);
        const points: number[] = [];
        for (let i = 1; i < divisions; i++) {
          const t = i / divisions;
          points.push(
            count * 0.065,
            -height / 2 + height * t,
            -width / 2,
            count * 0.065,
            -height / 2 + height * t,
            width / 2,
          );
          points.push(
            count * 0.065,
            -height / 2,
            -width / 2 + width * t,
            count * 0.065,
            height / 2,
            -width / 2 + width * t,
          );
        }
        const grid = new THREE.LineSegments(
          new THREE.BufferGeometry().setAttribute(
            "position",
            new THREE.Float32BufferAttribute(points, 3),
          ),
          new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: 0.12,
          }),
        );
        group.add(grid);
        if (node.data.kind === "convnext_stage") {
          const half = Math.max(0.24, count * 0.065);
          const loop = new THREE.CubicBezierCurve3(
            new THREE.Vector3(-half, height / 2, 0),
            new THREE.Vector3(-half, height / 2 + 0.8, 0),
            new THREE.Vector3(half, height / 2 + 0.8, 0),
            new THREE.Vector3(half, height / 2, 0),
          );
          group.add(
            new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(loop.getPoints(24)),
              new THREE.LineBasicMaterial({
                color: "#e8a286",
                transparent: true,
                opacity: 0.65,
              }),
            ),
          );
        }
      } else if (
        node.data.kind === "concat" ||
        node.data.kind === "residual" ||
        node.data.kind === "flatten"
      ) {
        const geometry =
          node.data.kind === "residual"
            ? new THREE.TorusGeometry(0.45, 0.06, 8, 36)
            : new THREE.OctahedronGeometry(0.4);
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.3,
            metalness: 0.35,
            roughness: 0.28,
          }),
        );
        if (node.data.kind === "residual") mesh.rotation.y = Math.PI / 2;
        halfDepth.set(id, 0.4);
        mesh.userData.nodeId = id;
        meshes.push(mesh);
        group.add(mesh);
        if (node.data.kind === "residual") {
          const plus = labelSprite("+", color, 0.55);
          plus.position.set(0, 0, 0.1);
          group.add(plus);
        }
      } else {
        const count = Math.max(1, Math.min(13, info.shape[0]));
        const height = isOutput(node.data.kind) ? 2.2 : 3.2;
        halfDepth.set(id, 0.15);
        const positions: THREE.Vector3[] = [];
        const geometry = new THREE.SphereGeometry(0.075, 12, 10);
        const dotMaterial = new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.65,
          transparent: true,
          opacity,
        });
        const dots = new THREE.InstancedMesh(geometry, dotMaterial, count);
        dots.userData.nodeId = id;
        const dotTransform = new THREE.Object3D();
        const haloPositions: THREE.Matrix4[] = [];
        for (let i = 0; i < count; i++) {
          const y = count === 1 ? 0 : (i / (count - 1) - 0.5) * height;
          dotTransform.position.set(0, y, 0);
          dotTransform.updateMatrix();
          dots.setMatrixAt(i, dotTransform.matrix);
          haloPositions.push(dotTransform.matrix.clone());
          positions.push(new THREE.Vector3(0, y, 0));
        }
        dots.instanceMatrix.needsUpdate = true;
        meshes.push(dots);
        group.add(dots);
        if (focused) {
          const halos = new THREE.InstancedMesh(
            new THREE.SphereGeometry(0.13, 8, 6),
            new THREE.MeshBasicMaterial({
              color,
              transparent: true,
              opacity: 0.08,
            }),
            count,
          );
          haloPositions.forEach((matrix, index) =>
            halos.setMatrixAt(index, matrix),
          );
          halos.instanceMatrix.needsUpdate = true;
          group.add(halos);
        }
        neurons.set(id, positions);
      }
      const block = graph.mode === "schematic" ? node.id.match(/_block(\d+)_(conv1|conv2|add)$/) : null;
      const shortName = typeof node.data.view_label === "string" ? node.data.view_label : block ? `B${block[1]} · ${block[2] === "add" ? "Add" : `${block[2] === "conv1" ? "Conv 1" : "Conv 2"} · d${node.data.params.dilation}`}` : node.data.label;
      const details = labelPairSprite(
        shortName,
        focused ? labelStrong : labelInk,
        shapeText(info.display_shape ?? info.shape),
        color,
        graph.mode === "schematic" ? 0.6 : 0.68,
      );
      details.position.set(0, -3.0, 0);
      group.add(details);
      if (isInput(node.data.kind) && (graph.mode !== "schematic" || !info.dependencies)) {
        const badge = labelSprite("INPUT", badgeInk, 0.4);
        badge.position.set(0, 2.75, 0);
        group.add(badge);
      }
      if (isOutput(node.data.kind) && graph.mode !== "schematic") {
        const badge = labelSprite(
          node.data.kind === "classification"
            ? "ONE-HOT"
            : node.data.kind === "embedding"
              ? "EMBEDDING"
              : "REGRESSION",
          outputInk,
          0.4,
        );
        badge.position.set(0, 2.5, 0);
        group.add(badge);
      }
      scene.add(group);
    }

    const worldPoint = (id: string) => {
      const base = pointMap.get(id);
      return base ? base.clone().add(manualOffset(id)) : undefined;
    };
    const worldNeurons = (id: string) => {
      const origin = worldPoint(id);
      const local = neurons.get(id);
      return origin && local ? local.map((point) => origin.clone().add(point)) : undefined;
    };

    const edgeLayer = new THREE.Group();
    scene.add(edgeLayer);
    const disposeBranch = (object: THREE.Object3D) => {
      object.traverse((child) => {
        const item = child as THREE.Mesh;
        item.geometry?.dispose();
        const materials = item.material
          ? Array.isArray(item.material)
            ? item.material
            : [item.material]
          : [];
        for (const material of materials) {
          (material as THREE.SpriteMaterial).map?.dispose();
          material.dispose();
        }
      });
    };
      const makeSpline = (
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
    const addSpline = (
      group: THREE.Group,
      start: THREE.Vector3,
      end: THREE.Vector3,
      color: string,
      opacity: number,
      bow = 0,
    ) => {
      const curve = makeSpline(start, end, bow);
      group.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)),
          new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity,
          }),
        ),
      );
      return curve;
    };
    const addSplineBatch = (
      group: THREE.Group,
      curves: THREE.Vector3[][],
      color: string,
      opacity: number,
    ) => {
      const positions: number[] = [];
      for (const points of curves)
        for (let i = 1; i < points.length; i++)
          positions.push(
            ...points[i - 1].toArray(),
            ...points[i].toArray(),
          );
      group.add(
        new THREE.LineSegments(
          new THREE.BufferGeometry().setAttribute(
            "position",
            new THREE.Float32BufferAttribute(positions, 3),
          ),
          new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity,
          }),
        ),
      );
    };

    const addRibbon = (
      group: THREE.Group,
      source: THREE.Vector3[],
      target: THREE.Vector3[],
      startColor: string,
      endColor: string,
    ) => {
      if (!source.length || !target.length) return;
      const extremes = (points: THREE.Vector3[]) => {
        let top = points[0];
        let bottom = points[0];
        for (const point of points) {
          if (point.y > top.y) top = point;
          if (point.y < bottom.y) bottom = point;
        }
        top = top.clone();
        bottom = bottom.clone();
        if (top.y - bottom.y < 0.24) {
          const middle = (top.y + bottom.y) / 2;
          top.y = middle + 0.12;
          bottom.y = middle - 0.12;
        }
        return { top, bottom };
      };
      const start = extremes(source);
      const end = extremes(target);
      const top = makeSpline(start.top, end.top, 0.18).getPoints(28);
      const bottom = makeSpline(start.bottom, end.bottom, 0.18).getPoints(28);
      const positions: number[] = [];
      const colors: number[] = [];
      const from = new THREE.Color(startColor);
      const to = new THREE.Color(endColor);
      for (let i = 0; i < top.length; i++) {
        const t = i / (top.length - 1);
        const tint = from.clone().lerp(to, t);
        positions.push(...top[i].toArray(), ...bottom[i].toArray());
        colors.push(
          tint.r,
          tint.g,
          tint.b,
          tint.r,
          tint.g,
          tint.b,
        );
      }
      const indices: number[] = [];
      for (let i = 0; i < top.length - 1; i++) {
        const a = i * 2;
        const b = a + 1;
        const c = a + 2;
        const d = a + 3;
        indices.push(a, b, c, b, d, c);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3),
      );
      geometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(colors, 3),
      );
      geometry.setIndex(indices);
      group.add(
        new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.17,
            side: THREE.DoubleSide,
            depthWrite: false,
          }),
        ),
      );
    };

    const buildEdge = (
      edge: (typeof graph.edges)[number],
      group: THREE.Group,
    ) => {
      const a = worldPoint(edge.source),
        b = worldPoint(edge.target);
      if (!a || !b) return;
      const sourceHalf = halfDepth.get(edge.source) ?? 0.15;
      const targetHalf = halfDepth.get(edge.target) ?? 0.15;
      const aFace = a.clone().add(new THREE.Vector3(sourceHalf, 0, 0));
      const bFace = b.clone().sub(new THREE.Vector3(targetHalf, 0, 0));
      const target = byId.get(edge.target)!;
      const residual = edge.data?.role === "identity_skip" || edge.data?.role === "detail_skip" ||
        (target.data.kind === "residual" &&
        (depths.get(edge.target)! - depths.get(edge.source)! > 1 ||
          byId.get(edge.source)!.data.params.role === "skip_projection"));
      const an = worldNeurons(edge.source)?.map((point) =>
        point.clone().add(new THREE.Vector3(sourceHalf, 0, 0)),
      );
      const bn = worldNeurons(edge.target)?.map((point) =>
        point.clone().sub(new THREE.Vector3(targetHalf, 0, 0)),
      );
      const color = displayColor(
        residual
          ? "#e8a286"
          : catalog[byId.get(edge.source)!.data.kind].color,
        theme,
      );
      const isContext =
        analysis.nodes[target.id].merge === "query_context" &&
        graph.edges.filter((e) => e.target === target.id)[1]?.id === edge.id;
      if (
        analysis.nodes[edge.source].layout === "streams" &&
        analysis.nodes[edge.target].layout === "streams"
      ) {
        const streamCurves: THREE.Vector3[][] = [];
        for (let lane = 0; lane < 4; lane++) {
          const offset = (lane - 1.5) * 0.9;
          const start = aFace.clone().add(new THREE.Vector3(0.02, 0, offset));
          const end = bFace.clone().add(new THREE.Vector3(-0.02, 0, offset));
          streamCurves.push(makeSpline(start, end, 0.35).getPoints(24));
        }
        addSplineBatch(
          group,
          streamCurves,
          displayColor("#e8a286", theme),
          0.55,
        );
      } else if (isContext) {
        const curve = addSpline(group, aFace, bFace, displayColor("#ddadd2", theme), 0.75, 3);
        const label = labelSprite("CONTEXT · K / V", displayColor("#ddadd2", theme), 0.4);
        label.position.copy(curve.getPoint(0.5));
        label.position.y += 0.35;
        label.visible = labelsValue.current;
        group.add(label);
      } else if (an && bn && !residual) {
        if (
          ["dense", "classification", "regression"].includes(target.data.kind) &&
          (graph.mode !== "schematic" || analysis.nodes[target.id].parameters > 0)
        ) {
          addRibbon(
            group,
            an,
            bn,
            color,
            displayColor(catalog[target.data.kind].color, theme),
          );
        } else {
          const pairs: [THREE.Vector3, THREE.Vector3][] = [];
          for (let i = 0; i < Math.min(an.length, bn.length); i++)
            pairs.push([an[i], bn[i]]);
          addSplineBatch(
            group,
            pairs.map(([start, end]) =>
              makeSpline(start, end, 0.18).getPoints(20),
            ),
            color,
            0.2,
          );
        }
      } else if (residual) {
        const curve = makeSpline(aFace, bFace, 4.2);
        const geometry = new THREE.TubeGeometry(curve, 48, 0.025, 6, false);
        group.add(
          new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({
              color,
              transparent: true,
              opacity: 0.8,
            }),
          ),
        );
        const skipLabel = labelSprite(edge.data?.role === "detail_skip" ? "HIGH-RES DETAIL · CONCAT" : "RESIDUAL STREAM", displayColor("#c99b83", theme), 0.42);
        skipLabel.position.copy(curve.getPoint(0.5));
        skipLabel.position.y += 0.35;
        skipLabel.visible = labelsValue.current;
        group.add(skipLabel);
      } else {
        const sourceShape = analysis.nodes[edge.source].shape,
          targetShape = analysis.nodes[edge.target].shape;
        if (sourceShape.length === 3 && targetShape.length === 3) {
          const startH = spatialScale(sourceShape[1]) / 2,
            startW = spatialScale(sourceShape[2]) / 2;
          const endH = spatialScale(targetShape[1]) / 2,
            endW = spatialScale(targetShape[2]) / 2;
          const cornerCurves: THREE.Vector3[][] = [];
          for (const sy of [-1, 1])
            for (const sz of [-1, 1])
              cornerCurves.push(
                makeSpline(
                  new THREE.Vector3(
                    aFace.x,
                    a.y + sy * startH,
                    a.z + sz * startW,
                  ),
                  new THREE.Vector3(
                    bFace.x,
                    b.y + sy * endH,
                    b.z + sz * endW,
                  ),
                  0.15,
                ).getPoints(24),
              );
          addSplineBatch(group, cornerCurves, color, 0.18);
        } else {
          addSpline(group, aFace, bFace, color, 0.45, 0.18);
        }
      }
    };

    const edgeGroups = new Map<string, THREE.Group>();
    const edgeById = new Map<string, (typeof graph.edges)[number]>();
    const incidentEdges = new Map<string, Set<string>>();
    const clearEdgeGroup = (group: THREE.Group) => {
      for (const child of [...group.children]) {
        group.remove(child);
        disposeBranch(child);
      }
    };
    const drawEdges = () => {
      for (const child of [...edgeLayer.children]) {
        edgeLayer.remove(child);
        disposeBranch(child);
      }
      edgeGroups.clear();
      edgeById.clear();
      incidentEdges.clear();
      graph.edges.forEach((edge, index) => {
        const key = edge.id || `edge-${index}`;
        const group = new THREE.Group();
        edgeLayer.add(group);
        edgeGroups.set(key, group);
        edgeById.set(key, edge);
        for (const node of [edge.source, edge.target]) {
          const set = incidentEdges.get(node) ?? new Set<string>();
          set.add(key);
          incidentEdges.set(node, set);
        }
        buildEdge(edge, group);
      });
    };
    const rebuildEdgesFor = (node: string) => {
      for (const key of incidentEdges.get(node) ?? []) {
        const group = edgeGroups.get(key);
        const edge = edgeById.get(key);
        if (!group || !edge) continue;
        clearEdgeGroup(group);
        buildEdge(edge, group);
      }
    };
    drawEdges();

    const reset = () => {
      camera.position.set(
        centerX + sceneExtent * 0.78,
        centerY + sceneExtent * 0.22,
        centerZ + sceneExtent * 0.55,
      );
      controls.target.set(centerX, centerY, centerZ);
      camera.zoom = 1;
      camera.updateProjectionMatrix();
      controls.update();
    };
    reset();
    const shapeSignature = validNodes.join(",");
    if (cameraView.current?.signature === shapeSignature) {
      camera.position.fromArray(cameraView.current.position);
      controls.target.fromArray(cameraView.current.target);
      camera.zoom = cameraView.current.zoom;
      camera.updateProjectionMatrix();
      controls.update();
    }
    const setLabelVisibility = (visible: boolean) => {
      scene.traverse((object) => {
        if (object.userData.isLabel) object.visible = visible;
      });
    };
    setLabelVisibility(labelsValue.current);
    api.current = {
      reset: () => {
        reset();
        renderer.render(scene, camera);
      },
      resetLayout: () => {
        readout.hidden = true;
        for (const id of Object.keys(manual)) delete manual[id];
        for (const [id, group] of layerGroups) {
          const base = pointMap.get(id);
          if (base) group.position.copy(base);
        }
        drawEdges();
        setLabelVisibility(labelsValue.current);
        storeManualLayout(graph.name, manual);
        renderer.render(scene, camera);
      },
      setRotate: (value) => {
        controls.autoRotate = value;
      },
      setLabels: (value) => {
        setLabelVisibility(value);
        renderer.render(scene, camera);
      },
    };
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      renderer.setSize(width, height);
      const aspect = width / height;
      camera.updateMatrixWorld();
      const right = new THREE.Vector3().setFromMatrixColumn(
        camera.matrixWorld,
        0,
      );
      const up = new THREE.Vector3().setFromMatrixColumn(
        camera.matrixWorld,
        1,
      );
      const projectedWidth =
        Math.abs(right.x) * frameSizeX +
        Math.abs(right.y) * frameSizeY +
        Math.abs(right.z) * frameSizeZ;
      const projectedHeight =
        Math.abs(up.x) * frameSizeX +
        Math.abs(up.y) * frameSizeY +
        Math.abs(up.z) * frameSizeZ;
      const span = Math.max(
        projectedWidth * 0.58,
        projectedHeight * aspect * 0.58,
        5.2 * aspect,
      );
      camera.left = -span;
      camera.right = span;
      camera.top = span / aspect;
      camera.bottom = -span / aspect;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    const renderOnChange = () => renderer.render(scene, camera);
    controls.addEventListener("change", renderOnChange);
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const setPointer = (event: { clientX: number; clientY: number }) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
    };
    const pickNode = (event: { clientX: number; clientY: number }) => {
      setPointer(event);
      raycaster.setFromCamera(pointer, camera);
      return raycaster.intersectObjects(meshes, false)[0]?.object.userData
        .nodeId as string | undefined;
    };
    type DragState = {
      id: string;
      startX: number;
      startY: number;
      startPoint: THREE.Vector3;
      plane: THREE.Plane;
      planeStart: THREE.Vector3;
      moved: boolean;
      autoRotate: boolean;
    };
    let drag: DragState | null = null;
    let down = { x: 0, y: 0 };
    let edgeFrame = 0;
    let edgeDirty: string | null = null;
    const scheduleEdgeRebuild = (node: string) => {
      edgeDirty = node;
      if (edgeFrame) return;
      edgeFrame = requestAnimationFrame(() => {
        edgeFrame = 0;
        const id = edgeDirty;
        edgeDirty = null;
        if (id) rebuildEdgesFor(id);
        renderer.render(scene, camera);
      });
    };
    const coordinate = (value: number) =>
      (Math.abs(value) < 0.05 ? 0 : value).toFixed(1);
    const updateReadout = (point: THREE.Vector3) => {
      readout.textContent = `GRID ${SNAP_STEP} · X ${coordinate(point.x)} · Y ${coordinate(point.y)} · Z ${coordinate(point.z)}`;
    };

    const pointerDown = (event: PointerEvent) => {
      down = { x: event.clientX, y: event.clientY };
      if (event.button !== 0) return;
      const id = pickNode(event);
      if (!id) return;
      const base = pointMap.get(id);
      if (!base) return;
      const startPoint = base.clone().add(manualOffset(id));
      const normal = new THREE.Vector3();
      camera.getWorldDirection(normal);
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
        normal,
        startPoint,
      );
      setPointer(event);
      raycaster.setFromCamera(pointer, camera);
      const planeStart = new THREE.Vector3();
      if (!raycaster.ray.intersectPlane(plane, planeStart))
        planeStart.copy(startPoint);
      drag = {
        id,
        startX: event.clientX,
        startY: event.clientY,
        startPoint,
        plane,
        planeStart,
        moved: false,
        autoRotate: controls.autoRotate,
      };
      updateReadout(startPoint);
      readout.hidden = false;
      controls.enabled = false;
      controls.autoRotate = false;
      renderer.domElement.style.cursor = "grabbing";
      try {
        renderer.domElement.setPointerCapture(event.pointerId);
      } catch {
        // Pointer capture is best-effort; dragging still works inside the canvas.
      }
      event.stopImmediatePropagation();
      event.preventDefault();
    };

    const pointerMove = (event: PointerEvent) => {
      if (!drag) return;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (!drag.moved && Math.hypot(dx, dy) < 2) return;
      drag.moved = true;
      const base = pointMap.get(drag.id);
      if (!base) return;
      const next = drag.startPoint.clone();
      if (event.shiftKey) {
        // Shift locks X and Z and moves only in world Y, using the projected
        // screen direction of the Y axis so the drag tracks the pointer.
        const rect = renderer.domElement.getBoundingClientRect();
        const p0 = drag.startPoint.clone().project(camera);
        const p1 = drag.startPoint
          .clone()
          .add(new THREE.Vector3(0, 1, 0))
          .project(camera);
        const screenX = ((p1.x - p0.x) * rect.width) / 2;
        const screenY = (-(p1.y - p0.y) * rect.height) / 2;
        const lengthSquared = screenX * screenX + screenY * screenY;
        if (lengthSquared > 1)
          next.y =
            drag.startPoint.y +
            (dx * screenX + dy * screenY) / lengthSquared;
        next.y = Math.round(next.y / SNAP_STEP) * SNAP_STEP;
      } else {
        setPointer(event);
        raycaster.setFromCamera(pointer, camera);
        const hit = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(drag.plane, hit))
          next.copy(drag.startPoint).add(hit.sub(drag.planeStart));
        next.set(
          Math.round(next.x / SNAP_STEP) * SNAP_STEP,
          Math.round(next.y / SNAP_STEP) * SNAP_STEP,
          Math.round(next.z / SNAP_STEP) * SNAP_STEP,
        );
      }
      updateReadout(next);
      const offset = new THREE.Vector3().subVectors(next, base);
      manual[drag.id] = [offset.x, offset.y, offset.z];
      layerGroups.get(drag.id)?.position.copy(next);
      scheduleEdgeRebuild(drag.id);
      event.stopImmediatePropagation();
      event.preventDefault();
    };

    const finishDrag = (event: PointerEvent, cancelled = false) => {
      if (!drag) return false;
      const { id, moved, autoRotate } = drag;
      drag = null;
      controls.enabled = true;
      controls.autoRotate = autoRotate;
      renderer.domElement.style.cursor = "";
      readout.hidden = true;
      try {
        renderer.domElement.releasePointerCapture(event.pointerId);
      } catch {
        // The pointer may already have been released by the browser.
      }
      if (moved) storeManualLayout(graph.name, manual);
      if (!cancelled) select.current(id);
      return true;
    };

    const pointerUp = (event: PointerEvent) => {
      if (event.button !== 0) return;
      if (finishDrag(event)) {
        event.stopImmediatePropagation();
        event.preventDefault();
        return;
      }
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5)
        return;
      const id = pickNode(event);
      if (id) select.current(id);
    };

    const pointerCancel = (event: PointerEvent) => {
      finishDrag(event, true);
      event.stopImmediatePropagation();
    };

    const doubleClick = (event: MouseEvent) => {
      const id = pickNode(event);
      const base = id ? pointMap.get(id) : undefined;
      if (!id || !base) return;
      delete manual[id];
      layerGroups.get(id)?.position.copy(base);
      rebuildEdgesFor(id);
      setLabelVisibility(labelsValue.current);
      storeManualLayout(graph.name, manual);
      renderer.render(scene, camera);
      event.stopImmediatePropagation();
      event.preventDefault();
    };

    element.addEventListener("pointerdown", pointerDown, true);
    element.addEventListener("pointermove", pointerMove, true);
    element.addEventListener("pointerup", pointerUp, true);
    element.addEventListener("pointercancel", pointerCancel, true);
    element.addEventListener("dblclick", doubleClick, true);
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      if (document.hidden) return;
      controls.update();
    };
    frame = requestAnimationFrame(animate);
    return () => {
      cameraView.current = {
        position: camera.position.toArray(),
        target: controls.target.toArray(),
        zoom: camera.zoom,
        signature: shapeSignature,
      };
      cancelAnimationFrame(frame);
      cancelAnimationFrame(edgeFrame);
      observer.disconnect();
      controls.removeEventListener("change", renderOnChange);
      controls.dispose();
      element.removeEventListener("pointerdown", pointerDown, true);
      element.removeEventListener("pointermove", pointerMove, true);
      element.removeEventListener("pointerup", pointerUp, true);
      element.removeEventListener("pointercancel", pointerCancel, true);
      element.removeEventListener("dblclick", doubleClick, true);
      readout.remove();
      scene.traverse((object) => {
        const item = object as THREE.Mesh;
        item.geometry?.dispose();
        const materials = item.material
          ? Array.isArray(item.material)
            ? item.material
            : [item.material]
          : [];
        for (const material of materials) {
          (material as THREE.SpriteMaterial).map?.dispose();
          material.dispose();
        }
      });
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      api.current = null;
    };
  }, [signature]);

  return (
    <section className={`architecture-panel ${expanded ? "expanded" : ""}`}>
      <div className="panel-header">
        <div className="panel-title">
          <Box size={14} />
          <span>Architecture</span>
          <span className="live-badge">
            <i />
            LIVE 3D
          </span>
        </div>
        <div className="panel-tools">
          <button
            className={`icon-button ${labels ? "active" : ""}`}
            onClick={() => {
              setLabels(!labels);
              api.current?.setLabels(!labels);
            }}
            title="Toggle layer labels"
            aria-label="Toggle layer labels"
          >
            {labels ? <Eye size={14} /> : <EyeOff size={14} />}
          </button>
          <button
            className={`icon-button ${rotating ? "active" : ""}`}
            onClick={() => {
              setRotating(!rotating);
              api.current?.setRotate(!rotating);
            }}
            title="Auto-orbit"
            aria-label="Auto-orbit"
          >
            <Rotate3D size={15} />
          </button>
          <button
            className="icon-button"
            onClick={() => api.current?.reset()}
            title="Reset camera"
            aria-label="Reset camera"
          >
            <RotateCcw size={14} />
          </button>
          <button
            className="icon-button"
            onClick={() => api.current?.resetLayout()}
            title="Reset manually moved layers"
            aria-label="Reset manually moved layers"
          >
            <Undo2 size={14} />
          </button>
          <span className="tool-separator" />
          <button
            className="icon-button"
            onClick={onExpand}
            title={expanded ? "Split view" : "Expand 3D view"}
            aria-label={expanded ? "Split view" : "Expand 3D view"}
          >
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>
      <div className="architecture-canvas" ref={host} />
      {(!graph.nodes.length || failed) && (
        <div className="architecture-empty">
          <Box size={40} strokeWidth={1} />
          <h3>{failed ? "3D view needs WebGL" : "An idea, taking shape."}</h3>
          <p>
            {failed
              ? "The graph editor and model engine are still available."
              : "Add a building block to start assembling your model."}
          </p>
        </div>
      )}
      <div className="architecture-caption">
        <span>
          <i className="legend-dot" style={{ background: "#7acfd6" }} />
          Spatial
        </span>
        <span>
          <i className="legend-dot" style={{ background: "#baabec" }} />
          Dense
        </span>
        <span>
          <i className="legend-dot" style={{ background: "#e8a286" }} />
          Composition
        </span>
        {graph.nodes.some((n) =>
          [
            "attention",
            "cross_attention",
            "transformer_block",
            "deepseek_stage",
          ].includes(n.data.kind),
        ) && (
          <span title="Sampled connectivity; no learned attention weights are loaded">
            <i className="legend-dot" style={{ background: "#d79fdb" }} />{" "}
            Attention · sampled
          </span>
        )}
        <span className="architecture-hint">
          Drag layer: snap 3D grid <b>·</b> Shift: height <b>·</b> Background:
          orbit
        </span>
      </div>
      <div className="view-axis">
        <span className="axis-y">Y</span>
        <span className="axis-z">Z</span>
        <span className="axis-x">X</span>
        <i />
        <b />
      </div>
    </section>
  );
}
