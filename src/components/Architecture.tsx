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
  Pause,
  Play,
  Undo2,
} from "lucide-react";
import {
  catalog,
  isInput,
  isOutput,
  isRecurrent,
  shapeText,
  type Analysis,
  type ModelGraph,
} from "../model";
import { sequenceGeometry } from "./sequenceGeometry";
import { architectureGlyph, concatSegments, denseRunBoundaries, hasLearnedBias, straightChains, sharesDenseInputColumn } from "./architectureSemantics";
import { compactGridColumns, separateAlignedPaths } from "./architectureLayout";
import { recurrentGeometry, recurrentMetrics, spatialScale, type SpatialFace } from "./recurrentGeometry";
import {
  connectionExtent,
  makeConnectionCurve,
  pairConnectionPorts,
  spatialConnection,
} from "./connectionGeometry";
import {
  applyFlowAttributes,
  createFlowMaterial,
  flowProfile,
  pathProgress,
  type FlowAppearance,
} from "./connectionFlow";

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
const FLOW_SPEED_KEY = "vizml:architecture:flow-speed";
const SNAP_STEP = 2;

function readFlowSpeed() {
  try {
    const value = Number(localStorage.getItem(FLOW_SPEED_KEY) ?? 1);
    return Number.isFinite(value) ? THREE.MathUtils.clamp(value, 0.25, 40) : 1;
  } catch {
    return 1;
  }
}

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
  const nameWidth = context.measureText(name.slice(0, 40)).width;
  context.font = "500 27px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillStyle = detailColor;
  context.fillText(detail.slice(0, 40), 384, 108);
  const textWidth = Math.max(nameWidth, context.measureText(detail.slice(0, 40)).width);
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
  sprite.userData.textWidthFraction = Math.min(1, textWidth / canvas.width);
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
  const [flowing, setFlowing] = useState(
    () => !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [flowSpeed, setFlowSpeed] = useState(readFlowSpeed);
  const [failed, setFailed] = useState(false);
  const select = useRef(onSelect);
  select.current = onSelect;
  const rotateValue = useRef(rotating);
  rotateValue.current = rotating;
  const labelsValue = useRef(labels);
  labelsValue.current = labels;
  const flowingValue = useRef(flowing);
  flowingValue.current = flowing;
  const flowSpeedValue = useRef(flowSpeed);
  flowSpeedValue.current = flowSpeed;
  // Keep the current in place when selecting a layer or rebuilding the scene.
  const flowTime = useRef(0);
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
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setFlowing(!preference.matches);
    preference.addEventListener("change", update);
    return () => preference.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(FLOW_SPEED_KEY, String(flowSpeed));
    } catch {
      // The slider still works when preference storage is unavailable.
    }
  }, [flowSpeed]);

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
    const denseRuns = denseRunBoundaries(validNodes, graph.edges, id => architectureGlyph(byId.get(id)!.data.kind) === "dense");
    const needsInputColumn = (id: string) => {
      if (!denseRuns.starts.has(id)) return false;
      const incoming = graph.edges.filter(edge => edge.target === id && byId.has(edge.source));
      if (incoming.length !== 1) return true;
      const source = byId.get(incoming[0].source)!;
      // A vector input supplies the first column only when it represents the
      // declared input to this layer. Schematic dependencies can describe a
      // projection from a small slice of a much larger observation.
      return !sharesDenseInputColumn(isInput(source.data.kind), analysis.nodes[source.id]?.shape, byId.get(id)!.data.params.input_features);
    };
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
    const pointMap = new Map<string, THREE.Vector3>();
    const neurons = new Map<string, THREE.Vector3[]>();
    const inputNeurons = new Map<string, THREE.Vector3[]>();
    const spatialFaces = new Map<string, { input: SpatialFace; output: SpatialFace }>();
    const halfDepth = new Map<string, number>();
    const meshes: THREE.Object3D[] = [];

    // Information density and bottleneck throughput. A node's density is the
    // log2 of its tensor size; its throughput is the widest path that can reach
    // it from a source and still reach the sea (an output).
    const information = new Map<string, number>();
    for (const id of validNodes) {
      const elements = Math.max(1, analysis.nodes[id].elements || 0);
      information.set(id, Math.log2(elements + 1));
    }
    const upstream = new Map<string, number>();
    for (const id of validNodes) {
      const own = information.get(id) ?? 1;
      const parents = graph.edges
        .filter(
          (edge) =>
            edge.target === id &&
            byId.has(edge.source) &&
            upstream.has(edge.source),
        )
        .map((edge) => upstream.get(edge.source)!);
      upstream.set(
        id,
        parents.length ? Math.max(...parents.map((value) => Math.min(value, own))) : own,
      );
    }
    const downstream = new Map<string, number>();
    for (const id of [...validNodes].reverse()) {
      const own = information.get(id) ?? 1;
      const children = graph.edges
        .filter(
          (edge) =>
            edge.source === id &&
            byId.has(edge.target) &&
            downstream.has(edge.target),
        )
        .map((edge) => downstream.get(edge.target)!);
      downstream.set(
        id,
        children.length
          ? Math.max(...children.map((value) => Math.min(value, own)))
          : own,
      );
    }
    const throughput = new Map<string, number>();
    let maxThroughput = 1;
    for (const id of validNodes) {
      const value = Math.min(
        upstream.get(id) ?? 0,
        downstream.get(id) ?? 0,
      );
      throughput.set(id, value);
      maxThroughput = Math.max(maxThroughput, value);
    }
    const flow = (id: string) =>
      (throughput.get(id) ?? 0) / maxThroughput;

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
      if (architectureGlyph(byId.get(id)!.data.kind) === "dense") return "vector";
      if (info.layout === "streams" || info.shape.length === 2)
        return "sequence";
      if (info.shape.length === 3 || info.shape.length === 4) return "spatial";
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
          flow: flow(edge.source),
          height: flowHeight.get(edge.source) ?? 0,
          depth: depths.get(edge.source) ?? 0,
        }));
      if (!candidates.length) continue;
      candidates.sort(
        (a, b) =>
          Number(a.family !== family) - Number(b.family !== family) ||
          Number(a.skip) - Number(b.skip) ||
          b.flow - a.flow ||
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

    type FlowLane = {
      id: number;
      level: number;
      size: number;
      flow: number;
      z: number;
    };
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
      lanes[lane].flow += flow(node);
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
            flow: 0,
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
      lanes.push({ id: next, level, size: 0, flow: 0, z: 0 });
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
          b.flow - a.flow ||
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
    for (const id of validNodes) {
      const node = byId.get(id)!;
      const info = analysis.nodes[id];
      if (isRecurrent(node.data.kind)) {
        halfDepth.set(id, recurrentMetrics(node, info).halfDepth);
      } else if (info.shape.length === 3 || info.shape.length === 4) {
        const count =
          info.shape.length === 4 ? Math.min(6, info.shape[0]) : node.data.kind === "convnext_stage"
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
    // Pack each path independently instead of widening an entire depth when
    // one unrelated branch needs room for an MLP input column.
    const gridX = compactGridColumns(validNodes, graph.edges, id => {
      const glyph = architectureGlyph(byId.get(id)!.data.kind);
      const operatorDepth = glyph === "gather" || glyph === "activation" ? 0.55
        : glyph === "pool" ? 0.6 : glyph === "dot" ? 0.38 : undefined;
      const half = operatorDepth ?? halfDepth.get(id) ?? 0.15;
      return { left: half + (needsInputColumn(id) ? 1.7 : 0), right: half };
    }, SNAP_STEP);
    const maxX = Math.max(1, ...gridX.values());
    const pointFor = (id: string) => {
      const depth = depths.get(id)!;
      const lane = lanes[laneOfNode.get(id) ?? 0];
      return new THREE.Vector3(
        gridX.get(id) ?? depth * SNAP_STEP,
        0.3 -
          (displayLevel.get(id) ?? lane?.level ?? 0) * flowLevelSpacing,
        displayLateral.get(id) ??
          (lane?.z ?? 0) + (lateralOfNode.get(id) ?? 0),
      );
    };
    const halfWidthOf = (id: string) => {
      const shape = analysis.nodes[id]?.shape ?? [];
      const node = byId.get(id)!;
      if (isRecurrent(node.data.kind)) return recurrentMetrics(node, analysis.nodes[id]).radius;
      if (shape.length === 3 || shape.length === 4) return spatialScale(shape.at(-1)!) / 2;
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
        if (!source || source.data.kind !== "dense" || !/bias|condition/i.test(`${source.data.label} ${source.data.params.role ?? ""}`)) continue;
        if (graph.edges.filter(e => e.source === edge.source && byId.has(e.target)).length !== 1) continue;
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
    // Least-action relaxation. X stays on the flow axis; Y/Z settle under
    // information-weighted edge springs, channel cohesion, and clearance
    // repulsion. Sources and the sea stay near their headwater/mouth positions.
    const positions = new Map<string, THREE.Vector3>();
    const initial = new Map<string, THREE.Vector3>();
    const channelOf = new Map<string, string>();
    const hasIncoming = new Set<string>();
    const hasOutgoing = new Set<string>();
    for (const edge of graph.edges) {
      if (byId.has(edge.source) && byId.has(edge.target)) {
        hasOutgoing.add(edge.source);
        hasIncoming.add(edge.target);
      }
    }
    for (const id of validNodes) {
      const point = (parkedAt.get(id) ?? pointFor(id)).clone();
      positions.set(id, point.clone());
      initial.set(id, point.clone());
      channelOf.set(
        id,
        parkedAt.has(id) || displayLevel.has(id)
          ? `anchor:${id}`
          : `${laneOfNode.get(id)}:${Math.round((lateralOfNode.get(id) ?? 0) * 10)}`,
      );
    }
    const maxAbsZBefore = Math.max(
      1,
      ...[...positions.values()].map((point) => Math.abs(point.z)),
    );
    let maxLevelBefore = 0;
    for (const lane of lanes) maxLevelBefore = Math.max(maxLevelBefore, lane.level);
    const velocity = new Map<string, THREE.Vector2>();
    const iterations =
      validNodes.length > 180 ? 70 : validNodes.length > 100 ? 110 : 170;
    for (let iteration = 0; iteration < iterations; iteration++) {
      const forces = new Map<string, THREE.Vector2>();
      for (const id of validNodes) forces.set(id, new THREE.Vector2());
      const forceFor = (id: string) => forces.get(id)!;

      // High-information connections pull hardest.
      for (const edge of graph.edges) {
        const a = positions.get(edge.source);
        const b = positions.get(edge.target);
        if (!a || !b) continue;
        const weight =
          0.05 *
          (0.25 +
            0.75 * Math.min(flow(edge.source), flow(edge.target)));
        const dy = b.y - a.y;
        const dz = b.z - a.z;
        forceFor(edge.source).x += dy * weight;
        forceFor(edge.source).y += dz * weight;
        forceFor(edge.target).x -= dy * weight;
        forceFor(edge.target).y -= dz * weight;
      }

      // Each channel wants to stay a river, not a scatter of beads.
      const sums = new Map<string, { y: number; z: number; w: number }>();
      for (const id of validNodes) {
        const key = channelOf.get(id)!;
        const point = positions.get(id)!;
        const weight = 0.4 + flow(id);
        const sum = sums.get(key) ?? { y: 0, z: 0, w: 0 };
        sum.y += point.y * weight;
        sum.z += point.z * weight;
        sum.w += weight;
        sums.set(key, sum);
      }
      for (const id of validNodes) {
        const key = channelOf.get(id)!;
        const sum = sums.get(key)!;
        const point = positions.get(id)!;
        const pull = 0.022 * (0.35 + flow(id));
        forceFor(id).x += (sum.y / sum.w - point.y) * pull;
        forceFor(id).y += (sum.z / sum.w - point.z) * pull;
      }

      // Clearance only between different channels. Nodes along one channel
      // are meant to share a line.
      for (let i = 0; i < validNodes.length; i++) {
        const aId = validNodes[i];
        const a = positions.get(aId)!;
        const aChannel = channelOf.get(aId)!;
        for (let j = i + 1; j < validNodes.length; j++) {
          const bId = validNodes[j];
          const bChannel = channelOf.get(bId)!;
          if (aChannel === bChannel) continue;
          const b = positions.get(bId)!;
          if (Math.abs(a.x - b.x) > 6) continue;
          let dy = a.y - b.y;
          let dz = a.z - b.z;
          const distance = Math.hypot(dy, dz);
          if (distance > 6.5) continue;
          const clearance =
            3.4 + 2.2 * Math.max(flow(aId), flow(bId));
          if (distance >= clearance) continue;
          const push =
            ((clearance - distance) / clearance) *
            0.16 *
            (0.5 + Math.max(flow(aId), flow(bId)));
          if (distance < 0.05) {
            dy = i % 2 ? 1 : -1;
            dz = j % 2 ? 1 : -1;
          } else {
            dy /= distance;
            dz /= distance;
          }
          forceFor(aId).x += dy * push;
          forceFor(aId).y += dz * push;
          forceFor(bId).x -= dy * push;
          forceFor(bId).y -= dz * push;
        }
      }

      // Keep headwaters and mouths legible and prevent global drift.
      for (const id of validNodes) {
        const point = positions.get(id)!;
        const anchor = initial.get(id)!;
        const special = parkedAt.has(id) || displayLevel.has(id);
        const endpoint = !hasIncoming.has(id) || !hasOutgoing.has(id);
        const pull =
          special ? 0.11 : endpoint ? 0.045 : 0.014;
        forceFor(id).x += (anchor.y - point.y) * pull;
        forceFor(id).y += (anchor.z - point.z) * pull;
      }

      // Damped Verlet-style integration in the YZ plane.
      for (const id of validNodes) {
        const point = positions.get(id)!;
        let v = velocity.get(id);
        if (!v) {
          v = new THREE.Vector2();
          velocity.set(id, v);
        }
        const force = forceFor(id);
        v.x = (v.x + force.x) * 0.78;
        v.y = (v.y + force.y) * 0.78;
        point.y += v.x * 0.32;
        point.z += v.y * 0.32;
        point.y = Math.max(
          0.3 - maxLevelBefore * flowLevelSpacing - 12,
          Math.min(8, point.y),
        );
        point.z = Math.max(
          -maxAbsZBefore - 18,
          Math.min(maxAbsZBefore + 18, point.z),
        );
      }
    }

    // Keep sequential operations on one centerline, then resolve collisions
    // between whole paths so alignment cannot undo the clearance pass.
    const chains = straightChains(validNodes, graph.edges, id => {
      const glyph = architectureGlyph(byId.get(id)!.data.kind);
      if (["concat", "add", "gather", "dot"].includes(glyph) || parkedAt.has(id)) return "other";
      return layoutFamily(id);
    });
    separateAlignedPaths(validNodes, chains, positions, halfWidthOf, new Set(parkedAt.keys()), SNAP_STEP);
    // Place each condition/bias beside its own dependent, with separate slots
    // when an add has more than one such input.
    const parkedCounts = new Map<string, number>();
    for (const [id] of parkedAt) {
      const edge = graph.edges.find(e => e.source === id && ["add", "residual"].includes(byId.get(e.target)?.data.kind ?? ""));
      const target = edge && positions.get(edge.target);
      const point = positions.get(id);
      if (target && point && edge) {
        const slot = parkedCounts.get(edge.target) ?? 0;
        parkedCounts.set(edge.target, slot + 1);
        point.x = target.x - SNAP_STEP;
        point.y = target.y;
        point.z = target.z - (Math.ceil((halfWidthOf(edge.target) + 1.5) / SNAP_STEP) + slot) * SNAP_STEP;
      }
    }

    const layoutPoint = (id: string) =>
      positions.get(id) ?? parkedAt.get(id) ?? pointFor(id);
    let minZ = Number.POSITIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    let minX = Number.POSITIVE_INFINITY;
    let maxBoundX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const id of validNodes) {
      const point = layoutPoint(id);
      minZ = Math.min(minZ, point.z);
      maxZ = Math.max(maxZ, point.z);
      minX = Math.min(minX, point.x);
      maxBoundX = Math.max(maxBoundX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
    const frameMinX = Math.min(-4, minX - 4);
    const frameMaxX = Math.max(maxX + 4, maxBoundX + 4);
    const frameMinY = minY - 4.2;
    const frameMaxY = maxY + 4.8;
    const frameMinZ = Number.isFinite(minZ) ? minZ - 4 : -4;
    const frameMaxZ = Number.isFinite(maxZ) ? maxZ + 4 : 4;
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
    const nodeLabels: { id: string; sprite: THREE.Sprite }[] = [];
    const recurrentLoops = new Map<string, THREE.Vector3[][]>();

    for (const id of validNodes) {
      const node = byId.get(id)!;
      const info = analysis.nodes[id];
      const point = layoutPoint(id);
      pointMap.set(id, point);
      const group = new THREE.Group();
      group.position.copy(point).add(manualOffset(id));
      layerGroups.set(id, group);
      const glyph = architectureGlyph(node.data.kind);
      const terminal = !hasOutgoing.has(id);
      const nodeColor = isInput(node.data.kind) || (glyph === "dense" && denseRuns.ends.has(id)) || (terminal && isOutput(node.data.kind))
        ? "#e3b74f" : glyph === "dense" ? "#ad78df" : catalog[node.data.kind].color;
      const color = displayColor(nodeColor, theme);
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
      if (["concat", "add", "gather", "dot", "normalize", "pool", "activation", "embedding"].includes(glyph)) {
        // These are operators, not output tensors: show the same glyph at every rank.
        const opMaterial = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.16, roughness: 0.4 });
        const addMesh = (geometry: THREE.BufferGeometry, x = 0, y = 0, z = 0) => {
          const mesh = new THREE.Mesh(geometry, opMaterial);
          mesh.position.set(x, y, z); mesh.userData.nodeId = id; meshes.push(mesh); group.add(mesh);
          return mesh;
        };
        if (glyph === "concat") {
          addMesh(new THREE.BoxGeometry(0.18, 4, 0.22));
          const incoming = graph.edges.filter(e => e.target === id && byId.has(e.source));
          const segments = concatSegments(incoming.length);
          for (const [, end] of segments.slice(0, -1))
            addMesh(new THREE.BoxGeometry(0.22, 0.025, 0.34), 0, 2 - end * 4, 0);
          halfDepth.set(id, 0.09);
        } else if (glyph === "add") {
          addMesh(new THREE.TorusGeometry(0.5, 0.065, 8, 32));
          addMesh(new THREE.BoxGeometry(0.08, 0.65, 0.1));
          addMesh(new THREE.BoxGeometry(0.65, 0.08, 0.1));
          halfDepth.set(id, 0.12);
        } else if (glyph === "gather") {
          // A slotted bracket suggests selection from a larger collection.
          addMesh(new THREE.BoxGeometry(0.12, 2.4, 0.12));
          addMesh(new THREE.BoxGeometry(0.55, 0.12, 0.12), 0.22, 1.15);
          addMesh(new THREE.BoxGeometry(0.55, 0.12, 0.12), 0.22, -1.15);
          for (let i = -1; i <= 1; i++) addMesh(new THREE.SphereGeometry(0.13, 10, 8), 0.3, i * 0.55);
          halfDepth.set(id, 0.55);
        } else if (glyph === "dot") {
          addMesh(new THREE.IcosahedronGeometry(0.38, 0));
          halfDepth.set(id, 0.38);
        } else if (glyph === "normalize") {
          // A framed wave denotes rescaling, distinct from neuron columns.
          addMesh(new THREE.TorusGeometry(0.55, 0.045, 8, 4)).rotation.z = Math.PI / 4;
          for (let i = 0; i < 3; i++) addMesh(new THREE.BoxGeometry(0.38, 0.05, 0.08), 0, (i - 1) * 0.2);
          halfDepth.set(id, 0.15);
        } else if (glyph === "pool") {
          addMesh(new THREE.ConeGeometry(0.6, 1.15, 4, 1, true)).rotation.z = -Math.PI / 2;
          halfDepth.set(id, 0.6);
        } else if (glyph === "activation") {
          addMesh(new THREE.CapsuleGeometry(0.22, 0.65, 4, 8)).rotation.z = Math.PI / 2;
          halfDepth.set(id, 0.55);
        } else {
          for (let row = -1; row <= 1; row++) for (let col = -1; col <= 1; col++)
            addMesh(new THREE.BoxGeometry(0.12, 0.18, 0.18), 0, row * 0.28, col * 0.28);
          halfDepth.set(id, 0.12);
        }
      } else if (isRecurrent(node.data.kind)) {
        const glyph = recurrentGeometry(group, node, info, color, focused, meshes, opacity);
        if (glyph.ports) neurons.set(id, glyph.ports);
        if (glyph.inputPorts) inputNeurons.set(id, glyph.inputPorts);
        if (glyph.spatialFaces) spatialFaces.set(id, glyph.spatialFaces);
        recurrentLoops.set(id, glyph.loops);
        halfDepth.set(id, glyph.halfDepth);
        const badge = labelSprite(`${node.data.params.layers ?? 1}L · ${glyph.directions === 2 ? "↔ · " : ""}${glyph.cell === "lstm" ? "h + c" : "h"} · ${info.recurrent?.steps ?? info.input_shape[0]} STEPS`, badgeInk, 0.36);
        badge.position.set(0, Math.max(2.35, glyph.radius + 0.4), 0);
        group.add(badge);
      } else if (
        glyph !== "dense" && (info.shape.length === 2 ||
        info.layout === "streams" ||
        node.data.kind === "token_input")
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
      } else if (glyph !== "dense" && (info.shape.length === 3 || info.shape.length === 4)) {
        const [channels, h, w] = info.shape.slice(-3);
        const height = spatialScale(h),
          width = spatialScale(w);
        const count =
          info.shape.length === 4 ? Math.min(6, info.shape[0]) : node.data.kind === "convnext_stage"
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
        const count = Math.max(1, Math.min(13, glyph === "dense" ? info.shape.at(-1)! : info.shape[0]));
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
        if (glyph === "dense" && needsInputColumn(id)) {
          // The first column is the MLP's input, followed by the outputs of
          // each dense operation. These are sampled, not extra graph nodes.
          const width = Number(node.data.params.input_features) || info.input_shape.at(-1) || 1;
          const inputCount = Math.max(1, Math.min(13, width));
          const inputPositions = Array.from({ length: inputCount }, (_, i) =>
            new THREE.Vector3(-1.7, inputCount === 1 ? 0 : (i / (inputCount - 1) - 0.5) * height, 0));
          const inputColor = displayColor("#e3b74f", theme);
          const inputDots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.075, 12, 10),
            new THREE.MeshStandardMaterial({ color: inputColor, emissive: inputColor, emissiveIntensity: 0.65, transparent: true, opacity }), inputCount);
          const marker = new THREE.Object3D();
          inputPositions.forEach((position, i) => { marker.position.copy(position); marker.updateMatrix(); inputDots.setMatrixAt(i, marker.matrix); });
          inputDots.instanceMatrix.needsUpdate = true; inputDots.userData.nodeId = id; meshes.push(inputDots); group.add(inputDots);
          inputNeurons.set(id, inputPositions);
          const threads: number[] = [];
          for (const start of inputPositions) for (const end of positions) threads.push(...start.toArray(), ...end.toArray());
          group.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(threads, 3)),
            new THREE.LineBasicMaterial({ color: displayColor("#b294d6", theme), transparent: true, opacity: (theme === "light" ? 0.23 : 0.17) * opacity })));
        }
        if (glyph === "dense" && hasLearnedBias(node, graph.mode === "schematic")) {
          const biases = new THREE.InstancedMesh(new THREE.SphereGeometry(0.038, 10, 8),
            new THREE.MeshStandardMaterial({ color: "#268fff", emissive: "#268fff", emissiveIntensity: 0.5 }), count);
          const marker = new THREE.Object3D();
          positions.forEach((position, i) => {
            marker.position.copy(position).add(new THREE.Vector3(0.025, 0.025, 0.067));
            marker.updateMatrix(); biases.setMatrixAt(i, marker.matrix);
          });
          biases.instanceMatrix.needsUpdate = true; group.add(biases);
        }
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
      nodeLabels.push({ id, sprite: details });
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
    const worldNeurons = (id: string, input = false) => {
      const origin = worldPoint(id);
      const local = (input ? inputNeurons.get(id) : undefined) ?? neurons.get(id);
      return origin && local ? local.map((point) => origin.clone().add(point)) : undefined;
    };

    const edgeLayer = new THREE.Group();
    scene.add(edgeLayer);
    const sharedMaterials = new Set<THREE.Material>();
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
          if (!sharedMaterials.has(material)) material.dispose();
        }
      });
    };
    const flowMaterial = createFlowMaterial(theme);
    flowMaterial.uniforms.uTime.value = flowTime.current;
    sharedMaterials.add(flowMaterial);
    for (const [id, loops] of recurrentLoops) {
      const color = displayColor(catalog[byId.get(id)!.data.kind].color, theme);
      for (const points of loops) {
        const { progress, length } = pathProgress(points);
        const geometry = applyFlowAttributes(new THREE.BufferGeometry().setFromPoints(points), progress,
          { profile: flowProfile(flow(id), id, length), color, opacity: (selected && selected !== id ? 0.65 : 1) * 0.14 });
        layerGroups.get(id)!.add(new THREE.Line(geometry, flowMaterial));
      }
    }
    const addSpline = (
      group: THREE.Group,
      start: THREE.Vector3,
      end: THREE.Vector3,
      appearance: FlowAppearance,
      bow = 0,
    ) => {
      const curve = makeConnectionCurve(start, end, bow);
      const points = curve.getPoints(40);
      group.add(
        new THREE.Line(
          applyFlowAttributes(
            new THREE.BufferGeometry().setFromPoints(points),
            pathProgress(points).progress,
            appearance,
          ),
          flowMaterial,
        ),
      );
      return curve;
    };
    const addSplineBatch = (
      group: THREE.Group,
      curves: THREE.Vector3[][],
      appearance: FlowAppearance,
    ) => {
      const positions: number[] = [];
      const along: number[] = [];
      for (const points of curves) {
        const { progress } = pathProgress(points);
        for (let i = 1; i < points.length; i++) {
          positions.push(
            ...points[i - 1].toArray(),
            ...points[i].toArray(),
          );
          along.push(progress[i - 1], progress[i]);
        }
      }
      group.add(
        new THREE.LineSegments(
          applyFlowAttributes(
            new THREE.BufferGeometry().setAttribute(
              "position",
              new THREE.Float32BufferAttribute(positions, 3),
            ),
            along,
            appearance,
          ),
          flowMaterial,
        ),
      );
    };

    const addRibbon = (
      group: THREE.Group,
      source: THREE.Vector3[],
      target: THREE.Vector3[],
      appearance: FlowAppearance,
      exactStart = false,
      exactEnd = false,
    ) => {
      if (!source.length || !target.length) return;
      const exactExtent = (points: THREE.Vector3[]) => {
        const ordered = [...points].sort((a, b) => b.y - a.y);
        return { top: ordered[0].clone(), bottom: ordered[ordered.length - 1].clone() };
      };
      const start = exactStart ? exactExtent(source) : connectionExtent(source);
      const end = exactEnd ? exactExtent(target) : connectionExtent(target);
      const top = makeConnectionCurve(start.top, end.top, 0.18).getPoints(28);
      const bottom = makeConnectionCurve(start.bottom, end.bottom, 0.18).getPoints(28);
      const { progress } = pathProgress(top.map((point, i) =>
        point.clone().add(bottom[i]).multiplyScalar(0.5),
      ));
      const positions: number[] = [];
      const along: number[] = [];
      const across: number[] = [];
      for (let i = 0; i < top.length; i++) {
        positions.push(...top[i].toArray(), ...bottom[i].toArray());
        along.push(progress[i], progress[i]);
        across.push(0, 1);
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
      geometry.setIndex(indices);
      applyFlowAttributes(geometry, along, { ...appearance, surface: true }, across);
      group.add(new THREE.Mesh(geometry, flowMaterial));
    };

    const buildEdge = (
      edge: (typeof graph.edges)[number],
      group: THREE.Group,
      key: string,
    ) => {
      const a = worldPoint(edge.source),
        b = worldPoint(edge.target);
      if (!a || !b) return;
      const sourceHalf = halfDepth.get(edge.source) ?? 0.15;
      const targetHalf = halfDepth.get(edge.target) ?? 0.15;
      const aFace = a.clone().add(new THREE.Vector3(sourceHalf, 0, 0));
      const bFace = b.clone().sub(new THREE.Vector3(targetHalf + (inputNeurons.has(edge.target) ? 1.7 : 0), 0, 0));
      const sourceSpatialFace = spatialFaces.get(edge.source)?.output;
      const targetSpatialFace = spatialFaces.get(edge.target)?.input;
      if (sourceSpatialFace) aFace.add(sourceSpatialFace.center);
      if (targetSpatialFace) bFace.add(targetSpatialFace.center);
      const target = byId.get(edge.target)!;
      const residual = edge.data?.role === "identity_skip" || edge.data?.role === "detail_skip" ||
        (target.data.kind === "residual" &&
        (depths.get(edge.target)! - depths.get(edge.source)! > 1 ||
          byId.get(edge.source)!.data.params.role === "skip_projection"));
      const an = worldNeurons(edge.source)?.map((point) =>
        point.clone().add(new THREE.Vector3(sourceHalf, 0, 0)),
      );
      const bn = worldNeurons(edge.target, true)?.map((point) =>
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
      const profile = flowProfile(
        Math.min(flow(edge.source), flow(edge.target)),
        key,
        pathProgress(makeConnectionCurve(aFace, bFace, residual ? 4.2 : isContext ? 3 : 0.18).getPoints(40)).length,
      );
      const focus = selected && selected !== edge.source && selected !== edge.target ? 0.65 : 1;
      const appearance = (opacity: number, tint?: string): FlowAppearance => ({
        profile,
        color: tint ?? color,
        endColor: tint ?? (residual ? color : displayColor(catalog[target.data.kind].color, theme)),
        opacity: opacity * focus,
      });
      const longConnection = Math.abs(bFace.x - aFace.x) > SNAP_STEP * 4;
      const sourceGlyph = architectureGlyph(byId.get(edge.source)!.data.kind);
      const targetGlyph = architectureGlyph(target.data.kind);
      if (sourceGlyph === "concat" || targetGlyph === "concat") {
        const extent = (center: THREE.Vector3, points: THREE.Vector3[] | undefined, spatial: SpatialFace | undefined) => {
          if (points?.length) return points;
          const height = spatial?.dimensions[0] ?? 1.2;
          return [center.clone().add(new THREE.Vector3(0, height / 2, 0)), center.clone().add(new THREE.Vector3(0, -height / 2, 0))];
        };
        const bar = (center: THREE.Vector3, start: number, end: number) => [
          center.clone().add(new THREE.Vector3(0, 2 - 4 * start, 0)),
          center.clone().add(new THREE.Vector3(0, 2 - 4 * end, 0)),
        ];
        let from = extent(aFace, an, sourceSpatialFace);
        let to = extent(bFace, bn, targetSpatialFace);
        if (sourceGlyph === "concat") from = bar(aFace, 0, 1);
        if (targetGlyph === "concat") {
          const incoming = graph.edges.filter(e => e.target === edge.target && byId.has(e.source));
          const segments = concatSegments(incoming.length);
          const index = incoming.indexOf(edge);
          const [start, end] = segments[index] ?? [0, 1];
          to = bar(bFace, start, end);
        }
        addRibbon(group, from, to, appearance(longConnection ? (theme === "light" ? 0.075 : 0.06) : (theme === "light" ? 0.24 : 0.17)), sourceGlyph === "concat", targetGlyph === "concat");
        if (sourceGlyph === "concat" && targetGlyph === "dense" && bn && !inputNeurons.has(edge.target) &&
            (graph.mode !== "schematic" || analysis.nodes[target.id].parameters > 0)) {
          const samples = Array.from({ length: 13 }, (_, i) => aFace.clone().add(new THREE.Vector3(0, 2 - i * 4 / 12, 0)));
          addSplineBatch(group, samples.flatMap(start => bn.map(end => makeConnectionCurve(start, end, 0).getPoints(16))),
            appearance(theme === "light" ? 0.20 : 0.15));
        }
      } else if (
        analysis.nodes[edge.source].layout === "streams" &&
        analysis.nodes[edge.target].layout === "streams"
      ) {
        const streamCurves: THREE.Vector3[][] = [];
        for (let lane = 0; lane < 4; lane++) {
          const offset = (lane - 1.5) * 0.9;
          const start = aFace.clone().add(new THREE.Vector3(0.02, 0, offset));
          const end = bFace.clone().add(new THREE.Vector3(-0.02, 0, offset));
          streamCurves.push(makeConnectionCurve(start, end, 0.35).getPoints(24));
        }
        addSplineBatch(
          group,
          streamCurves,
          appearance(theme === "light" ? 0.38 : 0.28, displayColor("#e8a286", theme)),
        );
      } else if (isContext) {
        const curve = addSpline(group, aFace, bFace,
          appearance(theme === "light" ? 0.48 : 0.38, displayColor("#ddadd2", theme)), 3);
        const label = labelSprite("CONTEXT · K / V", displayColor("#ddadd2", theme), 0.4);
        label.position.copy(curve.getPoint(0.5));
        label.position.y += 0.35;
        label.visible = labelsValue.current;
        group.add(label);
      } else if (bn && !residual && targetGlyph === "dense" && !inputNeurons.has(edge.target) &&
          (graph.mode !== "schematic" || analysis.nodes[target.id].parameters > 0)) {
        // Sample flattened spatial inputs too: every shown source connects to
        // every shown target, but these threads are not actual weight values.
        const sourceCount = Math.max(1, Math.min(13, analysis.nodes[edge.source].elements || 1));
        const height = sourceSpatialFace?.dimensions[0] ?? 2.4;
        const samples = an ?? Array.from({ length: sourceCount }, (_, i) =>
          aFace.clone().add(new THREE.Vector3(0, sourceCount === 1 ? 0 : (i / (sourceCount - 1) - 0.5) * height, 0)));
        addSplineBatch(group, samples.flatMap(start => bn.map(end =>
          makeConnectionCurve(start, end, 0).getPoints(16))),
          appearance(theme === "light" ? 0.20 : 0.15));
      } else if (an && bn && !residual && longConnection) {
        // Long data dependencies are bundles, not weight matrices. Show one
        // translucent ribbon and its centerline instead of a stack of threads
        // stretching across unrelated layers.
        addRibbon(group, an, bn, appearance(theme === "light" ? 0.06 : 0.05));
        addSpline(group, aFace, bFace, appearance(theme === "light" ? 0.20 : 0.16), 0.18);
      } else if (an && bn && !residual) {
          const pairs = pairConnectionPorts(an, bn);
          addSplineBatch(
            group,
            pairs.map(([start, end]) =>
              makeConnectionCurve(start, end, 0.18).getPoints(20),
            ),
            appearance(theme === "light" ? 0.22 : 0.16),
          );
      } else if (residual) {
        const curve = makeConnectionCurve(aFace, bFace, 4.2);
        const geometry = new THREE.TubeGeometry(curve, 48, 0.025, 6, false);
        // TubeGeometry samples by arc length; its U coordinate follows the path.
        const uv = geometry.attributes.uv;
        const along = Float32Array.from({ length: uv.count }, (_, i) => uv.getX(i));
        applyFlowAttributes(geometry, along, appearance(theme === "light" ? 0.48 : 0.38));
        group.add(
          new THREE.Mesh(
            geometry,
            flowMaterial,
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
        if ([3, 4].includes(sourceShape.length) && [3, 4].includes(targetShape.length)) {
          const volume = spatialConnection(
            aFace,
            bFace,
            sourceSpatialFace?.dimensions ?? [spatialScale(sourceShape.at(-2)!), spatialScale(sourceShape.at(-1)!)],
            targetSpatialFace?.dimensions ?? [spatialScale(targetShape.at(-2)!), spatialScale(targetShape.at(-1)!)],
          );
          const { progress } = pathProgress(volume.centerline);
          applyFlowAttributes(
            volume.geometry,
            volume.stations.map(station => progress[station]),
            { ...appearance(theme === "light" ? 0.045 : 0.035), surface: true },
            volume.across,
          );
          const faceAppearance = volume.geometry.attributes.aAppearance as THREE.BufferAttribute;
          volume.faceOpacity.forEach((opacity, i) =>
            faceAppearance.setX(i, faceAppearance.getX(i) * opacity),
          );
          group.add(new THREE.Mesh(volume.geometry, flowMaterial));
          addSplineBatch(group, volume.curves, appearance(theme === "light" ? 0.22 : 0.16));
        } else {
          addSpline(group, aFace, bFace, appearance(theme === "light" ? 0.38 : 0.26), 0.18);
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
        buildEdge(edge, group, key);
      });
    };
    const rebuildEdgesFor = (node: string) => {
      for (const key of incidentEdges.get(node) ?? []) {
        const group = edgeGroups.get(key);
        const edge = edgeById.get(key);
        if (!group || !edge) continue;
        clearEdgeGroup(group);
        buildEdge(edge, group, key);
      }
    };
    drawEdges();

    const reset = () => {
      camera.position.set(
        centerX + sceneExtent * 0.18,
        centerY + sceneExtent * 0.12,
        centerZ + sceneExtent * 1.0,
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
    // Labels share screen space even when their nodes are separated in 3D.
    // Keep the selected label and reveal additional non-overlapping labels as
    // the camera zooms in, instead of drawing an unreadable wall of text.
    const declutterLabels = () => {
      if (!labelsValue.current) return;
      const width = renderer.domElement.clientWidth, height = renderer.domElement.clientHeight;
      if (!width || !height) return;
      camera.updateMatrixWorld();
      scene.updateMatrixWorld();
      const occupied: { left: number; right: number; top: number; bottom: number }[] = [];
      const ordered = [...nodeLabels].sort((a, b) => Number(b.id === selected) - Number(a.id === selected));
      for (const { id, sprite } of ordered) {
        const point = new THREE.Vector3();
        sprite.getWorldPosition(point).project(camera);
        const x = (point.x + 1) * width / 2, y = (1 - point.y) * height / 2;
        const pixelScale = width * camera.zoom / (camera.right - camera.left);
        const w = sprite.scale.x * pixelScale * sprite.userData.textWidthFraction + 8;
        const h = sprite.scale.y * pixelScale * 0.72 + 4;
        const box = { left: x - w / 2, right: x + w / 2, top: y - h / 2, bottom: y + h / 2 };
        const visible = id === selected || (h >= 8 && point.z >= -1 && point.z <= 1 &&
          box.right >= 0 && box.left <= width && box.bottom >= 0 && box.top <= height &&
          !occupied.some(other => box.left < other.right && box.right > other.left && box.top < other.bottom && box.bottom > other.top));
        sprite.visible = visible;
        if (visible) occupied.push(box);
      }
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
    let previousTime: number | null = null;
    const animate = (time: number) => {
      frame = requestAnimationFrame(animate);
      const delta = previousTime === null ? 0 : Math.min((time - previousTime) / 1000, 0.1);
      previousTime = time;
      if (document.hidden) return;
      controls.update();
      declutterLabels();
      if (flowingValue.current) flowTime.current += delta * flowSpeedValue.current;
      flowMaterial.uniforms.uTime.value = flowTime.current;
      renderer.render(scene, camera);
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
          if (!sharedMaterials.has(material)) material.dispose();
        }
      });
      for (const material of sharedMaterials) material.dispose();
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
            className={`icon-button ${flowing ? "active" : ""}`}
            onClick={() => setFlowing(!flowing)}
            title={flowing ? "Pause flow animation" : "Resume flow animation"}
            aria-label={flowing ? "Pause flow animation" : "Resume flow animation"}
          >
            {flowing ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <label className="flow-speed" title="Animation speed">
            <span>Flow</span>
            <input
              type="range"
              min="0.25"
              max="40"
              step="0.25"
              value={flowSpeed}
              onChange={(event) => setFlowSpeed(Number(event.target.value))}
              aria-label="Animation speed"
              aria-valuetext={`${flowSpeed}× speed`}
            />
            <output aria-hidden="true">{flowSpeed}×</output>
          </label>
          <span className="tool-separator" />
          <button
            className={`icon-button ${labels ? "active" : ""}`}
            onClick={() => {
              setLabels(!labels);
              api.current?.setLabels(!labels);
            }}
            title="Toggle layer labels · zoom in to reveal more"
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
          Hidden · sampled neurons
        </span>
        <span><i className="legend-dot" style={{ background: "#e3b74f" }} /> Input / output</span>
        <span><i className="legend-dot" style={{ background: "#268fff" }} /> Bias</span>
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
