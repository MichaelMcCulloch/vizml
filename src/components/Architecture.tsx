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
  selected: string | null;
  onSelect: (id: string) => void;
  expanded: boolean;
  onExpand: () => void;
};
type SceneAPI = {
  reset: () => void;
  setRotate: (enabled: boolean) => void;
  setLabels: (enabled: boolean) => void;
};

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

export default function Architecture({
  graph,
  analysis,
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
        powerPreference: "low-power",
      });
    } catch {
      setFailed(true);
      return;
    }
    setFailed(false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x101610, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive 3D model architecture. Drag to orbit, scroll to zoom, click a layer to inspect.",
    );
    renderer.domElement.setAttribute("role", "img");
    element.appendChild(renderer.domElement);
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
    scene.add(new THREE.AmbientLight(0xc6d7c4, 1.7));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3);
    keyLight.position.set(-6, 12, 8);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x76c9b3, 2.5);
    rimLight.position.set(12, 5, -8);
    scene.add(rimLight);

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
    const totalWidth = maxX + 3;
    camera.far = Math.max(300, totalWidth * 3);
    const centerX = maxX / 2;
    const pointMap = new Map<string, THREE.Vector3>();
    const neurons = new Map<string, THREE.Vector3[]>();
    const meshes: THREE.Object3D[] = [];
    const labelObjects: THREE.Sprite[] = [];
    const laneCounts = new Map<number, number>();
    const laneMap = new Map<string, number>();
    for (const id of [...validNodes].sort(
      (a, b) =>
        depths.get(a)! - depths.get(b)! ||
        Number(byId.get(a)!.data.params.role === "skip_projection") -
          Number(byId.get(b)!.data.params.role === "skip_projection") ||
        analysis.nodes[b].shape.length - analysis.nodes[a].shape.length,
    )) {
      const depth = depths.get(id)!;
      const lane = laneCounts.get(depth) ?? 0;
      laneMap.set(id, lane);
      laneCounts.set(depth, lane + 1);
    }
    const layerGroups = new Map<string, THREE.Group>();
    const spatialScale = (v: number) =>
      Math.max(0.65, Math.min(5.3, Math.sqrt(v / 32) * 3.4));

    for (const id of validNodes) {
      const node = byId.get(id)!;
      const info = analysis.nodes[id];
      const depth = depths.get(id)!;
      const lane = laneMap.get(id) ?? 0;
      const point = new THREE.Vector3(
        xForDepth(depth),
        lane ? -3.8 * lane : 0.3,
        0,
      );
      pointMap.set(id, point);
      const group = new THREE.Group();
      group.position.copy(point);
      layerGroups.set(id, group);
      const color = catalog[node.data.kind].color;
      const focused = selected === id;
      const opacity = selected && !focused ? 0.72 : 1;
      const material = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: 0.26 * opacity,
        roughness: 0.38,
        metalness: 0.16,
        emissive: color,
        emissiveIntensity: focused ? 0.2 : 0.045,
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
            labelObjects.push(label);
          },
        );
        neurons.set(id, positions);
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
        for (let i = 0; i < count; i++) {
          const geometry = new THREE.BoxGeometry(0.07, height, width);
          const box = new THREE.Mesh(geometry, material);
          box.position.x = (i - (count - 1) / 2) * 0.13;
          box.userData.nodeId = id;
          meshes.push(box);
          group.add(box);
          const outline = new THREE.LineSegments(
            new THREE.EdgesGeometry(geometry),
            new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: focused ? 0.95 : 0.62 * opacity,
            }),
          );
          outline.position.copy(box.position);
          group.add(outline);
        }
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
        const positions: THREE.Vector3[] = [];
        const geometry = new THREE.SphereGeometry(0.075, 12, 10);
        const dotMaterial = new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.65,
          transparent: true,
          opacity,
        });
        for (let i = 0; i < count; i++) {
          const y = count === 1 ? 0 : (i / (count - 1) - 0.5) * height;
          const dot = new THREE.Mesh(geometry, dotMaterial);
          dot.position.set(0, y, 0);
          dot.userData.nodeId = id;
          meshes.push(dot);
          group.add(dot);
          positions.push(new THREE.Vector3(point.x, point.y + y, 0));
          if (focused) {
            const halo = new THREE.Mesh(
              new THREE.SphereGeometry(0.13, 8, 6),
              new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.08,
              }),
            );
            halo.position.copy(dot.position);
            group.add(halo);
          }
        }
        neurons.set(id, positions);
      }
      const block = graph.mode === "schematic" ? node.id.match(/_block(\d+)_(conv1|conv2|add)$/) : null;
      const shortName = typeof node.data.view_label === "string" ? node.data.view_label : block ? `B${block[1]} · ${block[2] === "add" ? "Add" : `${block[2] === "conv1" ? "Conv 1" : "Conv 2"} · d${node.data.params.dilation}`}` : node.data.label;
      const name = labelSprite(
        shortName,
        focused ? "#f0f2e9" : "#b0b8aa",
        graph.mode === "schematic" ? 0.6 : 0.68,
      );
      name.position.set(0, -2.75, 0);
      group.add(name);
      labelObjects.push(name);
      const dimensions = labelSprite(shapeText(info.display_shape ?? info.shape), color, 0.57);
      dimensions.position.set(0, -3.12, 0);
      group.add(dimensions);
      labelObjects.push(dimensions);
      if (isInput(node.data.kind) && (graph.mode !== "schematic" || !info.dependencies)) {
        const badge = labelSprite("INPUT", "#6d7869", 0.4);
        badge.position.set(0, 2.75, 0);
        group.add(badge);
        labelObjects.push(badge);
      }
      if (isOutput(node.data.kind) && graph.mode !== "schematic") {
        const badge = labelSprite(
          node.data.kind === "classification"
            ? "ONE-HOT"
            : node.data.kind === "embedding"
              ? "EMBEDDING"
              : "REGRESSION",
          "#a9ce7d",
          0.4,
        );
        badge.position.set(0, 2.5, 0);
        group.add(badge);
        labelObjects.push(badge);
      }
      scene.add(group);
    }

    for (const edge of graph.edges) {
      const a = pointMap.get(edge.source),
        b = pointMap.get(edge.target);
      if (!a || !b) continue;
      const target = byId.get(edge.target)!;
      const residual = edge.data?.role === "identity_skip" || edge.data?.role === "detail_skip" ||
        (target.data.kind === "residual" &&
        (depths.get(edge.target)! - depths.get(edge.source)! > 1 ||
          byId.get(edge.source)!.data.params.role === "skip_projection"));
      const an = neurons.get(edge.source),
        bn = neurons.get(edge.target);
      const color = residual
        ? "#e8a286"
        : catalog[byId.get(edge.source)!.data.kind].color;
      const isContext =
        analysis.nodes[target.id].merge === "query_context" &&
        graph.edges.filter((e) => e.target === target.id)[1]?.id === edge.id;
      if (
        analysis.nodes[edge.source].layout === "streams" &&
        analysis.nodes[edge.target].layout === "streams"
      ) {
        for (let lane = 0; lane < 4; lane++) {
          const z = (lane - 1.5) * 0.9;
          const start = a.clone().add(new THREE.Vector3(0.2, 0, z));
          const end = b.clone().add(new THREE.Vector3(-0.2, 0, z));
          const curve = new THREE.CubicBezierCurve3(
            start,
            start.clone().add(new THREE.Vector3(0.6, 0.15, 0)),
            end.clone().sub(new THREE.Vector3(0.6, 0.15, 0)),
            end,
          );
          scene.add(
            new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)),
              new THREE.LineBasicMaterial({
                color: "#e8a286",
                transparent: true,
                opacity: 0.5,
              }),
            ),
          );
        }
      } else if (isContext) {
        const curve = new THREE.CubicBezierCurve3(
          a,
          a.clone().add(new THREE.Vector3(1, 3, 0)),
          b.clone().add(new THREE.Vector3(-1, 3, 0)),
          b,
        );
        scene.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(curve.getPoints(32)),
            new THREE.LineBasicMaterial({
              color: "#ddadd2",
              transparent: true,
              opacity: 0.7,
            }),
          ),
        );
        const label = labelSprite("CONTEXT · K / V", "#ddadd2", 0.4);
        label.position.copy(curve.getPoint(0.5));
        label.position.y += 0.35;
        scene.add(label);
        labelObjects.push(label);
      } else if (an && bn && !residual) {
        const points: number[] = [];
        if (
          ["dense", "classification", "regression"].includes(target.data.kind) &&
          (graph.mode !== "schematic" || analysis.nodes[target.id].parameters > 0)
        ) {
          for (const start of an)
            for (const end of bn)
              points.push(...start.toArray(), ...end.toArray());
        } else {
          for (let i = 0; i < Math.min(an.length, bn.length); i++)
            points.push(...an[i].toArray(), ...bn[i].toArray());
        }
        scene.add(
          new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute(
              "position",
              new THREE.Float32BufferAttribute(points, 3),
            ),
            new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: 0.18,
            }),
          ),
        );
      } else if (residual || a.y !== b.y) {
        const lift = residual ? 4.2 : -1.2;
        const curve = new THREE.CubicBezierCurve3(
          a,
          new THREE.Vector3(a.x + 1.5, a.y + lift, 0),
          new THREE.Vector3(b.x - 1.5, b.y + lift, 0),
          b,
        );
        const geometry = new THREE.TubeGeometry(
          curve,
          40,
          residual ? 0.025 : 0.012,
          6,
          false,
        );
        scene.add(
          new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({
              color,
              transparent: true,
              opacity: residual ? 0.8 : 0.35,
            }),
          ),
        );
        if (residual) {
          const skipLabel = labelSprite(edge.data?.role === "detail_skip" ? "HIGH-RES DETAIL · CONCAT" : "RESIDUAL STREAM", "#c99b83", 0.42);
          skipLabel.position.copy(curve.getPoint(0.5));
          skipLabel.position.y += 0.35;
          scene.add(skipLabel);
          labelObjects.push(skipLabel);
        }
      } else {
        const sourceShape = analysis.nodes[edge.source].shape,
          targetShape = analysis.nodes[edge.target].shape;
        if (sourceShape.length === 3 && targetShape.length === 3) {
          const startH = spatialScale(sourceShape[1]) / 2,
            startW = spatialScale(sourceShape[2]) / 2;
          const endH = spatialScale(targetShape[1]) / 2,
            endW = spatialScale(targetShape[2]) / 2;
          const points: number[] = [];
          for (const sy of [-1, 1])
            for (const sz of [-1, 1])
              points.push(
                a.x + 0.4,
                a.y + sy * startH,
                sy * 0 + sz * startW,
                b.x - 0.4,
                b.y + sy * endH,
                sz * endW,
              );
          scene.add(
            new THREE.LineSegments(
              new THREE.BufferGeometry().setAttribute(
                "position",
                new THREE.Float32BufferAttribute(points, 3),
              ),
              new THREE.LineBasicMaterial({
                color,
                transparent: true,
                opacity: 0.18,
              }),
            ),
          );
        } else {
          const curve = new THREE.CubicBezierCurve3(
            a,
            a.clone().add(new THREE.Vector3(0.9, 0, 0)),
            b.clone().sub(new THREE.Vector3(0.9, 0, 0)),
            b,
          );
          scene.add(
            new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(curve.getPoints(30)),
              new THREE.LineBasicMaterial({
                color,
                transparent: true,
                opacity: 0.45,
              }),
            ),
          );
        }
      }
    }

    const grid = new THREE.GridHelper(
      Math.max(60, totalWidth * 2),
      Math.min(100, Math.round(totalWidth * 3)),
      0x253122,
      0x1a241a,
    );
    grid.position.set(centerX, -5.8, 0);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.45;
    scene.add(grid);
    const reset = () => {
      camera.position.set(
        centerX + totalWidth * 0.48,
        totalWidth * 0.28,
        totalWidth * 0.85,
      );
      controls.target.set(centerX, -0.7, 0);
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
    labelObjects.forEach((label) => {
      label.visible = labelsValue.current;
    });
    api.current = {
      reset: () => {
        reset();
        renderer.render(scene, camera);
      },
      setRotate: (value) => {
        controls.autoRotate = value;
      },
      setLabels: (value) => {
        labelObjects.forEach((label) => {
          label.visible = value;
        });
        renderer.render(scene, camera);
      },
    };
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      renderer.setSize(width, height);
      const aspect = width / height;
      const span = Math.max(totalWidth * 0.6, 5.2 * aspect);
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
    let down = { x: 0, y: 0 };
    const pointerDown = (event: PointerEvent) => {
      down = { x: event.clientX, y: event.clientY };
    };
    const pointerUp = (event: PointerEvent) => {
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5)
        return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(meshes, false)[0];
      if (hit?.object.userData.nodeId)
        select.current(hit.object.userData.nodeId);
    };
    renderer.domElement.addEventListener("pointerdown", pointerDown);
    renderer.domElement.addEventListener("pointerup", pointerUp);
    let frame = 0;
    let previous = 0;
    const animate = (time: number) => {
      frame = requestAnimationFrame(animate);
      if (time - previous < 1000 / 40 || document.hidden) return;
      previous = time;
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
      observer.disconnect();
      controls.removeEventListener("change", renderOnChange);
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
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
          Drag to orbit <b>·</b> Scroll to zoom
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
