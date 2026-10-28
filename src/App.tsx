import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  MiniMap,
  applyNodeChanges,
  applyEdgeChanges,
  useReactFlow,
  useUpdateNodeInternals,
  type Connection,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import {
  ChevronDown,
  ChevronRight,
  Check,
  Expand,
  GitBranch,
  LayoutGrid,
  Loader2,
  Minus,
  Plus,
  Redo2,
  Undo2,
  X,
  CircleAlert,
  CheckCircle2,
  PanelLeftClose,
  ArrowUpRight,
  FileSearch,
  Moon,
  Monitor,
  Sun,
} from "lucide-react";
import Architecture from "./components/Architecture";
import ResizablePanels from "./components/ResizablePanels";
import StatusBar from "./components/StatusBar";
import TemplateMenu from "./components/TemplateMenu";
import WorkspaceControls from "./components/WorkspaceControls";
import CheckpointControls from "./components/CheckpointControls";
import { preferredSnapshotScope, snapshotScopes } from "./snapshotView";
import Library from "./components/Library";
import Inspector from "./components/Inspector";
import LayerNode, { AnalysisContext } from "./components/LayerNode";
import SkipEdge from "./components/SkipEdge";
import {
  autoLayout,
  canConnect,
  projectResidualSkips,
  expandConvNeXtStage,
  catalog,
  cleanGraph,
  editedSchematic,
  examples,
  makeNode,
  parseGraph,
  shapeText,
  type Kind,
  type LayerNode as LayerNodeType,
  type ModelGraph,
  type NodeAnalysis,
  type Params,
} from "./model";
import { download, fileName, useEngine } from "./useEngine";
import { withNodeMeasurements } from "./graphView";
import { useProjectBaseline } from "./useProjectBaseline";

const nodeTypes = { layer: LayerNode };
const edgeTypes = { skip: SkipEdge };
const STORAGE = "vizml.graph.v1";
const THEME_STORAGE = "vizml.theme";
type ThemeMode = "light" | "dark" | "system";
function themeInk(hex: string, theme: "light" | "dark") {
  if (theme === "dark") return hex;
  const value = hex.replace("#", "");
  const expanded =
    value.length === 3
      ? value
          .split("")
          .map((part) => part + part)
          .join("")
      : value.slice(0, 6);
  const number = Number.parseInt(expanded, 16);
  if (!Number.isFinite(number)) return "#4d5b6b";
  const red = Math.round(((number >> 16) & 255) * 0.55);
  const green = Math.round(((number >> 8) & 255) * 0.55);
  const blue = Math.round((number & 255) * 0.55);
  return `rgb(${red} ${green} ${blue})`;
}

function initialTheme(): ThemeMode {
  try {
    const value = localStorage.getItem(THEME_STORAGE);
    if (value === "light" || value === "dark" || value === "system")
      return value;
  } catch {
    /* Use the system preference. */
  }
  return "system";
}
function initialGraph() {
  try {
    const value = localStorage.getItem(STORAGE);
    if (value) return parseGraph(value);
  } catch {
    /* Recover with the starter graph. */
  }
  return examples.multimodal();
}

export default function App() {
  const [graph, setGraph] = useState<ModelGraph>(initialGraph);
  const [themeMode, setThemeMode] = useState<ThemeMode>(initialTheme);
  const [systemDark, setSystemDark] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
  const effectiveTheme: "light" | "dark" =
    themeMode === "system" ? (systemDark ? "dark" : "light") : themeMode;
  const workspaceVersion = useRef(0);
  const graphRef = useRef(graph);
  graphRef.current = graph;
  const [selected, setSelected] = useState<string | null>(
    () => graph.nodes.find((n) => n.id === "conv2")?.id ?? null,
  );
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [toast, setToast] = useState<{
    message: string;
    error: boolean;
  } | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [exampleMenu, setExampleMenu] = useState(false);
  const [dialog, setDialog] = useState<"help" | "issues" | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [showLibrary, setShowLibrary] = useState(() => window.innerWidth > 900);
  const [saved, setSaved] = useState(true);
  const [projectId, setProjectId] = useState<string | null>(() => {
    try { return localStorage.getItem("vizml.project.id"); } catch { return null; }
  });
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;
  const { baseline: projectBaseline, status: baselineStatus, clear: clearProjectBaseline, saved: markProjectSaved } = useProjectBaseline(projectId);
  const projectGraph = useMemo(() => projectId ? JSON.stringify(cleanGraph(graph)) : null, [projectId, graph]);
  const changeProjectId = useCallback((id: string | null) => {
    if (projectIdRef.current !== id) clearProjectBaseline();
    projectIdRef.current = id;
    setProjectId(id);
    try { if (id) localStorage.setItem("vizml.project.id", id); else localStorage.removeItem("vizml.project.id"); } catch { /* workspace still works for this session */ }
  }, [clearProjectBaseline]);
  const [, setHistoryVersion] = useState(0);
  const history = useRef<{
    past: { graph: ModelGraph; projectId: string | null }[];
    future: { graph: ModelGraph; projectId: string | null }[];
    lastEdit: number;
  }>({ past: [], future: [], lastEdit: 0 });
  const flow = useReactFlow<LayerNodeType>();
  const schematic = graph.mode === "schematic";
  const [showSnapshotInfo, setShowSnapshotInfo] = useState(false);
  const [scope, setScope] = useState(() => localStorage.getItem("vizml.snapshot.focus") ?? "all");
  const scopes = useMemo(() => snapshotScopes(graph), [graph]);
  const viewGraph = useMemo(() => {
    if (!schematic || scope === "all") return graph;
    const ids = new Set(scopes.find(s => s.id === scope)?.ids ?? graph.nodes.map(n => n.id));
    return { ...graph, nodes: graph.nodes.filter(n => ids.has(n.id)), edges: graph.edges.filter(e => ids.has(e.source) && ids.has(e.target)) };
  }, [graph, schematic, scope, scopes]);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const change = () => setSystemDark(media.matches);
    media.addEventListener("change", change);
    return () => media.removeEventListener("change", change);
  }, []);
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = effectiveTheme;
    try {
      localStorage.setItem(THEME_STORAGE, themeMode);
    } catch {
      /* Theme still applies for this session. */
    }
    const meta = document.querySelector('meta[name="theme-color"]');
    meta?.setAttribute(
      "content",
      effectiveTheme === "dark" ? "#0e1219" : "#f8fafc",
    );
  }, [themeMode, effectiveTheme]);

  useEffect(() => {
    localStorage.setItem("vizml.snapshot.focus", scope);
    if (schematic && !scopes.some(s => s.id === scope)) setScope("all");
  }, [scope, schematic, scopes]);
  const graphPane = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const notice = useCallback((message: string, error = false) => {
    setToast({ message, error });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), error ? 6500 : 4500);
  }, []);
  const engine = useEngine(graph, notice);
  const { analysis } = engine;
  const updateNodeInternals = useUpdateNodeInternals();
  const nodeGeometry = useRef(new Map<string, { data: LayerNodeType["data"]; info: NodeAnalysis | undefined }>());
  useEffect(() => {
    const next = new Map<string, { data: LayerNodeType["data"]; info: NodeAnalysis | undefined }>();
    const changed: string[] = [];
    for (const node of viewGraph.nodes) {
      const info = analysis?.nodes[node.id];
      const previous = nodeGeometry.current.get(node.id);
      if (!previous || previous.data !== node.data || previous.info !== info) changed.push(node.id);
      next.set(node.id, { data: node.data, info });
    }
    nodeGeometry.current = next;
    // A reused ID can gain/lose ports without resizing. Refresh after edits
    // and async shape labels reach the DOM, in one batch instead of per node.
    if (changed.length) updateNodeInternals(changed);
  }, [viewGraph.nodes, analysis, updateNodeInternals]);
  const snapshot = useCallback(() => {
    history.current.past.push({ graph: cleanGraph(graphRef.current), projectId: projectIdRef.current });
    if (history.current.past.length > 70) history.current.past.shift();
    history.current.future = [];
    setHistoryVersion((v) => v + 1);
  }, []);
  const commit = useCallback(
    (next: ModelGraph, remember = true, coalesce = false) => {
      if (
        remember &&
        (!coalesce || Date.now() - history.current.lastEdit > 650)
      )
        snapshot();
      history.current.lastEdit = coalesce ? Date.now() : 0;
      const measured = withNodeMeasurements(next, graphRef.current);
      graphRef.current = measured;
      setGraph(measured);
    },
    [snapshot],
  );
  const fit = useCallback(() => {
    requestAnimationFrame(() =>
      flow.fitView({ padding: 0.2, duration: 450, maxZoom: 0.9 }),
    );
  }, [flow]);
  const select = useCallback((id: string | null) => {
    setShowSnapshotInfo(false);
    if (id && graphRef.current.mode === "schematic") {
      const node = graphRef.current.nodes.find(n => n.id === id);
      if (node) {
        if (!viewGraph.nodes.some(n => n.id === id)) setScope("all");
        setTimeout(() => flow.setCenter(node.position.x + 105, node.position.y + 65, { zoom: 0.85, duration: 400 }), 30);
      }
    }
    setSelected(id);
    setSelectedEdge(null);
  }, [flow, viewGraph.nodes]);

  useEffect(() => {
    setSaved(false);
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(STORAGE, JSON.stringify(cleanGraph(graph)));
        setSaved(true);
      } catch {
        notice(
          "Browser storage is full. Save a graph file to keep your changes.",
          true,
        );
      }
    }, 350);
    return () => clearTimeout(timer);
  }, [graph, notice]);
  useEffect(() => {
    if (!exampleMenu) return;
    const outside = (event: MouseEvent) => {
      if (!menu.current?.contains(event.target as globalThis.Node))
        setExampleMenu(false);
    };
    document.addEventListener("mousedown", outside);
    return () => document.removeEventListener("mousedown", outside);
  }, [exampleMenu]);

  const addNode = (kind: Kind, position?: { x: number; y: number }) => {
    if (graph.nodes.length >= 300) {
      notice("A graph can contain up to 300 layers.", true);
      return;
    }
    const rect = graphPane.current?.getBoundingClientRect();
    const point =
      position ??
      flow.screenToFlowPosition({
        x: (rect?.left ?? 230) + (rect?.width ?? 700) * 0.45,
        y: (rect?.top ?? 400) + (rect?.height ?? 300) * 0.4,
      });
    const layer = makeNode(
      kind,
      point.x + Math.random() * 30,
      point.y + Math.random() * 30,
    );
    if (schematic) {
      layer.data = { ...layer.data, output_shape: [1], parameter_count: 0, declaration_pending: true,
        description: "New schematic block. Declare its output shape and parameter count." };
      setScope("all");
    }
    commit(editedSchematic({ ...graph, nodes: [...graph.nodes, layer] }));
    select(layer.id);
    if (expanded) setExpanded(false);
    if (window.innerWidth <= 900) setShowLibrary(false);
    notice(
      schematic ? "Block added to schematic. Set its declared shape and connect it." : kind === "residual"
        ? "Connect the skip path and a transformed branch. Both tensors must have matching shapes."
        : `${catalog[kind].title} added. Drag between ports to connect it.`,
    );
  };
  const connect = useCallback(
    (source: string, target: string) => {
      const current = graphRef.current;
      if (!canConnect(current, source, target)) {
        notice(
          "That connection creates a cycle, duplicates a dependency, or connects an output to an input.",
          true,
        );
        return;
      }
      if (current.edges.length >= 2000) {
        notice("A graph can contain up to 2,000 connections.", true);
        return;
      }
      commit(editedSchematic({
        ...current,
        edges: [...current.edges, { id: crypto.randomUUID(), source, target }],
      }));
    },
    [commit, notice],
  );
  const updateNode = (id: string, label: string, params: Params) =>
    commit(
      {
        ...graph,
        nodes: graph.nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, label, params } } : n,
        ),
      },
      true,
      true,
    );
  const deleteNode = useCallback(
    (id: string) => {
      const current = graphRef.current;
      commit(editedSchematic({
        ...current,
        nodes: current.nodes.filter((n) => n.id !== id),
        edges: current.edges.filter((e) => e.source !== id && e.target !== id),
      }));
      setSelected(null);
    },
    [commit],
  );
  const disconnect = useCallback(
    (id: string) => {
      const current = graphRef.current;
      commit(editedSchematic({ ...current, edges: current.edges.filter((e) => e.id !== id) }));
      setSelectedEdge(null);
    },
    [commit],
  );
  const duplicate = (id: string) => {
    if (graph.nodes.length >= 300) {
      notice("A graph can contain up to 300 layers.", true);
      return;
    }
    const source = graph.nodes.find((n) => n.id === id)!;
    const copy = makeNode(
      source.data.kind,
      source.position.x + 35,
      source.position.y + 180,
    );
    copy.data = {
      ...source.data,
      label: `${source.data.label} copy`.slice(0, 100),
      params: { ...source.data.params },
    };
    if (schematic) {
      copy.data = { ...copy.data, parameter_tensors: [], declaration_pending: true };
      setScope("all");
    }
    commit(editedSchematic({ ...graph, nodes: [...graph.nodes, copy] }));
    select(copy.id);
  };
  const updateSchematicNode = (id: string, patch: Partial<LayerNodeType["data"]>) => {
    const current = graphRef.current;
    const next = { ...current, nodes: current.nodes.map(n => n.id === id ? { ...n, data: { ...n.data, ...patch } } : n) };
    commit(Object.keys(patch).every(key => key === "label" || key === "description") ? next : editedSchematic(next), true, true);
  };
  const projectResidual = (id: string) => {
    if (engine.checking) {
      notice("Waiting for the latest tensor shapes. Try again in a moment.");
      return;
    }
    const next = projectResidualSkips(graphRef.current, id, analysis);
    if (!next) {
      notice(
        "A 1×1 convolution cannot match these branches automatically. Check the input shapes or graph size limit.",
        true,
      );
      return;
    }
    commit(next);
    select(id);
    notice(
      "Added explicit 1×1 skip projections. Select them to inspect channels and stride.",
    );
  };
  const expandStage = (id: string) => {
    const next = expandConvNeXtStage(graphRef.current, id);
    if (!next) {
      notice(
        "This stage cannot be expanded within the graph size limit.",
        true,
      );
      return;
    }
    commit(next);
    select(id);
    notice(
      "Stage expanded into individually editable residual blocks. Undo restores the grouped stage.",
    );
  };
  const swapInputs = (id: string) => {
    const current = graphRef.current;
    const incoming = current.edges.filter((e) => e.target === id);
    if (incoming.length !== 2) return;
    let index = 0;
    commit({
      ...current,
      edges: current.edges.map((e) =>
        e.target === id ? incoming[1 - index++] : e,
      ),
    });
  };
  const undo = useCallback(() => {
    const previous = history.current.past.pop();
    if (!previous) return;
    workspaceVersion.current++;
    history.current.future.push({ graph: cleanGraph(graphRef.current), projectId: projectIdRef.current });
    const measured = withNodeMeasurements(previous.graph, graphRef.current);
    graphRef.current = measured;
    setGraph(measured);
    setHistoryVersion((v) => v + 1);
    select(null);
    changeProjectId(previous.projectId);
  }, [select, changeProjectId]);
  const redo = useCallback(() => {
    const next = history.current.future.pop();
    if (!next) return;
    workspaceVersion.current++;
    history.current.past.push({ graph: cleanGraph(graphRef.current), projectId: projectIdRef.current });
    const measured = withNodeMeasurements(next.graph, graphRef.current);
    graphRef.current = measured;
    setGraph(measured);
    setHistoryVersion((v) => v + 1);
    select(null);
    changeProjectId(next.projectId);
  }, [select, changeProjectId]);
  const saveFile = useCallback(() => {
    const current = graphRef.current;
    download(
      JSON.stringify(cleanGraph(current), null, 2),
      `${fileName(current.name)}.vizml.json`,
    );
    notice("Architecture saved as a portable graph file.");
  }, [notice]);
  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.key === "Escape") {
        setDialog(null);
        setExampleMenu(false);
        select(null);
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key === "s") {
        event.preventDefault();
        window.dispatchEvent(new Event("vizml:save-project"));
        return;
      }
      if (
        ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) ||
        target.isContentEditable
      )
        return;
      if (event.key === "Delete" || event.key === "Backspace") {
        if (selected) {
          event.preventDefault();
          deleteNode(selected);
        } else if (selectedEdge) {
          event.preventDefault();
          disconnect(selectedEdge);
        }
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
      if (event.key.toLowerCase() === "f") fit();
      if (event.key === "?") setDialog("help");
    };
    window.addEventListener("keydown", keyDown);
    return () => window.removeEventListener("keydown", keyDown);
  }, [
    selected,
    selectedEdge,
    deleteNode,
    disconnect,
    undo,
    redo,
    saveFile,
    fit,
    select,
  ]);

  const onNodesChange = useCallback(
    (changes: NodeChange<LayerNodeType>[]) => {
      const changesToApply = changes.filter((c) => c.type !== "select");
      if (!changesToApply.length) return;
      const current = graphRef.current;
      commit(
        changesToApply.some(c => c.type === "remove") ? editedSchematic({ ...current, nodes: applyNodeChanges(changesToApply, current.nodes) }) : { ...current, nodes: applyNodeChanges(changesToApply, current.nodes) },
        false,
      );
    },
    [commit],
  );
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const changesToApply = changes.filter((c) => c.type !== "select");
      if (changesToApply.length)
        commit(
          editedSchematic({
            ...graphRef.current,
            edges: applyEdgeChanges(changesToApply, graphRef.current.edges),
          }),
          false,
        );
    },
    [commit],
  );
  const loadExample = (key: string) => {
    workspaceVersion.current++;
    const next = examples[key]();
    commit(next);
    changeProjectId(null);
    select(null);
    setScope(preferredSnapshotScope(next));
    setExampleMenu(false);
    setExpanded(false);
    setTimeout(fit, 50);
    notice(
      key === "blank"
        ? "A fresh canvas. Start with an input."
        : "Example loaded. Your previous graph is available with Undo.",
    );
  };
  const displayNodes = useMemo(() => viewGraph.nodes.map((n) => ({
    ...n,
    selected: n.id === selected,
  })), [viewGraph.nodes, selected]);
  const displayEdges = viewGraph.edges.map((e) => {
    const target = graph.nodes.find((n) => n.id === e.target);
    const source = graph.nodes.find((n) => n.id === e.source);
    const cross =
      target?.data.kind === "cross_attention" ||
      (target?.data.kind === "transformer_block" &&
        target.data.params.mode === "cross");
    const context =
      cross &&
      graph.edges.filter((link) => link.target === e.target)[1]?.id === e.id;
    const color =
      target?.data.kind === "residual"
        ? catalog.residual.color
        : source
          ? catalog[source.data.kind].color
          : "#8a98a8";
    const stroke = themeInk(color, effectiveTheme);
    const selectedStroke = effectiveTheme === "light" ? "#4055c7" : "#d8edba";
    return {
      ...e,
      sourceHandle: "out",
      targetHandle: "in",
      type: ["identity_skip", "detail_skip"].includes(String(e.data?.role)) ? "skip" : "default",
      selected: e.id === selectedEdge,
      label: cross ? (context ? "K / V" : "Query") : e.label,
      labelStyle: {
        fill: themeInk("#ddadd2", effectiveTheme),
        fontSize: 9,
      },
      labelBgStyle: {
        fill: effectiveTheme === "light" ? "#ffffff" : "#1c231a",
        fillOpacity: 0.95,
      },
      style: {
        stroke: e.id === selectedEdge ? selectedStroke : stroke,
        strokeWidth: e.id === selectedEdge ? 2.5 : 1.6,
        opacity: e.id === selectedEdge ? 1 : effectiveTheme === "light" ? 0.78 : 0.65,
        strokeDasharray: context || ["identity_skip", "detail_skip"].includes(String(e.data?.role)) ? "5 4" : undefined,
      },
    };
  });
  const activeNode = graph.nodes.find((n) => n.id === selected);
  const issues = [...(analysis?.errors ?? []), ...(analysis?.warnings ?? [])];
  const canBuild =
    !!analysis?.valid &&
    analysis.executable !== false &&
    analysis.outputs.length > 0 &&
    engine.status === "connected" &&
    !engine.checking;
  const helpItems = [
    ["Add a block", "Click the library or drag a block onto the graph."],
    [
      "Connect layers",
      "Drag from a right port to a left port. A layer can have multiple dependencies.",
    ],
    [
      "Compose branches",
      "Incoming branches concatenate in connection order. Grid + vector becomes one feature vector.",
    ],
    [
      "Add a residual",
      "Connect two matching tensors to a Residual stream block.",
    ],
    [
      "Inspect the model",
      "Select a layer in either view. Drag the 3D view to orbit and scroll to zoom.",
    ],
    [
      "Build with PyTorch",
      "Verify builds a real nn.Module on the meta device. Export downloads standalone Python.",
    ],
  ];

  return (
    <div className={`app-shell ${schematic ? "snapshot-mode" : ""}`}>
      <header className="topbar">
        <a className="brand" href="/" aria-label="vizml home">
          <span className="brand-mark">
            <svg viewBox="0 0 28 28" fill="none">
              <path
                d="m5 8 9 14 9-14M10 8l4 6 4-6"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span>
            vizml<span className="brand-period">.</span>
          </span>
          <span className="studio-tag">STUDIO</span>
        </a>
        <div className="project-breadcrumb">
          <span className="topbar-divider" />
          <span className="workspace-label">{projectId ? "Project" : "Workspace"}</span>
          <ChevronRight size={13} />
          <input
            aria-label="Architecture name"
            value={graph.name}
            maxLength={100}
            onChange={(e) =>
              commit({ ...graph, name: e.target.value }, true, true)
            }
          />
          <span
            className="saved-status"
            title={saved ? "Saved in this browser" : "Saving…"}
          >
            {saved ? <Check size={12} /> : <span className="saving-dot" />}
          </span>
          {projectId && <span className="project-save-state" title="Browser autosave and the saved project are separate">{baselineStatus === "checking" ? "Checking project" : baselineStatus === "unavailable" ? "Project status unavailable" : projectGraph === projectBaseline ? "Project saved" : "Unsaved changes"}</span>}
        </div>
        <div className="topbar-actions">
          <div className="theme-switch" role="group" aria-label="Color theme">
            <button
              className={themeMode === "light" ? "active" : ""}
              onClick={() => setThemeMode("light")}
              title="Light theme"
              aria-label="Light theme"
              aria-pressed={themeMode === "light"}
            >
              <Sun size={13} />
            </button>
            <button
              className={themeMode === "system" ? "active" : ""}
              onClick={() => setThemeMode("system")}
              title="Follow system theme"
              aria-label="Follow system theme"
              aria-pressed={themeMode === "system"}
            >
              <Monitor size={13} />
            </button>
            <button
              className={themeMode === "dark" ? "active" : ""}
              onClick={() => setThemeMode("dark")}
              title="Dark theme"
              aria-label="Dark theme"
              aria-pressed={themeMode === "dark"}
            >
              <Moon size={13} />
            </button>
          </div>
          <span className="topbar-divider" />
          <WorkspaceControls graph={graph} projectId={projectId} onProjectId={changeProjectId} onNotice={notice}
            getWorkspaceVersion={() => workspaceVersion.current}
            onSaved={(id, savedGraph, previousName, version) => {
              if (version !== workspaceVersion.current) return false;
              if (savedGraph.name !== previousName && graphRef.current.name === previousName) commit({ ...graphRef.current, name: savedGraph.name });
              changeProjectId(id); markProjectSaved(savedGraph); return true;
            }}
            onNew={() => { workspaceVersion.current++; commit(examples.blank()); changeProjectId(null); select(null); setScope("all"); setExpanded(false); }}
            onOpen={(opened, id, version, originalGraph) => { if (version !== workspaceVersion.current || graphRef.current !== originalGraph) return false; workspaceVersion.current++; commit(opened); changeProjectId(id); markProjectSaved(opened); select(null); setScope(preferredSnapshotScope(opened)); setExpanded(false); setTimeout(fit, 100); return true; }} />
          <div className="example-picker" ref={menu}>
            <button
              className={`quiet-button example-trigger ${exampleMenu ? "active" : ""}`}
              onClick={() => setExampleMenu(!exampleMenu)}
            >
              <LayoutGrid size={14} />
              Templates
              <ChevronDown size={12} />
            </button>
            {exampleMenu && <TemplateMenu graph={graph} onLoad={loadExample} onNotice={notice} onLoadSaved={opened => { workspaceVersion.current++; commit(opened); changeProjectId(null); select(null); setScope(preferredSnapshotScope(opened)); setExampleMenu(false); setExpanded(false); setTimeout(fit, 100); notice("Template loaded. Save as a project to keep your changes."); }} />}
          </div>
          <span className="topbar-divider" />
          <CheckpointControls graph={graph} valid={!!analysis?.valid && !engine.checking} canBuild={canBuild && !engine.busy}
            onSave={saveFile} onPython={() => engine.request("export")} onNotice={notice}
            onImport={imported => { workspaceVersion.current++; commit(imported); changeProjectId(null); select(null); setExpanded(false); setScope(preferredSnapshotScope(imported)); setTimeout(fit, 100); }} />
        </div>
      </header>

      <div className={`workbench ${showLibrary ? "" : "library-hidden"}`}>
        {showLibrary && <Library onAdd={addNode} schematic={schematic} onClose={() => setShowLibrary(false)} />}
        <main className={`main-workspace ${expanded ? "model-expanded" : ""}`}>
          <ResizablePanels
            expanded={expanded}
            architecture={
              <Architecture
                graph={viewGraph}
                fullGraph={graph}
                analysis={analysis}
                theme={effectiveTheme}
                selected={selected}
                onSelect={select}
                expanded={expanded}
                onExpand={() => setExpanded(!expanded)}
                branchScope={schematic ? { value: scope, options: scopes, onChange: value => { setScope(value); select(null); setTimeout(fit, 80); } } : undefined}
              />
            }
          >
            <section className="graph-panel">
              <div className="panel-header">
                <div className="panel-title">
                  <GitBranch size={14} />
                  <span>Model graph</span>
                  <span className="panel-subtitle">
                    {schematic ? graph.metadata?.proposal ? "Untrained proposal" : "Supplied schematic" : "Compose your architecture"}
                  </span>
                </div>
                <div className="panel-tools">
                  {schematic && <select className="snapshot-scope" aria-label="Focus model branch" value={scope} onChange={e => { setScope(e.target.value); select(null); setTimeout(fit, 80); }}>
                    {scopes.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
                  </select>}
                  {schematic && <button className="icon-button" aria-label="Inspect supplied schematic" title="Source descriptions and snapshot claims" onClick={() => { select(null); setShowSnapshotInfo(!showSnapshotInfo); }}><FileSearch size={14} /></button>}
                  <button
                    className={`quiet-button add-block-button ${showLibrary ? "active" : ""}`}
                    onClick={() => setShowLibrary(!showLibrary)}
                    aria-label="Add block"
                    title="Toggle building blocks"
                  >
                    {showLibrary ? (
                      <PanelLeftClose size={14} />
                    ) : (
                      <Plus size={14} />
                    )}
                    Add block
                  </button>
                  <button
                    className="icon-button"
                    disabled={!history.current.past.length}
                    onClick={undo}
                    aria-label="Undo"
                    title="Undo · ⌘Z"
                  >
                    <Undo2 size={14} />
                  </button>
                  <button
                    className="icon-button"
                    disabled={!history.current.future.length}
                    onClick={redo}
                    aria-label="Redo"
                    title="Redo · ⌘⇧Z"
                  >
                    <Redo2 size={14} />
                  </button>
                  <span className="tool-separator" />
                  <button
                    className="quiet-button layout-button"
                    onClick={() => {
                      commit(autoLayout(graph, analysis));
                      setTimeout(fit, 30);
                    }}
                  >
                    <LayoutGrid size={13} />
                    Auto layout
                  </button>
                  <button
                    className="icon-button"
                    onClick={fit}
                    title="Fit graph · F"
                    aria-label="Fit graph"
                  >
                    <Expand size={14} />
                  </button>
                </div>
              </div>
              <div
                className="graph-canvas"
                ref={graphPane}
                onDragOver={(event) => {
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "copy";
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  const kind = event.dataTransfer.getData(
                    "application/vizml",
                  ) as Kind;
                  if (Object.hasOwn(catalog, kind))
                    addNode(
                      kind,
                      flow.screenToFlowPosition({
                        x: event.clientX,
                        y: event.clientY,
                      }),
                    );
                }}
              >
                <AnalysisContext.Provider value={analysis}>
                  <ReactFlow<LayerNodeType>
                    nodes={displayNodes}
                    edges={displayEdges}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onNodeClick={(_, node) => select(node.id)}
                    onEdgeClick={(_, edge) => {
                      setSelectedEdge(edge.id);
                      setSelected(null);
                    }}
                    onPaneClick={() => select(null)}
                    onNodeDragStart={snapshot}
                    onConnect={(connection: Connection) => {
                      if (connection.source && connection.target)
                        connect(connection.source, connection.target);
                    }}
                    isValidConnection={(connection) =>
                      canConnect(
                        graphRef.current,
                        connection.source,
                        connection.target,
                      )
                    }
                    defaultViewport={{ x: 0, y: 0, zoom: 0.7 }}
                    onInit={(instance) => {
                      if (graphRef.current.mode === "schematic") {
                        setTimeout(() => instance.fitView({ padding: 0.2, maxZoom: 0.9 }), 100);
                        return;
                      }
                      const target =
                        graphRef.current.nodes.find((n) => n.id === "conv2") ??
                        graphRef.current.nodes[
                          Math.floor(graphRef.current.nodes.length / 2)
                        ];
                      if (target)
                        setTimeout(
                          () =>
                            instance.setCenter(
                              target.position.x + 150,
                              target.position.y + 120,
                              { zoom: 0.7 },
                            ),
                          80,
                        );
                    }}
                    minZoom={schematic ? 0.025 : 0.15}
                    nodesConnectable
                    maxZoom={1.7}
                    deleteKeyCode={null}
                    panOnScroll
                    colorMode={effectiveTheme}
                    connectionLineStyle={{
                      stroke: "#c1ed84",
                      strokeWidth: 1.5,
                    }}
                  >
                    <Background
                      variant={BackgroundVariant.Dots}
                      gap={20}
                      size={1}
                      color="#354031"
                    />
                    <MiniMap
                      nodeColor={(node) =>
                        catalog[(node.data as LayerNodeType["data"]).kind].color
                      }
                      maskColor="#101510c4"
                      bgColor="#151b14"
                      nodeStrokeWidth={0}
                      pannable
                      zoomable
                      position="bottom-right"
                    />
                  </ReactFlow>
                </AnalysisContext.Provider>
                {!graph.nodes.length && (
                  <div className="graph-empty">
                    <Plus size={23} />
                    <span>Your canvas is open.</span>
                    <small>Choose an input from the building blocks.</small>
                  </div>
                )}
                <div className="graph-zoom">
                  <button
                    onClick={() => flow.zoomOut({ duration: 200 })}
                    aria-label="Zoom out"
                  >
                    <Minus size={14} />
                  </button>
                  <span>ZOOM</span>
                  <button
                    onClick={() => flow.zoomIn({ duration: 200 })}
                    aria-label="Zoom in"
                  >
                    <Plus size={14} />
                  </button>
                </div>
                <span className="graph-instruction">
                  {schematic ? `${viewGraph.nodes.length} of ${graph.nodes.length} nodes · edit declared connections` : "Connect ports to declare dependencies"}
                </span>
              </div>
            </section>
          </ResizablePanels>
        </main>
        <Inspector
          showSnapshotOverview={showSnapshotInfo}
          node={activeNode}
          graph={graph}
          analysis={analysis}
          onUpdate={updateNode}
          onUpdateSchematic={updateSchematicNode}
          onSelect={select}
          onDelete={deleteNode}
          onDuplicate={duplicate}
          onConnect={connect}
          onDisconnect={disconnect}
          onProjectResidual={projectResidual}
          onExpandStage={expandStage}
          onSwapInputs={swapInputs}
        />
      </div>
      <StatusBar
        graph={graph}
        analysis={analysis}
        engineStatus={engine.status}
        checking={engine.checking}
        busy={engine.busy}
        verified={!!engine.compilation}
        canBuild={canBuild}
        saved={saved}
        onIssues={() => setDialog("issues")}
        onVerify={() => engine.request("compile")}
        onHelp={() => setDialog("help")}
      />
      {toast && (
        <div
          className={`toast ${toast.error ? "toast-error" : ""}`}
          role="status"
        >
          {toast.error ? <CircleAlert size={16} /> : <CheckCircle2 size={16} />}
          <span>{toast.message}</span>
          <button onClick={() => setToast(null)} aria-label="Dismiss message">
            <X size={14} />
          </button>
        </div>
      )}
      {dialog && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setDialog(null);
          }}
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dialog-title"
          >
            <button
              className="modal-close icon-button"
              onClick={() => setDialog(null)}
              aria-label="Close dialog"
            >
              <X size={18} />
            </button>
            {dialog === "help" ? (
              <>
                <span className="eyebrow">A SHORT FIELD GUIDE</span>
                <h2 id="dialog-title">From blocks to a model.</h2>
                <p className="modal-intro">
                  The graph is your intention. The 3D view is what it becomes.
                </p>
                <div className="help-items">
                  {helpItems.map(([title, description], index) => (
                    <div key={title}>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <div>
                        <h3>{title}</h3>
                        <p>{description}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="shortcut-row">
                  <span>
                    <kbd>⌘ Z</kbd> Undo
                  </span>
                  <span>
                    <kbd>⌘ S</kbd> Save project
                  </span>
                  <span>
                    <kbd>F</kbd> Fit graph
                  </span>
                  <span>
                    <kbd>⌫</kbd> Delete
                  </span>
                </div>
              </>
            ) : (
              <>
                <span className="eyebrow">ARCHITECTURE CHECK</span>
                <h2 id="dialog-title">
                  {analysis?.valid
                    ? schematic ? "Snapshot structure checked." : "Everything adds up."
                    : "A few things to resolve."}
                </h2>
                <p className="modal-intro">
                  {schematic ? "Declared counts and tensor ownership are checked for consistency. Shapes, operations and forward connections remain source claims." : "Shapes are inferred analytically. No weights or training data are allocated."}
                </p>
                {issues.length ? (
                  <div className="issue-list">
                    {issues.map((issue, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          if (issue.node_id) {
                            select(issue.node_id);
                            setDialog(null);
                          }
                        }}
                      >
                        <CircleAlert size={16} />
                        <span>
                          {issue.node_id && (
                            <strong>
                              {
                                graph.nodes.find((n) => n.id === issue.node_id)
                                  ?.data.label
                              }
                            </strong>
                          )}
                          {issue.message}
                        </span>
                        {issue.node_id && <ArrowUpRight size={14} />}
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="all-good">
                    <CheckCircle2 size={28} />
                    <span>
                      All dependencies and tensor shapes are compatible.
                    </span>
                  </div>
                )}
                {engine.compilation && (
                  <div className="compile-result">
                    <h3>Verified with real PyTorch modules</h3>
                    <p>
                      {engine.compilation.parameters.toLocaleString()}{" "}
                      parameters · meta device · 0 bytes of weight storage
                    </p>
                    {Object.entries(engine.compilation.outputs).map(
                      ([id, shape]) => (
                        <div key={id}>
                          <span>
                            {graph.nodes.find((n) => n.id === id)?.data.label}
                          </span>
                          <code>{shapeText(shape)}</code>
                        </div>
                      ),
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
