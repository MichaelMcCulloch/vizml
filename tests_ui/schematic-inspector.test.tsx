import { afterAll, afterEach, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";

GlobalRegistrator.register();
const React = await import("react");
const { render, fireEvent, screen, cleanup } = await import("@testing-library/react");
const { default: SchematicInspector } = await import("../src/components/SchematicInspector");
const { parseGraph } = await import("../src/model");

afterEach(cleanup);
afterAll(() => GlobalRegistrator.unregister());

function fixture() {
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Schematic", mode: "schematic", nodes: [
    { id: "node", position: { x: 0, y: 0 }, data: { kind: "dense", label: "Projection", params: { units: 32, bias: true, layout: [1, 2] }, output_shape: ["B", "L", 32], parameter_count: 0, declaration_pending: true } },
  ], edges: [] }));
  const updates: any[] = [];
  const noop = () => {};
  function Editor() {
    const [node, setNode] = React.useState(graph.nodes[0]);
    return <SchematicInspector graph={graph} node={node} analysis={null} onSelect={noop} onDelete={noop} onDuplicate={noop} onDisconnect={noop}
      onUpdate={(_id, patch) => { updates.push(patch); setNode(current => ({ ...current, data: { ...current.data, ...patch } })); }} />;
  }
  render(<Editor />);
  return updates;
}

test("shape edits accept familiar notation and preserve invalid input for correction", () => {
  const updates = fixture();
  const shape = screen.getByLabelText("Declared output shape") as HTMLInputElement;
  expect(shape.value).toBe("B × L × 32");
  fireEvent.change(shape, { target: { value: "B, L, 64" } });
  fireEvent.blur(shape);
  expect(updates.at(-1)).toEqual({ output_shape: ["B", "L", 64], declaration_pending: false });
  fireEvent.change(shape, { target: { value: "B × -3" } });
  fireEvent.blur(shape);
  expect(shape.value).toBe("B × -3");
  expect(screen.getByRole("alert").textContent).toContain("nonnegative");
  expect(updates).toHaveLength(1);
});

test("simple settings preserve nested values; advanced drafts must be applied or discarded", () => {
  const updates = fixture();
  const units = screen.getByLabelText("units") as HTMLInputElement;
  const bias = screen.getByLabelText("bias") as HTMLInputElement;
  fireEvent.change(units, { target: { value: "64" } });
  fireEvent.blur(units);
  expect(updates.at(-1).params).toEqual({ units: 64, bias: true, layout: [1, 2] });
  fireEvent.click(bias);
  expect(updates.at(-1).params).toEqual({ units: 64, bias: false, layout: [1, 2] });
  const json = screen.getByLabelText("All operation settings") as HTMLTextAreaElement;
  fireEvent.change(json, { target: { value: "{" } });
  expect(units.disabled).toBe(true);
  expect(bias.disabled).toBe(true);
  fireEvent.click(screen.getByText("Apply settings"));
  expect(screen.getByRole("alert").textContent).toContain("JSON object");
  expect(json.value).toBe("{");
  fireEvent.click(screen.getByText("Discard changes"));
  expect(units.disabled).toBe(false);
  expect(JSON.parse(json.value)).toEqual({ units: 64, bias: false, layout: [1, 2] });
  fireEvent.change(json, { target: { value: '{"units": 96, "bias": false, "layout": [3, 4]}' } });
  fireEvent.click(screen.getByText("Apply settings"));
  expect(updates.at(-1).params).toEqual({ units: 96, bias: false, layout: [3, 4] });
  expect(units.disabled).toBe(false);
});

test("unfinished declarations survive switching blocks and do not overwrite changed settings", () => {
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Draft test", mode: "schematic", nodes: [
    { id: "a", position: { x: 0, y: 0 }, data: { kind: "dense", label: "A", params: { units: 32 }, output_shape: ["B", 32], parameter_count: 0 } },
    { id: "b", position: { x: 200, y: 0 }, data: { kind: "dense", label: "B", params: { units: 16 }, output_shape: ["B", 16], parameter_count: 0 } },
  ], edges: [] }));
  const updates: any[] = [];
  const noop = () => {};
  function Editor() {
    const [nodes, setNodes] = React.useState(graph.nodes);
    const [selected, setSelected] = React.useState("a");
    const current = nodes.find(n => n.id === selected)!;
    return <><button onClick={() => setSelected(selected === "a" ? "b" : "a")}>Switch block</button>
      <button onClick={() => setNodes(old => old.map(n => n.id === "a" ? { ...n, data: { ...n.data, params: { units: 48 } } } : n))}>Change outside editor</button>
      <SchematicInspector graph={graph} node={current} analysis={null} onSelect={noop} onDelete={noop} onDuplicate={noop} onDisconnect={noop}
        onUpdate={(id, patch) => { updates.push(patch); setNodes(old => old.map(n => n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)); }} /></>;
  }
  render(<Editor />);
  const shape = screen.getByLabelText("Declared output shape") as HTMLInputElement;
  fireEvent.change(shape, { target: { value: "B × -5" } });
  fireEvent.blur(shape);
  const settings = screen.getByLabelText("All operation settings") as HTMLTextAreaElement;
  fireEvent.change(settings, { target: { value: '{"units": 64}' } });
  fireEvent.click(screen.getByText("Switch block"));
  fireEvent.click(screen.getByText("Switch block"));
  expect((screen.getByLabelText("Declared output shape") as HTMLInputElement).value).toBe("B × -5");
  expect((screen.getByLabelText("All operation settings") as HTMLTextAreaElement).value).toBe('{"units": 64}');
  fireEvent.click(screen.getByText("Change outside editor"));
  fireEvent.click(screen.getByText("Apply settings"));
  expect(screen.getAllByRole("alert").some(element => element.textContent?.includes("Settings changed"))).toBe(true);
  expect(updates).toHaveLength(0);
  fireEvent.click(screen.getByText("Discard changes"));
  expect(JSON.parse((screen.getByLabelText("All operation settings") as HTMLTextAreaElement).value)).toEqual({ units: 48 });
});

test("parameter counts can be replaced and reject invalid declarations", () => {
  const updates = fixture();
  const count = screen.getByLabelText("Declared parameter count") as HTMLInputElement;
  fireEvent.change(count, { target: { value: "" } });
  expect(count.value).toBe("");
  fireEvent.change(count, { target: { value: "-1" } });
  fireEvent.blur(count);
  expect(screen.getByRole("alert").textContent).toContain("nonnegative whole");
  expect(updates).toHaveLength(0);
  fireEvent.change(count, { target: { value: "1024" } });
  fireEvent.blur(count);
  expect(updates.at(-1)).toEqual({ parameter_count: 1024 });
});

test("expanded architecture exposes branch navigation and reports changes", async () => {
  const { default: Architecture } = await import("../src/components/Architecture");
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Schematic", mode: "schematic", nodes: [], edges: [] }));
  const changes: string[] = [];
  const noop = () => {};
  const props = { graph, analysis: null, theme: "light" as const, selected: null, onSelect: noop, onExpand: noop,
    branchScope: { value: "all", options: [{id: "all", label: "Entire model"}, {id: "policy", label: "Policy"}], onChange: (value: string) => changes.push(value) } };
  const view = render(<Architecture {...props} expanded={true} />);
  fireEvent.change(screen.getByLabelText("Focus model branch"), { target: { value: "policy" } });
  expect(changes).toEqual(["policy"]);
  view.rerender(<Architecture {...props} selected="node" expanded={true} />);
  expect(screen.getByRole("button", {name: "Focus selected layer"})).toBeTruthy();
  view.rerender(<Architecture {...props} expanded={false} />);
  expect(screen.queryByLabelText("Focus model branch")).toBeNull();
  expect(screen.queryByRole("button", {name: "Focus selected layer"})).toBeNull();
});

test("project save state recovers after reload and ignores stale requests", async () => {
  const { renderHook, act, waitFor } = await import("@testing-library/react");
  const { useProjectBaseline } = await import("../src/useProjectBaseline");
  const { cleanGraph } = await import("../src/model");
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Saved", nodes: [], edges: [] }));
  let resolveFirst!: (value: {graph: typeof graph}) => void;
  const load = (_id: string) => new Promise<{graph: typeof graph}>(resolve => { resolveFirst = resolve; });
  const hook = renderHook(() => useProjectBaseline("project", load));
  expect(hook.result.current.status).toBe("checking");
  await act(async () => { resolveFirst({graph}); });
  await waitFor(() => expect(hook.result.current.baseline).toBe(JSON.stringify(cleanGraph(graph))));
  expect(hook.result.current.status).toBe("ready");

  hook.unmount();
  const second = renderHook(() => useProjectBaseline("project", load));
  const newlySaved = { ...graph, name: "Newer save" };
  act(() => second.result.current.saved(newlySaved));
  await act(async () => { resolveFirst({graph}); });
  expect(second.result.current.baseline).toBe(JSON.stringify(cleanGraph(newlySaved)));
});

test("unavailable saved projects report an unknown state", async () => {
  const { renderHook, waitFor } = await import("@testing-library/react");
  const { useProjectBaseline } = await import("../src/useProjectBaseline");
  const load = async () => { throw new Error("offline"); };
  const hook = renderHook(() => useProjectBaseline("project", load));
  await waitFor(() => expect(hook.result.current.status).toBe("unavailable"));
  expect(hook.result.current.baseline).toBeNull();
});

test("a delayed save reports the workspace it started in", async () => {
  const { default: WorkspaceControls } = await import("../src/components/WorkspaceControls");
  const { waitFor } = await import("@testing-library/react");
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Original", nodes: [], edges: [] }));
  const next = { ...graph, name: "Other workspace" };
  const originalFetch = globalThis.fetch;
  let finish!: (response: Response) => void;
  globalThis.fetch = (async (_url: any, init?: RequestInit) => {
    if (init?.method === "POST") return new Promise<Response>(resolve => { finish = resolve; });
    return new Response(JSON.stringify({ items: [] }), { status: 200 });
  }) as typeof fetch;
  let version = 1;
  const notices: string[] = [];
  const saves: Array<{version: number; name: string; previous: string}> = [];
  const noop = () => {};
  const props = { projectId: null, onProjectId: noop, onNew: noop, getWorkspaceVersion: () => version,
    onNotice: (message: string) => notices.push(message), onOpen: () => true,
    onSaved: (_id: string, saved: typeof graph, previous: string, started: number) => {
      saves.push({version: started, name: saved.name, previous}); return started === version;
    } };
  try {
    const view = render(<WorkspaceControls {...props} graph={graph} />);
    fireEvent.click(screen.getByRole("button", {name: "Projects"}));
    fireEvent.click(screen.getByText("Save project"));
    fireEvent.change(screen.getByLabelText("Project name"), {target: {value: "Saved copy"}});
    fireEvent.click(screen.getByRole("button", {name: "Save project"}));
    version = 2;
    view.rerender(<WorkspaceControls {...props} graph={next} />);
    finish(new Response(JSON.stringify({id:"saved-id",name:"Saved copy",graph}), {status:200}));
    await waitFor(() => expect(notices.some(message => message.includes("current workspace was left unchanged"))).toBe(true));
    expect(saves).toEqual([{version:1, name:"Saved copy", previous:"Original"}]);
  } finally { globalThis.fetch = originalFetch; }
});

test("a delayed template open does not replace a changed workspace", async () => {
  const { default: TemplateMenu } = await import("../src/components/TemplateMenu");
  const { waitFor } = await import("@testing-library/react");
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Original", nodes: [], edges: [] }));
  const originalFetch = globalThis.fetch;
  let finish!: (response: Response) => void;
  globalThis.fetch = (async (url: any) => String(url).endsWith("/templates")
    ? new Response(JSON.stringify({ items: [{ id: "t", name: "Saved", updated_at: "2026-01-01" }] }), { status: 200 })
    : new Promise<Response>(resolve => { finish = resolve; })) as typeof fetch;
  let version = 1;
  const notices: string[] = [];
  const attempts: number[] = [];
  const props = { onLoad: () => {}, getWorkspaceVersion: () => version,
    onNotice: (message: string) => notices.push(message),
    onLoadSaved: (_opened: typeof graph, original: typeof graph, started: number) => {
      attempts.push(started); return started === version && original.name === "Other";
    } };
  try {
    const view = render(<TemplateMenu {...props} graph={graph} />);
    await waitFor(() => expect(screen.getByText("Saved template")).toBeTruthy());
    fireEvent.click(screen.getByText("Saved template"));
    version = 2;
    view.rerender(<TemplateMenu {...props} graph={{ ...graph, name: "Other" }} />);
    finish(new Response(JSON.stringify({ id: "t", name: "Saved", graph }), { status: 200 }));
    await waitFor(() => expect(notices.some(message => message.includes("workspace changed"))).toBe(true));
    expect(attempts).toEqual([1]);
  } finally { globalThis.fetch = originalFetch; }
});

test("dismissing the template menu cancels a pending open", async () => {
  const { default: TemplateMenu } = await import("../src/components/TemplateMenu");
  const { waitFor, act } = await import("@testing-library/react");
  const graph = parseGraph(JSON.stringify({ version: 1, name: "Original", nodes: [], edges: [] }));
  const originalFetch = globalThis.fetch;
  let finish!: (response: Response) => void;
  globalThis.fetch = (async (url: any) => String(url).endsWith("/templates")
    ? new Response(JSON.stringify({ items: [{ id: "t", name: "Saved", updated_at: "2026-01-01" }] }), { status: 200 })
    : new Promise<Response>(resolve => { finish = resolve; })) as typeof fetch;
  let opened = false;
  try {
    const view = render(<TemplateMenu graph={graph} onLoad={() => {}} getWorkspaceVersion={() => 1}
      onNotice={() => {}} onLoadSaved={() => { opened = true; return true; }} />);
    await waitFor(() => expect(screen.getByText("Saved template")).toBeTruthy());
    fireEvent.click(screen.getByText("Saved template"));
    view.unmount();
    await act(async () => { finish(new Response(JSON.stringify({ id: "t", name: "Saved", graph }), { status: 200 })); });
    expect(opened).toBe(false);
  } finally { globalThis.fetch = originalFetch; }
});
