import { useEffect, useRef, useState } from "react";
import { ArrowDownToLine, ArrowUpFromLine, ChevronDown, FileSearch, Loader2, X } from "lucide-react";
import { bytesText, cleanGraph, parseGraph, type ModelGraph } from "../model";

type TensorInfo = { name: string; shape: number[]; dtype: string; bytes: number; meta: boolean };
type Checkpoint = { id: string; filename: string; sha256: string; tensor_count: number; tensor_elements: number; tensor_bytes: number; file_bytes: number; has_weights: boolean; has_optimizer: boolean; architecture?: string; decisions?: number; graph?: ModelGraph; tensors: TensorInfo[]; download_url?: string; weights_origin?: string };
type Props = { graph: ModelGraph; valid: boolean; canBuild: boolean; onImport: (graph: ModelGraph) => void; onSave: () => void; onPython: () => void; onNotice: (message: string, error?: boolean) => void };
async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const value = await response.json();
  if (!response.ok) throw Error(typeof value.detail === "string" ? value.detail : "The checkpoint request could not be completed.");
  return value;
}
const jsonBody = (value: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });

export default function CheckpointControls({ graph, valid, canBuild, onImport, onSave, onPython, onNotice }: Props) {
  const file = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [info, setInfo] = useState<Checkpoint | null>(null);
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [pairError, setPairError] = useState("");
  useEffect(() => {
    const close = (event: KeyboardEvent) => { if (event.key === "Escape") { setMenu(false); setOpen(false); } };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  const weights = graph.metadata?.weights as { id: string; filename: string } | undefined;
  const bind = async (checkpoint: Checkpoint, target: ModelGraph) => api<{ graph: ModelGraph }>(`/api/checkpoints/${checkpoint.id}/bind`, jsonBody(cleanGraph(target)));
  const importFiles = async (files: File[]) => {
    setBusy(true); setPairError("");
    try {
      const jsons = files.filter(f => f.name.toLowerCase().endsWith(".json"));
      const pts = files.filter(f => /\.(pt|pth)$/i.test(f.name));
      if (jsons.length > 1 || pts.length > 1 || jsons.length + pts.length !== files.length) throw Error("Choose one architecture JSON and/or one .pt or .pth checkpoint.");
      if (jsons[0]?.size > 2_000_000) throw Error("Choose an architecture JSON smaller than 2 MB.");
      let target = jsons[0] ? parseGraph(await jsons[0].text()) : graph;
      let checkpoint = pts[0] ? null : (jsons.length && info && !info.graph ? info : null);
      if (pts[0]) {
        if (pts[0].size > 2 * 1024 ** 3) throw Error("Checkpoint imports are limited to 2 GiB.");
        onNotice("Reading checkpoint tensor metadata…");
        checkpoint = await api<Checkpoint>(`/api/checkpoints/import?filename=${encodeURIComponent(pts[0].name)}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pts[0] });
        setInfo(checkpoint);
        if (!jsons.length && checkpoint.graph) target = parseGraph(JSON.stringify(checkpoint.graph));
      }
      if (checkpoint?.has_weights) {
        try {
          const bound = await bind(checkpoint, target);
          target = parseGraph(JSON.stringify(bound.graph));
          onImport(target); setOpen(false);
          onNotice("Checkpoint paired by tensor names and shapes. Imported weights are preserved.");
        } catch (error) {
          if (jsons.length || checkpoint.graph) onImport(target);
          setPairError(error instanceof Error ? error.message : "The checkpoint does not match this graph.");
          setOpen(true);
          onNotice("Checkpoint inspected. A matching architecture is needed to show its forward connections.");
        }
      } else if (jsons.length || checkpoint?.graph) {
        onImport(target); setOpen(false);
        onNotice(target.mode === "schematic" ? "Supplied schematic loaded. Shapes and forward connections are source claims." : "Architecture imported.");
      } else if (checkpoint) {
        setOpen(true); onNotice("Tensor metadata loaded. This file contains no numerical weights.");
      }
    } catch (error) { onNotice(error instanceof Error ? error.message : "Could not import this file.", true); }
    finally { setBusy(false); }
  };
  const exportFile = async (mode: "architecture" | "checkpoint" | "state_dict") => {
    setMenu(false); setBusy(true);
    try {
      const result = await api<Checkpoint>("/api/checkpoints/export", jsonBody({ graph: cleanGraph(graph), mode }));
      const anchor = document.createElement("a"); anchor.href = result.download_url!; anchor.download = result.filename; anchor.click();
      onNotice(mode === "architecture" ? "Architecture exported as .pt without allocating weights." : result.weights_origin === "initialized" ? "Untrained, freshly initialized weights exported." : "Original checkpoint weights exported.");
    } catch (error) { onNotice(error instanceof Error ? error.message : "Export failed.", true); }
    finally { setBusy(false); }
  };
  const inspect = async () => {
    try { if (weights) setInfo(await api<Checkpoint>(`/api/checkpoints/${weights.id}`)); setPairError(""); setOpen(true); }
    catch (error) { onNotice((error as Error).message, true); }
  };
  return <>
    <button className="icon-button file-action" disabled={busy} onClick={() => file.current?.click()} aria-label="Import architecture or checkpoint" title="Import JSON, .pt, or both together"><ArrowUpFromLine size={16} /></button>
    <button className="icon-button file-action" onClick={onSave} aria-label="Save graph" title="Save graph JSON · ⌘S"><ArrowDownToLine size={16} /></button>
    {(weights || info) && <button className="icon-button file-action" disabled={busy} onClick={inspect} aria-label="Inspect checkpoint" title={weights?.filename ?? info?.filename}><FileSearch size={16} /></button>}
    <div className="checkpoint-export">
      <button className="export-button" disabled={busy} onClick={() => setMenu(!menu)} aria-expanded={menu} aria-label="Export model">{busy ? <Loader2 size={14} className="spin" /> : <ArrowDownToLine size={14} />} {busy ? "Working…" : "Export"}<ChevronDown size={12} /></button>
      {menu && <><button className="menu-dismiss" aria-label="Close export menu" onClick={() => setMenu(false)} /><div className="checkpoint-menu">
        <button onClick={() => { setMenu(false); onSave(); }}>Graph JSON<small>Portable diagram and annotations</small></button>
        <button disabled={!valid} onClick={() => exportFile("architecture")}>.pt architecture<small>Graph only · no weight allocation</small></button>
        <button disabled={!valid || (!weights && !canBuild)} onClick={() => exportFile("checkpoint")}>.pt checkpoint<small>{weights ? "Original weights + graph + training payload" : "Fresh, untrained weights + graph · up to 512 MiB"}</small></button>
        <button disabled={!valid || (!weights && !canBuild)} onClick={() => exportFile("state_dict")}>.pt state_dict<small>{weights ? "Original tensors with native names" : "Fresh, untrained tensors · up to 512 MiB"}</small></button>
        <button disabled={!canBuild} onClick={() => { setMenu(false); onPython(); }}>PyTorch source<small>Standalone executable Python</small></button>
      </div></>}
    </div>
    <input ref={file} type="file" hidden multiple accept=".json,.pt,.pth" onChange={event => { const files = Array.from(event.currentTarget.files ?? []); event.currentTarget.value = ""; if (files.length) void importFiles(files); }} />
    {open && info && <div className="modal-backdrop" onClick={() => setOpen(false)}><section className="checkpoint-dialog" role="dialog" aria-label="Checkpoint tensors" aria-modal="true" onClick={e => e.stopPropagation()}>
      <div className="checkpoint-dialog-heading"><div><span className="snapshot-kicker">LOCAL CHECKPOINT</span><h2>{info.filename}</h2></div><button className="icon-button" aria-label="Close checkpoint" onClick={() => setOpen(false)}><X size={16} /></button></div>
      <p>{info.tensor_count.toLocaleString()} tensors · {info.tensor_elements.toLocaleString()} stored elements · {bytesText(info.tensor_bytes)}{info.has_optimizer ? " · training payload retained" : ""}</p>
      {info.architecture && <p>{info.architecture}{info.decisions ? ` · ${info.decisions.toLocaleString()} decisions` : ""}</p>}
      <p className="checkpoint-explanation">Tensor names and shapes can be checked against a diagram. A state_dict alone does not record the forward connections. Numerical weights are retained for export; the 3D view shows architecture.</p>
      {pairError && <div className="checkpoint-note revision-mismatch"><strong>Architecture pairing</strong><span>{pairError}</span><span>Import a matching JSON diagram, or choose the JSON and .pt together.</span></div>}
      <details className="snapshot-detail"><summary>File SHA-256</summary><pre>{info.sha256}</pre></details>
      <label className="snapshot-search"><input autoFocus aria-label="Search checkpoint tensors" placeholder="Filter tensors by name…" value={search} onChange={e => setSearch(e.target.value)} /></label>
      <div className="tensor-table"><table><thead><tr><th>Tensor</th><th>Shape</th><th>Type</th></tr></thead><tbody>{info.tensors.filter(t => t.name.toLowerCase().includes(search.toLowerCase())).map(t => <tr key={t.name}><td>{t.name}</td><td>{t.shape.join(" × ") || "scalar"}</td><td>{t.dtype}{t.meta ? " · meta" : ""}</td></tr>)}</tbody></table></div>
    </section></div>}
  </>;
}
