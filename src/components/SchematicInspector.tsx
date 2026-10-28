import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { FileSearch, X, ArrowRight, Search } from "lucide-react";
import { catalog, formatNumber, shapeText, type Analysis, type LayerNode, type ModelGraph, type Params } from "../model";
import { formatDeclaredShape, parseDeclaredShape } from "../schematicEditing";

function Detail({ title, value }: { title: string; value: unknown }) {
  if (value == null) return null;
  return <details className="snapshot-detail"><summary>{title}</summary><pre>{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</pre></details>;
}

function ParameterField({ name, value, onChange, disabled = false, validate }: { name: string; value: unknown; onChange: (value: string | number | boolean) => void; disabled?: boolean; validate?: (value: number) => boolean }) {
  const [draft, setDraft] = useState(String(value));
  const [invalid, setInvalid] = useState(false);
  useEffect(() => { setDraft(String(value)); setInvalid(false); }, [value]);
  const label = name.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");
  if (typeof value === "boolean") return <label className="schematic-toggle"><input type="checkbox" checked={value} disabled={disabled} onChange={e => onChange(e.target.checked)} /> <span>{label}</span></label>;
  if (typeof value !== "number" && typeof value !== "string") return null;
  const commit = () => {
    if (typeof value === "number") {
      const number = Number(draft);
      if (!draft.trim() || !Number.isFinite(number) || (validate && !validate(number))) { setInvalid(true); return; }
      setInvalid(false);
      if (number !== value) onChange(number);
    } else {
      setInvalid(false);
      if (draft !== value) onChange(draft);
    }
  };
  return <label className="parameter-field schematic-parameter"><span>{label}</span><input className="text-field" type={typeof value === "number" ? "number" : "text"} step="any" value={draft}
    aria-invalid={invalid} disabled={disabled} onChange={e => { setDraft(e.target.value); setInvalid(false); }} onBlur={commit} onKeyDown={e => { if (e.key === "Enter") e.currentTarget.blur(); }} />
    {invalid && <small role="alert">{validate ? "Enter a nonnegative whole number." : "Enter a valid number."}</small>}</label>;
}

export default function SchematicInspector({ node, graph, analysis, onSelect, overview, onUpdate, onDelete, onDuplicate, onInsertDense, onDisconnect }: {
  node?: LayerNode; graph: ModelGraph; analysis: Analysis | null; onSelect: (id: string | null) => void; overview?: boolean;
  onUpdate: (id: string, patch: Partial<LayerNode["data"]>) => void;
  onDelete: (id: string) => void; onDuplicate: (id: string) => void; onInsertDense?: (id: string) => void; onDisconnect: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = 0; }, [node?.id, graph.name, graph.template]);
  const [shape, setShape] = useState("");
  const [settings, setSettings] = useState("");
  const [shapeError, setShapeError] = useState("");
  const [settingsError, setSettingsError] = useState("");
  const [settingsDirty, setSettingsDirty] = useState(false);
  type Draft = { text: string; baseline: string; error: string };
  const shapeDrafts = useRef(new Map<string, Draft>());
  const settingsDrafts = useRef(new Map<string, Draft>());
  const draftKey = node ? JSON.stringify([graph.name, graph.template ?? "", node.id]) : "";
  const shapeBaseline = JSON.stringify(node?.data.output_shape ?? []);
  const settingsBaseline = JSON.stringify(node?.data.params ?? {});
  useEffect(() => {
    const draft = shapeDrafts.current.get(draftKey);
    setShape(draft?.text ?? formatDeclaredShape(node?.data.output_shape));
    setShapeError(draft?.error ?? "");
  }, [draftKey, shapeBaseline]);
  useEffect(() => {
    const draft = settingsDrafts.current.get(draftKey);
    setSettings(draft?.text ?? JSON.stringify(node?.data.params ?? {}, null, 2));
    setSettingsDirty(!!draft); setSettingsError(draft?.error ?? "");
  }, [draftKey]);
  useEffect(() => {
    if (!settingsDrafts.current.has(draftKey)) setSettings(JSON.stringify(node?.data.params ?? {}, null, 2));
  }, [draftKey, settingsBaseline, settingsDirty]);
  const saveShape = () => {
    if (!node) return;
    const draft = shapeDrafts.current.get(draftKey);
    try {
      if (draft && draft.baseline !== shapeBaseline)
        throw new Error("The saved shape changed while you were editing. Discard the draft or copy it before continuing.");
      const value = parseDeclaredShape(shape);
      shapeDrafts.current.delete(draftKey);
      if (JSON.stringify(value) !== JSON.stringify(node.data.output_shape) || node.data.declaration_pending === true)
        onUpdate(node.id, { output_shape: value, declaration_pending: false });
      setShape(formatDeclaredShape(value)); setShapeError("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Enter a valid shape.";
      shapeDrafts.current.set(draftKey, { text: shape, baseline: draft?.baseline ?? shapeBaseline, error: message });
      setShapeError(message);
    }
  };
  const saveSettings = () => {
    if (!node) return;
    const draft = settingsDrafts.current.get(draftKey);
    try {
      if (draft && draft.baseline !== settingsBaseline)
        throw new Error("Settings changed while you were editing. Discard the draft or copy it before continuing.");
      const value = JSON.parse(settings);
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Operation settings must be a JSON object.");
      settingsDrafts.current.delete(draftKey);
      if (JSON.stringify(value) !== JSON.stringify(node.data.params)) onUpdate(node.id, { params: value });
      setSettingsDirty(false); setSettingsError("");
    } catch (error) {
      const message = error instanceof SyntaxError ? "Operation settings must be a JSON object." : error instanceof Error ? error.message : "Operation settings must be a JSON object.";
      settingsDrafts.current.set(draftKey, { text: settings, baseline: draft?.baseline ?? settingsBaseline, error: message });
      setSettingsError(message);
    }
  };
  const data = node?.data;
  const meta = graph.metadata ?? {};
  const proposal = meta.proposal as { baseline_parameters: number; proposed_parameters: number; parameter_reduction_percent: number; spatial_macs_per_encoder: number; spatial_mac_ratio: number } | undefined;
  const weights = meta.weights as { filename: string; matched: number; expected: number; snapshot_match: boolean | null } | undefined;
  const checkpoint = meta.checkpoint as { generation?: number; cumulative_decisions?: number } | undefined;
  const links = node ? graph.edges.filter(e => e.target === node.id || e.source === node.id) : [];
  const concatInputs = node?.data.kind === "concat" ? links.filter(edge => edge.target === node.id && graph.nodes.some(source => source.id === edge.source)) : [];
  return <aside className={`inspector snapshot-inspector ${node || overview ? "snapshot-selected" : ""}`}>
    <div className="inspector-eyebrow"><FileSearch size={13} /> {proposal ? "ARCHITECTURE PROPOSAL" : "SUPPLIED SCHEMATIC"}
      {(node || overview) && <button className="icon-button" aria-label="Close inspector" onClick={() => onSelect(null)}><X size={14} /></button>}
    </div>
    <div className="inspector-scroll" ref={scrollRef}>
      <div className="snapshot-section snapshot-provenance">
        <span className="snapshot-kicker">{node ? catalog[node.data.kind].title : proposal ? "Untrained · separate actor & value encoders" : "Architecture snapshot"}</span>
        <h3>{node?.data.label ?? (proposal ? "Catan · ConvNeXt Small" : "The model, as described")}</h3>
        <p>{node ? String(data?.description ?? "No description supplied.") : proposal ? "32 → 48 → 64 channels. Five residual blocks per encoder. Full-resolution entity detail joins a coarse map with whole-board reach." : "Shapes, operations and connections come from the supplied JSON. You can inspect every claim here; this diagram does not execute the original forward pass."}</p>
        {!node && proposal && <>
          <div className="proposal-budget">
            <div><span>Original → proposal</span><strong>{(proposal.baseline_parameters / 1e6).toFixed(2)}M → {(proposal.proposed_parameters / 1e6).toFixed(2)}M</strong><small>{proposal.parameter_reduction_percent.toFixed(1)}% fewer parameters</small></div>
            <div><span>Spatial MACs / encoder</span><strong>{formatNumber(proposal.spatial_macs_per_encoder)}</strong><small>{proposal.spatial_mac_ratio.toFixed(1)}× fewer · arithmetic estimate</small></div>
            <div><span>Final nominal receptive field</span><strong>97 × 97</strong><small>Original board · 45 × 49</small></div>
          </div>
          <p className="proposal-scope-note">CNN encoder implemented and tested in PyTorch. Game-specific heads and PPG phase wiring retain the supplied integration contract. This model has no trained weights.</p>
          {graph.template === "catan_convnext_small" && <a className="snapshot-link" href="/api/templates/catan_convnext_small/encoder.py" download="catan_convnext_encoder.py">Download PyTorch encoder<ArrowRight size={12} /></a>}
        </>}
        {meta.source_edited === true && <p className="proposal-scope-note">Edited schematic: original source statistics and provenance describe the original snapshot. Review declared shapes and parameter counts after changes.</p>}
        {!node && checkpoint && <p>Snapshot generation {checkpoint.generation} · {checkpoint.cumulative_decisions?.toLocaleString()} decisions</p>}
        {weights && <div className={`checkpoint-note ${weights.snapshot_match === false ? "revision-mismatch" : ""}`}>
          <strong>{weights.matched}/{weights.expected} tensor names & shapes match</strong>
          <span>{weights.filename}{weights.snapshot_match === false ? ": file hash differs from the JSON snapshot. Supplied statistics and training details describe that snapshot." : ". Forward connections remain supplied claims."}</span>
        </div>}
      </div>
      {node && data && <>
        <div className="snapshot-section schematic-edit">
          <h4>Edit declaration</h4>
          <label className="field-label" htmlFor="schematic-name">Name</label>
          <input id="schematic-name" className="text-field" maxLength={100} value={node.data.label} onChange={e => onUpdate(node.id, { label: e.target.value || "Layer" })} />
          <label className="field-label" htmlFor="schematic-shape">Declared output shape</label>
          <input id="schematic-shape" className="text-field" value={shape} placeholder="B × L × 64" aria-invalid={!!shapeError} onChange={e => { const text = e.target.value; const draft = shapeDrafts.current.get(draftKey); shapeDrafts.current.set(draftKey, { text, baseline: draft?.baseline ?? shapeBaseline, error: "" }); setShape(text); setShapeError(""); }} onBlur={saveShape} onKeyDown={e => { if (e.key === "Enter") e.currentTarget.blur(); }} />
          <small className="schematic-field-hint">Separate dimensions with × or commas. Symbols such as B and L are allowed.</small>
          {shapeError && <><p className="schematic-error" role="alert">{shapeError}</p><button className="quiet-button" onClick={() => { shapeDrafts.current.delete(draftKey); setShape(formatDeclaredShape(node.data.output_shape)); setShapeError(""); }}>Discard shape draft</button></>}
          <ParameterField key={`${node.id}:parameter-count`} name="Declared parameter count" value={Number(data.parameter_count ?? 0)}
            validate={value => Number.isSafeInteger(value) && value >= 0} onChange={value => onUpdate(node.id, { parameter_count: value })} />
          <label className="field-label" htmlFor="schematic-description">Description</label>
          <textarea id="schematic-description" className="text-field" rows={3} value={typeof data.description === "string" ? data.description : ""} onChange={e => onUpdate(node.id, { description: e.target.value })} />
          <div className="schematic-settings-heading">Operation settings</div>
          {settingsDirty && <p className="context-note" role="status">An advanced settings draft is waiting below. Apply or discard it to edit individual fields.</p>}
          {Object.entries(data.params).map(([key, value]) => <ParameterField key={`${node.id}:${key}`} name={key} value={value} disabled={settingsDirty} onChange={value => onUpdate(node.id, { params: { ...node.data.params, [key]: value } as Params })} />)}
          <details className="snapshot-detail schematic-advanced"><summary>Advanced settings · JSON</summary>
            <label className="field-label" htmlFor="schematic-settings">All operation settings</label>
            <textarea id="schematic-settings" className="text-field" rows={5} value={settings} onChange={e => { const text = e.target.value; const draft = settingsDrafts.current.get(draftKey); settingsDrafts.current.set(draftKey, { text, baseline: draft?.baseline ?? settingsBaseline, error: "" }); setSettings(text); setSettingsDirty(true); setSettingsError(""); }} />
            {settingsDirty && <div className="schematic-actions"><button className="quiet-button" onClick={() => { settingsDrafts.current.delete(draftKey); setSettings(JSON.stringify(node.data.params, null, 2)); setSettingsDirty(false); setSettingsError(""); }}>Discard changes</button><button className="quiet-button" onClick={saveSettings}>Apply settings</button></div>}
            {settingsError && <p className="schematic-error" role="alert">{settingsError}</p>}
            <small className="schematic-field-hint">Drafts stay here when you switch blocks, but are not saved to the project until applied. Apply or discard before changing individual fields.</small>
          </details>
          {(analysis?.errors ?? []).filter(issue => issue.node_id === node.id).map((issue, index) => <p className="schematic-error" role="alert" key={index}>{issue.message}</p>)}
          {data.declaration_pending === true && <p className="context-note">Placeholder shape and count: review and save the declared shape.</p>}
          <p className="context-note">These are editable declarations. The original operation is not executed or verified.</p>
          {data?.kind === "dense" && onInsertDense && <div className="snapshot-section"><button className="quiet-button" onClick={() => onInsertDense(node.id)}>Insert dense layer after</button><p>Existing outgoing connections pass through the new layer. Review its declarations.</p></div>}
          <div className="schematic-actions"><button className="quiet-button" onClick={() => onDuplicate(node.id)}>Duplicate</button><button className="quiet-button" onClick={() => onDelete(node.id)}>Delete block</button></div>
        </div>
        <div className="snapshot-section snapshot-metrics"><span>{proposal ? "Output shape" : "Declared output"}<strong>{shapeText(analysis?.nodes[node.id]?.display_shape ?? data.output_shape as (number | string)[])}</strong></span><span>Owned parameters<strong>{Number(data.parameter_count).toLocaleString()}</strong></span></div>
        <div className="snapshot-section">
          <Detail title="Pre-operations bundled into this node" value={data.pre_operations} />
          <Detail title="Module paths" value={data.module_paths} />
          <Detail title="Owned parameter tensors" value={data.parameter_tensors} />
          <Detail title="Weight statistics · supplied snapshot" value={data.weight_statistics} />
          <Detail title="CPU example · supplied claim" value={data.verified_cpu_example} />
          {Object.entries(data).filter(([key]) => !["kind", "label", "params", "description", "output_shape", "parameter_count", "module_paths", "parameter_tensors", "weight_statistics", "verified_cpu_example", "pre_operations"].includes(key)).map(([key, value]) => <Detail key={key} title={key.replaceAll("_", " ")} value={value} />)}
        </div>
        <div className="snapshot-section"><h4>Connections</h4>{links.map(edge => {
          const incoming = edge.target === node.id;
          const other = graph.nodes.find(n => n.id === (incoming ? edge.source : edge.target));
          const inputIndex = concatInputs.indexOf(edge);
          return <div className="schematic-connection" key={edge.id}><button className="snapshot-link" onClick={() => other && onSelect(other.id)}><span>{inputIndex >= 0 ? `${inputIndex + 1} · ` : ""}{incoming ? "From" : "To"} · {other?.data.label}<small>{String(edge.label ?? edge.data?.role ?? "dependency")}</small></span><ArrowRight size={12} /></button><button className="icon-button" aria-label={`Remove connection ${other?.data.label ?? ""}`} onClick={() => onDisconnect(edge.id)}><X size={12} /></button></div>;
        })}</div>
      </>}
      {!node && <>
        <div className="snapshot-section">
          <Detail title="What the source says matters" value={meta.important_details} />
          <Detail title={proposal ? "Parameter budget" : "Parameter breakdown · declared"} value={meta.parameter_counts} />
          <Detail title="Inputs and receptive field" value={{ inputs: meta.input_dimensions, receptive_field: meta.receptive_field }} />
          <Detail title={proposal ? "Encoder checks · scope and shapes" : "Validation claims from the source"} value={meta.validation} />
          <Detail title="Checkpoint provenance · supplied" value={meta.checkpoint} />
          <Detail title="Format notes" value={meta.format_notes} />
        </div>
        <div className="snapshot-section"><label className="snapshot-search"><Search size={12} /><input aria-label="Find a snapshot node" placeholder="Find a node…" value={search} onChange={e => setSearch(e.target.value)} /></label>
          {graph.nodes.filter(n => `${n.id} ${n.data.label} ${n.data.module_paths}`.toLowerCase().includes(search.toLowerCase())).map(n => <button className="snapshot-link" key={n.id} onClick={() => onSelect(n.id)}><span>{n.data.label}<small>{n.id}</small></span><ArrowRight size={12} /></button>)}
        </div>
      </>}
    </div>
  </aside>;
}
