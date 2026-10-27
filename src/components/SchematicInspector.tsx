import { useEffect, useState } from "react";
import { FileSearch, X, ArrowRight, Search } from "lucide-react";
import { catalog, formatNumber, shapeText, type Analysis, type LayerNode, type ModelGraph } from "../model";

function Detail({ title, value }: { title: string; value: unknown }) {
  if (value == null) return null;
  return <details className="snapshot-detail"><summary>{title}</summary><pre>{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</pre></details>;
}

export default function SchematicInspector({ node, graph, analysis, onSelect, overview, onUpdate, onDelete, onDuplicate, onDisconnect }: {
  node?: LayerNode; graph: ModelGraph; analysis: Analysis | null; onSelect: (id: string | null) => void; overview?: boolean;
  onUpdate: (id: string, patch: Partial<LayerNode["data"]>) => void;
  onDelete: (id: string) => void; onDuplicate: (id: string) => void; onDisconnect: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [shape, setShape] = useState("");
  const [settings, setSettings] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    setShape(JSON.stringify(node?.data.output_shape ?? []));
    setSettings(JSON.stringify(node?.data.params ?? {}, null, 2));
    setError("");
  }, [node?.id, node?.data.output_shape, node?.data.params]);
  const saveShape = () => {
    if (!node) return;
    try {
      const value = JSON.parse(shape);
      if (!Array.isArray(value) || !value.length || value.length > 6 || value.some(x => !(typeof x === "string" && x.length > 0 && x.length <= 40) && !(Number.isSafeInteger(x) && x >= 0))) throw Error();
      if (JSON.stringify(value) !== JSON.stringify(node.data.output_shape) || node.data.declaration_pending === true)
        onUpdate(node.id, { output_shape: value, declaration_pending: false });
      setError("");
    } catch { setError('Use a JSON shape such as ["B", 32] or ["B", "L", 64].'); }
  };
  const saveSettings = () => {
    if (!node) return;
    try {
      const value = JSON.parse(settings);
      if (!value || typeof value !== "object" || Array.isArray(value)) throw Error();
      if (JSON.stringify(value) !== JSON.stringify(node.data.params)) onUpdate(node.id, { params: value });
      setError("");
    } catch { setError("Operation settings must be a JSON object."); }
  };
  const data = node?.data;
  const meta = graph.metadata ?? {};
  const proposal = meta.proposal as { baseline_parameters: number; proposed_parameters: number; parameter_reduction_percent: number; spatial_macs_per_encoder: number; spatial_mac_ratio: number } | undefined;
  const weights = meta.weights as { filename: string; matched: number; expected: number; snapshot_match: boolean | null } | undefined;
  const checkpoint = meta.checkpoint as { generation?: number; cumulative_decisions?: number } | undefined;
  const links = node ? graph.edges.filter(e => e.target === node.id || e.source === node.id) : [];
  return <aside className={`inspector snapshot-inspector ${node || overview ? "snapshot-selected" : ""}`}>
    <div className="inspector-eyebrow"><FileSearch size={13} /> {proposal ? "ARCHITECTURE PROPOSAL" : "SUPPLIED SCHEMATIC"}
      {(node || overview) && <button className="icon-button" aria-label="Close inspector" onClick={() => onSelect(null)}><X size={14} /></button>}
    </div>
    <div className="inspector-scroll">
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
          <label className="field-label" htmlFor="schematic-shape">Declared output shape · JSON</label>
          <input id="schematic-shape" className="text-field" value={shape} onChange={e => setShape(e.target.value)} onBlur={saveShape} />
          <label className="field-label" htmlFor="schematic-parameters">Declared parameter count</label>
          <input id="schematic-parameters" className="text-field" type="number" min={0} max={Number.MAX_SAFE_INTEGER} value={Number(data.parameter_count ?? 0)} onChange={e => { if (Number.isSafeInteger(e.target.valueAsNumber) && e.target.valueAsNumber >= 0) onUpdate(node.id, { parameter_count: e.target.valueAsNumber }); }} />
          <label className="field-label" htmlFor="schematic-settings">Operation settings · JSON</label>
          <textarea id="schematic-settings" className="text-field" rows={5} value={settings} onChange={e => setSettings(e.target.value)} onBlur={saveSettings} />
          {error && <p className="schematic-error" role="alert">{error}</p>}
          {data.declaration_pending === true && <p className="context-note">Placeholder shape and count: review and save the declared shape.</p>}
          <p className="context-note">These are editable declarations. The original operation is not executed or verified.</p>
          <div className="schematic-actions"><button className="quiet-button" onClick={() => onDuplicate(node.id)}>Duplicate</button><button className="quiet-button" onClick={() => onDelete(node.id)}>Delete block</button></div>
        </div>
        <div className="snapshot-section snapshot-metrics"><span>{proposal ? "Output shape" : "Declared output"}<strong>{shapeText(analysis?.nodes[node.id]?.display_shape ?? data.output_shape as (number | string)[])}</strong></span><span>Owned parameters<strong>{Number(data.parameter_count).toLocaleString()}</strong></span></div>
        <div className="snapshot-section">
          <Detail title="Operation settings" value={data.params} />
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
          return <div className="schematic-connection" key={edge.id}><button className="snapshot-link" onClick={() => other && onSelect(other.id)}><span>{incoming ? "From" : "To"} · {other?.data.label}<small>{String(edge.label ?? edge.data?.role ?? "dependency")}</small></span><ArrowRight size={12} /></button><button className="icon-button" aria-label={`Remove connection ${other?.data.label ?? ""}`} onClick={() => onDisconnect(edge.id)}><X size={12} /></button></div>;
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
