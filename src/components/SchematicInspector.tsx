import { useState } from "react";
import { FileSearch, X, ArrowRight, Search } from "lucide-react";
import { catalog, formatNumber, shapeText, type Analysis, type LayerNode, type ModelGraph } from "../model";

function Detail({ title, value }: { title: string; value: unknown }) {
  if (value == null) return null;
  return <details className="snapshot-detail"><summary>{title}</summary><pre>{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</pre></details>;
}

export default function SchematicInspector({ node, graph, analysis, onSelect, overview }: {
  node?: LayerNode; graph: ModelGraph; analysis: Analysis | null; onSelect: (id: string | null) => void; overview?: boolean;
}) {
  const [search, setSearch] = useState("");
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
        {!node && checkpoint && <p>Snapshot generation {checkpoint.generation} · {checkpoint.cumulative_decisions?.toLocaleString()} decisions</p>}
        {weights && <div className={`checkpoint-note ${weights.snapshot_match === false ? "revision-mismatch" : ""}`}>
          <strong>{weights.matched}/{weights.expected} tensor names & shapes match</strong>
          <span>{weights.filename}{weights.snapshot_match === false ? ": file hash differs from the JSON snapshot. Supplied statistics and training details describe that snapshot." : ". Forward connections remain supplied claims."}</span>
        </div>}
      </div>
      {node && data && <>
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
          return <button key={edge.id} className="snapshot-link" onClick={() => onSelect(other!.id)}><span>{incoming ? "From" : "To"} · {other?.data.label}<small>{String(edge.label ?? edge.data?.role ?? "dependency")}</small></span><ArrowRight size={12} /></button>;
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
