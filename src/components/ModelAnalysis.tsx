import { useState } from "react";
import { ArrowUpRight, CheckCircle2, CircleAlert, Download, Info, Loader2 } from "lucide-react";
import { bytesText, catalog, formatNumber, shapeText, type Analysis, type ModelGraph } from "../model";
import { download, fileName } from "../useEngine";

type Metric = "parameters" | "macs" | "outputs";
type Props = {
  graph: ModelGraph;
  analysis: Analysis | null;
  checking: boolean;
  selected: string | null;
  verifiedNodes?: number;
  onSelect: (id: string) => void;
};

export default function ModelAnalysis({ graph, analysis, checking, selected, verifiedNodes, onSelect }: Props) {
  const [metric, setMetric] = useState<Metric>("parameters");
  const [batch, setBatch] = useState(1);
  const [precision, setPrecision] = useState("fp32");
  const [showAll, setShowAll] = useState(false);
  const bytes = precision === "fp32" ? 4 : 2;
  const declared = graph.mode === "schematic";
  const profile = analysis?.profile;
  const selectedInfo = selected ? analysis?.nodes[selected] : undefined;
  const selectedNode = graph.nodes.find(node => node.id === selected);
  const roots = analysis?.errors.filter(issue => !issue.blocked_by?.length) ?? [];
  const blocked = analysis?.errors.filter(issue => issue.blocked_by?.length) ?? [];
  const rows = graph.nodes.map(node => {
    const info = analysis?.nodes[node.id];
    const value = !info ? null : metric === "parameters" ? info.parameters : metric === "macs" ? info.macs == null ? null : info.macs * batch
      : info.elements * (info.dtype === "int64" ? 8 : bytes) * batch;
    return { node, info, value };
  }).sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
  const maximum = Math.max(1, ...rows.map(row => row.value ?? 0));
  const total = rows.reduce((sum, row) => sum + (row.value ?? 0), 0);
  const outputBytes = profile ? (profile.float_output_elements * bytes + profile.integer_output_elements * 8) * batch : null;
  const format = (value: number) => metric === "outputs" ? bytesText(value) : formatNumber(value);
  return <section className="model-analysis" aria-label="Model analysis" aria-busy={checking}>
    <header className="analysis-heading">
      <div><span className="eyebrow">MODEL INSIGHTS</span><h2>Understand the cost.</h2></div>
      <button className="icon-button" aria-label="Download analysis report" title="Download analysis report" disabled={!analysis || checking}
        onClick={() => download(JSON.stringify({ name: graph.name, assumptions: { batch, precision, costs: "Logical matrix/conv MACs. Output tensor storage excludes internal intermediates, gradients, optimizer state and runtime workspaces.", basis: declared ? "Supplied declarations" : "Static shape inference" }, estimates: { weight_bytes: analysis!.parameters * bytes, output_tensor_bytes: outputBytes, macs: profile?.macs == null ? null : profile.macs * batch }, layers: graph.nodes.map(node => ({ id: node.id, label: node.data.label, kind: node.data.kind })), analysis_at_batch_1_fp32: analysis }, null, 2), `${fileName(graph.name)}_analysis.json`)}><Download size={15} /></button>
    </header>
    <div className={`analysis-state ${analysis?.valid ? "valid" : "invalid"}`}>
      {checking ? <Loader2 className="spin" size={13} /> : analysis?.valid ? <CheckCircle2 size={13} /> : <CircleAlert size={13} />}
      <span>{checking ? "Updating analysis…" : !analysis ? "Waiting for the engine" : !analysis.valid ? `${roots.length} ${roots.length === 1 ? "issue" : "issues"} to resolve` : declared ? "Source declarations · structure checked" : analysis.executable === false ? "Reference architecture · shape analysis" : verifiedNodes ? `${verifiedNodes} layers verified in PyTorch` : analysis.outputs.length ? "Shapes checked · ready to verify" : "Shapes checked · add an output"}</span>
    </div>
    <div className={`analysis-content ${checking ? "is-checking" : ""}`}>
      <div className="analysis-metrics">
        <div><span>{declared ? "Declared parameters" : "Parameters"}</span><strong>{analysis ? formatNumber(analysis.parameters) : "—"}</strong><small>{analysis ? `${bytesText(analysis.parameters * bytes)} weights` : "Awaiting analysis"}</small></div>
        <div><span>Compute / batch</span><strong>{profile?.macs != null ? formatNumber(profile.macs * batch) : "—"}</strong><small>{profile?.macs != null ? "matrix / conv MACs" : declared ? "Needs executable ops" : "Incomplete coverage"}</small></div>
        <div><span>Output tensors</span><strong>{outputBytes == null ? "—" : bytesText(outputBytes)}</strong><small>{profile?.complete ? "Sum across all nodes" : "Available shapes only"}</small></div>
        <div><span>Architecture</span><strong>{graph.nodes.length}<em> layers</em></strong><small>{graph.edges.length} links · {analysis?.depth ?? "—"} steps deep</small></div>
      </div>
      <div className="analysis-assumptions">
        <label>Batch size<input aria-label="Analysis batch size" type="number" min={1} max={65536} value={batch} onChange={event => { const value = event.currentTarget.valueAsNumber; if (Number.isInteger(value) && value >= 1 && value <= 65536) setBatch(value); }} /></label>
        <label>Storage<select aria-label="Analysis storage precision" value={precision} onChange={event => setPrecision(event.target.value)}><option value="fp32">FP32 · 4 bytes</option><option value="fp16">FP16 · 2 bytes</option><option value="bf16">BF16 · 2 bytes</option></select></label>
      </div>
      <p className="analysis-caption">Whole-model estimates · model unchanged · not peak memory.</p>
      {declared && <p className="analysis-basis"><Info size={14} />Declared shapes and weights. Compute is unknown.</p>}
      {profile && !declared && profile.macs === null && <p className="analysis-basis"><Info size={14} />{profile.unknown_nodes.length} layers need valid or implemented operations. {formatNumber(profile.known_macs * batch)} MACs are accounted for; the total is unavailable.</p>}
      <div className="analysis-section-title"><h3>Where it goes</h3><span>{metric === "parameters" ? "Weight count" : metric === "macs" ? "MACs / batch" : "Tensor storage"}</span></div>
      <div className="analysis-metric-tabs" role="group" aria-label="Rank layers by">
        {([['parameters', 'Parameters'], ['macs', 'Compute'], ['outputs', 'Outputs']] as const).map(([value, label]) => <button key={value} aria-pressed={metric === value} onClick={() => setMetric(value)}>{label}</button>)}
      </div>
      <div className="analysis-ranking">
        {(showAll ? rows : rows.slice(0, 8)).map(({ node, info, value }, index) => <button key={node.id} className={selected === node.id ? "selected" : ""} onClick={() => onSelect(node.id)} aria-label={`Inspect ${node.data.label}`}>
          <span className="analysis-rank">{String(index + 1).padStart(2, "0")}</span>
          <span className="analysis-row-main"><span className="analysis-row-name">{node.data.label}</span><span className="analysis-row-meta">{catalog[node.data.kind].title} <b>{value == null || !total ? "" : `${(100 * value / total).toFixed(1)}%`}</b></span><span className="analysis-bar"><i style={{ width: `${100 * (value ?? 0) / maximum}%` }} /></span></span>
          <span className="analysis-row-value" title={value == null ? "Unavailable" : value.toLocaleString()}>{value == null ? "—" : format(value)}<small>{shapeText(info?.display_shape ?? info?.shape)}</small></span>
        </button>)}
      </div>
      {rows.length > 8 && <button className="analysis-show-all" onClick={() => setShowAll(!showAll)}>{showAll ? "Show top 8" : `Show all ${rows.length} layers`}</button>}
      {selectedInfo && selectedNode && <div className="analysis-selection">
        <span className="eyebrow">SELECTED LAYER</span><h3>{selectedNode.data.label}</h3>
        <div><span>Output</span><code>{shapeText(selectedInfo.display_shape ?? selectedInfo.shape)}</code></div>
        {selectedInfo.receptive_field && <><div><span>Receptive field</span><code>{shapeText(selectedInfo.receptive_field.size)}</code></div><div><span>Effective stride</span><code>{shapeText(selectedInfo.receptive_field.stride)}</code></div><p>Theoretical support in {graph.nodes.find(node => node.id === selectedInfo.receptive_field!.source)?.data.label ?? "input"} pixels, including padding.</p></>}
      </div>}
      <details className="analysis-diagnostics" open={roots.length > 0 || undefined}>
        <summary>Diagnostics <span>{roots.length} issues · {analysis?.warnings.length ?? 0} notes</span></summary>
        {[...roots, ...(analysis?.warnings ?? [])].map((issue, i) => <button key={i} className={i < roots.length ? "is-error" : ""} disabled={!issue.node_id} onClick={() => issue.node_id && onSelect(issue.node_id)}>
          {i < roots.length ? <CircleAlert size={14} /> : <Info size={14} />}<span>{issue.node_id && <strong>{graph.nodes.find(node => node.id === issue.node_id)?.data.label}</strong>}{issue.message}</span>{issue.node_id && <ArrowUpRight size={13} />}
        </button>)}
        {!!blocked.length && <p className="analysis-caption">{blocked.length} downstream {blocked.length === 1 ? "layer is" : "layers are"} waiting on these fixes.</p>}
        {!!analysis && !roots.length && !analysis.warnings.length && <p className="analysis-caption">All dependencies and tensor shapes are compatible.</p>}
      </details>
      <details className="analysis-method"><summary>How these estimates work</summary><p>One MAC is one multiply-accumulate. Counts include linear maps, convolutions, recurrent transitions and dense attention products. They exclude biases, normalization, activations, pooling and data movement; they do not predict latency.</p><p>Output tensors count every visible layer output once, including inputs. They exclude internal temporaries, gradients, optimizer state and allocator overhead. Storage choices assume uniform floating-point storage; integer tokens stay at 8 bytes. These controls do not enable mixed-precision execution.</p>{declared && <p>Schematic shapes and weights remain source claims. Symbolic dimensions use source examples (L defaults to 14, other unknown symbols to 1); these are illustrative sizes, not guaranteed bounds.</p>}</details>
    </div>
  </section>;
}
