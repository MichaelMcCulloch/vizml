import {
  Copy,
  Trash2,
  Link2,
  X,
  Plus,
  SlidersHorizontal,
  ArrowRight,
} from "lucide-react";
import {
  catalog,
  formatNumber,
  isInput,
  shapeText,
  canConnect,
  residualProjectionPlan,
  type Analysis,
  type Kind,
  type LayerNode,
  type ModelGraph,
  type Params,
} from "../model";
import SequenceInspector from "./SequenceInspector";
import SchematicInspector from "./SchematicInspector";

type Props = {
  showSnapshotOverview?: boolean;
  node?: LayerNode;
  graph: ModelGraph;
  analysis: Analysis | null;
  onUpdate: (id: string, label: string, params: Params) => void;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
  onConnect: (source: string, target: string) => void;
  onDisconnect: (id: string) => void;
  onProjectResidual: (id: string) => void;
  onExpandStage: (id: string) => void;
  onSwapInputs: (id: string) => void;
};
type Field = {
  key: string;
  label: string;
  min?: number;
  max?: number;
  hint?: string;
  step?: number | "any";
};
const fields: Partial<Record<Kind, Field[]>> = {
  grid_input: [
    { key: "width", label: "Width", max: 256 },
    { key: "height", label: "Height", max: 256 },
    { key: "channels", label: "Channels" },
  ],
  feature_input: [
    {
      key: "features",
      label: "Feature count",
      min: 0,
      hint: "0–65,535 features",
    },
  ],
  sequence_input: [
    { key: "tokens", label: "Token count", max: 1048576 },
    { key: "features", label: "Embedding width" },
  ],
  token_input: [{ key: "tokens", label: "Token count", max: 1048576 }],
  token_embedding: [
    { key: "vocab", label: "Vocabulary size", min: 2, max: 1048576 },
    { key: "features", label: "Embedding width" },
  ],
  patch_embed: [
    { key: "patch", label: "Patch size", max: 256 },
    { key: "features", label: "Embedding width" },
    { key: "registers", label: "Storage tokens", min: 0, max: 64 },
  ],
  attention: [{ key: "heads", label: "Attention heads", max: 1024 }],
  cross_attention: [{ key: "heads", label: "Attention heads", max: 1024 }],
  feedforward: [
    {
      key: "expansion",
      label: "MLP expansion",
      min: 0.125,
      max: 64,
      step: "any",
    },
  ],
  transformer_block: [
    { key: "blocks", label: "Transformer blocks", max: 96 },
    { key: "heads", label: "Attention heads", max: 1024 },
    {
      key: "expansion",
      label: "MLP expansion",
      min: 0.125,
      max: 64,
      step: "any",
    },
  ],
  token_pool: [
    {
      key: "prefix",
      label: "Manual prefix length",
      min: 0,
      max: 65,
      hint: "Patch metadata takes priority",
    },
  ],
  token_head: [
    { key: "vocab", label: "Vocabulary / classes", min: 2, max: 1048576 },
  ],
  deepseek_stage: [
    { key: "start", label: "First layer index", min: 0, max: 39 },
    { key: "blocks", label: "Layers in group", max: 40 },
  ],
  conv2d: [
    { key: "channels", label: "Output channels" },
    { key: "kernel", label: "Kernel size", max: 256 },
    { key: "stride", label: "Stride", max: 256 },
    { key: "padding", label: "Padding", min: 0, max: 256 },
    { key: "dilation", label: "Dilation", max: 256, hint: "1 = no gaps" },
    { key: "groups", label: "Groups" },
  ],
  dense: [{ key: "units", label: "Output features" }],
  maxpool2d: [
    { key: "kernel", label: "Kernel size", max: 256 },
    { key: "stride", label: "Stride", max: 256 },
    { key: "padding", label: "Padding", min: 0, max: 256 },
    { key: "dilation", label: "Dilation", max: 256, hint: "1 = no gaps" },
  ],
  padding: [{ key: "padding", label: "Padding per side", min: 0, max: 256 }],
  upsample2d: [{ key: "height", label: "Output height", min: 1, max: 4096 }, { key: "width", label: "Output width", min: 1, max: 4096 }],
  classification: [{ key: "units", label: "Classes" }],
  regression: [{ key: "units", label: "Output dimensions" }],
  layernorm: [
    { key: "eps", label: "Epsilon", min: 1e-12, max: 1, step: "any" },
  ],
  convnext_stage: [
    { key: "blocks", label: "Residual blocks", max: 64 },
    {
      key: "kernel",
      label: "Depthwise kernel",
      max: 255,
      step: 2,
      hint: "Odd sizes preserve shape",
    },
    { key: "expansion", label: "MLP expansion", max: 16 },
    {
      key: "layer_scale",
      label: "LayerScale init",
      min: 0,
      max: 1,
      step: "any",
    },
    { key: "eps", label: "Norm epsilon", min: 1e-12, max: 1, step: "any" },
    { key: "drop_path", label: "Drop path", min: 0, max: 0.99, step: 0.01 },
  ],
};

export default function Inspector({
  showSnapshotOverview,
  node,
  graph,
  analysis,
  onUpdate,
  onSelect,
  onDelete,
  onDuplicate,
  onConnect,
  onDisconnect,
  onProjectResidual,
  onExpandStage,
  onSwapInputs,
}: Props) {
  if (graph.mode === "schematic") return <SchematicInspector node={node} graph={graph} analysis={analysis} onSelect={onSelect} overview={showSnapshotOverview} />;
  if (!node)
    return (
      <aside className="inspector">
        <div className="inspector-eyebrow">
          <SlidersHorizontal size={13} /> INSPECTOR
        </div>
        <div className="inspector-empty">
          <div className="empty-inspector-icon">
            <SlidersHorizontal size={25} strokeWidth={1.3} />
          </div>
          <h3>A closer look</h3>
          <p>
            Select a layer in either view to explore its dimensions and tune its
            parameters.
          </p>
          <span>Every connection has a consequence.</span>
        </div>
        <div className="quick-tip">
          <span>GOOD TO KNOW</span>
          <p>
            Multiple incoming branches concatenate automatically. Mix a grid
            with a vector and the grid flattens at the union.
          </p>
        </div>
      </aside>
    );
  const { kind, label, params } = node.data;
  const def = catalog[kind];
  const info = analysis?.nodes[node.id];
  const errors = analysis?.errors.filter((e) => e.node_id === node.id) ?? [];
  const incoming = graph.edges.filter((e) => e.target === node.id);
  const outgoing = graph.edges.filter((e) => e.source === node.id);
  const candidates = graph.nodes.filter((n) =>
    canConnect(graph, n.id, node.id),
  );
  const projection = residualProjectionPlan(graph, node.id, analysis);
  const nodeFields = fields[kind]?.filter(
    (field) =>
      kind !== "token_pool" ||
      field.key !== "prefix" ||
      (params.method === "patches" && !info?.grid),
  );
  const change = (key: string, value: number | string) =>
    onUpdate(node.id, label, { ...params, [key]: value });
  return (
    <aside className="inspector">
      <div className="inspector-eyebrow">
        <SlidersHorizontal size={13} /> INSPECTOR
        <button
          className="icon-button"
          onClick={() => onSelect(null)}
          aria-label="Close inspector"
        >
          <X size={14} />
        </button>
      </div>
      <div className="inspector-scroll">
        <div className="inspector-intro">
          <span
            className="inspector-layer-icon"
            style={{
              color: def.color,
              background: `${def.color}10`,
              borderColor: `${def.color}30`,
            }}
          >
            {def.symbol}
          </span>
          <div>
            <span style={{ color: def.color }}>{def.title}</span>
            <h2>{label}</h2>
          </div>
        </div>
        <div className="inspector-section">
          <label className="field-label" htmlFor="layer-name">
            Layer name
          </label>
          <input
            id="layer-name"
            className="text-field"
            value={label}
            maxLength={100}
            onChange={(e) =>
              onUpdate(node.id, e.target.value || "Layer", params)
            }
          />
        </div>
        <div className="inspector-section">
          <div className="section-title">
            TENSOR SHAPE{" "}
            <span>
              {info?.layout === "streams"
                ? "T × STREAMS × D"
                : info?.shape.length === 2
                  ? "T × D"
                  : kind === "token_input"
                    ? "INT64"
                    : info?.shape.length === 3
                      ? "C × H × W"
                      : "FEATURES"}
            </span>
          </div>
          <div className="shape-card">
            {!isInput(kind) && (
              <div>
                <span>Input</span>
                <strong>{shapeText(info?.input_shape)}</strong>
              </div>
            )}
            <div>
              <span>Output</span>
              <strong style={{ color: def.color }}>
                {shapeText(info?.shape)}
              </strong>
            </div>
          </div>
          {info && (
            <div className="shape-footnote">
              {formatNumber(info.elements)} elements <span>·</span>{" "}
              {formatNumber(info.parameters)} parameters
            </div>
          )}
        </div>
        {!!nodeFields?.length && (
          <div className="inspector-section">
            <div className="section-title">PARAMETERS</div>
            <div
              className={`parameter-grid ${nodeFields!.length === 1 ? "single-field" : ""}`}
            >
              {nodeFields!.map((field) => (
                <label className="parameter-field" key={field.key}>
                  <span>{field.label}</span>
                  <input
                    aria-label={field.label}
                    type="number"
                    min={field.min ?? 1}
                    max={field.max ?? 65535}
                    step={field.step ?? 1}
                    value={params[field.key] ?? 1}
                    onChange={(e) => {
                      if (Number.isFinite(e.target.valueAsNumber))
                        change(field.key, e.target.valueAsNumber);
                    }}
                  />
                  {field.hint && <small>{field.hint}</small>}
                </label>
              ))}
            </div>
          </div>
        )}
        {kind === "upsample2d" && <div className="inspector-section">
          <label className="field-label" htmlFor="resize-mode">Interpolation</label>
          <select className="text-field" id="resize-mode" value={params.mode ?? "bilinear"} onChange={e => change("mode", e.target.value)}>
            <option value="bilinear">Bilinear</option><option value="nearest">Nearest</option>
          </select>
          {params.mode !== "nearest" && <label className="resize-corners"><input type="checkbox" checked={!!Number(params.align_corners ?? 1)} onChange={e => change("align_corners", e.target.checked ? 1 : 0)} /> Align corner coordinates</label>}
        </div>}
        <SequenceInspector
          node={node}
          info={info}
          incoming={incoming.length}
          onChange={change}
          onExpand={() => onExpandStage(node.id)}
          onSwap={() => onSwapInputs(node.id)}
        />
        {["conv2d", "dense", "residual"].includes(kind) && (
          <div className="inspector-section">
            <label className="field-label" htmlFor="activation">
              Activation
            </label>
            <select
              id="activation"
              className="text-field"
              value={params.activation ?? "none"}
              onChange={(e) => change("activation", e.target.value)}
            >
              {["relu", "gelu", "silu", "tanh", "none"].map((a) => (
                <option key={a} value={a}>
                  {a === "none" ? "None / linear" : a.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        )}
        {kind === "conv2d" && Number(params.dilation) > 1 && (
          <div className="context-note">
            Dilation {params.dilation} gives this {params.kernel} ×{" "}
            {params.kernel} kernel a{" "}
            <strong>
              {Number(params.dilation) * (Number(params.kernel) - 1) + 1} ×{" "}
              {Number(params.dilation) * (Number(params.kernel) - 1) + 1}
            </strong>{" "}
            receptive field.
          </div>
        )}
        {kind === "classification" && (
          <div className="context-note">
            The PyTorch head returns logits. <code>predict()</code> converts the
            winning class to a one-hot feature vector.
          </div>
        )}
        {kind === "convnext_stage" && (
          <div className="convnext-guide">
            <div className="section-title">
              INSIDE EACH BLOCK <span>×{params.blocks}</span>
            </div>
            <ol>
              <li>
                <span>
                  {params.kernel}×{params.kernel} depthwise conv
                </span>
                <code>groups = C</code>
              </li>
              <li>
                <span>Channel LayerNorm</span>
                <code>ε {params.eps}</code>
              </li>
              <li>
                <span>Pointwise linear + GELU</span>
                <code>C → {params.expansion}C</code>
              </li>
              <li>
                <span>Pointwise linear</span>
                <code>{params.expansion}C → C</code>
              </li>
              <li>
                <span>LayerScale</span>
                <code>
                  {Number(params.layer_scale) > 0
                    ? `γ ${params.layer_scale}`
                    : "disabled"}
                </code>
              </li>
              {Number(params.drop_path) > 0 && (
                <li>
                  <span>Stochastic depth</span>
                  <code>p {params.drop_path}</code>
                </li>
              )}
              <li className="block-residual">
                <span>Add input skip</span>
                <code>x + F(x)</code>
              </li>
            </ol>
            {Number(params.blocks) > 1 && (
              <button
                className="projection-button"
                onClick={() => onExpandStage(node.id)}
              >
                <Plus size={13} />
                Expand {params.blocks} blocks
                <ArrowRight size={13} />
              </button>
            )}
          </div>
        )}
        {kind === "layernorm" && (
          <div className="context-note">
            Normalizes the channel axis at every grid location, or the feature
            axis for a vector or each token in a sequence. Includes learned
            scale and bias.
          </div>
        )}
        {kind === "global_avgpool" && (
          <div className="context-note">
            Averages height and width independently for each channel. Produces a
            C-dimensional vector.
          </div>
        )}
        {kind === "embedding" && (
          <div className="context-note">
            Returns backbone features directly. To add a task head, connect
            classification or regression to the upstream layer.
          </div>
        )}
        {kind === "residual" && (
          <div className="residual-guide">
            <div className="residual-formula">x + F(x)</div>
            <p>
              Connect the skip path and the transformed branch. Their channels,
              height, and width must match.
            </p>
            {incoming.map((edge) => {
              const source = graph.nodes.find((n) => n.id === edge.source)!;
              const shape = analysis?.nodes[source.id]?.shape;
              return (
                <div className="residual-branch-shape" key={edge.id}>
                  <span>{source.data.label}</span>
                  <code>{shapeText(shape)}</code>
                </div>
              );
            })}
            {projection && (
              <>
                <p className="projection-description">
                  Match the skip path to{" "}
                  <strong>
                    {
                      graph.nodes.find((n) => n.id === projection.targetId)
                        ?.data.label
                    }
                  </strong>{" "}
                  using{" "}
                  {projection.steps.length === 1
                    ? "a 1×1 convolution"
                    : "1×1 convolutions"}
                  {projection.steps.some((s) => s.stride > 1)
                    ? " with downsampling"
                    : ""}
                  .
                </p>
                <button
                  className="projection-button"
                  onClick={() => onProjectResidual(node.id)}
                >
                  <Plus size={13} />
                  Add skip projection
                  <ArrowRight size={13} />
                </button>
              </>
            )}
            {incoming.length < 2 && (
              <small>
                Add {2 - incoming.length} more{" "}
                {incoming.length ? "dependency" : "dependencies"} below.
              </small>
            )}
          </div>
        )}
        {!isInput(kind) && (
          <div className="inspector-section dependencies">
            <div className="section-title">
              <span>DEPENDENCIES</span>
              <span>{incoming.length}</span>
            </div>
            {incoming.map((edge, i) => {
              const source = graph.nodes.find((n) => n.id === edge.source)!;
              return (
                <div className="dependency" key={edge.id}>
                  <span className="dependency-index">{i + 1}</span>
                  <button onClick={() => onSelect(source.id)}>
                    <span style={{ color: catalog[source.data.kind].color }}>
                      {catalog[source.data.kind].symbol}
                    </span>
                    {source.data.label}
                  </button>
                  <button
                    className="icon-button"
                    onClick={() => onDisconnect(edge.id)}
                    aria-label={`Disconnect ${source.data.label}`}
                  >
                    <X size={12} />
                  </button>
                </div>
              );
            })}
            <div className="add-dependency">
              <Plus size={12} />
              <select
                aria-label="Add dependency"
                value=""
                onChange={(e) => {
                  if (e.target.value) onConnect(e.target.value, node.id);
                }}
              >
                <option value="">Add dependency…</option>
                {candidates.map((n) => (
                  <option value={n.id} key={n.id}>
                    {n.data.label}
                  </option>
                ))}
              </select>
            </div>
            {incoming.length > 1 && kind !== "residual" && (
              <div className="merge-explanation">
                <Link2 size={12} />
                <span>
                  {info?.merge === "flatten_concat"
                    ? "Flatten each branch, then concatenate features."
                    : "Concatenate along the channel axis."}{" "}
                  Connection order is preserved.
                </span>
              </div>
            )}
          </div>
        )}
        {!!outgoing.length && (
          <div className="inspector-section">
            <div className="section-title">
              DEPENDENTS <span>{outgoing.length}</span>
            </div>
            {outgoing.map((edge) => {
              const target = graph.nodes.find((n) => n.id === edge.target)!;
              return (
                <button
                  className="dependent-link"
                  key={edge.id}
                  onClick={() => onSelect(target.id)}
                >
                  <span>{target.data.label}</span>
                  <ArrowRight size={12} />
                </button>
              );
            })}
          </div>
        )}
        {errors.map((error, i) => (
          <div key={i} className="inspector-error">
            {error.message}
          </div>
        ))}
      </div>
      <div className="inspector-actions">
        <button onClick={() => onDuplicate(node.id)}>
          <Copy size={13} /> Duplicate
        </button>
        <button className="delete-layer" onClick={() => onDelete(node.id)}>
          <Trash2 size={13} /> Delete layer
        </button>
      </div>
    </aside>
  );
}
