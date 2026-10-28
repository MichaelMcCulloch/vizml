import { createContext, useContext, memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { ArrowRight, CircleAlert } from "lucide-react";
import {
  catalog,
  formatNumber,
  isInput,
  isOutput,
  isRecurrent,
  shapeText,
  type Analysis,
  type LayerNode as LayerNodeType,
} from "../model";

export const AnalysisContext = createContext<Analysis | null>(null);

function LayerNode({ id, data, selected }: NodeProps<LayerNodeType>) {
  const analysis = useContext(AnalysisContext);
  const info = analysis?.nodes[id];
  const issue = analysis?.errors.find((e) => e.node_id === id);
  const definition = catalog[data.kind];
  const p = data.params;
  const schematic = Array.isArray(data.output_shape);
  const schematicDetail = data.kind === "conv2d"
    ? `${p.kernel}×${p.kernel} · stride ${p.stride}${Number(p.dilation) > 1 ? ` · d${p.dilation}` : ""}${Array.isArray(data.pre_operations) ? " · pre-norm" : ""}`
    : data.kind === "convnext_stage" ? `${p.blocks} ${Number(p.blocks) === 1 ? "block" : "blocks"} · ${p.kernel}×${p.kernel} DW · ${p.expansion}× MLP`
    : data.kind === "upsample2d" ? `${p.mode} · ${p.height} × ${p.width}`
    : data.kind === "concat" ? `Concatenate ${p.axis ?? "features"}`
    : typeof data.current_role === "string" ? data.current_role : "Architecture schematic";
  const baseDetail =
    data.kind === "conv2d"
      ? `${p.kernel}×${p.kernel} · stride ${p.stride}${Number(p.dilation) > 1 ? ` · d${p.dilation}` : ""}`
      : data.kind === "maxpool2d"
        ? `${p.kernel}×${p.kernel} · stride ${p.stride}`
        : data.kind === "dense"
          ? `${p.units} units · ${String(p.activation).toUpperCase()}`
          : data.kind === "concat"
            ? info?.merge === "flatten_concat"
              ? "Flatten + concatenate"
              : "Concatenate channels"
            : data.kind === "residual"
              ? "Elementwise sum"
              : data.kind === "classification"
                ? `${p.units} classes · one-hot`
                : data.kind === "regression"
                  ? `${p.units} continuous outputs`
                  : data.kind === "padding"
                    ? `Zero padding · ${p.padding} per side`
                    : data.kind === "flatten"
                      ? "Preserve all features"
                      : data.kind === "grid_input"
                        ? "C × H × W"
                        : "Feature vector";
  const detail =
    data.kind === "convnext_stage"
      ? `${p.blocks} residual ${Number(p.blocks) === 1 ? "block" : "blocks"} · ${p.kernel}×${p.kernel} DW`
      : data.kind === "layernorm"
        ? `Channel norm · ε ${p.eps}`
        : data.kind === "global_avgpool"
          ? "Average H × W → C"
          : data.kind === "embedding"
            ? "Backbone feature vector"
            : baseDetail;
  const sequenceDetail: Partial<Record<typeof data.kind, string>> = {
    sequence_input: "T steps / tokens × D features",
    grid_sequence_input: "T steps × C × H × W",
    token_input: "T integer token IDs",
    token_embedding: `${formatNumber(Number(p.vocab))} vocabulary · ${p.features}-D`,
    patch_embed: `${p.patch}×${p.patch} patches${Number(p.cls_token) ? " · CLS" : ""}${Number(p.registers) ? ` · ${p.registers} storage` : ""}`,
    position_encoding: `${p.method === "learned" ? "Learned" : "Sinusoidal"} token positions`,
    attention: `${p.heads} heads · ${Number(p.causal) ? "causal" : "bidirectional"}${p.position !== "none" ? ` · ${p.position}` : ""}`,
    cross_attention: `${p.heads} heads · query ← context`,
    feedforward: `${p.expansion}× expansion · ${String(p.activation).toUpperCase()}`,
    transformer_block: `${p.blocks} ${Number(p.blocks) === 1 ? "block" : "blocks"} · ${p.heads} heads · ${p.mode === "cross" ? "self + cross" : p.mode === "decoder" ? "causal" : "bidirectional"}`,
    token_pool:
      p.method === "cls"
        ? "Select first / CLS token"
        : p.method === "patches"
          ? "Keep patch tokens; strip prefix"
          : "Average tokens → D",
    token_head: `${formatNumber(Number(p.vocab))} classes per token`,
    deepseek_stage: `Layers ${p.start}–${Number(p.start) + Number(p.blocks) - 1} · 384 experts · 4 streams`,
    deepseek_vision: "14×14 patches · ViT 32 · 3× aligner",
    deepseek_bridge: "Image placeholders → 4 residual lanes",
    deepseek_head: "129,280 token logits · reference",
    deepseek_draft: "Layers 37/38/39 → 5-token draft",
  };
  return (
    <div
      className={`layer-node ${selected ? "is-selected" : ""} ${issue ? "has-error" : ""}`}
      style={{ "--layer-color": definition.color } as React.CSSProperties}
    >
      {(schematic || !isInput(data.kind)) && (
        <Handle type="target" position={Position.Left} id="in" />
      )}
      <div className="layer-node-top">
        <span className="layer-symbol">{definition.symbol}</span>
        <span className="layer-kind">{schematic && data.kind === "embedding" ? "Embedding table" : definition.title}</span>
        {issue && <CircleAlert size={13} className="error-icon" />}
      </div>
      <div className="layer-node-name" title={data.label}>{data.label}</div>
      <div className="layer-detail">
        {schematic ? schematicDetail : isRecurrent(data.kind)
          ? `${p.hidden_size} hidden · ${p.layers} ${Number(p.layers) === 1 ? "layer" : "layers"}${Number(p.bidirectional) ? " · ↔" : ""} · ${Number(p.return_sequences ?? 1) ? "sequence" : "final state"}`
          : data.kind === "upsample2d" ? `${p.mode} → ${p.height} × ${p.width}` : sequenceDetail[data.kind] ??
          (data.kind === "embedding" && info?.shape.length === 2
            ? "Token features"
            : detail)}
      </div>
      <div className="layer-node-bottom">
        <span title={issue?.message ?? shapeText(info?.display_shape ?? info?.shape)}>
          {issue ? "Shape error" : shapeText(info?.display_shape ?? info?.shape)}
        </span>
        <span>
          {info?.parameters ? (
            `${formatNumber(info.parameters)} p`
          ) : (
            <ArrowRight size={12} />
          )}
        </span>
      </div>
      {(schematic || !isOutput(data.kind)) && (
        <Handle type="source" position={Position.Right} id="out" />
      )}
    </div>
  );
}

export default memo(LayerNode);
