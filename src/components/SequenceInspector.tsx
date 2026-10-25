import { ArrowRight, Plus, ArrowLeftRight, ExternalLink } from "lucide-react";
import type { LayerNode, NodeAnalysis } from "../model";

type Props = {
  node: LayerNode;
  info?: NodeAnalysis;
  incoming: number;
  onChange: (key: string, value: string | number) => void;
  onExpand: () => void;
  onSwap: () => void;
};

export default function SequenceInspector({
  node,
  info,
  incoming,
  onChange,
  onExpand,
  onSwap,
}: Props) {
  const { kind, params: p } = node.data;
  const transformer = kind === "transformer_block";
  const attention =
    transformer || kind === "attention" || kind === "cross_attention";
  const cross =
    kind === "cross_attention" || (transformer && p.mode === "cross");
  const reference = kind.startsWith("deepseek_");
  const select = (
    key: string,
    label: string,
    choices: [string | number, string][],
  ) => (
    <label className="sequence-option" key={key}>
      <span>{label}</span>
      <select
        className="text-field"
        aria-label={label}
        value={p[key]}
        onChange={(e) =>
          onChange(
            key,
            typeof choices[0][0] === "number"
              ? Number(e.target.value)
              : e.target.value,
          )
        }
      >
        {choices.map(([value, text]) => (
          <option key={value} value={value}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
  const numeric = (
    key: string,
    label: string,
    min: number,
    max: number,
    step: number | "any" = "any",
  ) => (
    <label className="parameter-field" key={key}>
      <span>{label}</span>
      <input
        type="number"
        aria-label={label}
        value={p[key] ?? 0}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          if (Number.isFinite(e.target.valueAsNumber))
            onChange(key, e.target.valueAsNumber);
        }}
      />
    </label>
  );
  return (
    <>
      {(attention ||
        kind === "feedforward" ||
        kind === "position_encoding" ||
        kind === "token_pool") && (
        <div className="inspector-section sequence-options">
          {transformer &&
            select("mode", "Block type", [
              ["encoder", "Encoder · bidirectional"],
              ["decoder", "Decoder · causal"],
              ["cross", "Decoder · self + cross"],
            ])}
          {kind === "attention" &&
            select("causal", "Attention mask", [
              [0, "Bidirectional"],
              [1, "Causal · past tokens only"],
            ])}
          {attention &&
            select("position", "Rotary positions", [
              ["none", "None / external encoding"],
              ["rope", "1D RoPE"],
              ["rope2d", "2D axial RoPE · image patches"],
            ])}
          {(transformer || kind === "feedforward") &&
            select("activation", "Feed-forward activation", [
              ["gelu", "GELU"],
              ["relu", "ReLU"],
              ["silu", "SiLU"],
              ["swiglu", "SwiGLU · gated"],
            ])}
          {kind === "position_encoding" &&
            select("method", "Position encoding", [
              ["sinusoidal", "Sinusoidal · fixed"],
              ["learned", "Learned position table"],
            ])}
          {kind === "token_pool" &&
            select("method", "Token selection", [
              ["mean", "Mean over all tokens"],
              ["cls", "First / CLS token"],
              ["patches", "Patch tokens · remove prefix"],
            ])}
        </div>
      )}
      {cross && (
        <div className="context-note">
          Dependency 1 supplies the <strong>query</strong>. Dependency 2
          supplies <strong>context</strong> for keys and values. Context may
          have a different length and width.
          {incoming === 2 && (
            <button className="projection-button" onClick={onSwap}>
              <ArrowLeftRight size={13} /> Swap query and context
            </button>
          )}
        </div>
      )}
      {kind === "patch_embed" && (
        <>
          <div className="inspector-section sequence-options">
            {select("cls_token", "Class token", [
              [1, "Include learned CLS token"],
              [0, "No class token"],
            ])}
          </div>
          <div className="context-note">
            A non-overlapping convolution projects each patch to one token.{" "}
            {info?.grid && (
              <>
                <strong>
                  {info.grid[0]} × {info.grid[1]}
                </strong>{" "}
                patches + <strong>{info.prefix_tokens}</strong> prefix
                tokens.{" "}
              </>
            )}
            Storage tokens follow CLS and precede image patches.
          </div>
          <details className="advanced-parameters">
            <summary>Mask token</summary>
            <div className="sequence-options">
              {select("mask_token", "Training mask parameter", [
                [0, "Omit mask token"],
                [1, "Include learned mask token"],
              ])}
              <p>
                The mask parameter is retained for DINOv3 architecture counts.
                This editor previews unmasked inference.
              </p>
            </div>
          </details>
        </>
      )}
      {attention && (
        <div className="convnext-guide attention-guide">
          <div className="section-title">
            {transformer ? "INSIDE EACH BLOCK" : "ATTENTION"}
            <span>{transformer ? `×${p.blocks}` : `${p.heads} heads`}</span>
          </div>
          <ol>
            {transformer && p.norm_style === "pre" && (
              <li>
                <span>LayerNorm · pre-norm</span>
                <code>ε {p.eps}</code>
              </li>
            )}
            <li>
              <span>Query / key / value projections</span>
              <code>
                {p.heads} ×{" "}
                {info ? info.input_shape.at(-1)! / Number(p.heads) : "dₖ"}
              </code>
            </li>
            <li>
              <span>
                {(transformer && p.mode !== "encoder") || Number(p.causal)
                  ? "Causal attention"
                  : "Bidirectional attention"}
              </span>
              <code>softmax(QKᵀ / √dₖ)V</code>
            </li>
            {p.position !== "none" && (
              <li>
                <span>
                  {p.position === "rope2d"
                    ? "Axial 2D rotary positions"
                    : "1D rotary positions"}
                </span>
                <code>Q, K only</code>
              </li>
            )}
            <li>
              <span>Concatenate heads + output map</span>
              <code>D → D</code>
            </li>
            {transformer && Number(p.layer_scale) > 0 && (
              <li>
                <span>Attention LayerScale</span>
                <code>γ {p.layer_scale}</code>
              </li>
            )}
            {transformer && (
              <>
                <li className="block-residual">
                  <span>Add attention residual</span>
                  <code>x + Attn(x)</code>
                </li>
                {p.norm_style === "post" && (
                  <li>
                    <span>Post-residual LayerNorm</span>
                    <code>ε {p.eps}</code>
                  </li>
                )}
                {cross && (
                  <li className="block-residual">
                    <span>Cross-attention + residual</span>
                    <code>Q ← context</code>
                  </li>
                )}
                <li>
                  <span>
                    {p.norm_style === "pre"
                      ? "LayerNorm + token MLP"
                      : "Token MLP"}
                  </span>
                  <code>D → {p.expansion}D → D</code>
                </li>
                {Number(p.layer_scale) > 0 && (
                  <li>
                    <span>MLP LayerScale</span>
                    <code>γ {p.layer_scale}</code>
                  </li>
                )}
                <li className="block-residual">
                  <span>Add MLP residual</span>
                  <code>x + MLP(x)</code>
                </li>
                {p.norm_style === "post" && (
                  <li>
                    <span>Post-residual LayerNorm</span>
                    <code>ε {p.eps}</code>
                  </li>
                )}
              </>
            )}
          </ol>
          {transformer && Number(p.blocks) > 1 && (
            <button className="projection-button" onClick={onExpand}>
              <Plus size={13} /> Expand {p.blocks} blocks{" "}
              <ArrowRight size={13} />
            </button>
          )}
        </div>
      )}
      {(attention || kind === "feedforward") && (
        <details className="advanced-parameters">
          <summary>Advanced parameters</summary>
          <div className="sequence-options">
            {transformer &&
              select("norm_style", "Normalization order", [
                ["pre", "Pre-norm"],
                ["post", "Post-norm"],
              ])}
            <div className="parameter-grid">
              {numeric("dropout", "Dropout", 0, 0.99, 0.01)}
              {transformer && numeric("eps", "Norm epsilon", 1e-20, 1)}
              {transformer && numeric("layer_scale", "LayerScale init", 0, 1)}
              {attention &&
                p.position !== "none" &&
                numeric("rope_base", "RoPE base", 1.01, 10000000)}
            </div>
            {attention &&
              select("qkv_bias", "QKV bias", [
                [1, "Enabled"],
                [0, "Disabled"],
              ])}
            {attention &&
              select("proj_bias", "Output projection bias", [
                [1, "Enabled"],
                [0, "Disabled"],
              ])}
            {(transformer || kind === "feedforward") &&
              select("ffn_bias", "Feed-forward bias", [
                [1, "Enabled"],
                [0, "Disabled"],
              ])}
            {attention &&
              select("mask_k_bias", "Key bias masking", [
                [0, "Disabled"],
                [1, "Zero key bias · DINOv3"],
              ])}
          </div>
        </details>
      )}
      {kind === "token_pool" && (
        <div className="context-note">
          {p.method === "patches"
            ? "Removes CLS and storage tokens using the patch embedding’s prefix metadata. The manual prefix is used for ordinary sequences."
            : p.method === "cls"
              ? "Selects token zero. Include a class token in the patch embedding when using this for ViT."
              : "Averages every token, including any prefix tokens."}
        </div>
      )}
      {kind === "token_head" && (
        <div className="context-note">
          Applies the same linear vocabulary projection to every token. Returns
          T × V logits; <code>predict()</code> returns a one-hot class for each
          token.
        </div>
      )}
      {reference && (
        <div className="reference-guide">
          <div className="section-title">REFERENCE ARCHITECTURE</div>
          {kind === "deepseek_stage" && (
            <>
              <dl>
                <div>
                  <dt>Width / heads</dt>
                  <dd>5,120 / 64 × 512</dd>
                </div>
                <div>
                  <dt>Query / output ranks</dt>
                  <dd>1,280 / 1,024 × 8</dd>
                </div>
                <div>
                  <dt>Sliding window</dt>
                  <dd>128 tokens</dd>
                </div>
                <div>
                  <dt>Routed experts</dt>
                  <dd>6 of 384 + 1 shared</dd>
                </div>
                <div>
                  <dt>SwiGLU width</dt>
                  <dd>2,304</dd>
                </div>
                <div>
                  <dt>Residual lanes</dt>
                  <dd>4 · Sinkhorn mixing</dd>
                </div>
              </dl>
              <p>
                Compressed KV sources: layers 2, 8, 14, 20. Index sources: 2, 8,
                14, 20, 24, 28, 32, 36. Layers 2–19 compress 2:1; layers 20–39
                reuse layer 20’s 1:1 cache.
              </p>
              {!!info?.engram_layers?.length && (
                <p className="reference-highlight">
                  Engram n-gram memory at layer {info.engram_layers.join(", ")}{" "}
                  · 2–4 grams, 8 hash heads.
                </p>
              )}
              {Number(p.blocks) > 1 && (
                <button className="projection-button" onClick={onExpand}>
                  <Plus size={13} /> Expand {p.blocks} layers{" "}
                  <ArrowRight size={13} />
                </button>
              )}
            </>
          )}
          {kind === "deepseek_vision" && (
            <p>
              32 bidirectional ViT blocks, width 1,024, 16 heads, 14×14 patches
              and 2D RoPE. A 3× spatial downsampling aligner projects to width
              5,120. The template uses a 252px preview; released image
              preprocessing normally uses larger images.
            </p>
          )}
          {kind === "deepseek_bridge" && (
            <>
              <p>
                First dependency: text embeddings with image placeholder
                positions. Second: aligned image tokens. Image features replace
                those positions, then the stream expands into four copies.
                Includes learned start, end and newline embeddings.
              </p>
              {incoming === 2 && (
                <button className="projection-button" onClick={onSwap}>
                  <ArrowLeftRight size={13} /> Swap input order
                </button>
              )}
            </>
          )}
          {kind === "deepseek_head" && (
            <p>
              Collapses the four residual lanes using the last block’s mixing
              coefficients, applies RMSNorm, then projects to 129,280 vocabulary
              logits. The output projection is independent of the token
              embedding.
            </p>
          )}
          {kind === "deepseek_draft" && (
            <p>
              Three DSpark stages read backbone layers 37, 38 and 39 to draft
              five tokens. Each stage routes 3 of 128 experts. Shared input
              embeddings and output projection are counted once in the main
              path. Includes the Markov and confidence heads.
            </p>
          )}
          <p>
            Shape and logical-weight blueprint. Native sparse attention,
            quantization, MoE routing, Engram hashing and mHC execution require
            DeepSeek’s runtime. Save/export JSON to keep this architecture.
          </p>
          <a
            href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/inference"
            target="_blank"
            rel="noreferrer"
          >
            Checkpoint reference <ExternalLink size={11} />
          </a>
        </div>
      )}
    </>
  );
}
