import type { Edge, Node } from "@xyflow/react";
import catanOriginal from "../templates/catan_original.vizml.json" with { type: "json" };
import catanSmall from "../templates/catan_convnext_small.vizml.json" with { type: "json" };

export const recurrentKinds = ["rnn", "gru", "lstm", "conv_rnn", "conv_gru", "conv_lstm"] as const;
export type RecurrentKind = typeof recurrentKinds[number];
export type Kind =
  | RecurrentKind
  | "grid_sequence_input"
  | "grid_input"
  | "feature_input"
  | "conv2d"
  | "dense"
  | "maxpool2d"
  | "padding"
  | "flatten"
  | "concat"
  | "residual"
  | "classification"
  | "regression"
  | "convnext_stage"
  | "layernorm"
  | "global_avgpool"
  | "embedding"
  | "sequence_input"
  | "token_input"
  | "token_embedding"
  | "patch_embed"
  | "position_encoding"
  | "attention"
  | "cross_attention"
  | "feedforward"
  | "transformer_block"
  | "token_pool"
  | "token_head"
  | "deepseek_stage"
  | "deepseek_vision"
  | "deepseek_bridge"
  | "deepseek_head"
  | "deepseek_draft"
  | "activation" | "add" | "dot" | "gather" | "globalavgpool2d" | "globalmaxpool2d"
  | "legal_policy" | "outcome_transform" | "standardize" | "checkpoint_module" | "upsample2d";
export type Params = Record<string, number | string>;
export type LayerData = {
  kind: Kind;
  label: string;
  params: Params;
  [key: string]: unknown;
};
export type LayerNode = Node<LayerData, "layer">;
export type ModelGraph = {
  version: 1;
  name: string;
  template?: string;
  mode?: "builder" | "schematic";
  metadata?: Record<string, unknown>;
  nodes: LayerNode[];
  edges: Edge[];
};
export type NodeAnalysis = {
  shape: number[];
  display_shape?: (number | string)[];
  input_shape: number[];
  parameters: number;
  elements: number;
  merge: string;
  dependencies: number;
  depth: number;
  grid?: number[];
  prefix_tokens?: number;
  dtype?: string;
  layout?: string;
  recurrent?: {
    cell: "rnn" | "gru" | "lstm";
    spatial: boolean;
    steps: number;
    layers: number;
    directions: number;
    hidden_size: number;
    return_sequences: boolean;
    state_shape: number[];
    states: string[];
  };
  context_shape?: number[];
  residual_blocks?: number;
  reference_only?: boolean;
  terminal?: boolean;
  kv_sources?: number[];
  index_sources?: number[];
  engram_layers?: number[];
};
export type Issue = { node_id: string | null; message: string };
export type Analysis = {
  valid: boolean;
  executable?: boolean;
  basis?: "declared";
  nodes: Record<string, NodeAnalysis>;
  order: string[];
  parameters: number;
  activation_bytes: number;
  depth: number;
  errors: Issue[];
  warnings: Issue[];
  outputs: string[];
};

type Definition = {
  title: string;
  subtitle: string;
  category: string;
  color: string;
  symbol: string;
  defaults: Params;
};
const recurrentDefaults: Params = { hidden_size: 64, layers: 1, bias: 1, bidirectional: 0, return_sequences: 1, dropout: 0 };
const spatialRecurrentDefaults: Params = { hidden_size: 32, layers: 1, bias: 1, return_sequences: 1, dropout: 0, kernel: 3, dilation: 1 };
export const catalog: Record<Kind, Definition> = {
  grid_sequence_input: {
    title: "Grid sequence", subtitle: "Video & spatial time series", category: "Inputs",
    color: "#8ecdf4", symbol: "▥", defaults: { steps: 8, channels: 3, height: 32, width: 32 },
  },
  rnn: {
    title: "RNN", subtitle: "Recurrent hidden state", category: "Recurrent", color: "#84cfc0", symbol: "◌",
    defaults: { ...recurrentDefaults, activation: "tanh" },
  },
  gru: {
    title: "GRU", subtitle: "Reset & update gates", category: "Recurrent", color: "#a9b6eb", symbol: "◎",
    defaults: { ...recurrentDefaults },
  },
  lstm: {
    title: "LSTM", subtitle: "Hidden state & cell memory", category: "Recurrent", color: "#d6b18d", symbol: "⊚",
    defaults: { ...recurrentDefaults },
  },
  conv_rnn: {
    title: "ConvRNN", subtitle: "Recurrent spatial features", category: "Recurrent", color: "#84cfc0", symbol: "◌",
    defaults: { ...spatialRecurrentDefaults, activation: "tanh" },
  },
  conv_gru: {
    title: "ConvGRU", subtitle: "Gated spatial memory", category: "Recurrent", color: "#a9b6eb", symbol: "◎",
    defaults: { ...spatialRecurrentDefaults },
  },
  conv_lstm: {
    title: "ConvLSTM", subtitle: "Spatial hidden & cell memory", category: "Recurrent", color: "#d6b18d", symbol: "⊚",
    defaults: { ...spatialRecurrentDefaults },
  },
  grid_input: {
    title: "Grid input",
    subtitle: "Images & spatial data",
    category: "Inputs",
    color: "#8ecdf4",
    symbol: "▦",
    defaults: { width: 32, height: 32, channels: 3 },
  },
  feature_input: {
    title: "Feature input",
    subtitle: "A vector of any length",
    category: "Inputs",
    color: "#8ecdf4",
    symbol: "⋮",
    defaults: { features: 12 },
  },
  conv2d: {
    title: "Convolution",
    subtitle: "Learn spatial features",
    category: "Layers",
    color: "#7acfd6",
    symbol: "▧",
    defaults: {
      channels: 32,
      kernel: 3,
      stride: 1,
      padding: 1,
      dilation: 1,
      groups: 1,
      activation: "relu",
    },
  },
  dense: {
    title: "Perceptron",
    subtitle: "Fully connected layer",
    category: "Layers",
    color: "#baabec",
    symbol: "⋈",
    defaults: { units: 64, activation: "relu", bias: 1 },
  },
  maxpool2d: {
    title: "Max pooling",
    subtitle: "Reduce spatial dimensions",
    category: "Layers",
    color: "#88abea",
    symbol: "⊞",
    defaults: { kernel: 2, stride: 2, padding: 0, dilation: 1 },
  },
  padding: {
    title: "Padding",
    subtitle: "Extend the spatial boundary",
    category: "Layers",
    color: "#d6bf83",
    symbol: "▣",
    defaults: { padding: 1 },
  },
  upsample2d: {
    title: "Resize grid", subtitle: "Align a spatial feature map", category: "Layers",
    color: "#d6bf83", symbol: "⤢", defaults: { height: 45, width: 49, mode: "bilinear", align_corners: 1 },
  },
  convnext_stage: {
    title: "ConvNeXt stage",
    subtitle: "Repeated residual channel MLPs",
    category: "Layers",
    color: "#84cfc0",
    symbol: "▥",
    defaults: {
      blocks: 3,
      kernel: 7,
      expansion: 4,
      layer_scale: 1e-6,
      eps: 1e-6,
      drop_path: 0,
    },
  },
  layernorm: {
    title: "Layer normalization",
    subtitle: "Normalize channels per position",
    category: "Layers",
    color: "#aeb9c8",
    symbol: "μ",
    defaults: { eps: 1e-6 },
  },
  global_avgpool: {
    title: "Global average pool",
    subtitle: "One feature per channel",
    category: "Layers",
    color: "#88abea",
    symbol: "⟨·⟩",
    defaults: {},
  },
  flatten: {
    title: "Flatten",
    subtitle: "Grid → feature vector",
    category: "Composition",
    color: "#a0b7c5",
    symbol: "≡",
    defaults: {},
  },
  concat: {
    title: "Concatenate",
    subtitle: "Join incoming branches",
    category: "Composition",
    color: "#e0b47e",
    symbol: "⋃",
    defaults: {},
  },
  residual: {
    title: "Residual stream",
    subtitle: "Add a skip connection",
    category: "Composition",
    color: "#e8a286",
    symbol: "↗",
    defaults: { activation: "none" },
  },
  classification: {
    title: "Classification",
    subtitle: "One-hot class predictions",
    category: "Outputs",
    color: "#e9c46a",
    symbol: "◉",
    defaults: { units: 10 },
  },
  regression: {
    title: "Regression",
    subtitle: "Continuous predictions",
    category: "Outputs",
    color: "#e9c46a",
    symbol: "∿",
    defaults: { units: 1 },
  },
  embedding: {
    title: "Embedding output",
    subtitle: "Expose backbone features",
    category: "Outputs",
    color: "#e9c46a",
    symbol: "⟶",
    defaults: {},
  },
  sequence_input: {
    title: "Sequence input",
    subtitle: "Time steps / tokens × features",
    category: "Inputs",
    color: "#8ecdf4",
    symbol: "▤",
    defaults: { tokens: 64, features: 128 },
  },
  token_input: {
    title: "Token IDs",
    subtitle: "Integer vocabulary indices",
    category: "Inputs",
    color: "#8ecdf4",
    symbol: "#",
    defaults: { tokens: 64 },
  },
  token_embedding: {
    title: "Token embedding",
    subtitle: "Vocabulary IDs → vectors",
    category: "Attention",
    color: "#b0b8ef",
    symbol: "E",
    defaults: { vocab: 32000, features: 128 },
  },
  patch_embed: {
    title: "Patch embedding",
    subtitle: "Image → visual tokens",
    category: "Attention",
    color: "#7acfd6",
    symbol: "▦",
    defaults: {
      patch: 16,
      features: 192,
      cls_token: 1,
      registers: 0,
      mask_token: 0,
    },
  },
  position_encoding: {
    title: "Position encoding",
    subtitle: "Learned or sinusoidal positions",
    category: "Attention",
    color: "#c4b2df",
    symbol: "∿",
    defaults: { method: "sinusoidal" },
  },
  attention: {
    title: "Self-attention",
    subtitle: "Multi-head token mixing",
    category: "Attention",
    color: "#d79fdb",
    symbol: "⊙",
    defaults: {
      heads: 8,
      causal: 0,
      qkv_bias: 1,
      proj_bias: 1,
      mask_k_bias: 0,
      dropout: 0,
      position: "none",
      rope_base: 10000,
    },
  },
  cross_attention: {
    title: "Cross-attention",
    subtitle: "Queries attend to context",
    category: "Attention",
    color: "#ddadd2",
    symbol: "⇄",
    defaults: {
      heads: 8,
      causal: 0,
      qkv_bias: 1,
      proj_bias: 1,
      mask_k_bias: 0,
      dropout: 0,
      position: "none",
      rope_base: 10000,
    },
  },
  feedforward: {
    title: "Token feed-forward",
    subtitle: "GELU, ReLU, SiLU or SwiGLU",
    category: "Attention",
    color: "#baabec",
    symbol: "⋈",
    defaults: { expansion: 4, activation: "gelu", ffn_bias: 1, dropout: 0 },
  },
  transformer_block: {
    title: "Transformer blocks",
    subtitle: "Attention + MLP + residuals",
    category: "Attention",
    color: "#d79fdb",
    symbol: "⫶",
    defaults: {
      blocks: 2,
      heads: 8,
      mode: "encoder",
      expansion: 4,
      activation: "gelu",
      norm_style: "pre",
      eps: 1e-5,
      layer_scale: 0,
      qkv_bias: 1,
      proj_bias: 1,
      ffn_bias: 1,
      mask_k_bias: 0,
      dropout: 0,
      position: "none",
      rope_base: 10000,
    },
  },
  token_pool: {
    title: "Token selection / pool",
    subtitle: "CLS, mean or patch features",
    category: "Composition",
    color: "#88abea",
    symbol: "⟨T⟩",
    defaults: { method: "mean", prefix: 1 },
  },
  token_head: {
    title: "Token predictions",
    subtitle: "One class per token",
    category: "Outputs",
    color: "#e9c46a",
    symbol: "T→",
    defaults: { vocab: 32000 },
  },
  deepseek_stage: {
    title: "DeepSeek V4.1 stage",
    subtitle: "Sparse attention · MoE · mHC",
    category: "Reference",
    color: "#89bfeb",
    symbol: "DS",
    defaults: { start: 0, blocks: 2 },
  },
  deepseek_vision: {
    title: "DeepSeek vision",
    subtitle: "32-layer ViT + image aligner",
    category: "Reference",
    color: "#7acfd6",
    symbol: "▦",
    defaults: {},
  },
  deepseek_bridge: {
    title: "Image / text bridge",
    subtitle: "Inject image slots · 4 streams",
    category: "Reference",
    color: "#e8a286",
    symbol: "⇉",
    defaults: {},
  },
  deepseek_head: {
    title: "DeepSeek LM head",
    subtitle: "Collapse streams → vocabulary",
    category: "Outputs",
    color: "#e9c46a",
    symbol: "DS→",
    defaults: {},
  },
  deepseek_draft: {
    title: "DSpark draft head",
    subtitle: "3 stages · 5-token draft",
    category: "Outputs",
    color: "#e9c46a",
    symbol: "↠",
    defaults: {},
  },
  activation: { title: "Activation", subtitle: "Declared pointwise function", category: "Snapshot", color: "#baabec", symbol: "ƒ", defaults: {} },
  add: { title: "Add / residual", subtitle: "Declared addition", category: "Snapshot", color: "#e8a286", symbol: "+", defaults: {} },
  dot: { title: "Dot product", subtitle: "Query and action score", category: "Snapshot", color: "#ddadd2", symbol: "·", defaults: {} },
  gather: { title: "Gather", subtitle: "Entity or index selection", category: "Snapshot", color: "#a0b7c5", symbol: "⋮", defaults: {} },
  globalavgpool2d: { title: "Global mean", subtitle: "Declared spatial average", category: "Snapshot", color: "#88abea", symbol: "⟨·⟩", defaults: {} },
  globalmaxpool2d: { title: "Global maximum", subtitle: "Declared spatial maximum", category: "Snapshot", color: "#88abea", symbol: "max", defaults: {} },
  legal_policy: { title: "Legal-action policy", subtitle: "Variable legal-action probabilities", category: "Snapshot", color: "#e9c46a", symbol: "π", defaults: {} },
  outcome_transform: { title: "Outcome transform", subtitle: "Signed and draw scores → logits", category: "Snapshot", color: "#ddadd2", symbol: "±", defaults: {} },
  standardize: { title: "Standardization", subtitle: "Declared feature normalization", category: "Snapshot", color: "#aeb9c8", symbol: "σ", defaults: {} },
  checkpoint_module: { title: "Checkpoint module", subtitle: "Stored tensor group", category: "Snapshot", color: "#89bfeb", symbol: "PT", defaults: {} },
};

export const formatNumber = (value: number) =>
  Intl.NumberFormat("en", {
    notation: value >= 100000 ? "compact" : "standard",
    maximumFractionDigits: 1,
  }).format(value);
export const shapeText = (shape?: (number | string)[]) =>
  shape ? shape.map((v) => v.toLocaleString()).join(" × ") : "—";
export const bytesText = (value: number) =>
  value >= 1e15
    ? `${(value / 1e15).toFixed(1)} PB`
    : value >= 1e12
      ? `${(value / 1e12).toFixed(1)} TB`
      : value >= 1e9
        ? `${(value / 1e9).toFixed(1)} GB`
        : value >= 1e6
          ? `${(value / 1e6).toFixed(1)} MB`
          : `${(value / 1e3).toFixed(1)} KB`;
export const isInput = (kind: Kind) =>
  ["grid_input", "feature_input", "sequence_input", "token_input", "grid_sequence_input"].includes(
    kind,
  );
export const isRecurrent = (kind: Kind): kind is RecurrentKind =>
  (recurrentKinds as readonly string[]).includes(kind);
export const isOutput = (kind: Kind) =>
  [
    "classification",
    "regression",
    "embedding",
    "token_head",
    "deepseek_head",
    "deepseek_draft",
  ].includes(kind);

export function makeNode(
  kind: Kind,
  x = 0,
  y = 0,
  id: string = crypto.randomUUID(),
): LayerNode {
  return {
    id,
    type: "layer",
    position: { x, y },
    data: {
      kind,
      label: catalog[kind].title,
      params: { ...catalog[kind].defaults },
    },
  };
}

function node(
  id: string,
  kind: Kind,
  label: string,
  x: number,
  y: number,
  params: Params = {},
): LayerNode {
  const result = makeNode(kind, x, y, id);
  result.data.label = label;
  result.data.params = { ...result.data.params, ...params };
  return result;
}
const edges = (pairs: [string, string][]): Edge[] =>
  pairs.map(([source, target]) => ({
    id: `${source}-${target}`,
    source,
    target,
  }));

export const dinoVariants = {
  tiny: { title: "Tiny", depths: [3, 3, 9, 3], dims: [96, 192, 384, 768] },
  small: { title: "Small", depths: [3, 3, 27, 3], dims: [96, 192, 384, 768] },
  base: { title: "Base", depths: [3, 3, 27, 3], dims: [128, 256, 512, 1024] },
  large: { title: "Large", depths: [3, 3, 27, 3], dims: [192, 384, 768, 1536] },
};
export type DinoVariant = keyof typeof dinoVariants;
export const templateInfo: Record<
  string,
  { title: string; subtitle: string; source: string }
> = Object.fromEntries(
  Object.entries(dinoVariants).map(([key, config]) => [
    `dinov3_convnext_${key}`,
    {
      title: `DINOv3 ConvNeXt ${config.title}`,
      subtitle: `${config.depths.join(" / ")} blocks · ${config.dims.at(-1)}-D embedding`,
      source:
        "https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py",
    },
  ]),
);

export function makeDinoConvNeXt(variant: DinoVariant = "tiny"): ModelGraph {
  const config = dinoVariants[variant];
  const nodes: LayerNode[] = [];
  const links: [string, string][] = [];
  const add = (id: string, kind: Kind, label: string, params: Params = {}) => {
    if (nodes.length) links.push([nodes.at(-1)!.id, id]);
    nodes.push(node(id, kind, label, nodes.length * 245, 80, params));
  };
  add("image", "grid_input", "Image · RGB", {
    width: 256,
    height: 256,
    channels: 3,
  });
  add("stem", "conv2d", "Patchify stem · 4×4", {
    channels: config.dims[0],
    kernel: 4,
    stride: 4,
    padding: 0,
    activation: "none",
  });
  add("stem_norm", "layernorm", "Stem normalization");
  config.depths.forEach((blocks, i) => {
    if (i > 0) {
      add(`down${i}_norm`, "layernorm", `Transition ${i} · normalize`);
      add(`down${i}`, "conv2d", `Transition ${i} · downsample`, {
        channels: config.dims[i],
        kernel: 2,
        stride: 2,
        padding: 0,
        activation: "none",
      });
    }
    add(
      `stage${i + 1}`,
      "convnext_stage",
      `Stage ${i + 1} · ${blocks} blocks`,
      { blocks, stage: i + 1 },
    );
  });
  add("global_pool", "global_avgpool", "Global average");
  add("final_norm", "layernorm", "Final normalization");
  add("embedding", "embedding", `${config.dims.at(-1)}-D image embedding`);
  return {
    version: 1,
    name: `DINOv3 · ConvNeXt ${config.title}`,
    template: `dinov3_convnext_${variant}`,
    nodes,
    edges: edges(links),
  };
}

export const dinoVitVariants = {
  small: { title: "S/16", dim: 384, heads: 6, depth: 12 },
  base: { title: "B/16", dim: 768, heads: 12, depth: 12 },
  large: { title: "L/16", dim: 1024, heads: 16, depth: 24 },
};
export type DinoVitVariant = keyof typeof dinoVitVariants;

function chain(
  name: string,
  template: string,
  specs: [string, Kind, string, Params?][],
): ModelGraph {
  const nodes = specs.map(([id, kind, label, params], i) =>
    node(id, kind, label, i * 250, 80, params),
  );
  return {
    version: 1,
    name,
    template,
    nodes,
    edges: edges(nodes.slice(1).map((n, i) => [nodes[i].id, n.id])),
  };
}

export function makeTransformer(
  mode: "encoder" | "decoder" | "seq2seq" = "encoder",
): ModelGraph {
  if (mode === "seq2seq") {
    const graph = chain(
      "Transformer · encoder–decoder",
      "transformer_seq2seq",
      [
        ["source", "token_input", "Source tokens", { tokens: 64 }],
        [
          "source_embed",
          "token_embedding",
          "Source embedding",
          { vocab: 8192, features: 256 },
        ],
        ["source_pos", "position_encoding", "Source positions"],
        [
          "encoder",
          "transformer_block",
          "Encoder · 3 blocks",
          { blocks: 3, heads: 8 },
        ],
        ["memory", "layernorm", "Encoder memory", { eps: 1e-5 }],
      ],
    );
    const decoder = chain("", "", [
      [
        "target",
        "token_input",
        "Target tokens · shifted right",
        { tokens: 32 },
      ],
      [
        "target_embed",
        "token_embedding",
        "Target embedding",
        { vocab: 8192, features: 256 },
      ],
      ["target_pos", "position_encoding", "Target positions"],
      [
        "decoder",
        "transformer_block",
        "Decoder · 3 blocks",
        { blocks: 3, mode: "cross", heads: 8 },
      ],
      ["decoder_norm", "layernorm", "Final normalization", { eps: 1e-5 }],
      ["tokens", "token_head", "Next-token logits", { vocab: 8192 }],
    ]);
    decoder.nodes.forEach((n) => {
      n.position.y = 310;
    });
    graph.nodes.push(...decoder.nodes);
    // Query is always the first dependency; encoder memory is the second.
    graph.edges.push(...decoder.edges, ...edges([["memory", "decoder"]]));
    return graph;
  }
  if (mode === "decoder")
    return chain("Transformer · causal language model", "transformer_decoder", [
      ["tokens", "token_input", "Prompt tokens", { tokens: 128 }],
      [
        "embed",
        "token_embedding",
        "Token embedding",
        { vocab: 32000, features: 256 },
      ],
      [
        "positions",
        "position_encoding",
        "Learned positions",
        { method: "learned" },
      ],
      [
        "decoder",
        "transformer_block",
        "Causal decoder · 6 blocks",
        { blocks: 6, mode: "decoder", heads: 8 },
      ],
      ["norm", "layernorm", "Final normalization", { eps: 1e-5 }],
      ["logits", "token_head", "Next-token logits", { vocab: 32000 }],
    ]);
  return chain("Transformer · sequence encoder", "transformer_encoder", [
    [
      "sequence",
      "sequence_input",
      "Embedded sequence",
      { tokens: 64, features: 256 },
    ],
    ["positions", "position_encoding", "Sinusoidal positions"],
    [
      "encoder",
      "transformer_block",
      "Bidirectional encoder · 6 blocks",
      { blocks: 6, heads: 8 },
    ],
    ["norm", "layernorm", "Final normalization", { eps: 1e-5 }],
    ["pool", "token_pool", "Mean token features"],
    ["classes", "classification", "Sequence classes", { units: 10 }],
  ]);
}

export function makeViT(): ModelGraph {
  return chain("Vision Transformer · Tiny / 16", "vit_tiny", [
    ["image", "grid_input", "Image · RGB", { width: 224, height: 224 }],
    [
      "patches",
      "patch_embed",
      "16×16 patches + CLS",
      { patch: 16, features: 192 },
    ],
    [
      "positions",
      "position_encoding",
      "Learned visual positions",
      { method: "learned" },
    ],
    [
      "encoder",
      "transformer_block",
      "ViT encoder · 12 blocks",
      { blocks: 12, heads: 3, eps: 1e-6 },
    ],
    ["norm", "layernorm", "Final normalization", { eps: 1e-6 }],
    ["cls", "token_pool", "Select CLS token", { method: "cls" }],
    ["classes", "classification", "Image classes", { units: 1000 }],
  ]);
}

export function makeDinoViT(variant: DinoVitVariant = "small"): ModelGraph {
  const config = dinoVitVariants[variant];
  const graph = chain(`DINOv3 · ViT-${config.title}`, `dinov3_vit_${variant}`, [
    ["image", "grid_input", "Image · RGB", { width: 224, height: 224 }],
    [
      "patches",
      "patch_embed",
      "Patch / CLS / storage tokens",
      {
        patch: 16,
        features: config.dim,
        cls_token: 1,
        registers: 4,
        mask_token: 1,
      },
    ],
    [
      "encoder",
      "transformer_block",
      `RoPE encoder · ${config.depth} blocks`,
      {
        blocks: config.depth,
        heads: config.heads,
        position: "rope2d",
        rope_base: 100,
        layer_scale: 1e-5,
        eps: 1e-5,
        mask_k_bias: 1,
      },
    ],
    ["norm", "layernorm", "Shared final normalization", { eps: 1e-5 }],
    ["cls", "token_pool", "Select CLS token", { method: "cls" }],
    ["embedding", "embedding", `${config.dim}-D image embedding`],
  ]);
  graph.nodes.push(
    node(
      "patch_tokens",
      "token_pool",
      "Select normalized patch tokens",
      1000,
      310,
      { method: "patches", prefix: 5 },
    ),
    node("patch_output", "embedding", "Dense visual features", 1250, 310),
  );
  graph.edges.push(
    ...edges([
      ["norm", "patch_tokens"],
      ["patch_tokens", "patch_output"],
    ]),
  );
  return graph;
}

export function makeDeepSeekFlash(): ModelGraph {
  const graph: ModelGraph = {
    version: 1,
    name: "DeepSeek · V4.1 Flash 1017",
    template: "deepseek_v41_flash_1017",
    nodes: [
      node("tokens", "token_input", "Text + image placeholder IDs", 0, 80, {
        tokens: 128,
      }),
      node("embed", "token_embedding", "Shared vocabulary embedding", 250, 80, {
        vocab: 129280,
        features: 5120,
      }),
      node("image", "grid_input", "Image · reduced preview", 0, 310, {
        width: 252,
        height: 252,
        channels: 3,
      }),
      node("vision", "deepseek_vision", "ViT · 32 blocks + aligner", 250, 310),
      node(
        "bridge",
        "deepseek_bridge",
        "Inject image · expand 4 streams",
        500,
        80,
      ),
    ],
    edges: edges([
      ["tokens", "embed"],
      ["image", "vision"],
      ["embed", "bridge"],
      ["vision", "bridge"],
    ]),
  };
  const starts = [0, 2, 8, 14, 20, 24, 28, 32, 36, 40];
  for (let i = 0; i < starts.length - 1; i++) {
    const start = starts[i],
      end = starts[i + 1] - 1,
      id = `layers_${start}_${end}`;
    const previous = i ? `layers_${starts[i - 1]}_${start - 1}` : "bridge";
    graph.nodes.push(
      node(
        id,
        "deepseek_stage",
        `${start < 20 ? "Encoder" : "Decoder"} · layers ${start}–${end}`,
        750 + i * 270,
        80,
        { start, blocks: end - start + 1 },
      ),
    );
    graph.edges.push(...edges([[previous, id]]));
  }
  const last = graph.nodes.at(-1)!;
  graph.nodes.push(
    node(
      "logits",
      "deepseek_head",
      "Main language-model logits",
      last.position.x + 270,
      80,
    ),
    node(
      "draft",
      "deepseek_draft",
      "DSpark · 3 stages / 5 tokens",
      last.position.x + 270,
      310,
    ),
  );
  graph.edges.push(
    ...edges([
      [last.id, "logits"],
      [last.id, "draft"],
    ]),
  );
  return graph;
}

export function makeRecurrent(kind: RecurrentKind): ModelGraph {
  const spatial = kind.startsWith("conv_");
  const cell = kind.replace("conv_", "") as RecurrentKind;
  const nodes = [
    spatial
      ? node("input", "grid_sequence_input", "Spatial observations", 0, 80, { steps: 8, channels: 3, height: 16, width: 24 })
      : node("input", "sequence_input", "Observations over time", 0, 80, { tokens: 24, features: 16 }),
    node("memory", kind, `${catalog[kind].title} · temporal memory`, 280, 80,
      { hidden_size: spatial ? 16 : 64, layers: 2, return_sequences: spatial ? 1 : 0 }),
  ];
  if (spatial) nodes.push(
    node("pool", "global_avgpool", "Spatial summary per step", 560, 80),
    node("readout", cell, `${catalog[cell].title} · sequence readout`, 840, 80, { hidden_size: 32, return_sequences: 0 }),
  );
  nodes.push(node("classes", "classification", "Sequence classification", spatial ? 1120 : 560, 80, { units: 5 }));
  return { version: 1, name: `${catalog[kind].title} · ${spatial ? "spatial" : "sequence"} memory`,
    template: kind, nodes, edges: edges(nodes.slice(1).map((n, i) => [nodes[i].id, n.id])) };
}

Object.assign(templateInfo, {
  ...Object.fromEntries(recurrentKinds.map(kind => [kind, {
    title: `${catalog[kind].title} · ${kind.startsWith("conv_") ? "spatial" : "sequence"} memory`,
    subtitle: kind.startsWith("conv_") ? "Grid sequence → spatial memory → recurrent readout" : "Two recurrent layers · sequence classification",
    source: "",
  }])),
  catan_convnext_small: {
    title: "Catan · ConvNeXt Small",
    subtitle: "Untrained · 1.03M params · 32 / 48 / 64 · PPG heads",
    source: "/api/templates/catan_convnext_small/encoder.py",
  },
  catan_original: {
    title: "Catan · original residual CNN",
    subtitle: "Supplied schematic · 3.68M params · 8 dilated blocks per encoder",
    source: "",
  },
  transformer_encoder: {
    title: "Transformer · encoder",
    subtitle: "6 blocks · 8 heads · sequence classification",
    source: "https://arxiv.org/abs/1706.03762",
  },
  transformer_decoder: {
    title: "Transformer · causal decoder",
    subtitle: "6 blocks · learned positions · language model",
    source: "https://arxiv.org/abs/1706.03762",
  },
  transformer_seq2seq: {
    title: "Transformer · encoder–decoder",
    subtitle: "Cross-attention · source and target tokens",
    source: "https://arxiv.org/abs/1706.03762",
  },
  vit_tiny: {
    title: "Vision Transformer · Tiny / 16",
    subtitle: "12 blocks · 3 heads · 16×16 patches",
    source: "https://arxiv.org/abs/2010.11929",
  },
  deepseek_v41_flash_1017: {
    title: "DeepSeek V4.1 Flash 1017",
    subtitle: "Reference · MoE / Engram / 4 residual streams",
    source: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017",
  },
  ...Object.fromEntries(
    Object.entries(dinoVitVariants).map(([key, config]) => [
      `dinov3_vit_${key}`,
      {
        title: `DINOv3 ViT-${config.title}`,
        subtitle: `${config.depth} blocks · 2D RoPE · 4 storage tokens`,
        source:
          "https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py",
      },
    ]),
  ),
});

export const examples: Record<string, () => ModelGraph> = {
  ...Object.fromEntries(recurrentKinds.map(kind => [kind, () => makeRecurrent(kind)])),
  catan_convnext_small: () => parseGraph(JSON.stringify(catanSmall)),
  catan_original: () => parseGraph(JSON.stringify(catanOriginal)),
  transformer_encoder: () => makeTransformer("encoder"),
  transformer_decoder: () => makeTransformer("decoder"),
  transformer_seq2seq: () => makeTransformer("seq2seq"),
  vit_tiny: makeViT,
  dinov3_vit_small: () => makeDinoViT("small"),
  dinov3_vit_base: () => makeDinoViT("base"),
  dinov3_vit_large: () => makeDinoViT("large"),
  deepseek_v41_flash_1017: makeDeepSeekFlash,
  dinov3_convnext_tiny: () => makeDinoConvNeXt("tiny"),
  dinov3_convnext_small: () => makeDinoConvNeXt("small"),
  dinov3_convnext_base: () => makeDinoConvNeXt("base"),
  dinov3_convnext_large: () => makeDinoConvNeXt("large"),
  multimodal: () => ({
    version: 1,
    name: "Vision + features",
    nodes: [
      node("image", "grid_input", "Image", 0, 30),
      node("conv1", "conv2d", "Spatial features", 240, 30, { channels: 16 }),
      node("pool1", "maxpool2d", "Downsample", 480, 30),
      node("conv2", "conv2d", "Deep features", 720, 30),
      node("pool2", "maxpool2d", "Downsample 2", 960, 30),
      node("features", "feature_input", "Context features", 960, 240),
      node("join", "concat", "Feature union", 1200, 110),
      node("dense", "dense", "Hidden layer", 1440, 110),
      node("classes", "classification", "Class prediction", 1680, 110),
    ],
    edges: edges([
      ["image", "conv1"],
      ["conv1", "pool1"],
      ["pool1", "conv2"],
      ["conv2", "pool2"],
      ["pool2", "join"],
      ["features", "join"],
      ["join", "dense"],
      ["dense", "classes"],
    ]),
  }),
  residual: () => ({
    version: 1,
    name: "Residual vision",
    nodes: [
      node("image", "grid_input", "Image", 0, 70),
      node("stem", "conv2d", "Stem", 240, 70),
      node("conv1", "conv2d", "Residual block · 1", 480, 160),
      node("conv2", "conv2d", "Residual block · 2", 720, 160, {
        activation: "none",
      }),
      node("skip", "residual", "Skip connection", 960, 70, {
        activation: "relu",
      }),
      node("pool", "maxpool2d", "Downsample", 1200, 70),
      node("hidden", "dense", "Hidden layer", 1440, 70),
      node("classes", "classification", "Class prediction", 1680, 70),
    ],
    edges: edges([
      ["image", "stem"],
      ["stem", "conv1"],
      ["conv1", "conv2"],
      ["conv2", "skip"],
      ["stem", "skip"],
      ["skip", "pool"],
      ["pool", "hidden"],
      ["hidden", "classes"],
    ]),
  }),
  mlp: () => ({
    version: 1,
    name: "Perceptron regressor",
    nodes: [
      node("features", "feature_input", "Observations", 0, 80, {
        features: 24,
      }),
      node("hidden1", "dense", "Hidden layer · 1", 260, 80, {
        units: 128,
        activation: "gelu",
      }),
      node("hidden2", "dense", "Hidden layer · 2", 520, 80, {
        units: 64,
        activation: "gelu",
      }),
      node("hidden3", "dense", "Hidden layer · 3", 780, 80, { units: 16 }),
      node("value", "regression", "Predicted value", 1040, 80),
    ],
    edges: edges([
      ["features", "hidden1"],
      ["hidden1", "hidden2"],
      ["hidden2", "hidden3"],
      ["hidden3", "value"],
    ]),
  }),
  dilated: () => ({
    version: 1,
    name: "Dilated feature extractor",
    nodes: [
      node("image", "grid_input", "Image", 0, 60, { width: 64, height: 64 }),
      node("pad", "padding", "Spatial padding", 240, 60, { padding: 2 }),
      node("conv1", "conv2d", "Dilated convolution", 480, 60, {
        dilation: 2,
        padding: 0,
      }),
      node("pool", "maxpool2d", "Downsample", 720, 60),
      node("conv2", "conv2d", "Context convolution", 960, 60, {
        channels: 64,
        dilation: 3,
        padding: 3,
      }),
      node("hidden", "dense", "Hidden layer", 1200, 60),
      node("classes", "classification", "Class prediction", 1440, 60),
    ],
    edges: edges([
      ["image", "pad"],
      ["pad", "conv1"],
      ["conv1", "pool"],
      ["pool", "conv2"],
      ["conv2", "hidden"],
      ["hidden", "classes"],
    ]),
  }),
  blank: () => ({
    version: 1,
    name: "Untitled architecture",
    nodes: [],
    edges: [],
  }),
};

export function cleanGraph(graph: ModelGraph): ModelGraph {
  return {
    version: 1,
    name: graph.name,
    ...(graph.template ? { template: graph.template } : {}),
    ...(graph.mode === "schematic" ? { mode: graph.mode } : {}),
    ...(graph.metadata ? { metadata: graph.metadata } : {}),
    nodes: graph.nodes.map(({ id, position, data }) => ({
      id,
      type: "layer",
      position,
      data: { ...data, params: { ...data.params } },
    })),
    edges: graph.edges.map(({ id, source, target, label, data }) => ({
      id,
      source,
      target,
      ...(label ? { label } : {}),
      ...(data ? { data } : {}),
    })),
  };
}

export function parseGraph(raw: string): ModelGraph {
  const value = JSON.parse(raw);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw Error("Expected a vizml graph object.");
  const schematic = value.mode === "schematic" || (value.mode == null && Array.isArray(value.nodes) && value.nodes.length > 0 && value.nodes.every((n: LayerNode) => Array.isArray(n?.data?.output_shape) && Number.isInteger(n?.data?.parameter_count)));
  if (schematic) value.mode = "schematic";
  if ((value.mode != null && !["builder", "schematic"].includes(value.mode)) || (value.metadata != null && (typeof value.metadata !== "object" || Array.isArray(value.metadata)))) throw Error("Invalid graph metadata or mode.");
  if (
    value.template != null &&
    (typeof value.template !== "string" || value.template.length > 100)
  )
    throw Error("Invalid template metadata.");
  if (
    value.version !== 1 ||
    typeof value.name !== "string" ||
    value.name.length > 100 ||
    !Array.isArray(value.nodes) ||
    !Array.isArray(value.edges) ||
    value.nodes.length > 300 ||
    value.edges.length > 2000
  )
    throw Error("Expected a vizml graph (version 1, up to 300 layers).");
  const ids = new Set();
  for (const n of value.nodes) {
    if (
      typeof n.id !== "string" ||
      !n.id ||
      ids.has(n.id) ||
      !n.data ||
      !Object.hasOwn(catalog, n.data.kind) ||
      typeof n.data.label !== "string" ||
      !n.data.label ||
      n.data.label.length > 100 ||
      !n.data.params ||
      typeof n.data.params !== "object" ||
      Array.isArray(n.data.params) ||
      !Number.isFinite(n.position?.x) ||
      !Number.isFinite(n.position?.y)
    )
      throw Error("This file contains an invalid layer.");
    if (!schematic && (
      Object.values(n.data.params).some(
        (v) => !["number", "string"].includes(typeof v),
      )
    ))
      throw Error("Layer parameters must be numbers or strings.");
    ids.add(n.id);
  }
  const edgeIds = new Set();
  for (const edge of value.edges) {
    if (
      typeof edge.id !== "string" ||
      edgeIds.has(edge.id) ||
      !ids.has(edge.source) ||
      !ids.has(edge.target)
    )
      throw Error("This file contains an invalid connection.");
    edgeIds.add(edge.id);
  }
  return cleanGraph(value);
}

/** Insert one fully connected layer, routing all existing outgoing branches through it. */
export function insertDenseAfter(graph: ModelGraph, id: string): { graph: ModelGraph; id: string } | null {
  const source = graph.nodes.find(node => node.id === id);
  if (!source || source.data.kind !== "dense" || graph.nodes.length >= 300 || graph.edges.length >= 2000) return null;
  const next = makeNode("dense", source.position.x + 230, source.position.y);
  const downstream = new Set<string>();
  const queue = graph.edges.filter(edge => edge.source === id).map(edge => edge.target);
  for (let i = 0; i < queue.length; i++) {
    const child = queue[i];
    if (downstream.has(child) || child === id) continue;
    downstream.add(child);
    queue.push(...graph.edges.filter(edge => edge.source === child).map(edge => edge.target));
  }
  const nodes = graph.nodes.map(node => downstream.has(node.id) && node.position.x > source.position.x
    ? { ...node, position: { ...node.position, x: node.position.x + 230 } } : node);
  // Other branches can already occupy the new column. Keep those branches in
  // place and choose the nearest free row instead of drawing cards on top of
  // one another. The clearance includes the card and its connection handles.
  const rowStep = 140;
  const occupied = (y: number) => nodes.some(node =>
    Math.abs(node.position.x - next.position.x) < 195 &&
    Math.abs(node.position.y - y) < 120);
  if (occupied(next.position.y)) {
    for (let row = 1; row <= nodes.length + 1; row++) {
      const candidates = [source.position.y + row * rowStep, source.position.y - row * rowStep];
      const free = candidates.find(y => !occupied(y));
      if (free !== undefined) { next.position.y = free; break; }
    }
  }
  next.data = { ...next.data, label: "New dense layer", params: { ...next.data.params,
    units: source.data.params.units ?? next.data.params.units,
    bias: source.data.params.bias ?? next.data.params.bias } };
  if (graph.mode === "schematic") next.data = { ...next.data,
    output_shape: Array.isArray(source.data.output_shape) ? [...source.data.output_shape] : [1],
    parameter_count: 0, declaration_pending: true,
    description: "New schematic layer. Review its declared output shape and parameter count." };
  const edges = graph.edges.map(edge => edge.source === id ? { ...edge, source: next.id } : edge);
  edges.push({ id: crypto.randomUUID(), source: id, target: next.id });
  return { graph: editedSchematic({ ...graph, nodes: [...nodes, next], edges }), id: next.id };
}

export function canConnect(
  graph: ModelGraph,
  source: string,
  target: string,
): boolean {
  if (
    source === target ||
    !graph.nodes.some((n) => n.id === source && (graph.mode === "schematic" || !isOutput(n.data.kind))) ||
    !graph.nodes.some((n) => n.id === target && (graph.mode === "schematic" || !isInput(n.data.kind))) ||
    graph.edges.some((e) => e.source === source && e.target === target)
  )
    return false;
  const visited = new Set<string>();
  const queue = [target];
  while (queue.length) {
    const key = queue.pop()!;
    if (key === source) return false;
    if (visited.has(key)) continue;
    visited.add(key);
    queue.push(
      ...graph.edges.filter((e) => e.source === key).map((e) => e.target),
    );
  }
  return true;
}

/** Structural schematic edits invalidate weight pairing and original totals. */
export function editedSchematic(graph: ModelGraph): ModelGraph {
  if (graph.mode !== "schematic") return graph;
  const metadata: Record<string, unknown> = { ...graph.metadata, source_edited: true };
  delete metadata.weights;
  return { ...graph, metadata };
}

export type ResidualProjectionPlan = {
  targetId: string;
  targetShape: number[];
  steps: {
    edgeId: string;
    sourceId: string;
    sourceShape: number[];
    stride: number;
  }[];
};

/** Match CNN skip paths to the deepest incoming branch using explicit 1x1 convolutions. */
export function residualProjectionPlan(
  graph: ModelGraph,
  id: string,
  analysis: Analysis | null,
): ResidualProjectionPlan | null {
  if (
    !analysis ||
    graph.nodes.find((n) => n.id === id)?.data.kind !== "residual"
  )
    return null;
  const branches = graph.edges
    .filter((e) => e.target === id)
    .map((edge) => ({ edge, info: analysis.nodes[edge.source] }));
  if (
    branches.length < 2 ||
    branches.some((b) => !b.info || b.info.shape.length !== 3)
  )
    return null;
  const target = [...branches].sort(
    (a, b) =>
      b.info.depth - a.info.depth ||
      a.info.shape[1] * a.info.shape[2] - b.info.shape[1] * b.info.shape[2],
  )[0];
  const targetShape = target.info.shape;
  if (targetShape[0] > 65535) return null;
  const steps: ResidualProjectionPlan["steps"] = [];
  for (const branch of branches) {
    const shape = branch.info.shape;
    if (shape.every((v, i) => v === targetShape[i])) continue;
    let stride = 0;
    for (let s = 1; s <= 256; s++) {
      if (
        Math.floor((shape[1] - 1) / s) + 1 === targetShape[1] &&
        Math.floor((shape[2] - 1) / s) + 1 === targetShape[2]
      ) {
        stride = s;
        break;
      }
    }
    if (!stride) return null;
    steps.push({
      edgeId: branch.edge.id,
      sourceId: branch.edge.source,
      sourceShape: shape,
      stride,
    });
  }
  return steps.length
    ? { targetId: target.edge.source, targetShape, steps }
    : null;
}

export function projectResidualSkips(
  graph: ModelGraph,
  id: string,
  analysis: Analysis | null,
): ModelGraph | null {
  const plan = residualProjectionPlan(graph, id, analysis);
  const residual = graph.nodes.find((n) => n.id === id);
  if (
    !plan ||
    !residual ||
    graph.nodes.length + plan.steps.length > 300 ||
    graph.edges.length + plan.steps.length > 2000
  )
    return null;
  const next = cleanGraph(graph);
  plan.steps.forEach((step, i) => {
    const projection = makeNode(
      "conv2d",
      residual.position.x - 240,
      residual.position.y + 200 * (i + 1),
    );
    const source = graph.nodes.find((n) => n.id === step.sourceId)!;
    projection.data.label = `${source.data.label} · skip projection`.slice(
      0,
      100,
    );
    projection.data.params = {
      ...projection.data.params,
      channels: plan.targetShape[0],
      kernel: 1,
      padding: 0,
      dilation: 1,
      stride: step.stride,
      activation: "none",
      role: "skip_projection",
    };
    next.nodes.push(projection);
    next.edges = next.edges.map((e) =>
      e.id === step.edgeId ? { ...e, source: projection.id } : e,
    );
    next.edges.push({
      id: crypto.randomUUID(),
      source: step.sourceId,
      target: projection.id,
    });
  });
  return next;
}

export function expandConvNeXtStage(
  graph: ModelGraph,
  id: string,
): ModelGraph | null {
  const original = graph.nodes.find((n) => n.id === id);
  const count = Number(original?.data.params.blocks);
  if (
    !original ||
    !["convnext_stage", "transformer_block", "deepseek_stage"].includes(
      original.data.kind,
    ) ||
    !Number.isInteger(count) ||
    count < 2 ||
    count > 96 ||
    graph.nodes.length + count - 1 > 300 ||
    graph.edges.length +
      (count - 1) * (original.data.params.mode === "cross" ? 2 : 1) >
      2000
  )
    return null;
  const next = cleanGraph(graph);
  const blocks = Array.from({ length: count }, (_, i) => {
    const block = makeNode(
      original.data.kind,
      original.position.x + i * 245,
      original.position.y,
      i === 0 ? original.id : crypto.randomUUID(),
    );
    block.data = {
      ...original.data,
      label: `${original.data.label.replace(/ · \d+ blocks$/, "").slice(0, 65)} · Block ${i + 1}/${count}`,
      params: {
        ...original.data.params,
        blocks: 1,
        ...(original.data.kind === "deepseek_stage"
          ? { start: Number(original.data.params.start) + i }
          : {}),
      },
    };
    return block;
  });
  next.nodes = next.nodes.flatMap((n) =>
    n.id === id
      ? blocks
      : [
          {
            ...n,
            position: {
              ...n.position,
              x:
                n.position.x > original.position.x
                  ? n.position.x + (count - 1) * 245
                  : n.position.x,
            },
          },
        ],
  );
  next.edges = next.edges.map((e) =>
    e.source === id ? { ...e, source: blocks.at(-1)!.id } : e,
  );
  const context =
    original.data.params.mode === "cross"
      ? graph.edges.filter((e) => e.target === id)[1]
      : undefined;
  for (let i = 1; i < count; i++) {
    next.edges.push({
      id: crypto.randomUUID(),
      source: blocks[i - 1].id,
      target: blocks[i].id,
    });
    if (context)
      next.edges.push({
        ...context,
        id: crypto.randomUUID(),
        target: blocks[i].id,
      });
  }
  return next;
}

export function autoLayout(
  graph: ModelGraph,
  analysis: Analysis | null,
): ModelGraph {
  const depths = new Map<string, number>();
  const queue = graph.nodes
    .filter((n) => !graph.edges.some((e) => e.target === n.id))
    .map((n) => n.id);
  for (const id of queue) depths.set(id, 0);
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i];
    for (const edge of graph.edges.filter((e) => e.source === id)) {
      const incoming = graph.edges.filter((e) => e.target === edge.target);
      if (
        incoming.every((e) => depths.has(e.source)) &&
        !depths.has(edge.target)
      ) {
        depths.set(
          edge.target,
          Math.max(...incoming.map((e) => depths.get(e.source)!)) + 1,
        );
        queue.push(edge.target);
      }
    }
  }
  const lanes: Record<number, number> = {};
  return {
    ...graph,
    nodes: graph.nodes.map((n) => {
      const depth = analysis?.nodes[n.id]?.depth ?? depths.get(n.id) ?? 0;
      const lane = lanes[depth] ?? 0;
      lanes[depth] = lane + 1;
      return { ...n, position: { x: depth * 250, y: lane * 200 + 40 } };
    }),
  };
}
