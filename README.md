# vizml

A local visual architecture workbench for deep learning. Compose a directed graph on a ComfyUI-style canvas and inspect the assembled model in a separately generated Three.js view. FastAPI streams shape inference, validation, export, and PyTorch verification over a WebSocket.

## Run

Requires [Bun 1.4.4+](https://bun.sh/) and [uv](https://docs.astral.sh/uv/). uv selects Python 3.12–3.13 and installs it if needed.

```sh
bun install
uv sync --python 3.12
bun run dev
```

Open [localhost:5173](http://localhost:5173). Bun serves and hot-reloads the React UI directly from the HTML entrypoint and streams `/api` and `/ws` to the FastAPI/PyTorch engine on port 8000. Vite and Node.js are not required. Keep the development process running while using the app. Graphs are autosaved in your browser. Imported checkpoints are retained locally in `.vizml/checkpoints/`; no account or database is required.

For built assets served by Bun (with the Python engine started alongside it):

```sh
bun run build
bun run start
```

Then open [localhost:5173](http://localhost:5173). Port 8000 is the Python API; its root redirects to the Bun UI. The development and production launchers shut down the engine on exit. Use `PORT` for a different UI port or `VIZML_ENGINE_PORT` for a different engine port. For separate processes, use `bun run dev:backend` and `bun run dev:frontend` (or `bun run preview` for built files); the standalone Bun server accepts `VIZML_ENGINE_URL`.

## Workbench

- Click a library block or drag it onto the graph. Connect output ports on the right to dependency ports on the left. Inputs cannot have dependencies; output heads are terminal. Cycles and duplicate connections are rejected.
- Select a layer in the graph or the 3D view to edit it. The inspector also lets you add/remove dependencies and navigate dependents. Connections preserve insertion order, including across graph save/load.
- Incoming vectors concatenate. Matching grids concatenate channels. Matching token sequences concatenate their feature axes (T×D₁ + T×D₂ → T×(D₁+D₂)). Mixing a grid with a vector flattens every branch and concatenates features. For differently sized grid branches, insert Flatten nodes explicitly.
- Residual stream blocks perform elementwise addition and require identical shapes. The inspector shows every incoming branch shape. For CNN skips that cross channel changes or downsampling, **Add skip projection** inserts explicit 1×1 convolution layers with the matching channels and stride. The transformed branch stays unchanged. Each projection is editable, appears in both views, is included in PyTorch export, and can be undone. Spatial alignments that a 1×1 convolution cannot represent need manual adjustment. Skip paths are rendered as elevated arcs in 3D.
- Convolutions support kernel, stride, padding, dilation, groups, and activation. Pooling supports kernel, stride, padding, and dilation. Explicit zero-padding and flatten blocks are available too.
- Feature inputs accept 0–65,535 elements. Grid inputs accept widths and heights of 1–256 and 1–65,535 channels. Inferred intermediate dimensions can exceed those input limits.
- Classification heads expose a configurable class count. Generated modules return logits from `forward()` and one-hot predictions from `predict()`. Regression heads return continuous values from both methods.
- Three.js places layers by graph topology, independently of editor coordinates. Automatic placement estimates each layer’s information throughput (the widest source-to-output bottleneck) and relaxes the Y/Z plane under information-weighted edge springs, channel cohesion, and clearance repulsion; X remains forward flow. This keeps high-density trunks straight and coherent, treats inputs as headwaters and outputs as the sea, and leaves low-density side paths to settle around them. Connections carry a soft traveling highlight with a shared visual treatment for single curves, line bundles, ribbons, residual arcs, and spatial-grid connections. Translucent volumes join spatial layer faces and taper with downsampling. Endpoints share a consistent spatial order, so bundles and volumes do not invert between layer types. Strands in a connection move together; throughput gently influences pace and contrast. Dense layers use sampled purple neuron columns, with yellow input/output neurons and blue markers for declared learned biases. Lines between sampled neurons illustrate all-to-all connectivity rather than actual weights; a perceptron is one such layer, and an MLP is a chain of them. Concatenation is a segmented bar with independent input ribbons. Addition, gathering, dot products, normalization, pooling, activation and embeddings have distinct operator glyphs. Unbranched compatible layers share a straight layout lane. Motion follows the visible geometry, uses restrained blending in both themes, and can be paused from the architecture toolbar. A master Flow slider scales all connection motion from ¼× to 40× without changing relative throughput, and remembers the chosen speed. Reduced-motion preferences start the view still. Manual placement then snaps on top. Feature maps form a spatial tower; vector layers show sampled neurons and dense connections. Sequences show token/feature grids, attention heads show sampled connectivity with triangular causal masks, and transformer blocks show their residual loops. DeepSeek shows four parallel residual lanes and a sampled expert bank. These glyphs show architecture, not trained attention scores. Geometry is capped for large architectures; tensor dimensions and parameter estimates are still computed from the complete graph.
- Drag a layer in the 3D view to move it on the camera-facing plane; every position snaps to a 2-unit world-space 3D grid. Hold Shift while dragging to change only world Y, also snapped. Orbit the camera and drag again to reach the other axes. The live readout shows the snapped coordinate. Double-click a layer to release it back to its automatic position; the reset-positions button in the panel header releases all manual positions. Manual positions are stored locally per graph; automatic layout and camera pose are unchanged.
- The activation figure is the sum of all layer-output tensor sizes at batch 1, fp32. It is an estimate, not peak runtime memory; it excludes gradients, optimizer state, workspaces, and allocator overhead. Parameter storage is estimated separately.
- Graph JSON can be saved/imported. Browser autosave, undo/redo, graph layout, camera orbit, zoom, layer labels, and example architectures are built in. Loading an example is undoable.
- Metrics, graph health, connection state, and PyTorch verification live in the compact bottom status bar. In landscape viewports the model graph sits on the left and the 3D architecture on the right; in portrait viewports the architecture sits above the graph. Drag the divider along the matching axis to resize the panes; the split is saved locally. Focus the divider and use arrow keys to adjust it, Shift for larger steps, or Home/Enter to reset. Double-clicking also restores the default split.

Shortcuts: **⌘/Ctrl Z** undo, **⌘/Ctrl Shift Z** redo, **⌘/Ctrl S** save, **Delete/Backspace** remove selection, **F** fit graph, **/** search blocks, **?** guide, **Escape** close/clear selection.

## Recurrent memory

The **Recurrent** library and template group include **RNN, GRU, LSTM, ConvRNN, ConvGRU, and ConvLSTM**. All six are executable, editable architectures with untrained weights; portable presets live in `templates/`. ConvNeXt recurrence is deferred.

| Family | Input, excluding batch | Every-step output | Final hidden output |
| --- | --- | --- | --- |
| RNN / GRU / LSTM | T × features | T × (hidden · directions) | hidden · directions |
| ConvRNN / ConvGRU / ConvLSTM | T × C × H × W | T × hidden channels × H × W | hidden channels × H × W |

Use **Sequence input** or embedded tokens for vector recurrence. **Grid sequence** supplies spatial time series. Global average pooling preserves time while reducing spatial sequences to T × C; channel LayerNorm also supports spatial sequences. Multiple temporal dependencies concatenate features/channels only when time and spatial dimensions match.

The inspector exposes hidden width, up to 16 stacked layers, sequence/final output, bias, and inter-layer dropout (requires at least two layers). Vector cells support bidirectionality; final output concatenates both final hidden states. Spatial cells are forward-only, with odd kernels up to 31 and dilation up to 16; same padding preserves H/W. RNN and ConvRNN support tanh or ReLU.

Vector cells wrap native PyTorch [RNN](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html), [GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html), and [LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html). ConvRNN applies a convolution to input and previous hidden state before activation. ConvGRU uses reset-before convolution, as in [Ballas et al.](https://arxiv.org/abs/1511.06432), with `z` denoting retention here: `h = z*h_prev + (1-z)*candidate`. Native PyTorch GRU instead applies reset after its hidden projection. ConvLSTM uses input/forget/candidate/output convolutions and separate hidden/cell states, **without peepholes**; it is a variant of the [ConvLSTM architecture](https://arxiv.org/abs/1506.04214), not an exact reproduction of that paper's cell.

Each graph forward starts from zero state. Exported `VectorRecurrent` and `SpatialRecurrent` modules also accept an explicit state and `return_state=True` for chunked execution. State shape is (layers · directions) × batch × hidden for vector cells, and layers × batch × hidden × H × W for spatial cells; LSTM uses an `(h, c)` pair. Graph edges remain acyclic because time recurrence lives inside each module. Meta verification checks one shared transition and expands symbolic output shapes, so long sequences do not require unrolling for shape checks.

The 3D view uses quiet woven hoops with subtle hints of the cell structure: two stitches for GRU's gates, and a light inner memory thread for LSTM. Spatial variants place a faint grid and translucent volume inside the same woven silhouette. Sampled hoops suggest layer depth; the inspector explains the actual gates and equations. These are architectural illustrations, not measured activations. Restrained perimeter flow shares pause, reduced-motion behavior, and the master ¼×–40× speed slider with graph connections.

## DINOv3 ConvNeXt templates

Open **Templates → DINOv3 ConvNeXt Tiny**, Small, Base, or Large. Portable JSON presets also live in `templates/`. These are editable inference-backbone architectures with normalized embedding outputs and untrained weights. They follow [Meta's ConvNeXt implementation](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/convnext.py).

| Variant | Blocks per stage | Channels per stage     | Backbone parameters | Embedding |
| ------- | ---------------- | ---------------------- | ------------------- | --------- |
| Tiny    | 3 / 3 / 9 / 3    | 96 / 192 / 384 / 768   | 27,820,128          | 768       |
| Small   | 3 / 3 / 27 / 3   | 96 / 192 / 384 / 768   | 49,454,688          | 768       |
| Base    | 3 / 3 / 27 / 3   | 128 / 256 / 512 / 1024 | 87,566,464          | 1,024     |
| Large   | 3 / 3 / 27 / 3   | 192 / 384 / 768 / 1536 | 196,230,336         | 1,536     |

Each template takes a 3×256×256 input. A 4×4 stride-4 stem produces 64×64 feature maps; stage transitions halve the spatial dimensions to 32×32, 16×16, and 8×8. Every block computes a depthwise 7×7 convolution, channel LayerNorm, a per-pixel linear expansion to 4C, GELU, a projection to C, LayerScale, and an internal residual sum. Default epsilon and LayerScale initialization are 1e-6; drop path is zero. Downsampling uses LayerNorm followed by a 2×2 stride-2 convolution. Global average pooling and final LayerNorm produce the embedding.

Stages group repeated blocks to keep the graph readable. Select a stage to inspect its operations or **Expand N blocks** for individual editing. Repeats have separate weights. Expansion preserves parameter counts, connectivity, and output shapes. The 3D view draws stage stacks and internal skip loops. Activation estimates include logical operator outputs inside grouped stages; they remain estimates rather than peak memory predictions.

All four backbone parameter counts and output shapes were checked against the referenced Meta implementation using meta tensors. A CPU block with shared weights also matched numerically. Generated PyTorch uses the corresponding truncated-normal initialization for ConvNeXt templates; pretrained checkpoints and the DINOv3 training pipeline are separate from these architecture presets.

## Attention and transformer templates

The **Attention** library includes token embeddings, image patch embeddings, learned/sinusoidal positions, multi-head self-attention, cross-attention, per-token feed-forward networks, and repeated transformer blocks. Sequence inputs have shape **T × D**; token-ID inputs have shape **T** with integer dtype. Token counts can reach 1,048,576; embedding widths reach 65,535, and vocabulary sizes reach 1,048,576. Large designs remain practical to inspect through shape inference and meta verification.

Attention supports bidirectional/causal masks and optional 1D or axial 2D RoPE. A token feed-forward layer preserves token count and width, with GELU, ReLU, SiLU, or SwiGLU and configurable expansion. Transformer blocks combine attention, token MLPs, normalization, and internal residual additions, with pre/post norm, optional LayerScale, biases, and dropout. **Expand N blocks** creates independently editable repeats and preserves parameter counts and outputs.

Cross-attention and cross-attending decoder blocks use **dependency 1 as the query** and **dependency 2 as context**. These branches are not concatenated. Query and context can have different sequence lengths and embedding widths. Edges are labeled Query and K/V; the inspector can swap them. Expanded decoder stacks retain context at every block.

Open **Templates** and search for:

- **Transformer · encoder** — 6 bidirectional blocks, width 256, 8 heads, sinusoidal positions, mean pooling, and sequence classification.
- **Transformer · causal decoder** — 6 causal blocks, width 256, learned positions, and per-token vocabulary logits.
- **Transformer · encoder–decoder** — separate source/target embeddings, 3 encoder blocks, and 3 causal decoder blocks with cross-attention.
- **Vision Transformer · Tiny / 16** — 224px image, 16px patches, CLS token, learned positions, 12 blocks, width 192, 3 heads, and a 1,000-class head.

All are untrained, executable PyTorch architectures. Token heads return T×V logits; `predict()` produces one-hot classes per token. CLS/mean pooling produces a vector, while patch selection retains a sequence. Portable JSON versions live in `templates/`.

## DINOv3 ViT templates

**DINOv3 ViT-S/16**, **B/16**, and **L/16** follow [Meta’s published backbone configuration](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py). Each takes a 3×224×224 image and returns both a normalized CLS embedding and 196 normalized patch features.

| Variant | Width | Depth | Heads | Parameters |
| --- | ---: | ---: | ---: | ---: |
| S/16 | 384 | 12 | 6 | 21,601,152 |
| B/16 | 768 | 12 | 12 | 85,669,632 |
| L/16 | 1,024 | 24 | 16 | 303,154,176 |

Each includes a learned CLS token, four learned storage tokens, the training mask parameter, axial 2D RoPE on patch Q/K (base 100, separately normalized coordinates), masked key-projection bias, GELU MLPs with 4× expansion, pre-norm epsilon 1e-5, and LayerScale initialized to 1e-5. Storage and CLS tokens are excluded from rotary positions. No learned additive position table is used. The shared final norm feeds both feature outputs; patch selection automatically removes the five prefix tokens.

Counts and both output shapes match the official module on meta for all three sizes. A small CPU model loaded with shared reference weights matched both outputs exactly, including non-square patch grids. These templates represent unmasked inference with untrained weights. DINOv3 training augmentations, checkpoint loading, distillation, and training objectives are separate; the L template follows the LVD1689M architecture configuration.

## DeepSeek V4.1 Flash 1017 reference template

The **DeepSeek V4.1 Flash 1017** template follows the [published checkpoint configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json), [native inference modules](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/inference), and [architecture report](https://arxiv.org/html/2609.19969v1). The configuration snapshot is in `backend/references/`.

The graph includes the 32-layer vision encoder and spatial aligner, token embedding and image-placeholder injection, a 20-layer causal encoder followed by a 20-layer decoder, four-lane mHC residual streams, compressed sparse attention and index reuse groups, 384 routed experts (6 active) plus a shared expert, Engram at layers 1 and 14, the vocabulary head, and the three-stage DSpark draft head. Stage groups expand to individual layers. The image branch uses a reduced 252px preview to respect the editor’s input grid limit; released preprocessing uses larger images. The 128-token sequence includes image placeholders, rather than concatenating image embeddings onto the feature width.

This is an explicitly labeled **reference blueprint**. The local PyTorch builder does not implement DeepSeek’s native sparse kernels, quantized runtime, MoE routing, Engram hashing, image-span protocol, or mHC execution. PyTorch verification/export is therefore disabled for graphs using these reference blocks; graph editing, shape analysis, JSON save/import, and 3D inspection work normally. Grouped cache sharing is described in each stage’s inspector. The draft node summarizes its reads from layers 37/38/39.

The status bar shows **763,205,315,794 logical weights** for the full graph, including Engram memory, vision, and DSpark. This was independently checked against the native module constructed on meta, excluding quantizer-scale tensors and counting shared embedding/head weights once. It differs from the report’s rounded 552B backbone figure because those additional components are included. The fp32 figure is a storage equivalent, not the mixed-precision checkpoint’s download size. Reference activation estimates count visible operator outputs only; generic transformer estimates also include logical internal outputs and attention score matrices. Neither is a peak-memory estimate.

## PyTorch

For executable graphs, **Verify in PyTorch** constructs and runs a real `torch.nn.Module` using PyTorch's meta device. No weight storage or training data is allocated. This supports designs that would be too large to materialize on your computer.

**Export PyTorch** downloads standalone Python with a `VisualModel`, an ordered DAG forward pass, and the full graph specification. The code is generated from validated node kinds; labels and IDs are serialized as data, never interpolated as executable expressions.

```python
from vision_features import build_model

model = build_model()                 # default: allocation-free meta tensors
model = build_model(device="cpu")     # allocates and initializes real weights
# model({"image": image_nchw, "features": features_nf}) -> {"classes": logits}
# model.predict(inputs) -> {"classes": one_hot_predictions}
```

Input/output dictionary keys are graph node IDs; inspect `GRAPH` or the saved JSON. Real CPU/CUDA construction is explicit because very large designs can exhaust memory. This first version has no training or numerical activation visualization.

## Catan ConvNeXt Small proposal

**Templates → Catan · PPG** contains both the original supplied residual-CNN schematic and the new **Catan · ConvNeXt Small** proposal. The proposal opens on the actor pyramid; the branch menu also exposes the value pyramid, full encoders with context, local/coarse fusion, legal-action policy, and PPG outcome heads.

Each independent encoder projects the original `1390×45×49` board planes to 32 channels, conditions them on the same 578-feature context embedding, then uses ConvNeXt stages with **32/48/64 channels and 1/1/3 blocks**. Blocks have padded 7×7 depthwise convolutions, dilation 1, 2× channel expansion, LayerNorm, GELU, LayerScale, and residual additions. Padded 3×3 stride-2 transitions yield `45×49 → 23×25 → 12×13`. The nominal final receptive field is 97×97 original cells; this is theoretical support, not a measured effective receptive field.

For local action features, the 32-channel full-resolution stage bypasses downsampling and concatenates with the 64-channel coarse map resized to the original coordinates. `align_corners=True` aligns every stride-4 sample with its original cell on this canvas. The existing 96→32 entity projection, fixed 526×32 entity-table contract, ship attributes, variable legal list, 82,819 action IDs, 49 families, 256-wide global hidden interface, and distributional PPG heads are preserved in the schematic. The actor and value weights remain independent; phase-specific losses and gradient isolation belong to the original trainer.

The standalone, untrained encoder implementation is `backend/catan_convnext.py`, downloadable from the proposal inspector. It constructs on meta by default and also runs on CPU. Each encoder has **446,128 parameters**; adding both encoders and the unchanged head budget gives **1,028,301 parameters**, versus 3,680,301 originally (72.1% fewer). This includes the retained, unused value entity/ship adapters. Value-only inference can skip their computation with `include_entities=False`. The spatial convolution budget is **138,616,368 MACs per encoder**, versus 3,220,499,520 (23.2× fewer); norms, activations, interpolation, entity readout and heads are excluded, and throughput is unmeasured.

This is a full-model **architecture proposal**, with a tested executable CNN encoder. Original custom featurization, entity assembly, legal-policy normalization, outcome transforms, and the PPG trainer remain explicit integration contracts from the supplied schematic. The full game forward is not executable in the generic builder. No old checkpoint weights, optimizer state, or channel-specific stem refinement offsets are attached. Full-diagram export supports JSON and architecture-only `.pt`; the encoder can be downloaded separately as Python.

The portable proposal and original reference live in `templates/catan_convnext_small.vizml.json` and `templates/catan_original.vizml.json`. To regenerate them from a supplied source diagram, run `uv run python -m scripts.build_catan_proposal /path/to/catan_latest.vizml.json`. Tests check exact parameter ownership against the encoder implementation, every intermediate shape on meta, finite CPU outputs, original-coordinate alignment, separate actor/value parameters, preserved head interfaces, and the receptive-field/MAC calculations.

## Checkpoints and supplied schematics

Use **Import** to choose a `.pt`/`.pth` checkpoint, a graph JSON, or both together. A vizml `.pt` bundle carries its diagram and reopens directly. A plain `state_dict` or training checkpoint opens a searchable tensor inventory; import its matching JSON to supply forward connections. Tensor storage alone does not identify an arbitrary model's forward pass. Dictionary wrappers `model`, `model_state_dict`, `state_dict`, `weights`, and `net` are recognized, along with common `module.` / `_orig_mod.` name prefixes.

**Export** offers graph JSON, allocation-free `.pt architecture`, `.pt checkpoint`, `.pt state_dict`, and standalone Python for executable graphs. A checkpoint bundle contains `format="vizml.checkpoint"`, `format_version=1`, `graph`, `model_state_dict`, and a `weights` origin (`none`, `initialized`, or `imported`). Checkpoint bundles can also retain an `original_checkpoint` with optimizer, configuration, and training metadata. Exporting `state_dict` produces just the native tensor dictionary. Existing numerical weights are preserved when paired; without imported weights, weighted exports are explicitly **fresh, untrained initialization**, limited to 512 MiB. Architecture-only exports allocate no weights and support very large designs.

For executable vizml graphs, export both **PyTorch source** and **.pt checkpoint**, then load:

```python
import torch
from my_architecture import build_model

bundle = torch.load("my_architecture.pt", weights_only=True, map_location="cpu")
model = build_model(device="meta")
model.load_state_dict(bundle["model_state_dict"], strict=True, assign=True)
model.eval()
```

Generated source defines the same `GRAPH` and node IDs as the bundle. Import and pairing check every expected tensor name and shape. Changing graph semantics invalidates the attachment for weighted export; labels and layout may change freely. External model checkpoints retain their native keys and still require the original application model class to run.

Rich third-party diagrams with per-node `output_shape` and `parameter_count` automatically open as **descriptive schematics**. Their descriptions, complex operation settings, bundled pre-operations, symbolic dimensions, tensor ownership, checkpoint provenance, and edge roles are preserved. The inspector labels these as supplied claims. Structural checks validate a DAG and internally consistent parameter ownership; pairing independently checks tensor names and shapes. It does **not** verify custom operations or forward connections. A different checkpoint SHA-256 is shown as a revision mismatch, even if all tensor shapes match; supplied statistics continue to refer to the JSON snapshot. The Catan-style schematic exposes separate actor/value CNNs, per-block views, legal-action policy, and outcome heads. View filtering does not change the saved graph. Symbolic `L` remains symbolic in labels; sampling uses the source's CPU-example action count, or 14 when absent.

Imports use `torch.load(weights_only=True)` and FakeTensorMode for metadata inspection. Python model classes, TorchScript, quantized/sparse tensors, and arbitrary pickle globals are unsupported. No unsafe pickle fallback or uploaded source execution is used. Limits are 2 GiB per checkpoint, 30,000 named tensors, and 24 MiB of pickle metadata. Files remain on this local server; their cache IDs are local references, while exported `.pt` bundles are portable. JSON contains the diagram and annotations, not embedded numerical tensor data. Binary file transfers use HTTP; graph edits and analysis remain event driven over WebSocket.

## Architecture

```text
React + React Flow  ── graph.changed ──► Bun WebSocket proxy ──► FastAPI
     │                                      │
     │                               Pure shape inference
     │                                      │
     ◄──────────── graph.analyzed ────────────┘
     │
     └── Three.js topology/shape renderer

model.compile ──► PyTorch module + meta forward pass
model.export  ──► Standalone PyTorch source
```

The backend owns shape inference. Editor movements stay local. Semantic changes are debounced and tagged with request IDs; stale responses are ignored. The socket reconnects automatically and resubmits the latest graph. JSON imports are bounded to 300 layers, 2,000 edges, and 2 MB.

Bun serves the UI and proxies binary HTTP and graph WebSocket traffic; Python continues to own model inference and checkpoint processing. Runtime entrypoints are `scripts/dev.ts` and `scripts/server.ts`; `scripts/build.ts` bundles production HTML/CSS/JS with Bun.

The checked-in Bun patch for React Flow schedules resize measurements on animation frames, avoiding ResizeObserver feedback loops during panel resizing.

Main modules: `backend/graph.py`, `backend/attention.py`, and `backend/recurrent.py` for shape inference, `backend/deepseek.py` for reference analysis, `backend/builder.py`, `backend/torch_attention.py`, and `backend/torch_recurrent.py` for PyTorch, `backend/main.py` for transport, `src/model.ts` for graph types/catalog/examples, and `src/components/Architecture.tsx` for the independent renderer. New blocks extend the schema, inference, builder, catalog, inspector, and renderer as appropriate.

## Verify

```sh
bun run build
bun run test
```

Tests cover analytical shapes against real PyTorch execution, mixed-branch concatenation, residual compatibility, dilation/groups/padding/pooling, zero and maximum inputs, logits/one-hot/regression semantics, malformed/cyclic graphs, export, and WebSocket error recovery. Sequence tests compare attention numerically with PyTorch, enforce causal isolation from future tokens, exercise cross-attention with different context sizes, check 2D RoPE metadata through residuals, and verify all portable transformer templates. Recurrent tests check native parity, bidirectional final states, convolutional gate equations, causality, gradients, chunked state, parameter counts, exported weight round trips, and template switching over WebSocket. Renderer tests verify bounded geometry, consistent connection orientation, closed flow paths, and selection of open woven cells. Node tests also cover template round trips and expanded decoder context wiring.

## Roadmap

More attention variants, executable specialized MoE/sparse architectures, symbolic shape constraints, and `.safetensors` interchange. Current interchange supports graph JSON, generated Python source, and `.pt` architecture/checkpoint/state_dict files.

Bun integration tests exercise development/production HTML serving, binary request/download forwarding, WebSocket greeting and event ordering, and engine failures.
