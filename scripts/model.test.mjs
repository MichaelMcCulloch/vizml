import { test } from "bun:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import {
  makeNode,
  residualProjectionPlan,
  projectResidualSkips,
  canConnect,
  makeDinoConvNeXt,
  makeTransformer,
  examples,
  dinoVariants,
  expandConvNeXtStage,
  cleanGraph,
  parseGraph,
} from "../src/model.ts";

function fixture(sourceShape = [3, 32, 32], targetShape = [16, 16, 16]) {
  const nodes = [
    makeNode("grid_input", 0, 0, "image"),
    makeNode("conv2d", 250, 0, "main"),
    makeNode("residual", 500, 0, "sum"),
    makeNode("regression", 750, 0, "out"),
  ];
  nodes[0].data.params = {
    channels: sourceShape[0],
    height: sourceShape[1],
    width: sourceShape[2],
  };
  nodes[1].data.params = {
    ...nodes[1].data.params,
    channels: targetShape[0],
    stride: 2,
  };
  const graph = {
    version: 1,
    name: "Projected CNN residual",
    nodes,
    edges: [
      ["image", "main"],
      ["main", "sum"],
      ["image", "sum"],
      ["sum", "out"],
    ].map(([source, target], i) => ({ id: String(i), source, target })),
  };
  const analysis = {
    nodes: {
      image: { shape: sourceShape, depth: 0 },
      main: { shape: targetShape, depth: 1 },
    },
  };
  return { graph, analysis };
}

test("CNN skip projection matches changed channels and pooling dimensions", () => {
  const { graph, analysis } = fixture();
  const plan = residualProjectionPlan(graph, "sum", analysis);
  assert.deepEqual(plan.targetShape, [16, 16, 16]);
  assert.equal(plan.steps.length, 1);
  assert.equal(plan.steps[0].sourceId, "image");
  assert.equal(plan.steps[0].stride, 2);
});

test("projection is an explicit, connected, linear 1x1 convolution", () => {
  const { graph, analysis } = fixture();
  const projected = projectResidualSkips(graph, "sum", analysis);
  const projection = projected.nodes.find(
    (n) => n.data.params.role === "skip_projection",
  );
  assert.equal(projection.data.params.kernel, 1);
  assert.equal(projection.data.params.stride, 2);
  assert.equal(projection.data.params.channels, 16);
  assert.equal(projection.data.params.padding, 0);
  assert.equal(projection.data.params.activation, "none");
  assert.ok(
    projected.edges.some(
      (e) => e.source === "image" && e.target === projection.id,
    ),
  );
  assert.ok(
    projected.edges.some(
      (e) => e.source === projection.id && e.target === "sum",
    ),
  );
  assert.equal(graph.nodes.length, 4); // Undo can retain the original graph unchanged.
});

test("projected CNN compiles and runs in real PyTorch", () => {
  const { graph, analysis } = fixture();
  const projected = projectResidualSkips(graph, "sum", analysis);
  const output = execFileSync(
    "uv",
    [
      "run",
      "python",
      "-c",
      "import sys,json; from backend.schema import Graph; from backend.builder import compile_meta; print(json.dumps(compile_meta(Graph.model_validate(json.load(sys.stdin)))))",
    ],
    {
      cwd: fileURLToPath(new URL("..", import.meta.url)),
      input: JSON.stringify(projected),
      encoding: "utf8",
    },
  );
  const result = JSON.parse(output);
  assert.deepEqual(result.outputs, { out: [1, 1] });
  assert.equal(result.parameters, 4609);
  assert.equal(result.allocated_bytes, 0);
});

test("matching branches require no projection", () => {
  const { graph, analysis } = fixture([16, 16, 16], [16, 16, 16]);
  assert.equal(residualProjectionPlan(graph, "sum", analysis), null);
});

test("upsampling or incompatible spatial ratios do not suggest an invalid convolution", () => {
  for (const [source, target] of [
    [
      [3, 16, 16],
      [16, 32, 32],
    ],
    [
      [3, 32, 40],
      [16, 16, 24],
    ],
    [
      [3, 32, 32],
      [16, 10, 10],
    ],
  ]) {
    const { graph, analysis } = fixture(source, target);
    assert.equal(residualProjectionPlan(graph, "sum", analysis), null);
  }
});

test("projection uses the actual convolution shape formula for non-even grids", () => {
  const { graph, analysis } = fixture([3, 33, 49], [16, 17, 25]);
  assert.equal(
    residualProjectionPlan(graph, "sum", analysis).steps[0].stride,
    2,
  );
});

test("connection validation rejects cycles and duplicate dependencies", () => {
  const { graph } = fixture();
  assert.equal(canConnect(graph, "sum", "main"), false);
  assert.equal(canConnect(graph, "image", "sum"), false);
  assert.equal(canConnect(graph, "out", "main"), false);
});

test("DINOv3 templates preserve official stage depths, widths, and normalized embedding outputs", () => {
  for (const [variant, config] of Object.entries(dinoVariants)) {
    const graph = makeDinoConvNeXt(variant);
    const saved = JSON.parse(
      readFileSync(
        new URL(
          `../templates/dinov3_convnext_${variant}.vizml.json`,
          import.meta.url,
        ),
        "utf8",
      ),
    );
    assert.deepEqual(saved, graph);
    assert.equal(graph.nodes.length, 16);
    assert.deepEqual(
      graph.nodes
        .filter((n) => n.data.kind === "convnext_stage")
        .map((n) => n.data.params.blocks),
      config.depths,
    );
    assert.deepEqual(
      graph.nodes
        .filter((n) => n.data.kind === "conv2d")
        .map((n) => n.data.params.channels),
      config.dims,
    );
    assert.equal(graph.nodes[0].data.params.width, 256);
    assert.equal(graph.nodes.at(-3).data.kind, "global_avgpool");
    assert.equal(graph.nodes.at(-2).data.kind, "layernorm");
    assert.equal(graph.nodes.at(-1).data.kind, "embedding");
    assert.deepEqual(
      parseGraph(JSON.stringify(cleanGraph(graph))),
      cleanGraph(graph),
    );
    assert.equal(canConnect(graph, "embedding", "stage1"), false);
  }
});

test("all DINOv3 variants match reference backbone parameter counts and run on the meta device", () => {
  // Official convnext.py, facebookresearch/dinov3 commit 6876159a11b4df116f30f667f8c9888617df0751.
  const graphs = Object.fromEntries(
    Object.keys(dinoVariants).map((variant) => [
      variant,
      makeDinoConvNeXt(variant),
    ]),
  );
  const output = execFileSync(
    "uv",
    [
      "run",
      "python",
      "-c",
      "import sys,json; from backend.schema import Graph; from backend.builder import compile_meta; print(json.dumps({key:compile_meta(Graph.model_validate(value)) for key,value in json.load(sys.stdin).items()}))",
    ],
    {
      cwd: fileURLToPath(new URL("..", import.meta.url)),
      input: JSON.stringify(graphs),
      encoding: "utf8",
    },
  );
  const results = JSON.parse(output);
  const counts = {
    tiny: 27820128,
    small: 49454688,
    base: 87566464,
    large: 196230336,
  };
  for (const [variant, config] of Object.entries(dinoVariants)) {
    assert.equal(results[variant].parameters, counts[variant]);
    assert.deepEqual(results[variant].outputs, {
      embedding: [1, config.dims.at(-1)],
    });
    assert.equal(results[variant].allocated_bytes, 0);
  }
});

test("expanding a ConvNeXt stage preserves repeat count, connectivity, and PyTorch architecture", () => {
  const original = makeDinoConvNeXt("tiny");
  const expanded = expandConvNeXtStage(original, "stage3");
  assert.equal(original.nodes.length, 16);
  assert.equal(expanded.nodes.length, 24);
  assert.equal(
    expanded.nodes
      .filter((n) => n.data.kind === "convnext_stage")
      .reduce((n, layer) => n + layer.data.params.blocks, 0),
    18,
  );
  assert.equal(expanded.edges.filter((e) => e.target === "stage3").length, 1);
  assert.ok(
    !expanded.edges.some(
      (e) => e.source === "stage3" && e.target === "down3_norm",
    ),
  );
  const output = execFileSync(
    "uv",
    [
      "run",
      "python",
      "-c",
      "import sys,json; from backend.schema import Graph; from backend.builder import compile_meta; print(json.dumps(compile_meta(Graph.model_validate(json.load(sys.stdin)))))",
    ],
    {
      cwd: fileURLToPath(new URL("..", import.meta.url)),
      input: JSON.stringify(expanded),
      encoding: "utf8",
    },
  );
  assert.equal(JSON.parse(output).parameters, 27820128);
  assert.deepEqual(JSON.parse(output).outputs, { embedding: [1, 768] });
});

test("attention and ViT templates round-trip through portable JSON", () => {
  for (const key of [
    "transformer_encoder",
    "transformer_decoder",
    "transformer_seq2seq",
    "vit_tiny",
    "dinov3_vit_small",
    "dinov3_vit_base",
    "dinov3_vit_large",
    "deepseek_v41_flash_1017",
  ]) {
    const fixture = JSON.parse(
      readFileSync(
        new URL(`../templates/${key}.vizml.json`, import.meta.url),
        "utf8",
      ),
    );
    assert.deepEqual(
      parseGraph(JSON.stringify(fixture)),
      cleanGraph(examples[key]()),
    );
  }
});

test("expanded transformer decoder preserves query/context dependencies and parameters", () => {
  const grouped = makeTransformer("seq2seq");
  const expanded = expandConvNeXtStage(grouped, "decoder");
  assert.ok(expanded);
  const blocks = expanded.nodes.filter(
    (n) =>
      n.data.kind === "transformer_block" && n.data.params.mode === "cross",
  );
  assert.equal(blocks.length, 3);
  for (let i = 0; i < blocks.length; i++) {
    const incoming = expanded.edges.filter((e) => e.target === blocks[i].id);
    assert.equal(incoming[0].source, i ? blocks[i - 1].id : "target_pos");
    assert.equal(incoming[1].source, "memory");
  }
  const result = JSON.parse(
    execFileSync(
      "uv",
      [
        "run",
        "python",
        "-c",
        "import sys,json; from backend.schema import Graph; from backend.builder import compile_meta; print(json.dumps([compile_meta(Graph.model_validate(g)) for g in json.load(sys.stdin)]))",
      ],
      {
        cwd: fileURLToPath(new URL("..", import.meta.url)),
        input: JSON.stringify([grouped, expanded]),
        encoding: "utf8",
      },
    ),
  );
  assert.equal(result[0].parameters, result[1].parameters);
  assert.deepEqual(result[0].outputs, result[1].outputs);
  assert.equal(result[0].allocated_bytes, 0);
  assert.equal(result[1].allocated_bytes, 0);
  assert.equal(result[0].checked_nodes, grouped.nodes.length);
  assert.equal(result[1].checked_nodes, expanded.nodes.length);
});

test("DeepSeek grouped stages expand into consecutive native layer indices", () => {
  const grouped = examples.deepseek_v41_flash_1017();
  const expanded = expandConvNeXtStage(grouped, "layers_14_19");
  assert.ok(expanded);
  const blocks = expanded.nodes.filter(
    (n) =>
      n.data.kind === "deepseek_stage" &&
      Number(n.data.params.start) >= 14 &&
      Number(n.data.params.start) < 20,
  );
  assert.deepEqual(
    blocks.map((n) => n.data.params.start),
    [14, 15, 16, 17, 18, 19],
  );
  assert.ok(blocks.every((n) => n.data.params.blocks === 1));
  const a = JSON.parse(
    execFileSync(
      "uv",
      [
        "run",
        "python",
        "-c",
        "import sys,json; from backend.schema import Graph; from backend.graph import analyze; print(json.dumps([analyze(Graph.model_validate(g)) for g in json.load(sys.stdin)]))",
      ],
      {
        cwd: fileURLToPath(new URL("..", import.meta.url)),
        input: JSON.stringify([grouped, expanded]),
        encoding: "utf8",
      },
    ),
  );
  assert.ok(a[1].valid);
  assert.equal(a[0].parameters, a[1].parameters);
  assert.deepEqual(a[0].nodes.logits.shape, a[1].nodes.logits.shape);
});

test("descriptive schematics preserve source claims, complex settings, and edge roles", () => {
  const source = {
    version: 1, name: "A supplied snapshot", metadata: { checkpoint: { sha256: "claimed" } },
    nodes: [
      { id: "raw", type: "layer", position: { x: 0, y: 0 }, data: { kind: "feature_input", label: "Raw", params: {}, output_shape: ["B", 3], parameter_count: 0 } },
      { id: "operation", type: "layer", position: { x: 300, y: 0 }, data: { kind: "gather", label: "Custom operation", params: { enabled: true, indices: [0, 2] }, output_shape: ["B", "L", 2], parameter_count: 0, description: "A source claim", module_paths: ["model.operation"] } },
    ],
    edges: [{ id: "edge", source: "raw", target: "operation", label: "Selected features", data: { role: "data" } }],
  };
  const parsed = parseGraph(JSON.stringify(source));
  assert.equal(parsed.mode, "schematic");
  assert.deepEqual(parsed.nodes, source.nodes);
  assert.deepEqual(parsed.edges, source.edges);
  assert.deepEqual(parsed.metadata, source.metadata);
  assert.deepEqual(parseGraph(JSON.stringify(cleanGraph(parsed))), parsed);
  assert.equal(canConnect(parsed, "operation", "raw"), false);
  const duplicate = structuredClone(source);
  duplicate.nodes[1].id = "raw";
  assert.throws(() => parseGraph(JSON.stringify(duplicate)), /invalid layer/);
});

test("builder graph parameter validation remains strict", () => {
  const graph = examples.multimodal();
  graph.nodes[0].data.params.extra = [1, 2];
  assert.throws(() => parseGraph(JSON.stringify(graph)), /numbers or strings/);
});

test("Catan proposal and original templates retain their interfaces and independent parameter budgets", () => {
  const original = examples.catan_original();
  const proposal = examples.catan_convnext_small();
  assert.equal(proposal.mode, "schematic");
  assert.equal(proposal.nodes.length, 81);
  assert.equal(proposal.metadata.parameter_counts.total, 1028301);
  assert.equal(original.metadata.parameter_counts.total, 3680301);
  assert.deepEqual(proposal.metadata.input_dimensions, original.metadata.input_dimensions);
  assert.equal(proposal.metadata.proposal.untrained, true);
  assert.equal(proposal.metadata.weights, undefined);
  for (const branch of ["actor", "value"]) {
    assert.deepEqual(proposal.nodes.filter(n => n.id.startsWith(`${branch}_stage`)).map(n => n.data.params.blocks), [1,1,3]);
    assert.deepEqual(proposal.edges.filter(e => e.target === `${branch}_local_fusion`).map(e => e.source), [`${branch}_stage1`, `${branch}_upsample`]);
  }
  for (const graph of [original, proposal]) assert.deepEqual(parseGraph(JSON.stringify(graph)), graph);
});
