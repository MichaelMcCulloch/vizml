import pytest

from backend.builder import compile_meta
from backend.graph import analyze
from backend.tests.test_graph import graph, node


def test_grouped_dilated_conv_cost_and_receptive_field():
    spec = graph([node("x", "grid_input", channels=8, height=16, width=16),
                  node("conv", "conv2d", channels=12, groups=4, kernel=3, dilation=2, stride=2, padding=2),
                  node("pool", "global_avgpool"), node("out", "regression", units=3)],
                 [("x", "conv"), ("conv", "pool"), ("pool", "out")])
    a = analyze(spec)
    assert a["nodes"]["conv"]["macs"] == 8 * 8 * 12 * 2 * 9
    assert a["nodes"]["out"]["macs"] == 36
    assert a["profile"]["macs"] == 13860
    assert a["nodes"]["conv"]["receptive_field"] == {"source": "x", "size": [5, 5], "stride": [2, 2], "offset": [0.5, 0.5]}
    assert a["nodes"]["pool"]["receptive_field"]["size"] == [19, 19]


def test_convnext_cost_and_repeated_receptive_field():
    spec = graph([node("x", "grid_input", channels=8, height=5, width=7), node("stage", "convnext_stage", blocks=3, expansion=2)], [("x", "stage")])
    a = analyze(spec)
    assert a["profile"]["macs"] == 3 * 5 * 7 * (8 * 49 + 2 * 8 * 16)
    assert a["nodes"]["stage"]["receptive_field"]["size"] == [19, 19]


def test_attention_cross_decoder_counts_both_attention_paths_and_swiglu():
    spec = graph([node("q", "sequence_input", tokens=3, features=8), node("kv", "sequence_input", tokens=5, features=12),
                  node("dec", "transformer_block", mode="cross", blocks=2, heads=2, activation="swiglu", expansion=2)], [("q", "dec"), ("kv", "dec")])
    a = analyze(spec)
    cross = 2 * 3 * 8**2 + 2 * 5 * 12 * 8 + 2 * 3 * 5 * 8
    self_attention = 4 * 3 * 8**2 + 2 * 3**2 * 8
    ffn = 3 * 3 * 8 * 16
    assert a["profile"]["macs"] == 2 * (cross + self_attention + ffn)


@pytest.mark.parametrize("kind,gates", [("rnn", 1), ("gru", 3), ("lstm", 4)])
def test_recurrent_counts_all_steps_layers_and_directions(kind, gates):
    spec = graph([node("x", "sequence_input", tokens=7, features=5), node("r", kind, hidden_size=3, layers=2, bidirectional=1, return_sequences=0)], [("x", "r")])
    assert analyze(spec)["profile"]["macs"] == 7 * 2 * gates * 3 * ((5 + 3) + (6 + 3))


def test_spatial_recurrent_kernel_and_area():
    spec = graph([node("x", "grid_sequence_input", steps=4, channels=2, height=5, width=6), node("r", "conv_gru", hidden_size=3, kernel=3)], [("x", "r")])
    assert analyze(spec)["profile"]["macs"] == 4 * 5 * 6 * 3 * 3 * (2 + 3) * 9


def test_token_storage_stays_int64_and_patch_prefix_has_no_conv_cost():
    spec = graph([node("tokens", "token_input", tokens=4), node("embed", "token_embedding", features=8, vocab=32)], [("tokens", "embed")])
    a = analyze(spec)
    assert a["nodes"]["tokens"]["output_bytes"] == 32
    assert a["profile"]["integer_output_elements"] == 4
    assert a["profile"]["float_output_elements"] == 32
    assert a["activation_bytes"] == 4 * 8 + 32 * 4
    assert a["profile"]["macs"] == 0
    patches = graph([node("x", "grid_input", channels=3, height=8, width=8), node("patch", "patch_embed", patch=4, features=8, registers=2, cls_token=1)], [("x", "patch")])
    assert analyze(patches)["profile"]["macs"] == 4 * 8 * 3 * 16


def test_receptive_fields_merge_only_on_aligned_source_grids():
    spec = graph([node("x", "grid_input", channels=4), node("conv", "conv2d", channels=4), node("sum", "residual")], [("x", "conv"), ("x", "sum"), ("conv", "sum")])
    assert analyze(spec)["nodes"]["sum"]["receptive_field"]["size"] == [3, 3]
    spec.nodes[1].data.kind = "upsample2d"
    spec.nodes[1].data.params = {"height": 32, "width": 32}
    assert "receptive_field" not in analyze(spec)["nodes"]["sum"]


def test_invalid_graph_has_partial_costs_and_root_causes():
    spec = graph([node("x", "feature_input", features=3), node("bad", "conv2d"), node("next", "dense"), node("out", "regression")], [("x", "bad"), ("bad", "next"), ("next", "out")])
    a = analyze(spec)
    assert a["profile"]["macs"] is None
    assert not a["profile"]["complete"]
    assert a["errors"][1]["blocked_by"] == ["bad"]
    assert a["errors"][2]["blocked_by"] == ["bad"]


def test_schematic_compute_remains_unknown():
    spec = graph([node("x", "dense")], [])
    spec.mode = "schematic"
    spec.nodes[0].data.output_shape = ["L", 8]
    spec.nodes[0].data.parameter_count = 100
    a = analyze(spec)
    assert a["profile"]["macs"] is None
    assert a["profile"]["unknown_nodes"] == ["x"]
    assert a["profile"]["complete"]


@pytest.mark.parametrize("kind", ["dot", "gather", "activation", "legal_policy", "checkpoint_module"])
def test_schematic_operators_cannot_silently_compile_as_identity(kind):
    spec = graph([node("x", "feature_input"), node("op", kind), node("out", "regression")], [("x", "op"), ("op", "out")])
    assert not analyze(spec)["valid"]
    with pytest.raises(ValueError, match="Fix graph errors"):
        compile_meta(spec)


def test_meta_verification_checks_every_intermediate():
    spec = graph([node("x", "sequence_input", tokens=3, features=8), node("attn", "attention", heads=2), node("pool", "token_pool"), node("out", "regression")], [("x", "attn"), ("attn", "pool"), ("pool", "out")])
    result = compile_meta(spec)
    assert result["checked_nodes"] == 4
    assert result["nodes"] == {"x": [1, 3, 8], "attn": [1, 3, 8], "pool": [1, 8], "out": [1, 1]}
    assert result["allocated_bytes"] == 0


def test_meta_verification_rejects_a_wrong_intermediate_inference(monkeypatch):
    import backend.builder as builder
    original = builder.analyze
    def wrong_shape(spec):
        analysis = original(spec)
        analysis["nodes"]["hidden"]["shape"] = [9]
        return analysis
    spec = graph([node("x", "feature_input", features=4), node("hidden", "dense", units=8), node("out", "regression")], [("x", "hidden"), ("hidden", "out")])
    monkeypatch.setattr(builder, "analyze", wrong_shape)
    with pytest.raises(ValueError, match="hidden: PyTorch produced"):
        builder.compile_meta(spec)
