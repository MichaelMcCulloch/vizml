"""Sequence shape checks plus numerical/causal tests of the exported runtime."""
from pathlib import Path

import pytest
import torch
from torch import nn

from backend.builder import compile_meta, export_python
from backend.deepseek import block_parameters, draft_parameters, vision_parameters
from backend.graph import analyze
from backend.schema import Graph
from backend.torch_attention import MultiHeadAttention


def node(key, kind, **params):
    return {"id": key, "data": {"kind": kind, "label": key, "params": params}}


def graph(nodes, links):
    return Graph.model_validate({"nodes": nodes, "edges": [{"id": str(i), "source": a, "target": b} for i, (a, b) in enumerate(links)]})


def built(spec, device="cpu"):
    scope = {"__name__": "attention_test"}
    exec(export_python(spec), scope)
    return scope["build_model"](device).eval()


def test_attention_matches_torch_multihead_attention_numerically():
    torch.manual_seed(13)
    ours = MultiHeadAttention(12, heads=3).eval()
    reference = nn.MultiheadAttention(12, 3, batch_first=True).eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(ours.qkv.weight)
        reference.in_proj_bias.copy_(ours.qkv.bias)
        reference.out_proj.load_state_dict(ours.proj.state_dict())
    x = torch.randn(2, 7, 12)
    torch.testing.assert_close(ours(x), reference(x, x, x, need_weights=False)[0])


def test_cross_attention_uses_separate_context_width_and_token_count():
    spec = graph([node("q", "sequence_input", tokens=3, features=12), node("ctx", "sequence_input", tokens=7, features=20), node("cross", "cross_attention", heads=3), node("out", "embedding")], [("q", "cross"), ("ctx", "cross"), ("cross", "out")])
    a = analyze(spec)
    assert a["valid"]
    assert a["nodes"]["cross"]["merge"] == "query_context"
    assert a["nodes"]["cross"]["shape"] == [3, 12]
    assert a["parameters"] == 816 == compile_meta(spec)["parameters"]
    model = built(spec)
    inputs = {"q": torch.randn(2, 3, 12), "ctx": torch.randn(2, 7, 20)}
    output = model(inputs)["out"]
    assert output.shape == (2, 3, 12)
    assert not torch.allclose(output, model({**inputs, "ctx": inputs["ctx"] + 1})["out"])
    ours = next(m for m in model.modules() if m.__class__.__name__ == "MultiHeadAttention")
    reference = nn.MultiheadAttention(12, 3, kdim=20, vdim=20, batch_first=True).eval()
    with torch.no_grad():
        reference.q_proj_weight.copy_(ours.q.weight)
        reference.k_proj_weight.copy_(ours.kv.weight[:12])
        reference.v_proj_weight.copy_(ours.kv.weight[12:])
        reference.in_proj_bias.copy_(torch.cat([ours.q.bias, ours.kv.bias]))
        reference.out_proj.load_state_dict(ours.proj.state_dict())
    torch.testing.assert_close(output, reference(inputs["q"], inputs["ctx"], inputs["ctx"], need_weights=False)[0])


@pytest.mark.parametrize("norm_style", ["pre", "post"])
def test_exported_causal_transformer_cannot_read_future_tokens(norm_style):
    spec = graph([node("ids", "token_input", tokens=6), node("embed", "token_embedding", vocab=32, features=16), node("pos", "position_encoding", method="learned"), node("block", "transformer_block", blocks=2, heads=4, mode="decoder", norm_style=norm_style), node("out", "token_head", vocab=32)], [("ids", "embed"), ("embed", "pos"), ("pos", "block"), ("block", "out")])
    a = analyze(spec)
    assert a["valid"]
    assert a["parameters"] == compile_meta(spec)["parameters"]
    model = built(spec)
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 4, 11, 12]])
    a, b = model({"ids": first})["out"], model({"ids": changed})["out"]
    torch.testing.assert_close(a[:, :4], b[:, :4])
    assert not torch.allclose(a[:, 4:], b[:, 4:])
    predictions = model.predict({"ids": first})["out"]
    assert predictions.shape == (1, 6, 32)
    assert torch.equal(predictions.sum(-1), torch.ones(1, 6))


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu", "swiglu"])
def test_token_feedforward_and_cross_decoder_parameter_counts(activation):
    spec = graph([node("query", "sequence_input", tokens=4, features=16), node("context", "sequence_input", tokens=9, features=12), node("decoder", "transformer_block", heads=4, mode="cross", blocks=2, expansion=2.5, activation=activation, layer_scale=1e-5), node("out", "embedding")], [("query", "decoder"), ("context", "decoder"), ("decoder", "out")])
    a = analyze(spec)
    assert a["valid"]
    assert compile_meta(spec)["parameters"] == a["parameters"]
    result = built(spec)({"query": torch.randn(2, 4, 16), "context": torch.randn(2, 9, 12)})
    assert result["out"].shape == (2, 4, 16)


def test_sequences_concatenate_features_and_residuals_preserve_patch_metadata():
    spec = graph([node("image", "grid_input", channels=3, height=32, width=48), node("patch", "patch_embed", patch=8, features=24, registers=4), node("ffn", "feedforward", expansion=2), node("residual", "residual"), node("attn", "attention", heads=3, position="rope2d", rope_base=100), node("norm", "layernorm"), node("pool", "token_pool", method="patches"), node("out", "embedding")], [("image", "patch"), ("patch", "ffn"), ("patch", "residual"), ("ffn", "residual"), ("residual", "attn"), ("attn", "norm"), ("norm", "pool"), ("pool", "out")])
    a = analyze(spec)
    assert a["valid"], a["errors"]
    assert a["nodes"]["attn"]["grid"] == [4, 6]
    assert a["nodes"]["attn"]["prefix_tokens"] == 5
    assert compile_meta(spec)["outputs"] == {"out": [1, 24, 24]}
    assert a["parameters"] == compile_meta(spec)["parameters"]
    assert built(spec)({"image": torch.randn(1, 3, 32, 48)})["out"].shape == (1, 24, 24)
    joined = graph([node("a", "sequence_input", tokens=8, features=5), node("b", "sequence_input", tokens=8, features=7), node("join", "concat"), node("out", "embedding")], [("a", "join"), ("b", "join"), ("join", "out")])
    assert analyze(joined)["nodes"]["join"]["shape"] == [8, 12]
    assert compile_meta(joined)["outputs"] == {"out": [1, 8, 12]}
    joined.nodes[1].data.params["tokens"] = 7
    assert not analyze(joined)["valid"]


@pytest.mark.parametrize("kind,params", [("attention", {"heads": 3}), ("attention", {"heads": 0}), ("attention", {"heads": 4, "position": "rope2d"}), ("attention", {"dropout": 1}), ("transformer_block", {"mode": "mystery"}), ("transformer_block", {"blocks": 0}), ("feedforward", {"expansion": 0}), ("feedforward", {"activation": "bad"}), ("token_embedding", {})])
def test_invalid_attention_configuration_reports_a_local_error(kind, params):
    spec = graph([node("x", "sequence_input", tokens=8, features=16), node("block", kind, **params), node("out", "embedding")], [("x", "block"), ("block", "out")])
    a = analyze(spec)
    assert not a["valid"]
    assert a["errors"][0]["node_id"] == "block"


def test_token_ids_require_embedding_and_patch_sizes_must_tile_the_image():
    ids = graph([node("x", "token_input"), node("out", "regression")], [("x", "out")])
    assert not analyze(ids)["valid"]
    patches = graph([node("x", "grid_input", height=30), node("patch", "patch_embed", patch=16)], [("x", "patch")])
    assert "divisible" in analyze(patches)["errors"][0]["message"]


TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


@pytest.mark.parametrize("key,expected", [("dinov3_vit_small", 21601152), ("dinov3_vit_base", 85669632), ("dinov3_vit_large", 303154176), ("vit_tiny", 5717416), ("transformer_encoder", 4741642), ("transformer_decoder", 21187840), ("transformer_seq2seq", 11830272)])
def test_portable_transformer_templates_compile_without_weight_storage(key, expected):
    g = Graph.model_validate_json((TEMPLATES / f"{key}.vizml.json").read_text())
    a = analyze(g)
    assert a["valid"], a["errors"]
    assert a["executable"]
    assert a["parameters"] == expected == compile_meta(g)["parameters"]
    if key.startswith("dinov3_vit"):
        d = {"small": 384, "base": 768, "large": 1024}[key.removeprefix("dinov3_vit_")]
        assert compile_meta(g)["outputs"] == {"embedding": [1, d], "patch_output": [1, 196, d]}


def test_deepseek_reference_counts_match_the_native_module_without_claiming_execution():
    g = Graph.model_validate_json((TEMPLATES / "deepseek_v41_flash_1017.vizml.json").read_text())
    a = analyze(g)
    assert a["valid"], a["errors"]
    assert not a["executable"]
    # Independently counted from the published, unquantized reference nn.Module
    # on meta, excluding quantization-scale tensors and deduplicating shared weights.
    assert a["parameters"] == 763205315794
    assert sum(block_parameters(i) for i in range(40)) == 747170852464
    assert vision_parameters() == 485253120
    assert draft_parameters() == 14225362530
    assert a["nodes"]["layers_14_19"]["engram_layers"] == [14]
    assert a["nodes"]["layers_20_23"]["kv_sources"] == [20]
    assert a["nodes"]["logits"]["shape"] == [128, 129280]
    assert a["nodes"]["draft"]["shape"] == [5, 129280]
    assert any("reference architecture" in note["message"] for note in a["warnings"])
    with pytest.raises(ValueError, match="reference operators"):
        export_python(g)


def test_patch_embedding_concatenates_matching_grid_channels():
    spec = graph([node("a", "grid_input", channels=3), node("b", "grid_input", channels=5), node("patch", "patch_embed", patch=8, features=24), node("out", "embedding")], [("a", "patch"), ("b", "patch"), ("patch", "out")])
    a = analyze(spec)
    assert a["valid"]
    assert a["nodes"]["patch"]["input_shape"] == [8, 32, 32]
    assert a["nodes"]["patch"]["merge"] == "channels"
    assert a["parameters"] == compile_meta(spec)["parameters"]
    assert compile_meta(spec)["outputs"] == {"out": [1, 17, 24]}
