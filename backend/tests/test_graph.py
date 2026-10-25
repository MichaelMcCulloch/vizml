import pytest
import torch
from fastapi.testclient import TestClient

from backend.builder import compile_meta, export_python
from backend.graph import analyze
from backend.main import app
from backend.schema import Graph


def node(key, kind, **params):
    return {"id": key, "data": {"kind": kind, "label": key, "params": params}}


def graph(nodes, edges):
    return Graph.model_validate({"nodes": nodes, "edges": [{"id": str(i), "source": a, "target": b} for i, (a, b) in enumerate(edges)]})


def multimodal():
    return graph([
        node("image", "grid_input", width=32, height=32, channels=3),
        node("conv", "conv2d", channels=8, kernel=3, padding=1),
        node("pool", "maxpool2d"), node("features", "feature_input", features=12),
        node("union", "concat"), node("hidden", "dense", units=16),
        node("classes", "classification", units=5),
    ], [("image", "conv"), ("conv", "pool"), ("pool", "union"), ("features", "union"), ("union", "hidden"), ("hidden", "classes")])


def test_grid_and_vector_union_flattens_to_one_long_feature_set():
    result = analyze(multimodal())
    assert result["valid"]
    assert result["nodes"]["pool"]["shape"] == [8, 16, 16]
    assert result["nodes"]["union"]["shape"] == [2060]
    assert result["nodes"]["union"]["merge"] == "flatten_concat"
    assert result["parameters"] == (3 * 3 * 3 + 1) * 8 + (2060 + 1) * 16 + (16 + 1) * 5


def test_actual_pytorch_meta_forward_matches_shape_analysis():
    spec = multimodal()
    result = compile_meta(spec)
    assert result["outputs"] == {"classes": [1, 5]}
    assert result["parameters"] == analyze(spec)["parameters"]
    assert result["allocated_bytes"] == 0


def test_generated_model_cpu_logits_and_one_hot_predictions():
    namespace = {"__name__": "test_generated"}
    exec(export_python(multimodal()), namespace)
    model = namespace["build_model"]("cpu").eval()
    inputs = {"image": torch.randn(2, 3, 32, 32), "features": torch.randn(2, 12)}
    logits = model(inputs)["classes"]
    predictions = model.predict(inputs)["classes"]
    assert logits.shape == (2, 5)
    assert predictions.shape == (2, 5)
    assert torch.all(predictions.sum(dim=1) == 1)
    assert torch.equal(predictions.argmax(1), logits.argmax(1))


def test_grid_branches_concatenate_channels_instead_of_flattening():
    spec = graph([node("a", "grid_input", channels=3), node("b", "grid_input", channels=7), node("merge", "conv2d", channels=12), node("out", "regression")], [("a", "merge"), ("b", "merge"), ("merge", "out")])
    result = analyze(spec)
    assert result["valid"]
    assert result["nodes"]["merge"]["input_shape"] == [10, 32, 32]
    assert compile_meta(spec)["parameters"] == result["parameters"]


def test_dilation_padding_and_grouped_convolution_match_torch():
    spec = graph([node("x", "grid_input", width=48, height=32, channels=8), node("pad", "padding", padding=2), node("conv", "conv2d", channels=16, kernel=3, stride=2, dilation=2, padding=0, groups=4), node("pool", "maxpool2d", kernel=3, stride=2, dilation=2, padding=1), node("out", "regression", units=3)], [("x", "pad"), ("pad", "conv"), ("conv", "pool"), ("pool", "out")])
    result = analyze(spec)
    assert result["valid"]
    assert result["nodes"]["conv"]["shape"] == [16, 16, 24]
    assert result["nodes"]["pool"]["shape"] == [16, 7, 11]
    compiled = compile_meta(spec)
    assert compiled["outputs"] == {"out": [1, 3]}
    assert compiled["parameters"] == result["parameters"]


def test_residual_add_requires_matching_shapes_and_runs_in_pytorch():
    spec = graph([node("x", "grid_input", channels=8), node("conv", "conv2d", channels=8), node("skip", "residual", activation="relu"), node("out", "regression")], [("x", "conv"), ("x", "skip"), ("conv", "skip"), ("skip", "out")])
    result = analyze(spec)
    assert result["valid"]
    assert result["nodes"]["skip"]["merge"] == "add"
    assert result["nodes"]["skip"]["shape"] == [8, 32, 32]
    assert compile_meta(spec)["outputs"] == {"out": [1, 1]}
    spec.nodes[1].data.params["channels"] = 16
    result = analyze(spec)
    assert not result["valid"]
    assert any("identical shapes" in issue["message"] for issue in result["errors"])


def test_maximum_input_is_allocation_free():
    spec = graph([node("grid", "grid_input", width=256, height=256, channels=65535), node("features", "feature_input", features=65535), node("head", "classification", units=65535)], [("grid", "head"), ("features", "head")])
    result = analyze(spec)
    assert result["valid"]
    assert result["nodes"]["head"]["input_shape"] == [256 * 256 * 65535 + 65535]
    assert result["parameters"] > 100_000_000_000_000
    assert compile_meta(spec)["allocated_bytes"] == 0


def test_zero_feature_input_is_supported():
    spec = graph([node("x", "feature_input", features=0), node("out", "regression")], [("x", "out")])
    assert analyze(spec)["valid"]
    assert compile_meta(spec)["parameters"] == 1


@pytest.mark.parametrize("kind,params", [("feature_input", {"features": 65536}), ("grid_input", {"width": 257}), ("grid_input", {"channels": 0}), ("feature_input", {"features": 2.5}), ("feature_input", {"features": True})])
def test_input_bounds_are_enforced(kind, params):
    assert not analyze(graph([node("x", kind, **params)], []))["valid"]


def test_cycles_dangling_edges_duplicates_and_terminal_outputs_are_rejected():
    spec = graph([node("a", "dense"), node("b", "dense")], [("a", "b"), ("b", "a")])
    assert any("cycle" in issue["message"] for issue in analyze(spec)["errors"])
    spec = graph([node("a", "feature_input"), node("b", "regression"), node("c", "dense")], [("a", "b"), ("b", "c"), ("a", "b"), ("missing", "c")])
    result = analyze(spec)
    assert not result["valid"]
    assert len(result["errors"]) >= 3


def test_generated_source_handles_untrusted_labels_and_ids_as_data():
    unusual_id = "__import__('os').system('echo unsafe')"
    spec = graph([node(unusual_id, "feature_input", features=2), node("out", "regression")], [(unusual_id, "out")])
    assert compile_meta(spec)["outputs"] == {"out": [1, 1]}


def test_websocket_edit_compile_and_error_recovery():
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "graph.changed", "request_id": "r1", "graph": multimodal().model_dump()})
        event = ws.receive_json()
        assert event["type"] == "graph.analyzed"
        assert event["request_id"] == "r1"
        assert event["analysis"]["valid"]
        ws.send_json({"type": "model.compile", "request_id": "r2", "graph": multimodal().model_dump()})
        assert ws.receive_json()["result"]["outputs"] == {"classes": [1, 5]}
        ws.send_json({"type": "model.export", "request_id": "r3", "graph": multimodal().model_dump()})
        source = ws.receive_json()["source"]
        assert "class VisualModel(nn.Module)" in source
        ws.send_json({"type": "ping", "request_id": "alive"})
        assert ws.receive_json() == {"type": "pong", "request_id": "alive"}


def test_regression_prediction_preserves_continuous_outputs():
    spec = graph([node("x", "feature_input", features=4), node("out", "regression", units=2)], [("x", "out")])
    namespace = {"__name__": "test_generated"}
    exec(export_python(spec), namespace)
    model = namespace["build_model"]("cpu")
    inputs = {"x": torch.randn(3, 4)}
    assert torch.equal(model(inputs)["out"], model.predict(inputs)["out"])


def test_mismatched_grid_sizes_need_explicit_flatten():
    spec = graph([node("a", "grid_input", width=16), node("b", "grid_input", width=32), node("union", "concat")], [("a", "union"), ("b", "union")])
    assert not analyze(spec)["valid"]


def test_invalid_upstream_nodes_produce_localized_errors():
    spec = graph([node("x", "feature_input", features=12), node("conv", "conv2d"), node("out", "classification")], [("x", "conv"), ("conv", "out")])
    result = analyze(spec)
    assert {e["node_id"] for e in result["errors"]} == {"conv", "out"}


@pytest.mark.parametrize("kind", ["conv2d", "maxpool2d"])
def test_dilation_one_is_standard_and_zero_is_invalid(kind):
    spec = graph([node("x", "grid_input"), node("layer", kind, dilation=1), node("out", "regression")], [("x", "layer"), ("layer", "out")])
    assert analyze(spec)["valid"]
    assert compile_meta(spec)["outputs"] == {"out": [1, 1]}
    spec.nodes[1].data.params["dilation"] = 0
    result = analyze(spec)
    assert not result["valid"]
    assert any("Dilation must be an integer from 1" in e["message"] for e in result["errors"])


def generated_model(spec, device="cpu"):
    namespace = {"__name__": "test_generated"}
    exec(export_python(spec), namespace)
    return namespace["build_model"](device), namespace


def test_global_average_pool_and_embedding_preserve_continuous_features():
    spec = graph([node("x", "grid_input", channels=4, height=5, width=7), node("pool", "global_avgpool"), node("features", "embedding")], [("x", "pool"), ("pool", "features")])
    model, _ = generated_model(spec)
    x = torch.randn(2, 4, 5, 7)
    torch.testing.assert_close(model({"x": x})["features"], x.mean((-2, -1)))
    torch.testing.assert_close(model.predict({"x": x})["features"], x.mean((-2, -1)))
    assert analyze(spec)["nodes"]["pool"]["shape"] == [4]
    assert compile_meta(spec)["parameters"] == 0


@pytest.mark.parametrize("spatial", [True, False])
def test_layernorm_normalizes_channels_without_flattening_spatial_axes(spatial):
    input_node = node("x", "grid_input", channels=4, height=5, width=7) if spatial else node("x", "feature_input", features=4)
    spec = graph([input_node, node("norm", "layernorm", eps=1e-6), node("features", "embedding")], [("x", "norm"), ("norm", "features")])
    model, _ = generated_model(spec)
    x = torch.randn(2, 4, 5, 7) if spatial else torch.randn(2, 4)
    actual = model({"x": x})["features"]
    expected = torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1) if spatial else x, (4,), eps=1e-6)
    if spatial:
        expected = expected.permute(0, 3, 1, 2)
    torch.testing.assert_close(actual, expected)
    assert analyze(spec)["parameters"] == 8


def test_convnext_stage_has_internal_residuals_and_per_pixel_channel_mlps():
    spec = graph([node("x", "grid_input", channels=8, height=9, width=11), node("stage", "convnext_stage", blocks=2), node("features", "embedding")], [("x", "stage"), ("stage", "features")])
    analysis = analyze(spec)
    assert analysis["valid"]
    assert analysis["parameters"] == 2 * (8 * 8 * 8 + 58 * 8)
    assert analysis["nodes"]["stage"]["shape"] == [8, 9, 11]
    assert compile_meta(spec)["outputs"] == {"features": [1, 8, 9, 11]}
    model, _ = generated_model(spec)
    stage = model.layers[model.keys["stage"]]
    assert len(stage) == 2
    for block in stage:
        assert block.dwconv.groups == 8
        assert block.dwconv.kernel_size == (7, 7)
        assert block.pwconv1.out_features == 32
        assert block.pwconv2.out_features == 8
        assert block.norm.eps == 1e-6
        torch.testing.assert_close(block.gamma, torch.full((8,), 1e-6))
        with torch.no_grad():
            block.pwconv2.weight.zero_()
            block.pwconv2.bias.zero_()
    x = torch.randn(2, 8, 9, 11)
    torch.testing.assert_close(model({"x": x})["features"], x, rtol=0, atol=0)


def test_disabling_layerscale_removes_its_parameters():
    spec = graph([node("x", "grid_input", channels=8), node("stage", "convnext_stage", blocks=1, layer_scale=0), node("features", "embedding")], [("x", "stage"), ("stage", "features")])
    assert analyze(spec)["parameters"] == 8 * 8 * 8 + 57 * 8
    model, _ = generated_model(spec)
    assert model.layers[model.keys["stage"]][0].gamma is None


@pytest.mark.parametrize("params", [{"blocks": 0}, {"kernel": 6}, {"expansion": 17}, {"layer_scale": -0.1}, {"eps": 0}, {"drop_path": 1}])
def test_convnext_parameter_constraints(params):
    spec = graph([node("x", "grid_input"), node("stage", "convnext_stage", **params), node("features", "embedding")], [("x", "stage"), ("stage", "features")])
    assert not analyze(spec)["valid"]
