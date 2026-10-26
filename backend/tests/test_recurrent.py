import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn
from torch.nn import functional as F

from backend.builder import compile_meta, export_python
from backend.graph import analyze
from backend.schema import Graph
from backend.main import app
from backend.torch_recurrent import SpatialRecurrent, SpatialRecurrentCell, VectorRecurrent


def node(key, kind, **params):
    return {"id": key, "data": {"kind": kind, "label": key, "params": params}}


def graph(nodes, links):
    return Graph.model_validate({"nodes": nodes, "edges": [
        {"id": str(i), "source": a, "target": b} for i, (a, b) in enumerate(links)]})


def built(spec, device="cpu"):
    scope = {"__name__": "recurrent_test"}
    exec(export_python(spec), scope)
    return scope["build_model"](device).eval()


@pytest.fixture(autouse=True, scope="module")
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_vector_outputs_and_final_states_match_native_torch(cell, bidirectional, bias):
    torch.manual_seed(21)
    ours = VectorRecurrent(cell, 3, 5, layers=2, bias=bias, bidirectional=bidirectional).eval()
    reference = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell](
        3, 5, num_layers=2, bias=bias, bidirectional=bidirectional, batch_first=True).eval()
    reference.load_state_dict(ours.core.state_dict())
    x = torch.randn(2, 6, 3)
    state = torch.randn(4 if bidirectional else 2, 2, 5)
    state = (state, torch.randn_like(state)) if cell == "lstm" else state
    expected, final = reference(x, state)
    actual, actual_state = ours(x, state, return_state=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, final)
    ours.return_sequences = False
    hidden = final[0] if cell == "lstm" else final
    expected_last = torch.cat([hidden[-2], hidden[-1]], -1) if bidirectional else hidden[-1]
    torch.testing.assert_close(ours(x, state), expected_last)


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
def test_spatial_gate_equations_independently(cell):
    torch.manual_seed(12)
    layer = SpatialRecurrentCell(cell, 2, 3, kernel=3, dilation=2)
    x, h, c = torch.randn(2, 2, 5, 7), torch.randn(2, 3, 5, 7), torch.randn(2, 3, 5, 7)
    def conv(value, module):
        return F.conv2d(value, module.weight, module.bias, padding=2, dilation=2)
    gates = conv(torch.cat((x, h), 1), layer.gates)
    if cell == "rnn":
        expected_h, expected_c = gates.tanh(), None
    elif cell == "gru":
        reset, update = gates.sigmoid().split(3, 1)
        candidate = conv(torch.cat((x, reset * h), 1), layer.candidate).tanh()
        expected_h, expected_c = update * h + (1 - update) * candidate, None
    else:
        i, f, g, o = gates.split(3, 1)
        expected_c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        expected_h = o.sigmoid() * expected_c.tanh()
    actual_h, actual_c = layer(x, h, c)
    torch.testing.assert_close(actual_h, expected_h)
    torch.testing.assert_close(actual_c, expected_c)


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
def test_spatial_causality_chunked_state_and_gradients(cell):
    torch.manual_seed(29)
    layer = SpatialRecurrent(cell, 2, 3, layers=2, dilation=2).eval()
    x = torch.randn(2, 5, 2, 4, 7, requires_grad=True)
    initial_h = torch.randn(2, 2, 3, 4, 7)
    initial = (initial_h, torch.randn_like(initial_h)) if cell == "lstm" else initial_h
    output, final = layer(x, initial, return_state=True)
    first, middle = layer(x[:, :2], initial, return_state=True)
    last, chunked_final = layer(x[:, 2:], middle, return_state=True)
    torch.testing.assert_close(torch.cat((first, last), 1), output)
    torch.testing.assert_close(chunked_final, final)
    changed = x.detach().clone()
    changed[:, 3:] += 10
    torch.testing.assert_close(layer(changed, initial)[:, :3], output[:, :3])
    assert not torch.allclose(layer(changed, initial)[:, 3:], output[:, 3:])
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in layer.parameters())
    # A normal forward does not retain state from previous calls.
    zero_first = layer(x)
    layer(changed)
    torch.testing.assert_close(layer(x), zero_first)
    layer.return_sequences = False
    torch.testing.assert_close(layer(x, initial), output[:, -1])


@pytest.mark.parametrize("kind", ["rnn", "gru", "lstm", "conv_rnn", "conv_gru", "conv_lstm"])
@pytest.mark.parametrize("sequence", [0, 1])
@pytest.mark.parametrize("bias", [0, 1])
def test_analysis_export_meta_and_weight_roundtrip(kind, sequence, bias):
    spatial = kind.startswith("conv_")
    source = node("x", "grid_sequence_input", steps=4, channels=2, height=3, width=5) if spatial else node("x", "sequence_input", tokens=4, features=2)
    spec = graph([source, node("cell", kind, hidden_size=3, layers=2, bias=bias,
        bidirectional=0 if spatial else 1, return_sequences=sequence, dropout=0.2), node("out", "embedding")],
        [("x", "cell"), ("cell", "out")])
    a = analyze(spec)
    assert a["valid"], a["errors"]
    output_shape = ([4] if sequence else []) + ([3, 3, 5] if spatial else [6])
    assert a["nodes"]["cell"]["shape"] == output_shape
    compiled = compile_meta(spec)
    assert compiled["parameters"] == a["parameters"]
    assert compiled["outputs"] == {"out": [1, *output_shape]}
    model = built(spec)
    assert sum(p.numel() for p in model.parameters()) == a["parameters"]
    x = torch.randn(2, *a["nodes"]["x"]["shape"])
    result = model({"x": x})["out"]
    assert list(result.shape) == [2, *output_shape]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = built(spec)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored({"x": x})["out"], result)


def test_grid_sequence_merge_norm_pool_and_vector_recurrence():
    spec = graph([node("a", "grid_sequence_input", steps=3, channels=2, height=4, width=5),
        node("b", "grid_sequence_input", steps=3, channels=1, height=4, width=5),
        node("join", "concat"), node("norm", "layernorm"), node("spatial", "conv_gru", hidden_size=4),
        node("pool", "global_avgpool"), node("vector", "lstm", hidden_size=6, return_sequences=0),
        node("out", "classification", units=2)],
        [("a", "join"), ("b", "join"), ("join", "norm"), ("norm", "spatial"),
         ("spatial", "pool"), ("pool", "vector"), ("vector", "out")])
    a = analyze(spec)
    assert a["valid"], a["errors"]
    assert a["nodes"]["join"]["shape"] == [3, 3, 4, 5]
    assert a["nodes"]["norm"]["parameters"] == 6
    assert a["nodes"]["pool"]["shape"] == [3, 4]
    assert compile_meta(spec)["parameters"] == a["parameters"]
    x, y = torch.randn(2, 3, 2, 4, 5), torch.randn(2, 3, 1, 4, 5)
    model = built(spec)
    assert model({"a": x, "b": y})["out"].shape == (2, 2)
    norm = model.layers[model.keys["norm"]]
    merged = torch.cat((x, y), 2)
    expected = F.layer_norm(merged.permute(0, 1, 3, 4, 2), (3,), eps=1e-6).permute(0, 1, 4, 2, 3)
    torch.testing.assert_close(norm(merged), expected)
    # Direct multiple dependencies use the same channel merge as an explicit union.
    spec.edges = [edge for edge in spec.edges if edge.target != "join" and edge.source not in {"join", "norm"}]
    from backend.schema import GraphEdge
    spec.edges.extend([GraphEdge(id="a-spatial", source="a", target="spatial"), GraphEdge(id="b-spatial", source="b", target="spatial")])
    spec.nodes = [n for n in spec.nodes if n.id not in {"join", "norm"}]
    assert analyze(spec)["nodes"]["spatial"]["merge"] == "sequence_channels"
    assert built(spec)({"a": x, "b": y})["out"].shape == (2, 2)


@pytest.mark.parametrize("kind,source,params", [
    ("rnn", "grid_input", {}), ("gru", "token_input", {}), ("lstm", "feature_input", {}),
    ("conv_rnn", "sequence_input", {}), ("conv_gru", "grid_input", {}),
    ("conv_lstm", "grid_sequence_input", {"kernel": 2}),
    ("conv_rnn", "grid_sequence_input", {"bidirectional": 1}),
    ("conv_lstm", "grid_sequence_input", {"layers": 0}),
    ("gru", "sequence_input", {"dropout": 0.2}),
    ("rnn", "sequence_input", {"activation": "gelu"}),
    ("lstm", "sequence_input", {"hidden_size": 1.5}),
    ("gru", "sequence_input", {"return_sequences": 2}),
])
def test_invalid_recurrence_reports_local_error(kind, source, params):
    spec = graph([node("x", source), node("cell", kind, **params)], [("x", "cell")])
    a = analyze(spec)
    assert not a["valid"]
    assert a["errors"][0]["node_id"] == "cell"


@pytest.mark.parametrize("target", ["conv_lstm", "concat"])
@pytest.mark.parametrize("change", [{"steps": 2}, {"height": 3}, {"width": 3}])
def test_temporal_merges_reject_incompatible_axes(target, change):
    spec = graph([node("a", "grid_sequence_input"), node("b", "grid_sequence_input", **change),
                  node("join", target)], [("a", "join"), ("b", "join")])
    assert not analyze(spec)["valid"]


@pytest.mark.parametrize("kind", ["rnn", "conv_lstm"])
def test_meta_validates_long_sequences_without_unrolling(kind):
    source = node("x", "sequence_input", tokens=1048576, features=2) if kind == "rnn" else node("x", "grid_sequence_input", steps=1048576, channels=2, height=3, width=5)
    spec = graph([source, node("cell", kind, hidden_size=3), node("out", "embedding")], [("x", "cell"), ("cell", "out")])
    assert compile_meta(spec)["outputs"]["out"][1] == 1048576


def test_spatial_state_shape_is_checked():
    layer = SpatialRecurrent("lstm", 2, 3, layers=2)
    x = torch.randn(2, 4, 2, 3, 5)
    with pytest.raises(ValueError, match="hidden, cell"):
        layer(x, torch.zeros(2, 2, 3, 3, 5))
    with pytest.raises(ValueError, match="State shape"):
        layer(x, (torch.zeros(1, 2, 3, 3, 5), torch.zeros(1, 2, 3, 3, 5)))


def test_websocket_switches_recurrent_templates_and_recovers_from_unknown_kind():
    templates = Path(__file__).resolve().parents[2] / "templates"
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "graph.changed", "request_id": "bad-kind", "graph": {
            "nodes": [node("unknown", "unknown_kind")], "edges": []}})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "Unsupported node type at nodes.0.data.kind" in error["message"]
        for kind in ["rnn", "gru", "lstm", "conv_rnn", "conv_gru", "conv_lstm"]:
            spec = json.loads((templates / f"{kind}.vizml.json").read_text())
            for type in ["graph.changed", "model.compile"]:
                ws.send_json({"type": type, "request_id": kind, "graph": spec})
                response = ws.receive_json()
                assert response["request_id"] == kind
                assert response["type"] != "error", response
                if type == "graph.changed":
                    assert response["analysis"]["valid"], response
                else:
                    assert response["result"]["outputs"] == {"classes": [1, 5]}
