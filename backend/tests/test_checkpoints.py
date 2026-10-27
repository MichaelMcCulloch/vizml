import copy
import io
import os
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend import checkpoints as cp
from backend.builder import export_python
from backend.graph import analyze
from backend.main import app
from backend.schema import Graph


@pytest.fixture(autouse=True)
def local_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZML_CHECKPOINT_DIR", str(tmp_path / "cache"))


def tiny_graph():
    return Graph.model_validate({"name": "Tiny", "nodes": [
        {"id": "input", "data": {"kind": "feature_input", "label": "Input", "params": {"features": 3}}},
        {"id": "out", "data": {"kind": "regression", "label": "Output", "params": {"units": 2}}},
    ], "edges": [{"id": "edge", "source": "input", "target": "out"}]})


def schematic():
    return {"version": 1, "name": "Supplied snapshot", "metadata": {"parameter_counts": {"total": 8}, "checkpoint": {"sha256": "different-revision"}}, "nodes": [
        {"id": "input", "position": {"x": 0, "y": 0}, "data": {"kind": "feature_input", "label": "Raw features", "params": {}, "output_shape": ["B", 3], "parameter_count": 0}},
        {"id": "custom", "position": {"x": 250, "y": 0}, "data": {"kind": "dense", "label": "Custom adapter", "params": {"bias": True, "indices": [0, 2]}, "description": "A supplied forward claim", "output_shape": ["B", "L", 2], "parameter_count": 8, "parameter_tensors": [{"name": "adapter.weight", "shape": [2, 3]}, {"name": "adapter.bias", "shape": [2]}]}}
    ], "edges": [{"id": "edge", "source": "input", "target": "custom", "label": "claimed connection", "data": {"role": "data"}}]}


def save(tmp_path, payload, filename="model.pt"):
    path = tmp_path / filename
    torch.save(payload, path)
    return path


def test_schematic_is_not_reinterpreted_as_builder():
    graph = Graph.model_validate(schematic())
    assert graph.mode == "schematic"
    a = analyze(graph)
    assert a["valid"] and not a["executable"]
    assert a["parameters"] == 8 and a["owned_tensors"] == 2
    assert a["nodes"]["custom"]["display_shape"] == ["L", 2]
    assert a["nodes"]["custom"]["shape"] == [14, 2]
    assert graph.model_dump()["nodes"][1]["data"]["description"] == "A supplied forward claim"
    assert graph.edges[0].data["role"] == "data"
    with pytest.raises(ValueError, match="escriptive"):
        export_python(graph)


def test_schematic_duplicate_ownership_and_count_and_cycle_are_checked():
    data = schematic()
    data["nodes"][0]["data"]["parameter_tensors"] = [{"name": "adapter.weight", "shape": [2, 3]}]
    data["nodes"][0]["data"]["parameter_count"] = 6
    assert not analyze(Graph.model_validate(data))["valid"]
    data = schematic()
    data["metadata"]["parameter_counts"]["total"] = 9
    assert not analyze(Graph.model_validate(data))["valid"]
    data = schematic()
    data["edges"].append({"id": "cycle", "source": "custom", "target": "input"})
    assert any("cycle" in e["message"] for e in analyze(Graph.model_validate(data))["errors"])


def test_schematic_shapes_use_bounded_preview_not_fixed_symbolic_dim():
    data = schematic()
    data["metadata"]["validation"] = {"actual_cpu_observation_shapes": {"legal": []}}
    assert analyze(Graph.model_validate(data))["nodes"]["custom"]["shape"] == [14, 2]


def test_safe_import_pair_and_export_preserve_tensors_and_training_payload(tmp_path):
    state = {"adapter.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "adapter.bias": torch.tensor([.5, -.5]), "lookup": torch.tensor([2, 4, 6], dtype=torch.int64)}
    payload = {"model": state, "optimizer": {"state": {3: {"step": torch.tensor(7.), "moment": torch.ones(2, 3)}}}, "decisions": 100, "config": {"feature_count": 3}, "architecture": "custom-test"}
    path = save(tmp_path, payload)
    before = path.read_bytes()
    info = cp.register_file(path, "original.pt")
    assert info["tensor_count"] == 3 and info["tensor_elements"] == 11
    assert info["has_weights"] and info["has_optimizer"]
    assert info["architecture"] == "custom-test" and info["decisions"] == 100
    paired = cp.bind_checkpoint(info["id"], Graph.model_validate(schematic()))
    assert paired["report"]["matched"] == 2
    assert paired["report"]["snapshot_match"] is False
    graph = Graph.model_validate(paired["graph"])
    exported = cp.export_checkpoint(graph, "checkpoint")
    actual = torch.load(cp.cache_path(exported["id"]), weights_only=True)
    for name, tensor in state.items():
        torch.testing.assert_close(actual["model_state_dict"][name], tensor, rtol=0, atol=0)
    torch.testing.assert_close(actual["original_checkpoint"]["optimizer"]["state"][3]["moment"], torch.ones(2, 3))
    assert "weights" not in actual["graph"]["metadata"]
    assert actual["graph"]["nodes"][1]["data"]["params"]["indices"] == [0, 2]
    reimport = cp.register_file(cp.cache_path(exported["id"]), "roundtrip.pt")
    assert reimport["architecture"] == "custom-test" and reimport["decisions"] == 100
    assert cp.compatibility(reimport, Graph.model_validate(reimport["graph"]))["compatible"]
    raw = cp.export_checkpoint(graph, "state_dict")
    assert set(torch.load(cp.cache_path(raw["id"]), weights_only=True)) == set(state)
    assert path.read_bytes() == before


def test_tensor_shape_mismatch_is_not_bound(tmp_path):
    info = cp.register_file(save(tmp_path, {"adapter.weight": torch.zeros(3, 2), "adapter.bias": torch.zeros(2)}), "bad.pt")
    report = cp.compatibility(info, Graph.model_validate(schematic()))
    assert not report["compatible"] and report["mismatches"][0]["expected"] == [2, 3]
    with pytest.raises(ValueError, match="does not match"):
        cp.bind_checkpoint(info["id"], Graph.model_validate(schematic()))


def test_generated_checkpoints_load_strictly_and_remain_attached_after_layout_edits():
    graph = tiny_graph()
    info = cp.export_checkpoint(graph, "checkpoint")
    assert info["weights_origin"] == "initialized"
    state = torch.load(cp.cache_path(info["id"]), weights_only=True)["model_state_dict"]
    cp._model(graph, "cpu").load_state_dict(state, strict=True)
    paired = Graph.model_validate(cp.bind_checkpoint(info["id"], graph)["graph"])
    paired.nodes[0].position.x += 123
    paired.name = "Renamed"
    export = cp.export_checkpoint(paired, "state_dict")
    for name, value in torch.load(cp.cache_path(export["id"]), weights_only=True).items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    paired.nodes[0].data.params["features"] = 4
    with pytest.raises(ValueError, match="semantics changed"):
        cp.export_checkpoint(paired, "checkpoint")


def test_common_training_prefix_is_matched(tmp_path):
    graph = tiny_graph()
    model = cp._model(graph, "cpu")
    info = cp.register_file(save(tmp_path, {"state_dict": {f"module._orig_mod.{k}": v for k, v in model.state_dict().items()}}), "wrapped.pth")
    assert cp.compatibility(info, graph)["compatible"]


def test_large_architecture_pt_is_allocation_free_and_initialization_is_bounded(monkeypatch):
    graph = tiny_graph()
    graph.nodes[0].data.params["features"] = 65535
    graph.nodes[1].data.params["units"] = 65535
    original = cp._model
    def forbid_cpu(g, device="meta"):
        assert device == "meta", "Large architecture must never allocate weights"
        return original(g, device)
    monkeypatch.setattr(cp, "_model", forbid_cpu)
    info = cp.export_checkpoint(graph, "architecture")
    assert info["tensor_count"] == 0 and info["file_bytes"] < 10000
    with pytest.raises(ValueError, match="512 MiB"):
        cp.export_checkpoint(graph, "checkpoint")


def test_imported_pickle_functions_are_never_executed(tmp_path):
    marker = tmp_path / "should_not_exist"
    class Untrusted:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)
    path = save(tmp_path, {"model": Untrusted()})
    with pytest.raises(ValueError, match="weights-only"):
        cp.register_file(path, "unsafe.pt")
    assert not marker.exists()
    assert not list(cp.cache_dir().glob("*.pt"))


def test_binary_api_upload_export_bind_download_and_error_recovery():
    client = TestClient(app)
    graph = tiny_graph().model_dump()
    result = client.post("/api/checkpoints/export", json={"graph": graph, "mode": "checkpoint"})
    assert result.status_code == 200, result.text
    export = result.json()
    downloaded = client.get(export["download_url"])
    assert downloaded.status_code == 200
    imported = client.post("/api/checkpoints/import?filename=model.pt", content=downloaded.content, headers={"Content-Type": "application/octet-stream"})
    assert imported.status_code == 200, imported.text
    info = imported.json()
    paired = client.post(f"/api/checkpoints/{info['id']}/bind", json=graph)
    assert paired.status_code == 200 and paired.json()["report"]["compatible"]
    assert client.get(f"/api/checkpoints/{info['id']}").json()["tensor_count"] == 2
    assert client.post("/api/checkpoints/import", content=b"not a checkpoint").status_code == 400
    assert client.get("/api/checkpoints/not-an-id/download").status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"


def test_edited_schematic_preserves_source_metadata_without_claiming_original_totals():
    data = schematic()
    data["mode"] = "schematic"
    data["metadata"]["source_edited"] = True
    data["nodes"].append({"id": "new", "position": {"x": 0, "y": 0}, "data": {
        "kind": "dense", "label": "New", "params": {}, "output_shape": ["B", 1],
        "parameter_count": 1, "declaration_pending": True,
    }})
    result = analyze(Graph.model_validate(data))
    assert result["valid"] and not result["executable"]
    assert result["parameters"] == 9
    assert any("original snapshot" in issue["message"] for issue in result["warnings"])
    assert any("placeholder" in issue["message"] for issue in result["warnings"])


def test_schematic_descriptions_do_not_change_checkpoint_signature():
    graph = Graph.model_validate(schematic())
    before = cp.signature(graph)
    graph.nodes[1].data.description = "A clearer description of the same operation"
    assert cp.signature(graph) == before
