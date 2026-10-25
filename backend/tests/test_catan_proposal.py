import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.builder import compile_meta, export_python
from backend.catan_convnext import CatanConvNeXtEncoder, build_encoders
from backend.graph import analyze
from backend.main import app
from backend.schema import Graph

ROOT = Path(__file__).resolve().parents[2]


def proposal():
    return Graph.model_validate_json((ROOT / "templates/catan_convnext_small.vizml.json").read_text())


def test_proposal_owns_every_encoder_parameter_and_matches_actual_intermediates():
    graph = proposal()
    analysis = analyze(graph)
    assert analysis["valid"] and analysis["parameters"] == 1_028_301
    assert graph.metadata["input_dimensions"] == json.loads((ROOT / "templates/catan_original.vizml.json").read_text())["metadata"]["input_dimensions"]
    assert not graph.metadata.get("weights") and not graph.metadata.get("checkpoint")
    model = CatanConvNeXtEncoder()
    state = dict(model.named_parameters())
    assert sum(p.numel() for p in state.values()) == 446_128
    observed = model(torch.empty(1,1390,45,49,device="meta"), torch.empty(1,578,device="meta"), torch.empty(1,12,79,device="meta"), return_intermediates=True)
    for branch, prefix in (("actor", "actor.encoder."), ("value", "value_encoder.")):
        owned = {}
        for key, value in observed.items():
            node = next(n for n in graph.nodes if n.id == f"{branch}_{key}")
            assert node.data.model_dump()["output_shape"] == ["B", *list(value.shape)[1:]]
            for tensor in node.data.model_dump().get("parameter_tensors", []):
                name = tensor["name"].removeprefix(prefix)
                assert name not in owned
                assert tensor["shape"] == list(state[name].shape)
                owned[name] = tensor
        assert set(owned) == set(state)
        assert [e.source for e in graph.edges if e.target == f"{branch}_local_fusion"] == [f"{branch}_stage1", f"{branch}_upsample"]
    assert not analysis["executable"]  # encoder verified, game-specific integration remains explicit


def test_original_heads_and_ppg_boundaries_are_preserved():
    original = Graph.model_validate_json((ROOT / "templates/catan_original.vizml.json").read_text())
    graph = proposal()
    by_id = {n.id: n for n in graph.nodes}
    ids = {n.id for n in original.nodes if n.id.startswith(("action_", "policy", "family_", "payload_")) or any(n.id.startswith(f"{branch}_{suffix}") for branch in ("actor", "value") for suffix in ("critic", "public", "board_score", "board_draw", "signed_score", "draw_score", "outcome", "mean_value", "entities"))}
    for node in original.nodes:
        if node.id not in ids: continue
        assert by_id[node.id].data.params == node.data.params
        for field in ("parameter_count", "parameter_tensors", "output_shape", "description", "usage"):
            assert by_id[node.id].data.model_dump().get(field) == node.data.model_dump().get(field)
    assert [(e.source,e.target,e.data) for e in graph.edges if e.source in ids and e.target in ids] == [(e.source,e.target,e.data) for e in original.edges if e.source in ids and e.target in ids]
    encoders = build_encoders()
    assert not ({id(p) for p in encoders["actor"].parameters()} & {id(p) for p in encoders["value"].parameters()})


def test_encoder_cpu_outputs_are_finite_and_detail_path_uses_original_coordinates():
    torch.manual_seed(9)
    model = CatanConvNeXtEncoder("cpu").eval()
    with torch.no_grad():
        values = model(torch.randn(1,1390,45,49), torch.randn(1,578), torch.randn(1,12,79), return_intermediates=True)
    assert values["mlp2"].shape == (1,256)
    assert values["entity"].shape == (1,2205,32)
    assert values["ship"].shape == (1,12,32)
    assert all(torch.isfinite(value).all() for value in values.values())
    torch.testing.assert_close(values["local_fusion"][:,:32], values["stage1"], rtol=0,atol=0)
    torch.testing.assert_close(values["local_fusion"][:,32:], values["upsample"], rtol=0,atol=0)
    # Every stride-4 center lands at precisely its original cell, including borders.
    torch.testing.assert_close(values["upsample"][:,:,::4,::4],values["final_norm"],rtol=0,atol=0)
    assert values["entity"].shape[1] == 45*49


def test_value_only_path_skips_unused_entity_work_and_rejects_wrong_interfaces():
    model = CatanConvNeXtEncoder()
    def unused(*_): raise AssertionError("Value-only inference should skip local readout")
    handle = model.entity.register_forward_hook(unused)
    result = model(torch.empty(2,1390,45,49,device="meta"),torch.empty(2,578,device="meta"),include_entities=False)
    handle.remove()
    assert list(result) == ["hidden"] and result["hidden"].shape == (2,256)
    with pytest.raises(ValueError, match="board planes"):
        model(torch.empty(1,3,45,49,device="meta"),torch.empty(1,578,device="meta"))
    with pytest.raises(ValueError, match="context"):
        model(torch.empty(1,1390,45,49,device="meta"),torch.empty(2,578,device="meta"))


def test_nominal_receptive_field_and_mac_budget_match_the_proposal():
    model = CatanConvNeXtEncoder()
    receptive_field, jump, macs = 1,1,0
    def account(module, inputs, output):
        nonlocal macs
        if isinstance(module,torch.nn.Conv2d):
            macs += output.numel()*module.weight.shape[1]*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,torch.nn.Linear) and inputs[0].ndim == 4:
            macs += output.numel()*module.in_features
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            receptive_field += (module.kernel_size[0]-1)*jump
            jump *= module.stride[0]
            assert module.dilation == (1,1)
        if isinstance(module,torch.nn.Conv2d) or any(part in name for part in ("stage1.","stage2.","stage3.")):
            module.register_forward_hook(account)
    assert receptive_field == 97 and jump == 4
    model(torch.empty(1,1390,45,49,device="meta"),torch.empty(1,578,device="meta"))
    assert macs == 138_616_368
    assert proposal().metadata["proposal"]["spatial_macs_per_encoder"] == macs


def test_generated_templates_are_reproducible_and_source_download_is_encoder_only():
    from scripts.build_catan_proposal import build_templates
    source = json.loads((ROOT / "templates/catan_original.vizml.json").read_text())
    _, rebuilt = build_templates(source)
    assert rebuilt == json.loads((ROOT / "templates/catan_convnext_small.vizml.json").read_text())
    response = TestClient(app).get("/api/templates/catan_convnext_small/encoder.py")
    assert response.status_code == 200
    assert "class CatanConvNeXtEncoder" in response.text
    assert "not implemented by this encoder" in response.text


def test_generic_resize_block_builds_real_pytorch_and_preserves_coordinates():
    graph = Graph.model_validate({"name":"Grid resize", "nodes":[
        {"id":"input", "data":{"kind":"grid_input","label":"Input","params":{"channels":2,"height":12,"width":13}}},
        {"id":"resize", "data":{"kind":"upsample2d","label":"Align","params":{"height":45,"width":49,"mode":"bilinear","align_corners":1}}},
        {"id":"out", "data":{"kind":"embedding","label":"Output","params":{}}}
    ], "edges":[{"id":"a","source":"input","target":"resize"},{"id":"b","source":"resize","target":"out"}]})
    assert compile_meta(graph)["outputs"] == {"out":[1,2,45,49]}
    namespace = {"__name__":"test_resize"}
    exec(export_python(graph),namespace)
    model = namespace["build_model"]("cpu")
    x = torch.arange(2*12*13,dtype=torch.float32).reshape(1,2,12,13)
    torch.testing.assert_close(model({"input":x})["out"][:,:,::4,::4],x,rtol=0,atol=0)
    graph.nodes[1].data.params["mode"] = "unsupported"
    assert not analyze(graph)["valid"]
