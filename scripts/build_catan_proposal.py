"""Build portable, untrained Catan templates from a supplied architecture JSON.

Run: uv run python -m scripts.build_catan_proposal path/to/source.vizml.json
This reads architecture annotations only; it never opens or copies a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from backend.catan_convnext import CatanConvNeXtEncoder
from backend.graph import analyze
from backend.schema import Graph

BASELINE = 3_680_301


def build_templates(source_graph: dict) -> tuple[dict, dict]:
    original = copy.deepcopy(source_graph)
    original.update(name="Catan · original residual CNN", template="catan_original", mode="schematic")
    # Templates carry architecture, never local cache attachments or weight stats.
    original["metadata"].pop("weights", None)
    original["metadata"].pop("source_files", None)
    original["metadata"].get("checkpoint", {}).pop("path", None)
    for node in original["nodes"]:
        node["data"].pop("weight_statistics", None)
        node["data"].pop("verified_cpu_example", None)
    graph = copy.deepcopy(original)
    graph.update(name="Catan · ConvNeXt Small · untrained", template="catan_convnext_small")
    model = CatanConvNeXtEncoder()
    state = dict(model.named_parameters())
    observed = model(torch.empty(1, 1390, 45, 49, device="meta"), torch.empty(1, 578, device="meta"), torch.empty(1, 12, 79, device="meta"), return_intermediates=True)
    replaced = {node["id"] for node in graph["nodes"] if any(node["id"].startswith(f"{b}_") and (node["id"][len(b)+1:].startswith("block") or node["id"][len(b)+1:] in observed) for b in ("actor", "value"))}
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in replaced]
    graph["edges"] = [edge for edge in graph["edges"] if (edge["source"] in replaced and edge["target"] not in replaced) or (edge["source"] not in replaced and edge["target"] not in replaced) or edge["target"] in ("actor_ship", "value_ship")]
    node_data = {}
    for node in graph["nodes"]:
        node["data"]["provenance"] = "Retained supplied Catan interface/head architecture; no checkpoint weights."
    for branch, base_y in (("actor", 320), ("value", 1180)):
        prefix = "actor.encoder" if branch == "actor" else "value_encoder"
        def add(key, kind, label, description, params=None, modules=(), x=0, y=0, **annotations):
            owned = [{"name": f"{prefix}.{name}", "shape": list(tensor.shape), "elements": tensor.numel()} for name, tensor in state.items() if any(name == path or name.startswith(path + ".") for path in modules)]
            d = {"kind": kind, "label": label, "params": params or {}, "description": description,
                 "output_shape": ["B", *list(observed[key].shape)[1:]], "parameter_count": sum(t["elements"] for t in owned),
                 "module_paths": [f"{prefix}.{path}" for path in modules], "provenance": "Untrained ConvNeXt proposal encoder; implemented in backend/catan_convnext.py.", **annotations}
            d["view_label"] = {"stem": "1,390 → 32", "conditioned": "+ known context", "stage1": "Stage 1 · 32 · 1 block", "down1": "↓2 · 32 → 48", "stage2": "Stage 2 · 48 · 1 block", "down2": "↓2 · 48 → 64", "stage3": "Stage 3 · 64 · 3 blocks", "final_norm": "Final channel norm", "upsample": "↑4 · align coordinates", "local_fusion": "32 local + 64 coarse", "entity": "32-D entities"}.get(key, label)
            if owned: d["parameter_tensors"] = owned
            result = {"id": f"{branch}_{key}", "type": "layer", "position": {"x": x, "y": base_y + y}, "data": d}
            graph["nodes"].append(result); node_data[result["id"]] = result
        def link(source, target, label=None, role="data"):
            source = source if source in ("board", "context") or source.startswith(f"{branch}_") else f"{branch}_{source}"
            target = f"{branch}_{target}"
            graph["edges"].append({"id": f"{source}-{target}", "source": source, "target": target, **({"label": label} if label else {}), "data": {"role": role}})
        add("stem", "conv2d", "Board projection · 1,390→32", "Project all original board feature planes at every original cell. This fresh 32-channel stem does not reuse the old checkpoint's learned channels or channel-specific refinement offsets.", {"channels": 32, "input_channels": 1390, "kernel": 1, "stride": 1, "padding": 0, "dilation": 1, "activation": "none"}, ("stem",), 260)
        add("context", "dense", "Context · 578→128", "The same non-board observation interface, embedded separately by each encoder.", {"units": 128, "input_features": 578, "activation": "silu"}, ("context.0",), 260, -240)
        add("context_norm", "layernorm", "Context norm · 128", "Normalize the context embedding per observation.", {"features": 128, "eps": 1e-5}, ("context.2",), 520, -240)
        add("condition", "dense", "Context biases · 128→32", "Broadcast 32 context-conditioned biases into the spatial stem.", {"units": 32, "input_features": 128, "activation": "none"}, ("condition",), 780, -240)
        add("conditioned", "add", "Board + known context", "Broadcast addition conditions every original board cell on known non-board information.", {}, (), 1040)
        for number, channels, blocks, shape_x in ((1, 32, 1, 1300), (2, 48, 1, 1820), (3, 64, 3, 2340)):
            add(f"stage{number}", "convnext_stage", f"Stage {number} · {channels} channels · {blocks} {'block' if blocks == 1 else 'blocks'}", "7×7 depthwise convolution → channel LayerNorm → 2× pointwise expansion → GELU → projection → LayerScale → residual addition. Dilation 1; padding 3. Dropout and stochastic depth are disabled.", {"blocks": blocks, "kernel": 7, "dilation": 1, "padding": 3, "expansion": 2, "layer_scale": 1e-6, "eps": 1e-6, "drop_path": 0}, (f"stage{number}",), shape_x, receptive_field=[7, 21, 97][number-1])
            if number < 3:
                out = 48 if number == 1 else 64
                add(f"down{number}", "conv2d", f"Downsample · {channels}→{out}", "Channel LayerNorm followed by a learned 3×3 stride-2 convolution, with padding 1. Odd grid sizes preserve original coordinate centers and border coverage.", {"input_channels": channels, "channels": out, "kernel": 3, "stride": 2, "padding": 1, "dilation": 1, "activation": "none"}, (f"down{number}",), shape_x + 260, pre_operations=[{"kind": "layernorm", "features": channels, "eps": 1e-5}])
        add("final_norm", "layernorm", "Coarse map · channel norm", "Normalize the final 64-channel, 12×13 map. Its nominal receptive field is 97×97 original cells.", {"features": 64, "eps": 1e-5}, ("final.norm",), 2600, receptive_field=97)
        add("mean", "globalavgpool2d", "Global mean · 64", "Pool all coarse locations, covering the complete input board.", {}, (), 2860)
        add("max", "globalmaxpool2d", "Global maximum · 64", "Retain strong spatial signals alongside the mean readout.", {}, (), 2860, 160)
        add("global_join", "concat", "Mean + max + context · 256", "Concatenate 64 means, 64 maxima and the 128-dimensional context embedding. This replaces the old 320-wide join while preserving the 256-wide hidden output.", {"axis": "features", "features": 256}, (), 3120)
        add("state", "dense", "Global hidden · 256→256", "Learn the same-width global representation used by the original policy and outcome heads.", {"units": 256, "input_features": 256, "activation": "silu"}, ("state.0",), 3380)
        add("state_norm", "layernorm", "Global norm · 256", "Normalize the global hidden state; no stochastic dropout is applied.", {"features": 256, "eps": 1e-5}, ("state.2",), 3640)
        add("mlp1", "dense", "Global MLP · 256→256", "Retain the first original serial global MLP layer with SiLU. This MLP has no residual bypass.", {"units": 256, "input_features": 256, "activation": "silu"}, ("global_residual.0",), 3900)
        add("mlp2", "dense", "Global output · 256", "Retain the final 256-wide interface for policy queries, family logits and PPG-compatible outcome heads.", {"units": 256, "input_features": 256, "activation": "none"}, ("global_residual.2",), 4160)
        add("upsample", "upsample2d", "Coarse → original coordinates", "Bilinearly align the 12×13 coarse map with the 45×49 board using align_corners=True. Coarse samples have centers 0,4,…,44 and 0,4,…,48, so this maps them back exactly to the original coordinate system.", {"height": 45, "width": 49, "mode": "bilinear", "align_corners": 1}, (), 2860, 400)
        add("local_fusion", "concat", "Local 32 + coarse 64 → 96", "Concatenate full-resolution stage-1 features and upsampled coarse features along channels. This is a detail-preserving concatenation, not a residual addition.", {"axis": "channels", "features": 96}, (), 3120, 400)
        add("entity", "dense", "Entity descriptors · 96→32", "Project each of the same 2,205 board coordinates to 32 features. The original fixed entity gather and per-legal-action head keep their existing interfaces.", {"units": 32, "input_features": 96, "activation": "none"}, ("entity",), 3380, 400, effective_use="Legal-action scoring" if branch == "actor" else "Stored, unused adapter retained for original module interface")
        add("ship", "dense", "Ship attributes · 79→32", "Keep the same 12×79 ship-attribute input and 32-wide projection before original entity-table assembly.", {"units": 32, "input_features": 79, "activation": "none"}, ("ship",), 3380, 600)
        link("stage1", "local_fusion", "High-resolution detail · 32 channels", "detail_skip")
        for source, target in [("board", "stem"), ("context", "context"), ("context", "context_norm"), ("context_norm", "condition"), ("stem", "conditioned"), ("condition", "conditioned"), ("conditioned", "stage1"), ("stage1", "down1"), ("down1", "stage2"), ("stage2", "down2"), ("down2", "stage3"), ("stage3", "final_norm"), ("final_norm", "mean"), ("final_norm", "max"), ("mean", "global_join"), ("max", "global_join"), ("context_norm", "global_join"), ("global_join", "state"), ("state", "state_norm"), ("state_norm", "mlp1"), ("mlp1", "mlp2"), ("final_norm", "upsample"), ("upsample", "local_fusion"), ("local_fusion", "entity")]:
            # The raw context input and the encoder's first context layer share a suffix.
            if source == "context" and target == "context_norm": source = f"{branch}_context"
            link(source, target)
    # Reposition retained application heads after the compact encoders.
    for node in graph["nodes"]:
        if node["id"] in node_data: continue
        key = node["id"]
        if key == "observation": node["position"] = {"x": -260, "y": 0}
        elif key == "board": node["position"] = {"x": 0, "y": 320}
        elif key == "context": node["position"] = {"x": 0, "y": 80}
        elif key == "legal": node["position"] = {"x": 3900, "y": -80}
        elif key in ("actor_entities", "value_entities"):
            node["position"] = {"x": 3640, "y": 720 if key.startswith("actor") else 1580}
        else:
            node["position"]["x"] = 4420 + max(0, node["position"]["x"] - 7600) * .65
            node["position"]["y"] = max(0, node["position"]["y"] - 1200) * .65
    per_encoder = sum(p.numel() for p in model.parameters())
    total = sum(n["data"]["parameter_count"] for n in graph["nodes"])
    scopes = []
    for branch in ("actor", "value"):
        title = branch.capitalize()
        scopes.append({"id": f"{branch}_pyramid", "label": f"{title} pyramid", "ids": ["board", *[f"{branch}_{key}" for key in ("stem", "conditioned", "stage1", "down1", "stage2", "down2", "stage3", "final_norm")]]})
        scopes.append({"id": f"{branch}_cnn", "label": f"{title} encoder + context", "ids": ["board", "context", *[f"{branch}_{key}" for key in observed if key != "ship"]]})
    scopes.insert(1, {"id": "actor_local", "label": "Local + global fusion", "ids": [f"actor_{key}" for key in ("stage1", "final_norm", "upsample", "local_fusion", "entity", "ship", "entities")]})
    scopes += [{"id": "policy", "label": "Legal-action policy", "ids": [n["id"] for n in graph["nodes"] if n["id"].startswith(("action_", "policy", "family_", "payload_")) or n["id"] in ("legal", "actor_entities", "actor_mlp2")]}, {"id": "outcomes", "label": "PPG outcome heads", "ids": [n["id"] for n in graph["nodes"] if any(n["id"].startswith(f"{b}_{key}") for b in ("actor", "value") for key in ("critic", "public", "board_score", "board_draw", "signed_score", "draw_score", "outcome", "mean_value", "mlp2"))]}]
    graph["metadata"] = {
        "description": "Untrained Catan ConvNeXt Small proposal. Verified encoder implementation; original supplied action/outcome architecture retained as a descriptive integration contract.",
        "architecture": "hex-convnext-small-ppg-proposal-v1", "algorithm": source_graph["metadata"]["algorithm"],
        "input_dimensions": copy.deepcopy(source_graph["metadata"]["input_dimensions"]),
        "parameter_counts": {"total": total, "actor_encoder": per_encoder, "independent_value_encoder": per_encoder, "actor_legal_policy": 134945, "actor_auxiliary_outcome_head": 550, "independent_outcome_head": 550, "stored_unused_value_adapters": 5664},
        "receptive_field": {"per_final_spatial_location_nominal": [97, 97], "stage_sizes": [[45,49,32],[23,25,48],[12,13,64]], "dilation": 1, "padding": "3 for 7×7 depthwise; 1 for 3×3 downsampling", "measured_effective_receptive_field": False},
        "proposal": {"untrained": True, "baseline_parameters": BASELINE, "proposed_parameters": total, "parameter_reduction_percent": 100*(1-total/BASELINE), "baseline_spatial_macs_per_encoder": 3220499520, "spatial_macs_per_encoder": 138616368, "spatial_mac_ratio": 3220499520/138616368, "encoder_parameters": per_encoder, "encoder_source": "/api/templates/catan_convnext_small/encoder.py", "validation_scope": "CNN encoder only. Full original Catan game forward and PPG trainer require application integration."},
        "important_details": ["The 1,390 board planes, 578 context features, original entity coordinates, 82,819 action IDs, 49 action families and variable legal list L are preserved.", "Separate actor and value encoders each use channels 32→48→64 and 1/1/3 ConvNeXt residual blocks, with 2× channel expansion and no dilation gaps.", "Padding is retained. Two stride-2 transitions produce 45×49→23×25→12×13; the final nominal receptive field is 97×97 original cells.", "The 45×49 stage-1 map bypasses downsampling. Its 32 channels concatenate with 64 upsampled coarse channels before the original 96→32 entity projection.", "The hidden output stays 256-wide, the entity table stays 526×32, and the action head scores only the supplied legal moves.", "Original PPG head structure and phase-specific gradient isolation remain the integration contract: independent critic training, actor auxiliary outcome head, and post-policy KL anchor.", "Every encoder weight is new. No pretrained checkpoint, old channel-specific stem refinement offsets, training statistics or optimizer state is attached.", "The value encoder's unused entity/ship adapters remain stored to preserve the original module interface; their 5,664 parameters are included.", "Spatial MAC counts exclude norms, activations, interpolation, entity projection, and heads. They are arithmetic estimates, not measured speedups."],
        "validation": {"encoder_meta_forward": {key: list(t.shape) for key,t in observed.items()}, "encoder_parameters": per_encoder, "full_game_forward_verified": False},
        "format_notes": "This is a descriptive full-model proposal with an executable standalone CNN encoder. Original custom featurization, entity assembly, legal-policy normalization and PPG trainer are preserved as interface specifications; they are not silently replaced by generic builder operations.",
        "default_scope": "actor_pyramid", "view_scopes": scopes,
    }
    assert per_encoder == 446128 and total == 1028301
    result = analyze(Graph.model_validate(graph))
    assert result["valid"], result["errors"]
    return original, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    original, proposal = build_templates(json.loads(args.source.read_text()))
    directory = Path(__file__).resolve().parents[1] / "templates"
    for graph in (original, proposal):
        target = directory / f"{graph['template']}.vizml.json"
        target.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
        print(f"{target.name}: {len(graph['nodes'])} nodes; {analyze(Graph.model_validate(graph))['parameters']:,} parameters")


if __name__ == "__main__":
    main()
