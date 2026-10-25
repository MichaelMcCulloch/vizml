"""Shape/count blueprint of the published V4.1-Flash-1017 inference graph.

These operators intentionally do not pretend to execute the model's custom
sparse/quantized kernels. Counts describe logical weights, excluding quantizer
scales; shared token embeddings and output heads are counted once.
"""

import json
from math import ceil, prod
from pathlib import Path

from .schema import GraphNode
from .validation import integer

CONFIG = json.loads(Path(__file__).with_name("references").joinpath("deepseek_v41_flash_1017.json").read_text())
TEXT = CONFIG["text_config"]
VISION = CONFIG["vision_config"]


def attention_parameters(layer: int) -> int:
    c = TEXT
    d, h, hd, qr, groups, out = (c[k] for k in ("hidden_size", "num_attention_heads", "head_dim", "q_lora_rank", "o_groups", "o_lora_rank"))
    count = h + d * qr + qr + qr * h * hd + d * hd + hd + (h * hd // groups) * (groups * out) + groups * out * d
    if layer in c["kv_source_layer_ids"]:
        count += d * hd * (2 if c["compress_ratios"][layer] > 1 else 1) + hd
    if layer in c["index_source_layer_ids"]:
        count += qr * c["index_n_heads"] * c["index_head_dim"] + d * c["index_n_heads"]
        if layer in c["kv_source_layer_ids"]:
            count += hd * c["index_head_dim"] + c["index_head_dim"]
    return count


def block_parameters(layer: int, draft=False) -> int:
    c = TEXT
    d, lanes, intermediate = c["hidden_size"], c["hc_mult"], c["moe_intermediate_size"]
    experts = c["dspark_n_routed_experts"] if draft else c["n_routed_experts"]
    ffn = 3 * d * intermediate * (experts + c["n_shared_experts"]) + experts * (d + 2)
    mixing = (2 + lanes) * lanes
    hc = 2 * (mixing * lanes * d + mixing + 3)
    count = attention_parameters(layer) + ffn + 2 * d + hc
    if not draft and layer in c["engram_layer_ids"]:
        rows = c["engram_num_embeddings"][c["engram_layer_ids"].index(layer)]
        cols = (c["engram_max_ngram_size"] - 1) * c["engram_n_heads"]
        count += rows * c["engram_head_dim"] + cols * c["engram_head_dim"] * d * (lanes + 1) + 2 * lanes * d
    return count


def vision_parameters() -> int:
    v, d = VISION, TEXT["hidden_size"]
    vd, inner, r = v["hidden_size"], v["intermediate_size"], v["downsample_ratio"]
    return (3 * v["patch_size"] ** 2 + 1) * vd + v["num_hidden_layers"] * (4 * vd * vd + 6 * vd + 3 * vd * inner) + vd + (vd * r * r + 1) * d + (d + 1) * d


def draft_parameters() -> int:
    c = TEXT
    d, v, rank = c["hidden_size"], c["vocab_size"], c["dspark_markov_rank"]
    return sum(block_parameters(c["num_hidden_layers"] + i, True) for i in range(c["num_nextn_predict_layers"])) + d * d * len(c["dspark_target_layer_ids"]) + 2 * d + 2 * v * rank + d + rank


def infer_reference(node: GraphNode, incoming: list[list[int]]) -> dict:
    kind, c = node.data.kind, TEXT
    d, lanes, params, merge = c["hidden_size"], c["hc_mult"], 0, "single"
    extra = {"reference_only": True}
    if kind == "deepseek_vision":
        if len(incoming) != 1 or len(incoming[0]) != 3 or incoming[0][0] != 3:
            raise ValueError("DeepSeek's vision branch needs one RGB grid.")
        _, h, w = incoming[0]
        p, r = VISION["patch_size"], VISION["downsample_ratio"]
        if h % p or w % p:
            raise ValueError("DeepSeek image dimensions must be multiples of its 14-pixel patch size.")
        gh, gw = ceil(h // p / r), ceil(w // p / r)
        shape, params = [gh * gw, d], vision_parameters()
        extra.update(grid=[gh, gw], prefix_tokens=0, preview_image=True)
    elif kind == "deepseek_bridge":
        if len(incoming) != 2 or any(len(s) != 2 or s[-1] != d for s in incoming):
            raise ValueError("Image injection needs text embeddings first and aligned image embeddings second, both 5,120 wide.")
        if incoming[0][0] < incoming[1][0] + 2:
            raise ValueError("The text sequence must reserve enough placeholder slots for the image span.")
        # Image slots replace text placeholders; they are not concatenated as features.
        shape, params, merge = [incoming[0][0], lanes, d], 3 * d, "image_injection"
        extra["layout"] = "streams"
    elif kind in {"deepseek_stage", "deepseek_head", "deepseek_draft"}:
        if len(incoming) != 1 or len(incoming[0]) != 3 or incoming[0][1:] != [lanes, d]:
            raise ValueError("This DeepSeek operator needs a T × 4 × 5,120 hyper-connection stream. Connect its image/token bridge or an upstream DeepSeek stage.")
        tokens = incoming[0][0]
        if kind == "deepseek_stage":
            start = integer(node, "start", 0, 0, 39)
            blocks = integer(node, "blocks", 2, high=40)
            if start + blocks > 40:
                raise ValueError("DeepSeek backbone layer indices run from 0 to 39.")
            shape, params = incoming[0].copy(), sum(block_parameters(i) for i in range(start, start + blocks))
            extra.update(layout="streams", residual_blocks=blocks,
                         kv_sources=[i for i in c["kv_source_layer_ids"] if start <= i < start + blocks],
                         index_sources=[i for i in c["index_source_layer_ids"] if start <= i < start + blocks],
                         engram_layers=[i for i in c["engram_layer_ids"] if start <= i < start + blocks])
        elif kind == "deepseek_head":
            shape, params = [tokens, c["vocab_size"]], d + d * c["vocab_size"]
        else:
            shape, params = [c["dspark_block_size"], c["vocab_size"]], draft_parameters()
    else:
        raise ValueError("Unknown DeepSeek reference operator.")
    return {"shape": shape, "input_shape": incoming[0] if incoming else [], "parameters": params,
            "elements": prod(shape), "activation_elements": prod(shape), "merge": merge,
            "dependencies": len(incoming), "dtype": "float32", **extra}
