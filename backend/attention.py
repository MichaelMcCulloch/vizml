"""Allocation-free inference for token sequences and transformer building blocks."""

from math import prod

from .schema import GraphNode
from .validation import choice, integer, real

KINDS = {"sequence_input", "token_input", "token_embedding", "patch_embed", "position_encoding",
         "attention", "cross_attention", "feedforward", "transformer_block", "token_pool", "token_head"}


def infer_sequence(node: GraphNode, incoming: list[list[int]], infos: list[dict]) -> dict:
    kind, p = node.data.kind, node.data.params
    params, merge, in_shape, extra = 0, "single", [], {}
    dtype = "float32"
    if kind in {"sequence_input", "token_input"}:
        if incoming:
            raise ValueError("Input nodes cannot have dependencies.")
        tokens = integer(node, "tokens", 64, high=1048576)
        shape = [tokens, integer(node, "features", 128)] if kind == "sequence_input" else [tokens]
        dtype = "int64" if kind == "token_input" else "float32"
    else:
        if not incoming:
            raise ValueError("Connect an input dependency to this layer.")
        cross = kind == "cross_attention" or (kind == "transformer_block" and p.get("mode", "encoder") == "cross")
        if cross:
            if len(incoming) != 2 or any(len(s) != 2 for s in incoming):
                raise ValueError("Cross-attention needs two sequences: first the query, then the context (keys and values).")
            merge, in_shape = "query_context", incoming[0].copy()
            extra["context_shape"] = incoming[1]
        elif len(incoming) == 1:
            in_shape = incoming[0].copy()
        elif kind == "patch_embed" and all(len(s) == 3 for s in incoming):
            if any(s[1:] != incoming[0][1:] for s in incoming):
                raise ValueError("Grid dependencies need matching height and width to concatenate channels before patch embedding.")
            in_shape, merge = [sum(s[0] for s in incoming), *incoming[0][1:]], "channels"
        elif all(len(s) == 2 and s[0] == incoming[0][0] for s in incoming):
            in_shape, merge = [incoming[0][0], sum(s[1] for s in incoming)], "features"
        else:
            raise ValueError("Sequence dependencies need matching token counts to concatenate features.")
        shape = in_shape.copy()
        if kind == "token_embedding":
            if len(incoming) != 1 or len(shape) != 1 or infos[0].get("dtype") != "int64":
                raise ValueError("Token embedding needs a Token IDs input, containing integer vocabulary indices.")
            vocab = integer(node, "vocab", 32000, 2, 1048576)
            dim = integer(node, "features", 128)
            params, shape = vocab * dim, [shape[0], dim]
        else:
            if any(info.get("dtype") == "int64" for info in infos):
                raise ValueError("Embed token IDs before using floating-point layers.")
            if kind == "patch_embed":
                if len(shape) != 3:
                    raise ValueError("Patch embedding needs a grid (C × H × W).")
                channels, h, w = shape
                patch = integer(node, "patch", 16, high=256)
                if h % patch or w % patch:
                    raise ValueError("Image height and width must be divisible by the patch size.")
                dim = integer(node, "features", 192)
                cls = integer(node, "cls_token", 1, 0, 1)
                registers = integer(node, "registers", 0, 0, 64)
                mask = integer(node, "mask_token", 0, 0, 1)
                params = (channels * patch * patch + 1 + cls + registers + mask) * dim
                shape = [h // patch * (w // patch) + cls + registers, dim]
                extra = {"grid": [h // patch, w // patch], "prefix_tokens": cls + registers}
            else:
                if len(shape) != 2:
                    raise ValueError("This layer needs a token sequence (T × D). Add a sequence input, token embedding, or patch embedding.")
                tokens, dim = shape
                if len(infos) == 1 or cross:
                    extra.update({k: infos[0][k] for k in ("grid", "prefix_tokens") if k in infos[0]})
                elif all(info.get("grid") == infos[0].get("grid") and info.get("prefix_tokens") == infos[0].get("prefix_tokens") for info in infos):
                    extra.update({k: infos[0][k] for k in ("grid", "prefix_tokens") if k in infos[0]})
                if kind == "position_encoding":
                    method = choice(node, "method", "sinusoidal", {"learned", "sinusoidal"})
                    params = tokens * dim if method == "learned" else 0
                elif kind == "token_pool":
                    method = choice(node, "method", "mean", {"mean", "cls", "patches"})
                    if method == "patches":
                        prefix = extra.get("prefix_tokens", integer(node, "prefix", 1, 0, 65))
                        if prefix >= tokens:
                            raise ValueError("At least one patch token must remain after removing the prefix.")
                        shape, extra = [tokens - prefix, dim], {"pool_prefix": prefix, "prefix_tokens": 0, **({"grid": extra["grid"]} if "grid" in extra else {})}
                    else:
                        shape, extra = [dim], {}
                elif kind == "token_head":
                    vocab = integer(node, "vocab", 32000, 2, 1048576)
                    params, shape, extra = (dim + 1) * vocab, [tokens, vocab], {}
                else:
                    if kind in {"attention", "cross_attention", "transformer_block"}:
                        heads = integer(node, "heads", 8, high=1024)
                        if dim % heads:
                            raise ValueError(f"Embedding width {dim:,} must be divisible by {heads} attention heads.")
                        pos = choice(node, "position", "none", {"none", "rope", "rope2d"})
                        if pos != "none" and (dim // heads) % (4 if pos == "rope2d" else 2):
                            raise ValueError(f"{pos.upper()} requires head width divisible by {4 if pos == 'rope2d' else 2}.")
                        if pos == "rope2d" and ("grid" not in extra or prod(extra["grid"]) + extra.get("prefix_tokens", 0) != tokens):
                            raise ValueError("2D RoPE needs patch-grid metadata from a patch embedding with no intervening flatten/concatenation.")
                        if cross and pos == "rope2d":
                            raise ValueError("2D RoPE is supported for self-attention. Use no rotary encoding for cross-attention.")
                        real(node, "rope_base", 100 if pos == "rope2d" else 10000, 1.01, 10000000)
                        qkv_bias = integer(node, "qkv_bias", 1, 0, 1)
                        proj_bias = integer(node, "proj_bias", 1, 0, 1)
                        integer(node, "mask_k_bias", 0, 0, 1)
                        real(node, "dropout", 0, 0, 0.99)
                        integer(node, "causal", 0, 0, 1)
                        context_dim = incoming[1][-1] if cross else dim
                        attention_params = 2 * dim * dim + 2 * context_dim * dim + (3 * qkv_bias + proj_bias) * dim
                    if kind in {"feedforward", "transformer_block"}:
                        real(node, "dropout", 0, 0, 0.99)
                        expansion = real(node, "expansion", 4, 0.125, 64)
                        hidden = max(1, int(dim * expansion))
                        act = choice(node, "activation", "gelu", {"relu", "gelu", "silu", "swiglu"})
                        bias = integer(node, "ffn_bias", 1, 0, 1)
                        ffn_params = (3 if act == "swiglu" else 2) * dim * hidden + bias * ((2 if act == "swiglu" else 1) * hidden + dim)
                    if kind in {"attention", "cross_attention"}:
                        params = attention_params
                    elif kind == "feedforward":
                        params = ffn_params
                    elif kind == "transformer_block":
                        blocks = integer(node, "blocks", 1, high=96)
                        mode = choice(node, "mode", "encoder", {"encoder", "decoder", "cross"})
                        choice(node, "norm_style", "pre", {"pre", "post"})
                        real(node, "eps", 1e-5, 1e-20, 1)
                        scale = real(node, "layer_scale", 0, 0, 1)
                        sublayers = 3 if mode == "cross" else 2
                        # Cross decoder also has its own causal self-attention.
                        self_params = 4 * dim * dim + (3 * qkv_bias + proj_bias) * dim if cross else 0
                        params = blocks * (attention_params + self_params + ffn_params + sublayers * dim * (2 + int(scale > 0)))
    result = {"shape": shape, "input_shape": in_shape, "parameters": params, "elements": prod(shape),
              "activation_elements": prod(shape), "merge": merge, "dependencies": len(incoming), "dtype": dtype, **extra}
    if kind in {"attention", "cross_attention", "transformer_block"}:
        blocks = int(p.get("blocks", 1)) if kind == "transformer_block" else 1
        t, d = in_shape
        context_tokens = incoming[1][0] if merge == "query_context" else t
        # Logical operator outputs, including the dense attention score matrices; not peak memory.
        scores = int(p.get("heads", 8)) * t * context_tokens
        operations = (8 * t * d + 2 * scores)
        if kind == "transformer_block":
            operations += int(t * d * (6 + 2 * float(p.get("expansion", 4))))
            if merge == "query_context":
                operations += 8 * t * d + 2 * int(p.get("heads", 8)) * t * t
        result["activation_elements"] = blocks * operations
        result["residual_blocks"] = blocks if kind == "transformer_block" else 0
    return result
