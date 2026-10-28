"""Static, allocation-free architecture costs. MACs count matrix/conv products only.

These are logical costs, not measured latency or peak accelerator memory. A
multiply-accumulate is one MAC; biases, nonlinearities, normalization, pooling,
interpolation and data movement are outside that convention. Attention counts
dense score matrices, even when the implementation can exploit a causal mask.
"""

from math import prod

from .schema import Graph, GraphNode


def node_macs(node: GraphNode, info: dict) -> int | None:
    kind, p = node.data.kind, node.data.params
    x, y = info["input_shape"], info["shape"]
    if kind.startswith("deepseek_") or info.get("reference_only"):
        return None
    if kind == "conv2d":
        return prod(y) * (x[0] // int(p.get("groups", 1))) * int(p.get("kernel", 3)) ** 2
    if kind in {"dense", "classification", "regression"}:
        return prod(x) * prod(y)
    if kind == "convnext_stage":
        c, h, w = x
        return int(p.get("blocks", 3)) * h * w * (c * int(p.get("kernel", 7)) ** 2 + 2 * c * c * int(p.get("expansion", 4)))
    if kind == "patch_embed":
        return prod(info["grid"]) * y[-1] * x[0] * int(p.get("patch", 16)) ** 2
    if kind == "token_head":
        return x[0] * x[1] * y[1]
    if kind in {"attention", "cross_attention", "feedforward", "transformer_block"}:
        t, d = x
        context_t, context_d = info.get("context_shape", x)
        # Q and output projections, K/V projections, QK^T and attention @ V.
        attention = 2 * t * d * d + 2 * context_t * context_d * d + 2 * t * context_t * d
        hidden = max(1, int(d * float(p.get("expansion", 4))))
        ffn = (3 if p.get("activation", "gelu") == "swiglu" else 2) * t * d * hidden
        if kind == "feedforward":
            return ffn
        if kind != "transformer_block":
            return attention
        extra_self = 4 * t * d * d + 2 * t * t * d if info["merge"] == "query_context" else 0
        return int(p.get("blocks", 1)) * (attention + ffn + extra_self)
    if "recurrent" in info:
        r = info["recurrent"]
        gates = {"rnn": 1, "gru": 3, "lstm": 4}[r["cell"]]
        h, directions = r["hidden_size"], r["directions"]
        kernel = int(p.get("kernel", 3)) ** 2 if r["spatial"] else 1
        return r["steps"] * prod(x[2:]) * sum(
            directions * gates * h * ((x[1] if layer == 0 else h * directions) + h) * kernel
            for layer in range(r["layers"])
        )
    return 0


def receptive_field(node: GraphNode, info: dict, parents: list[dict]) -> dict | None:
    """Theoretical spatial support in one original image's pixel coordinates.

Only propagate through known spatial operators on aligned grids. Resizing,
mixed sources, flattening and attention deliberately end this local analysis.
"""
    kind, p = node.data.kind, node.data.params
    if kind == "grid_input":
        return {"source": node.id, "size": [1, 1], "stride": [1, 1], "offset": [0.5, 0.5]}
    fields = [parent.get("receptive_field") for parent in parents]
    if not fields or any(field is None for field in fields):
        return None
    first = fields[0]
    if any(any(field[key] != first[key] for key in ("source", "stride", "offset")) for field in fields):
        return None
    allowed = {"conv2d", "maxpool2d", "padding", "convnext_stage", "layernorm", "residual", "concat", "embedding", "global_avgpool"}
    if kind not in allowed or len(info["input_shape"]) != 3:
        return None
    size = [max(field["size"][axis] for field in fields) for axis in range(2)]
    jump, offset = first["stride"].copy(), first["offset"].copy()
    if kind in {"conv2d", "maxpool2d"}:
        kernel = int(p.get("kernel", 3 if kind == "conv2d" else 2))
        effective = int(p.get("dilation", 1)) * (kernel - 1)
        padding = int(p.get("padding", 1 if kind == "conv2d" else 0))
        stride = int(p.get("stride", 1 if kind == "conv2d" else 2))
        size = [size[i] + effective * jump[i] for i in range(2)]
        offset = [offset[i] + (effective / 2 - padding) * jump[i] for i in range(2)]
        jump = [value * stride for value in jump]
    elif kind == "convnext_stage":
        reach = int(p.get("blocks", 3)) * (int(p.get("kernel", 7)) - 1)
        size = [size[i] + reach * jump[i] for i in range(2)]
    elif kind == "padding":
        offset = [offset[i] - int(p.get("padding", 1)) * jump[i] for i in range(2)]
    elif kind == "global_avgpool":
        spatial = info["input_shape"][-2:]
        size = [size[i] + (spatial[i] - 1) * jump[i] for i in range(2)]
        offset = [offset[i] + (spatial[i] - 1) * jump[i] / 2 for i in range(2)]
    return {"source": first["source"], "size": size, "stride": jump, "offset": offset}


def enrich_analysis(graph: Graph, analysis: dict) -> dict:
    nodes = {node.id: node for node in graph.nodes}
    parents = {key: [] for key in nodes}
    for edge in graph.edges:
        if edge.source in nodes and edge.target in nodes and edge.source not in parents[edge.target]:
            parents[edge.target].append(edge.source)
    results = analysis["nodes"]
    floating, integer, known = 0, 0, 0
    unknown = []
    for key in analysis["order"]:
        if key not in results:
            continue
        info = results[key]
        info["macs"] = None if graph.mode == "schematic" else node_macs(nodes[key], info)
        if info["macs"] is None:
            unknown.append(key)
        else:
            known += info["macs"]
        is_integer = info.get("dtype") == "int64"
        info["output_bytes"] = info["elements"] * (8 if is_integer else 4)
        if is_integer:
            integer += info["elements"]
        else:
            floating += info["elements"]
        if graph.mode != "schematic":
            field = receptive_field(nodes[key], info, [results.get(parent, {}) for parent in parents[key]])
            if field:
                info["receptive_field"] = field
    unknown.extend(key for key in nodes if key not in results)
    analysis["profile"] = {
        "macs": known if analysis["valid"] and not unknown else None,
        "known_macs": known, "unknown_nodes": unknown,
        "float_output_elements": floating, "integer_output_elements": integer,
        "output_bytes": floating * 4 + integer * 8,
        "complete": analysis["valid"] and len(results) == len(nodes),
    }
    return analysis
