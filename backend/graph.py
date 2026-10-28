"""Pure shape inference: even trillion-parameter designs allocate no tensors."""

from collections import deque
from math import prod

from .attention import KINDS as SEQUENCE_KINDS, infer_sequence
from .recurrent import KINDS as RECURRENT_KINDS, infer_recurrent
from .schema import Graph, GraphNode
from .validation import integer, real

ACTIVATIONS = {"relu", "gelu", "silu", "tanh", "none"}
INPUTS = {"feature_input", "grid_input", "sequence_input", "token_input", "grid_sequence_input"}
OUTPUTS = {"classification", "regression", "embedding", "token_head", "deepseek_head", "deepseek_draft"}


def merge_shapes(shapes: list[list[int]], residual: bool = False) -> tuple[list[int], str]:
    if not shapes:
        raise ValueError("Connect an input dependency to this layer.")
    if residual:
        if len(shapes) < 2:
            raise ValueError("A residual sum needs at least two dependencies.")
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError("Residual dependencies must have identical shapes. Use a projection layer to match them.")
        return shapes[0], "add"
    if len(shapes) == 1:
        return shapes[0], "single"
    if any(len(shape) == 4 for shape in shapes):
        if not all(len(shape) == 4 and shape[0] == shapes[0][0] and shape[2:] == shapes[0][2:] for shape in shapes):
            raise ValueError("Grid sequences need matching time, height and width to concatenate channels. Flatten each branch to join incompatible layouts.")
        return [shapes[0][0], sum(shape[1] for shape in shapes), *shapes[0][2:]], "sequence_channels"
    if all(len(shape) == 3 for shape in shapes):
        if any(shape[1:] != shapes[0][1:] for shape in shapes):
            raise ValueError("Grid dependencies need matching height and width to concatenate channels. Flatten each branch to join different grid sizes.")
        return [sum(shape[0] for shape in shapes), *shapes[0][1:]], "channels"
    if all(len(shape) == 2 for shape in shapes):
        if any(shape[0] != shapes[0][0] for shape in shapes):
            raise ValueError("Sequence dependencies need matching token counts to concatenate features.")
        return [shapes[0][0], sum(shape[1] for shape in shapes)], "features"
    return [sum(prod(shape) for shape in shapes)], "flatten_concat"


def infer_node(node: GraphNode, incoming: list[list[int]], infos: list[dict] | None = None) -> dict:
    kind, p = node.data.kind, node.data.params
    infos = infos or [{} for _ in incoming]
    if kind in RECURRENT_KINDS:
        return infer_recurrent(node, incoming, infos)
    if kind in SEQUENCE_KINDS:
        return infer_sequence(node, incoming, infos)
    if kind.startswith("deepseek_"):
        from .deepseek import infer_reference
        return infer_reference(node, incoming)
    if kind not in INPUTS | OUTPUTS | {"conv2d", "dense", "maxpool2d", "padding", "flatten", "concat", "residual", "convnext_stage", "layernorm", "global_avgpool", "upsample2d"}:
        raise ValueError(f"{kind} is a descriptive schematic operator; its forward operation is not implemented in the builder.")
    if any(info.get("dtype") == "int64" for info in infos):
        raise ValueError("Embed token IDs before using floating-point layers.")
    if any(info.get("layout") == "streams" for info in infos) and kind not in {"embedding", "residual"}:
        raise ValueError("Hyper-connection streams need a DeepSeek stage or head to preserve/collapse their four lanes.")
    params = 0
    merge = "single"
    in_shape = []
    activation_elements = None
    if kind in INPUTS:
        if incoming:
            raise ValueError("Input nodes cannot have dependencies.")
        if kind == "feature_input":
            shape = [integer(node, "features", 16, 0)]
        else:
            shape = [integer(node, "channels", 3), integer(node, "height", 32, high=256), integer(node, "width", 32, high=256)]
    else:
        in_shape, merge = merge_shapes(incoming, kind == "residual")
        shape = in_shape.copy()
        if kind in {"conv2d", "maxpool2d", "padding", "convnext_stage", "upsample2d"} and len(shape) != 3:
            raise ValueError("This layer needs a grid (C × H × W). A feature vector has no spatial axes.")
        if kind == "global_avgpool" and len(shape) not in {3, 4}:
            raise ValueError("Global average pooling needs a grid (C × H × W) or grid sequence (T × C × H × W).")
        if kind in {"conv2d", "maxpool2d"}:
            channels, height, width = shape
            kernel = integer(node, "kernel", 3 if kind == "conv2d" else 2, high=256)
            stride = integer(node, "stride", 1 if kind == "conv2d" else 2, high=256)
            dilation = integer(node, "dilation", 1, high=256)
            padding = integer(node, "padding", 1 if kind == "conv2d" else 0, 0, 256)
            if kind == "maxpool2d" and padding > kernel // 2:
                raise ValueError("Max-pool padding cannot exceed half its kernel size.")
            shape = [channels, (height + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1, (width + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1]
            if min(shape[1:]) < 1:
                raise ValueError("The effective kernel is larger than the padded input. Reduce kernel/dilation or increase padding.")
            if kind == "conv2d":
                out = integer(node, "channels", 32)
                groups = integer(node, "groups", 1)
                if channels % groups or out % groups:
                    raise ValueError("Input and output channels must be divisible by groups.")
                params = out * ((channels // groups) * kernel * kernel + 1)
                shape[0] = out
        elif kind == "padding":
            padding = integer(node, "padding", 1, 0, 256)
            shape = [shape[0], shape[1] + padding * 2, shape[2] + padding * 2]
        elif kind == "upsample2d":
            if p.get("mode", "bilinear") not in {"bilinear", "nearest"}:
                raise ValueError("Choose bilinear or nearest grid resizing.")
            integer(node, "align_corners", 1, 0, 1)
            shape = [shape[0], integer(node, "height", 45, high=4096), integer(node, "width", 49, high=4096)]
        elif kind in {"dense", "classification", "regression"}:
            size = integer(node, "units", 64 if kind == "dense" else (10 if kind == "classification" else 1))
            bias = integer(node, "bias", 1, 0, 1) if kind == "dense" else 1
            params = (prod(shape) + bias) * size
            shape = [size]
        elif kind == "flatten":
            shape = [prod(shape)]
        elif kind == "layernorm":
            real(node, "eps", 1e-6, 1e-12, 1)
            params = 2 * (shape[1] if len(shape) in {2, 4} else shape[0])
        elif kind == "global_avgpool":
            shape = shape[:-2]
        elif kind == "convnext_stage":
            blocks = integer(node, "blocks", 3, high=64)
            kernel = integer(node, "kernel", 7, high=255)
            expansion = integer(node, "expansion", 4, high=16)
            scale = real(node, "layer_scale", 1e-6, 0, 1)
            real(node, "eps", 1e-6, 1e-12, 1)
            real(node, "drop_path", 0, 0, 0.99)
            if kernel % 2 == 0:
                raise ValueError("ConvNeXt depthwise kernels must be odd to preserve residual shapes.")
            channels = shape[0]
            # Depthwise convolution, channel LN, two per-pixel linear maps, and layer scale.
            params = blocks * (channels * (kernel * kernel + 1) + 2 * channels + (channels + 1) * channels * expansion + (channels * expansion + 1) * channels + (channels if scale > 0 else 0))
            activation_elements = blocks * (4 + int(scale > 0) + 2 * expansion) * prod(shape)
        if kind in {"conv2d", "dense", "residual"} and p.get("activation", "relu" if kind != "residual" else "none") not in ACTIVATIONS:
            raise ValueError("Choose a supported activation: ReLU, GELU, SiLU, tanh, or none.")
    extra = {}
    if len(shape) == 4:
        extra["layout"] = "spatiotemporal"
    if len(infos) == 1 and kind in {"layernorm", "embedding"}:
        extra = {k: infos[0][k] for k in ("grid", "prefix_tokens", "layout") if k in infos[0]}
    if kind == "residual" and infos and infos[0].get("layout") == "streams":
        extra["layout"] = "streams"
    if kind in {"residual", "concat"} and infos and all(info.get("grid") == infos[0].get("grid") and info.get("prefix_tokens") == infos[0].get("prefix_tokens") for info in infos):
        extra.update({k: infos[0][k] for k in ("grid", "prefix_tokens") if k in infos[0]})
    return {"shape": shape, "input_shape": in_shape, "parameters": params, "elements": prod(shape), "activation_elements": activation_elements if activation_elements is not None else prod(shape), "merge": merge, "dependencies": len(incoming), "dtype": "float32", **extra}


def analyze(graph: Graph) -> dict:
    from .profiling import enrich_analysis
    return enrich_analysis(graph, _analyze(graph))


def _analyze(graph: Graph) -> dict:
    if graph.mode == "schematic":
        from .schematic import analyze_schematic
        return analyze_schematic(graph)
    nodes = {node.id: node for node in graph.nodes}
    errors: list[dict] = []
    warnings: list[dict] = []
    if len(nodes) != len(graph.nodes):
        return {"valid": False, "nodes": {}, "order": [], "errors": [{"node_id": None, "message": "Node IDs must be unique."}], "warnings": [], "parameters": 0, "activation_bytes": 0, "depth": 0, "outputs": []}
    dependencies: dict[str, list[str]] = {key: [] for key in nodes}
    children: dict[str, list[str]] = {key: [] for key in nodes}
    seen_edges = set()
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes:
            errors.append({"node_id": None, "message": "A connection references a missing node."})
            continue
        pair = (edge.source, edge.target)
        if pair in seen_edges:
            errors.append({"node_id": edge.target, "message": "Duplicate dependencies are not allowed."})
            continue
        seen_edges.add(pair)
        dependencies[edge.target].append(edge.source)
        children[edge.source].append(edge.target)
        if nodes[edge.source].data.kind in OUTPUTS:
            errors.append({"node_id": edge.source, "message": "Output heads are terminal. Connect this branch before its output head."})
    degrees = {key: len(value) for key, value in dependencies.items()}
    ready = deque(key for key in nodes if degrees[key] == 0)
    order: list[str] = []
    while ready:
        key = ready.popleft()
        order.append(key)
        for child in children[key]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        errors.append({"node_id": next(key for key in nodes if degrees[key]), "message": "This graph contains a cycle. Model dependencies must form a directed acyclic graph."})
    results = {}
    blocked_by: dict[str, list[str]] = {}
    depths: dict[str, int] = {}
    for key in order:
        depths[key] = max((depths.get(dep, 0) + 1 for dep in dependencies[key]), default=0)
        try:
            if any(dep not in results for dep in dependencies[key]):
                blocked_by[key] = list(dict.fromkeys(root for dep in dependencies[key] if dep not in results for root in blocked_by.get(dep, [dep])))
                raise ValueError("An upstream dependency has an invalid shape.")
            infos = [results[dep] for dep in dependencies[key]]
            results[key] = {**infer_node(nodes[key], [info["shape"] for info in infos], infos), "depth": depths[key]}
        except ValueError as error:
            message = str(error)
            if nodes[key].data.kind == "residual" and len(dependencies[key]) >= 2 and all(dep in results for dep in dependencies[key]):
                branches = "; ".join(f"{nodes[dep].data.label}: {' × '.join(str(d) for d in results[dep]['shape'])}" for dep in dependencies[key])
                message += f" Incoming branches — {branches}."
            errors.append({"node_id": key, "message": message, **({"blocked_by": blocked_by[key]} if key in blocked_by else {})})
    outputs = [key for key, node in nodes.items() if node.data.kind in OUTPUTS]
    if not outputs:
        warnings.append({"node_id": None, "message": "Add a classification, regression, or embedding output to complete the model."})
    if not nodes:
        warnings.append({"node_id": None, "message": "Add an input to start your architecture."})
    ancestors = set(outputs)
    queue = list(outputs)
    while queue:
        for dep in dependencies[queue.pop()]:
            if dep not in ancestors:
                ancestors.add(dep)
                queue.append(dep)
    for key in nodes:
        if outputs and key not in ancestors:
            warnings.append({"node_id": key, "message": "This branch does not contribute to an output."})
    total_params = sum(node["parameters"] for node in results.values())
    executable = not any(node.data.kind.startswith("deepseek_") for node in graph.nodes)
    if total_params > 100_000_000:
        warnings.append({"node_id": None, "message": "Large architecture: preview and meta-device compilation are allocation-free. Materializing weights requires sufficient memory." if executable else "Reference counts show unquantized logical weights, including Engram memory and DSpark. The fp32 figure is an equivalent storage estimate; the released checkpoint uses mixed quantization."})
    if not executable:
        warnings.append({"node_id": None, "message": "DeepSeek V4.1 Flash 1017 is a reference architecture: logical, unquantized parameter counts include Engram, vision, and DSpark (shared weights counted once). Native sparse attention, routing, hash memory, and hyper-connection kernels are not implemented in the PyTorch builder. JSON export is available."})
    return {
        "valid": not errors, "executable": executable, "nodes": results, "order": order, "errors": errors,
        "warnings": warnings, "parameters": total_params,
        "activation_bytes": sum(node["activation_elements"] * (8 if node.get("dtype") == "int64" else 4) for node in results.values()),
        "depth": max(depths.values(), default=0) + (1 if nodes else 0), "outputs": outputs,
    }
