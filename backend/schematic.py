"""Validate a descriptive diagram without reinterpreting it as executable code.

Shapes, operation descriptions and connectivity remain source assertions. Tensor
ownership is checked internally here and separately against uploaded checkpoints.
"""

from collections import deque
from math import prod

from .schema import Graph


def analyze_schematic(graph: Graph) -> dict:
    nodes = {n.id: n for n in graph.nodes}
    errors, warnings, results, order = [], [], {}, []
    parents = {key: [] for key in nodes}
    children = {key: [] for key in nodes}
    if len(nodes) != len(graph.nodes):
        errors.append({"node_id": None, "message": "Node IDs must be unique."})
    seen = set()
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes:
            errors.append({"node_id": edge.target, "message": "A connection references a missing node."})
        elif (edge.source, edge.target) in seen:
            errors.append({"node_id": edge.target, "message": "Duplicate dependencies are not allowed."})
        else:
            seen.add((edge.source, edge.target))
            parents[edge.target].append(edge.source)
            children[edge.source].append(edge.target)
    degree = {key: len(v) for key, v in parents.items()}
    ready = deque(key for key in nodes if degree[key] == 0)
    while ready:
        key = ready.popleft(); order.append(key)
        for child in children[key]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        errors.append({"node_id": None, "message": "The supplied schematic contains a cycle."})
    examples = graph.metadata.get("validation", {})
    examples = examples.get("actual_cpu_observation_shapes", {}) if isinstance(examples, dict) else {}
    legal = examples.get("legal", [1, 14]) if isinstance(examples, dict) else [1, 14]
    sample = legal[-1] if isinstance(legal, list) and legal else 14
    symbols = {"L": sample if type(sample) is int and 1 <= sample <= 1048576 else 14}
    owners, depths, symbolic = {}, {}, False
    for key in order:
        n = nodes[key]; d = n.data.model_dump()
        try:
            declared = d.get("output_shape")
            if not isinstance(declared, list) or not declared or len(declared) > 6:
                raise ValueError("A descriptive node needs an output_shape array.")
            display = declared[1:] if declared[0] == "B" else declared
            if not display or any(isinstance(v, bool) or not isinstance(v, (str, int)) or (isinstance(v, int) and not 0 <= v <= 2**53) or (isinstance(v, str) and not 1 <= len(v) <= 40) for v in display):
                raise ValueError("Declared shape dimensions must be nonnegative integers or symbolic names.")
            symbolic |= any(isinstance(v, str) for v in display)
            shape = [symbols.get(v, 1) if isinstance(v, str) else v for v in display]
            parameters = d.get("parameter_count", 0)
            if isinstance(parameters, bool) or not isinstance(parameters, int) or not 0 <= parameters <= 2**53:
                raise ValueError("Declared parameter_count must be a nonnegative integer.")
            tensors = d.get("parameter_tensors", [])
            if not isinstance(tensors, list):
                raise ValueError("parameter_tensors must be an array.")
            owned_count = 0
            for tensor in tensors:
                if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str) or not isinstance(tensor.get("shape"), list) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in tensor["shape"]):
                    raise ValueError("Parameter ownership entries need a tensor name and numeric shape.")
                name = tensor["name"]
                if name in owners:
                    raise ValueError(f"Tensor {name} is already owned by {owners[name]}.")
                owners[name] = key
                owned_count += prod(tensor["shape"])
            if tensors and owned_count != parameters:
                raise ValueError(f"Declared count {parameters:,} differs from owned tensor shapes ({owned_count:,}).")
            depths[key] = max((depths.get(p, 0) + 1 for p in parents[key]), default=0)
            is_sequence = len(shape) == 2 or (len(shape) == 3 and any(isinstance(v, str) for v in display))
            results[key] = {"shape": shape, "display_shape": display, "input_shape": results[parents[key][0]]["shape"] if parents[key] and parents[key][0] in results else [],
                            "parameters": parameters, "elements": prod(shape), "activation_elements": prod(shape), "dependencies": len(parents[key]),
                            "depth": depths[key], "merge": "declared", "reference_only": True, "layout": "sequence" if is_sequence else "declared", "terminal": not children[key]}
        except ValueError as e:
            errors.append({"node_id": key, "message": str(e)})
    parameters = sum(n["parameters"] for n in results.values())
    declared_total = graph.metadata.get("parameter_counts", {})
    declared_total = declared_total.get("total") if isinstance(declared_total, dict) else None
    if declared_total is not None and declared_total != parameters:
        errors.append({"node_id": None, "message": f"Node parameter counts sum to {parameters:,}; metadata declares {declared_total}."})
    warnings.append({"node_id": None, "message": "Untrained architecture proposal: a standalone CNN encoder is provided. Game-specific featurization, entity assembly, action/outcome heads and PPG phase routing remain a descriptive integration contract; full-game execution is unavailable here." if graph.metadata.get("proposal") else "Descriptive schematic: operation meanings, shapes, and forward connections are supplied claims. Structural checks and tensor ownership do not verify the original forward implementation. PyTorch execution is unavailable for this diagram."})
    if symbolic:
        warnings.append({"node_id": None, "message": f"Symbolic dimensions remain visible in labels. 3D sampling and output-size estimates use L={symbols['L']}; this is not a fixed action count."})
    return {"valid": not errors, "executable": False, "basis": "declared", "nodes": results, "order": order,
            "parameters": parameters, "activation_bytes": sum(n["elements"] * 4 for n in results.values()), "depth": max(depths.values(), default=0) + 1,
            "errors": errors, "warnings": warnings, "outputs": [key for key in nodes if not children[key]], "symbols": symbols, "owned_tensors": len(owners)}
