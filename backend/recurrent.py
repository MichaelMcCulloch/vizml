"""Shape and parameter inference for recurrent sequence layers."""
from math import prod

from .schema import GraphNode
from .validation import choice, integer, real

VECTOR_KINDS = {"rnn", "gru", "lstm"}
SPATIAL_KINDS = {"conv_rnn", "conv_gru", "conv_lstm"}
KINDS = VECTOR_KINDS | SPATIAL_KINDS | {"grid_sequence_input"}


def infer_recurrent(node: GraphNode, incoming: list[list[int]], infos: list[dict]) -> dict:
    kind = node.data.kind
    if kind == "grid_sequence_input":
        if incoming:
            raise ValueError("Input nodes cannot have dependencies.")
        shape = [integer(node, "steps", 8, high=1048576), integer(node, "channels", 3),
                 integer(node, "height", 32, high=256), integer(node, "width", 32, high=256)]
        return {"shape": shape, "input_shape": [], "parameters": 0, "elements": prod(shape),
                "activation_elements": prod(shape), "merge": "single", "dependencies": 0,
                "dtype": "float32", "layout": "spatiotemporal"}
    spatial = kind in SPATIAL_KINDS
    cell = kind.removeprefix("conv_")
    if not incoming:
        raise ValueError("Connect a grid sequence (T × C × H × W)." if spatial else "Connect a sequence (T × features).")
    if any(info.get("dtype") == "int64" for info in infos):
        raise ValueError("Embed token IDs before using a recurrent layer.")
    if any(info.get("layout") == "streams" for info in infos):
        raise ValueError("Recurrent layers do not accept hyper-connection streams.")
    rank = 4 if spatial else 2
    if any(len(shape) != rank for shape in incoming):
        raise ValueError("Convolutional recurrence needs T × C × H × W. Use a Grid sequence input, not a single image."
                         if spatial else "RNN, GRU and LSTM need T × features. Use a Sequence input or embed token IDs.")
    if any(shape[0] != incoming[0][0] or (spatial and shape[2:] != incoming[0][2:]) for shape in incoming):
        raise ValueError("Recurrent dependencies must have matching time steps and spatial dimensions.")
    shape_in = incoming[0].copy()
    shape_in[1] = sum(shape[1] for shape in incoming)
    steps, width = shape_in[:2]
    hidden = integer(node, "hidden_size", 32 if spatial else 64)
    layers = integer(node, "layers", 1, high=16)
    bias = integer(node, "bias", 1, 0, 1)
    sequence = integer(node, "return_sequences", 1, 0, 1)
    directions = 1 + integer(node, "bidirectional", 0, 0, 0 if spatial else 1)
    dropout = real(node, "dropout", 0, 0, 0.99)
    if layers == 1 and dropout:
        raise ValueError("Inter-layer dropout needs at least two recurrent layers.")
    if cell == "rnn":
        choice(node, "activation", "tanh", {"tanh", "relu"})
    kernel = integer(node, "kernel", 3, high=31) if spatial else 1
    if spatial:
        integer(node, "dilation", 1, high=16)
        if kernel % 2 == 0:
            raise ValueError("Recurrent convolution kernels must be odd to preserve the hidden-state grid.")
    gates = {"rnn": 1, "gru": 3, "lstm": 4}[cell]
    parameters = sum(directions * gates * hidden * (
        ((width if layer == 0 else hidden * directions) + hidden) * kernel ** 2 + bias * (1 if spatial else 2)
    ) for layer in range(layers))
    state_shape = [layers * directions, hidden, *shape_in[2:]]
    output = [hidden * directions, *shape_in[2:]]
    if sequence:
        output = [steps, *output]
    return {"shape": output, "input_shape": shape_in, "parameters": parameters,
            "elements": prod(output), "activation_elements": steps * layers * directions * hidden * prod(shape_in[2:]) * (gates + 2 + int(cell == "lstm")),
            "merge": "single" if len(incoming) == 1 else "sequence_channels" if spatial else "features",
            "dependencies": len(incoming), "dtype": "float32",
            **({"layout": "spatiotemporal"} if spatial and sequence else {}),
            "recurrent": {"cell": cell, "spatial": spatial, "steps": steps, "layers": layers,
                          "directions": directions, "hidden_size": hidden, "return_sequences": bool(sequence),
                          "state_shape": state_shape, "states": ["h", "c"] if cell == "lstm" else ["h"]}}
