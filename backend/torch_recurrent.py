"""Standalone recurrent runtime; embedded in exported models.

Vector cells use native PyTorch. ConvGRU uses reset-before convolution;
ConvLSTM uses input/forget/candidate/output gates without peepholes.
All layers start at zero state unless a caller supplies state explicitly.
"""
import torch
from torch import nn
from torch.nn import functional as F


class VectorRecurrent(nn.Module):
    def __init__(self, cell, input_size, hidden_size=64, layers=1, bias=True,
                 bidirectional=False, dropout=0, return_sequences=True, activation="tanh"):
        super().__init__()
        self.cell = cell
        self.return_sequences = return_sequences
        self.directions = 2 if bidirectional else 1
        options = {"nonlinearity": activation} if cell == "rnn" else {}
        self.core = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell](
            input_size, hidden_size, num_layers=layers, bias=bias, batch_first=True,
            bidirectional=bidirectional, dropout=dropout, **options)

    def forward(self, x, state=None, return_state=False):
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("Vector recurrence expects batch × time × features with at least one step.")
        # Meta validates a shared transition without unrolling a million steps.
        output, final = self.core(x[:, :1] if x.is_meta else x, state)
        if x.is_meta:
            output = output.expand(-1, x.shape[1], -1)
        hidden = final[0] if self.cell == "lstm" else final
        result = output if self.return_sequences else hidden[-self.directions:].transpose(0, 1).reshape(x.shape[0], -1)
        return (result, final) if return_state else result


class SpatialRecurrentCell(nn.Module):
    def __init__(self, cell, input_size, hidden_size, kernel=3, dilation=1, bias=True, activation="tanh"):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("Spatial recurrent kernels must be odd.")
        self.cell, self.hidden_size, self.activation = cell, hidden_size, activation
        def convolution(channels):
            return nn.Conv2d(input_size + hidden_size, channels, kernel,
                             padding=dilation * (kernel - 1) // 2, dilation=dilation, bias=bias)
        self.gates = convolution(hidden_size * {"rnn": 1, "gru": 2, "lstm": 4}[cell])
        self.candidate = convolution(hidden_size) if cell == "gru" else None

    def forward(self, x, h, c=None):
        gates = self.gates(torch.cat((x, h), dim=1))
        if self.cell == "rnn":
            return F.relu(gates) if self.activation == "relu" else torch.tanh(gates), None
        if self.cell == "gru":
            reset, update = gates.sigmoid().chunk(2, dim=1)
            candidate = torch.tanh(self.candidate(torch.cat((x, reset * h), dim=1)))
            return update * h + (1 - update) * candidate, None
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
        c = forget_gate.sigmoid() * c + input_gate.sigmoid() * candidate.tanh()
        return output_gate.sigmoid() * c.tanh(), c


class SpatialRecurrent(nn.Module):
    def __init__(self, cell, input_size, hidden_size=32, layers=1, kernel=3, dilation=1,
                 bias=True, dropout=0, return_sequences=True, activation="tanh"):
        super().__init__()
        self.cell, self.hidden_size = cell, hidden_size
        self.return_sequences, self.dropout = return_sequences, dropout
        self.cells = nn.ModuleList(SpatialRecurrentCell(cell, input_size if i == 0 else hidden_size,
            hidden_size, kernel, dilation, bias, activation) for i in range(layers))

    def forward(self, x, state=None, return_state=False):
        if x.ndim != 5 or x.shape[1] < 1:
            raise ValueError("Spatial recurrence expects batch × time × channels × height × width.")
        batch, steps, _, height, width = x.shape
        state_shape = (len(self.cells), batch, self.hidden_size, height, width)
        if state is None:
            hidden = x.new_zeros(state_shape)
            memory = x.new_zeros(state_shape) if self.cell == "lstm" else None
        else:
            if self.cell == "lstm":
                if not isinstance(state, (tuple, list)) or len(state) != 2:
                    raise ValueError("ConvLSTM state must be (hidden, cell).")
                hidden, memory = state
            else:
                hidden, memory = state, None
            if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != state_shape or (
                self.cell == "lstm" and (not isinstance(memory, torch.Tensor) or tuple(memory.shape) != state_shape)
            ):
                raise ValueError("State shape must be layers × batch × hidden channels × height × width.")
        final_h, final_c = [], []
        for layer, cell in enumerate(self.cells):
            h, c = hidden[layer], memory[layer] if memory is not None else None
            outputs = []
            # Each transition has identical shapes. Meta only needs one to
            # validate the real convolutions, gates and state arithmetic.
            for t in range(1 if x.is_meta else steps):
                h, c = cell(x[:, t], h, c)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)
            if x.is_meta:
                x = x.expand(-1, steps, -1, -1, -1)
            final_h.append(h)
            if c is not None:
                final_c.append(c)
            if layer < len(self.cells) - 1 and self.dropout:
                x = F.dropout(x, self.dropout, training=self.training)
        final = (torch.stack(final_h), torch.stack(final_c)) if self.cell == "lstm" else torch.stack(final_h)
        result = x if self.return_sequences else x[:, -1]
        return (result, final) if return_state else result


def recurrent_layer(kind, params, info):
    spatial = kind.startswith("conv_")
    options = dict(cell=kind.removeprefix("conv_"), input_size=info["input_shape"][1],
                   hidden_size=params.get("hidden_size", 32 if spatial else 64), layers=params.get("layers", 1),
                   bias=bool(params.get("bias", 1)), dropout=params.get("dropout", 0),
                   return_sequences=bool(params.get("return_sequences", 1)), activation=params.get("activation", "tanh"))
    if spatial:
        return SpatialRecurrent(**options, kernel=params.get("kernel", 3), dilation=params.get("dilation", 1))
    return VectorRecurrent(**options, bidirectional=bool(params.get("bidirectional", 0)))
