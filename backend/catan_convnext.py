"""Untrained ConvNeXt encoder proposal for the Catan observation interface.

Inputs are precomputed board planes [B,1390,45,49], context [B,578], and
optional ship attributes [B,12,79]. Outputs retain hidden width 256, per-cell
entity width 32, and ship width 32. Raw observation featurization, the fixed
526-entity assembly, legal-action scoring, and the PPG loss/phase controller
belong to the original application and are not implemented by this encoder.

The actor and value networks must instantiate separate copies; pass
include_entities=False for value-only inference to skip unused local readouts. No existing
checkpoint weights or channel-specific stem refinement offsets are reused.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 2 * channels)
        self.pwconv2 = nn.Linear(2 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x):
        branch = self.dwconv(x).permute(0, 2, 3, 1)
        branch = self.pwconv2(F.gelu(self.pwconv1(self.norm(branch))))
        return x + (branch * self.gamma).permute(0, 3, 1, 2)


class CatanConvNeXtEncoder(nn.Module):
    def __init__(self, device="meta"):
        super().__init__()
        with torch.device(device):
            self.stem = nn.Conv2d(1390, 32, 1)
            self.context = nn.Sequential(nn.Linear(578, 128), nn.SiLU(), nn.LayerNorm(128))
            self.condition = nn.Linear(128, 32)
            self.stage1 = nn.Sequential(ConvNeXtBlock(32))
            self.down1 = nn.Sequential(ChannelLayerNorm(32), nn.Conv2d(32, 48, 3, stride=2, padding=1))
            self.stage2 = nn.Sequential(ConvNeXtBlock(48))
            self.down2 = nn.Sequential(ChannelLayerNorm(48), nn.Conv2d(48, 64, 3, stride=2, padding=1))
            self.stage3 = nn.Sequential(*(ConvNeXtBlock(64) for _ in range(3)))
            self.final = ChannelLayerNorm(64)
            self.state = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256))
            self.global_residual = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
            self.entity = nn.Linear(96, 32)
            self.ship = nn.Linear(79, 32)

    def forward(self, board, context, ships=None, *, return_intermediates=False, include_entities=True):
        if board.ndim != 4 or tuple(board.shape[1:]) != (1390, 45, 49):
            raise ValueError("Expected precomputed board planes [B,1390,45,49].")
        if context.ndim != 2 or tuple(context.shape) != (board.shape[0], 578):
            raise ValueError("Expected context [B,578] with the same batch size.")
        if ships is not None and tuple(ships.shape) != (board.shape[0], 12, 79):
            raise ValueError("Expected ship attributes [B,12,79].")
        values = {"stem": self.stem(board), "context": F.silu(self.context[0](context))}
        values["context_norm"] = self.context[2](values["context"])
        values["condition"] = self.condition(values["context_norm"])
        values["conditioned"] = values["stem"] + values["condition"][:, :, None, None]
        values["stage1"] = self.stage1(values["conditioned"])
        values["down1"] = self.down1(values["stage1"])
        values["stage2"] = self.stage2(values["down1"])
        values["down2"] = self.down2(values["stage2"])
        values["stage3"] = self.stage3(values["down2"])
        values["final_norm"] = self.final(values["stage3"])
        values["mean"] = values["final_norm"].mean(dim=(-2, -1))
        values["max"] = values["final_norm"].amax(dim=(-2, -1))
        values["global_join"] = torch.cat([values["mean"], values["max"], values["context_norm"]], dim=-1)
        values["state"] = F.silu(self.state[0](values["global_join"]))
        values["state_norm"] = self.state[2](values["state"])
        values["mlp1"] = F.silu(self.global_residual[0](values["state_norm"]))
        values["mlp2"] = self.global_residual[2](values["mlp1"])
        # Stride-2 padded 3x3 downsamples preserve centers 0,2,... then 0,4,... .
        # On 45x49 -> 12x13, align_corners=True maps those centers exactly back
        # onto 0..44 / 0..48. Preserve entity coordinates rather than patch IDs.
        if include_entities or return_intermediates:
            values["upsample"] = F.interpolate(values["final_norm"], size=(45, 49), mode="bilinear", align_corners=True)
            values["local_fusion"] = torch.cat([values["stage1"], values["upsample"]], dim=1)
            values["entity"] = self.entity(values["local_fusion"].flatten(2).transpose(1, 2))
        if ships is not None and (include_entities or return_intermediates):
            values["ship"] = self.ship(ships)
        if return_intermediates:
            return values
        return {"hidden": values["mlp2"], **({"spatial_entities": values["entity"]} if include_entities else {}),
                **({"ships": values["ship"]} if ships is not None else {})}


def build_encoders(device="meta"):
    """Separate parameters preserve the actor/value gradient boundary for PPG."""
    return nn.ModuleDict({"actor": CatanConvNeXtEncoder(device), "value": CatanConvNeXtEncoder(device)})


if __name__ == "__main__":
    encoders = build_encoders()
    result = encoders["actor"](torch.empty(1, 1390, 45, 49, device="meta"), torch.empty(1, 578, device="meta"))
    print("Parameters per encoder:", sum(p.numel() for p in encoders["actor"].parameters()))
    print("Output shapes:", {key: list(value.shape) for key, value in result.items()})
