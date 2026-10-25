"""Portable PyTorch sequence operators, embedded verbatim in standalone exports.

DINOv3 configurations follow Meta's public architectural specification. This is
an independent inference implementation; it contains no pretrained weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, channels, dim, patch=16, cls_token=1, registers=0, mask_token=0):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.empty(1, cls_token, dim)) if cls_token else None
        self.storage_tokens = nn.Parameter(torch.empty(1, registers, dim)) if registers else None
        self.mask_token = nn.Parameter(torch.zeros(1, dim)) if mask_token else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        # Match DINOv3's patch-projection initialization.
        nn.init.uniform_(self.proj.weight, -math.sqrt(1 / (channels * patch * patch)), math.sqrt(1 / (channels * patch * patch)))
        nn.init.uniform_(self.proj.bias, -math.sqrt(1 / (channels * patch * patch)), math.sqrt(1 / (channels * patch * patch)))

    def forward(self, x):
        patches = self.proj(x).flatten(2).transpose(1, 2)
        prefix = []
        if self.cls_token is not None:
            cls = self.cls_token
            if self.mask_token is not None:
                cls = cls + 0 * self.mask_token
            prefix.append(cls.expand(x.shape[0], -1, -1))
        if self.storage_tokens is not None:
            prefix.append(self.storage_tokens.expand(x.shape[0], -1, -1))
        return torch.cat([*prefix, patches], dim=1) if prefix else patches


class PositionEncoding(nn.Module):
    def __init__(self, tokens, dim, method="sinusoidal"):
        super().__init__()
        if method == "learned":
            self.encoding = nn.Parameter(torch.empty(1, tokens, dim))
            nn.init.trunc_normal_(self.encoding, std=0.02)
        else:
            positions = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
            rates = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim))
            encoding = torch.empty(tokens, dim)
            encoding[:, 0::2] = torch.sin(positions * rates)
            encoding[:, 1::2] = torch.cos(positions * rates[:dim // 2])
            self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence exceeds the configured position table.")
        return x + self.encoding[:, :x.shape[1]].to(x.dtype)


class TokenPool(nn.Module):
    def __init__(self, method="mean", prefix=1):
        super().__init__()
        self.method, self.prefix = method, prefix

    def forward(self, x):
        if self.method == "cls":
            return x[:, 0]
        if self.method == "patches":
            return x[:, self.prefix:]
        return x.mean(dim=1)


class TokenFeedForward(nn.Module):
    def __init__(self, dim, expansion=4, activation="gelu", bias=True, dropout=0):
        super().__init__()
        hidden = max(1, int(dim * expansion))
        self.gated = activation == "swiglu"
        self.fc1 = nn.Linear(dim, hidden * (2 if self.gated else 1), bias=bias)
        self.fc2 = nn.Linear(hidden, dim, bias=bias)
        self.activation = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu, "swiglu": F.silu}[activation]
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        if self.gated:
            gate, up = x.chunk(2, dim=-1)
            x = self.activation(gate) * up
        else:
            x = self.activation(x)
        return self.drop(self.fc2(self.drop(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, context_dim=None, causal=False, qkv_bias=True,
                 proj_bias=True, mask_k_bias=False, dropout=0, position="none",
                 rope_base=10000, grid=None, prefix=0):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.causal, self.dropout, self.position = causal, dropout, position
        self.grid, self.prefix, self.rope_base = grid, prefix, rope_base
        self.cross = context_dim is not None
        if self.cross:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(dropout)
        self.mask_k_bias = mask_k_bias
        if mask_k_bias and not self.cross and qkv_bias:
            mask = torch.ones(dim * 3)
            mask[dim:2 * dim] = 0
            self.register_buffer("bias_mask", mask)

    @staticmethod
    def _rotate(x, sin, cos):
        dtype = x.dtype
        a, b = x.float().chunk(2, dim=-1)
        rotated = torch.cat([-b, a], dim=-1)
        return (x.float() * cos + rotated * sin).to(dtype)

    def _rotary(self, x, spatial=False):
        hd = self.head_dim
        if spatial:
            h, w = self.grid
            # DINOv3: axial periods; centers independently normalized to [-1, 1].
            ys = torch.arange(h, device=x.device, dtype=torch.float32).add(0.5) / h
            xs = torch.arange(w, device=x.device, dtype=torch.float32).add(0.5) / w
            coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).flatten(0, 1) * 2 - 1
            periods = self.rope_base ** (2 * torch.arange(hd // 4, device=x.device, dtype=torch.float32) / (hd // 2))
            angles = (2 * math.pi * coords[:, :, None] / periods).flatten(1, 2).repeat(1, 2)
            body = self._rotate(x[:, :, self.prefix:], angles.sin(), angles.cos())
            return torch.cat([x[:, :, :self.prefix], body], dim=-2)
        rates = self.rope_base ** (-torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd)
        angles = (torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * rates).repeat(1, 2)
        return self._rotate(x, angles.sin(), angles.cos())

    def forward(self, x, context=None):
        b, t, dim = x.shape
        if self.cross:
            if context is None:
                raise ValueError("Cross-attention requires a separate context sequence.")
            q = self.q(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
            kv = self.kv(context)
            if self.mask_k_bias and self.kv.bias is not None:
                bias = self.kv.bias.clone()
                bias[:dim] = 0
                kv = F.linear(context, self.kv.weight, bias)
            k, v = kv.reshape(b, context.shape[1], 2, self.heads, self.head_dim).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
        else:
            bias = self.qkv.bias
            if self.mask_k_bias and bias is not None:
                bias = bias * self.bias_mask.to(bias.dtype)
            qkv = F.linear(x, self.qkv.weight, bias).reshape(b, t, 3, self.heads, self.head_dim)
            q, k, v = [part.transpose(1, 2) for part in qkv.unbind(2)]
        if self.position != "none":
            q = self._rotary(q, self.position == "rope2d")
            k = self._rotary(k, self.position == "rope2d")
        mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        return self.proj_drop(self.proj(mixed.transpose(1, 2).reshape(b, t, dim)))


class TransformerUnit(nn.Module):
    def __init__(self, dim, p, grid=None, prefix=0, context_dim=None):
        super().__init__()
        mode = p.get("mode", "encoder")
        self.pre = p.get("norm_style", "pre") == "pre"
        eps = p.get("eps", 1e-5)
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
        common = dict(heads=p.get("heads", 8), qkv_bias=bool(p.get("qkv_bias", 1)),
                      proj_bias=bool(p.get("proj_bias", 1)), mask_k_bias=bool(p.get("mask_k_bias", 0)),
                      dropout=p.get("dropout", 0), position=p.get("position", "none"),
                      rope_base=p.get("rope_base", 100 if p.get("position") == "rope2d" else 10000), grid=grid, prefix=prefix)
        self.attn = MultiHeadAttention(dim, causal=mode in ("decoder", "cross"), **common)
        self.cross_attn = MultiHeadAttention(dim, context_dim=context_dim, **common) if mode == "cross" else None
        self.norm_cross = nn.LayerNorm(dim, eps=eps) if self.cross_attn is not None else None
        self.ffn = TokenFeedForward(dim, p.get("expansion", 4), p.get("activation", "gelu"), bool(p.get("ffn_bias", 1)), p.get("dropout", 0))
        scale = p.get("layer_scale", 0)
        self.ls1 = nn.Parameter(torch.full((dim,), float(scale))) if scale > 0 else None
        self.ls2 = nn.Parameter(torch.full((dim,), float(scale))) if scale > 0 else None
        self.ls_cross = nn.Parameter(torch.full((dim,), float(scale))) if scale > 0 and self.cross_attn is not None else None

    def _branch(self, x, fn, norm, scale):
        branch = fn(norm(x) if self.pre else x)
        if scale is not None:
            branch = branch * scale
        x = x + branch
        return x if self.pre else norm(x)

    def forward(self, x, context=None):
        x = self._branch(x, self.attn, self.norm1, self.ls1)
        if self.cross_attn is not None:
            x = self._branch(x, lambda q: self.cross_attn(q, context), self.norm_cross, self.ls_cross)
        return self._branch(x, self.ffn, self.norm2, self.ls2)


class TransformerStack(nn.Module):
    def __init__(self, dim, p, grid=None, prefix=0, context_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerUnit(dim, p, grid, prefix, context_dim) for _ in range(p.get("blocks", 1))])

    def forward(self, x, context=None):
        for block in self.blocks:
            x = block(x, context)
        return x


def sequence_layer(kind, p, info):
    if kind == "token_embedding":
        return nn.Embedding(p.get("vocab", 32000), p.get("features", 128))
    if kind == "patch_embed":
        return PatchEmbedding(info["input_shape"][0], p.get("features", 192), p.get("patch", 16),
                              p.get("cls_token", 1), p.get("registers", 0), p.get("mask_token", 0))
    dim = info["input_shape"][-1]
    if kind == "position_encoding":
        return PositionEncoding(info["input_shape"][0], dim, p.get("method", "sinusoidal"))
    if kind == "token_pool":
        return TokenPool(p.get("method", "mean"), info.get("pool_prefix", p.get("prefix", 1)))
    if kind == "token_head":
        return nn.Linear(dim, p.get("vocab", 32000))
    if kind == "feedforward":
        return TokenFeedForward(dim, p.get("expansion", 4), p.get("activation", "gelu"), bool(p.get("ffn_bias", 1)), p.get("dropout", 0))
    context = info.get("context_shape", [None])[-1]
    if kind == "transformer_block":
        return TransformerStack(dim, p, info.get("grid"), info.get("prefix_tokens", 0), context)
    return MultiHeadAttention(dim, p.get("heads", 8), context, bool(p.get("causal", 0)),
                              bool(p.get("qkv_bias", 1)), bool(p.get("proj_bias", 1)), bool(p.get("mask_k_bias", 0)),
                              p.get("dropout", 0), p.get("position", "none"),
                              p.get("rope_base", 100 if p.get("position") == "rope2d" else 10000), info.get("grid"), info.get("prefix_tokens", 0))
