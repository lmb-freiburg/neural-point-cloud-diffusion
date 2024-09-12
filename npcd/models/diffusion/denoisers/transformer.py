import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(ndim, device=device, dtype=dtype)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self, 
        *, 
        heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        
        self.use_flash_attn = use_flash_attn
        
        if not self.use_flash_attn:
            self.attn_dropout = nn.Dropout(dropout)

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        if self.use_flash_attn:
            out = flash_attn_func(q, k, v, causal=False, dropout_p=self.dropout if self.training else 0)
        else:
            scale = 1 / math.sqrt(math.sqrt(attn_ch))
            weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
            weight = torch.softmax(weight, dim=-1)
            weight = self.attn_dropout(weight)
            out = torch.einsum("bhts,bshc->bthc", weight, v)
    
        out = out.reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        bias: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=bias)
        self.c_proj = nn.Linear(width, width, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.attention = QKVMultiheadAttention(heads=heads,
                                               dropout=dropout,
                                               use_flash_attn=use_flash_attn)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        x = self.resid_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(
        self, 
        *, 
        width: int, 
        dropout: float = 0.0, 
        init_scale: float = 1.0, 
        bias: bool = True,
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, bias=bias)
        self.c_proj = nn.Linear(width * 4, width, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        dropout: float = 0.0,
        init_scale: float = 1.0,
        bias: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
            dropout=dropout,
            init_scale=init_scale,
            bias=bias,
            use_flash_attn=use_flash_attn,
        )
        if bias:  # use PyTorch LayerNorm class
            self.ln_1 = nn.LayerNorm(width)
            self.ln_2 = nn.LayerNorm(width)
        else:  # use custom LayerNorm class
            self.ln_1 = LayerNorm(width, bias=bias)
            self.ln_2 = LayerNorm(width, bias=bias)
        self.mlp = MLP(width=width, bias=bias, dropout=dropout, init_scale=init_scale)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        layers: int,
        heads: int,
        dropout: float = 0.0,
        init_scale: float = 0.25,
        bias: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                    init_scale=init_scale,
                    bias=bias,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class NPCDTransformer(nn.Module):
    def __init__(
        self,
        *,
        coords_dim: int,
        feats_dim: int,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.coords_dim = coords_dim
        self.feats_dim = feats_dim
        self.input_channels = coords_dim + feats_dim
        self.output_channels = coords_dim + feats_dim
        
        self.time_embed = MLP(width=width, init_scale=init_scale * math.sqrt(1.0 / width))
        
        self.ln_pre = nn.LayerNorm(width)
        self.backbone = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            use_flash_attn=use_flash_attn,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width)
        self.input_proj = nn.Linear(self.input_channels, width)
        self.output_proj = nn.Linear(width, self.output_channels)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, coords: torch.Tensor, feats: torch.Tensor, t: torch.Tensor):
        x = torch.cat([coords, feats], dim=1)  # [N, C, num_points]
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))  # [N, C]
        extra_tokens = [(t_embed, True)]
        
        pred = self._forward_with_cond(x, extra_tokens)
        
        pred_coords = pred[:, :self.coords_dim]
        pred_feats = pred[:, self.coords_dim:]
        return pred_coords, pred_feats

    def _forward_with_cond(self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # [N, num_points, width]
        
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)  # [N, num_points+(num_extra_tokens), width]

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]  # remove extra tokens
        h = self.output_proj(h)
        return h.permute(0, 2, 1)
