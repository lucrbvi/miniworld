"""
Full implementation of our Latent World Model
1. Encoder: A single layer CNN
2. Transformer:
    - Positional Encoding (TODO: try RoPE)
    - GELU activation function (TODO: try SwiGLU)
3. Decoder: A simple MLP
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kernels import get_kernel

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

class VisionEncoder(nn.Module):
    """A very simple CNN layer to patchify images and generate N patches of D dimensions"""

    def __init__(self, config: dict):
        super().__init__()
        assert (
            config["height"] % config["patch_size"] == 0
            and config["width"] % config["patch_size"] == 0
        ), "patch_size must divide height and width!"

        self.cnn = nn.Conv2d(
            in_channels=3,  # RGB
            out_channels=config["dim"],
            kernel_size=config["patch_size"],
            stride=config["patch_size"],
        )

    def forward(self, x: Tensor):
        C = x.size(1)
        assert C == 3, (
            f"Vision Encoder only support tensor with 3 channels (RGB), found: {C}"
        )

        # (B, D, H/patch, W/patch) -> (B, N, D)
        return self.cnn(x).flatten(2).transpose(1, 2)

# Positional Encoding
def pe(x: Tensor):
    shape = x.shape
    seq_len, dim = shape[-2], shape[-1]
    pos = torch.arange(seq_len, device=x.device).reshape(-1, 1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=x.device).float() * (-math.log(10000.0) / dim)
    )
    pe_table = torch.zeros(seq_len, dim, device=x.device)
    n_even = pe_table[:, 0::2].shape[1]
    n_odd = pe_table[:, 1::2].shape[1]
    pe_table[:, 0::2] = torch.sin(pos * div_term[:n_even])
    pe_table[:, 1::2] = torch.cos(pos * div_term[:n_odd])
    return x + pe_table

# Attention
def attn(q: Tensor, k: Tensor, v: Tensor, dim: int, mask: Tensor | None = None):
    core = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
    if mask is not None:
        core = core + mask
    return F.softmax(core, dim=-1) @ v

class MHAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.n_heads = config["n_heads"]
        self.h_dim = self.dim // self.n_heads
        self.hidden = self.dim * config["ffn_mult"]
        self.bound = 1 / math.sqrt(self.dim)

        self.blocks = nn.ModuleList(
            [self._new_block() for _ in range(config["n_blocks"])]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, config["ffn_mult"] * self.dim),
                    nn.GELU(),  # TODO: consider using SwiGLU
                    nn.Linear(config["ffn_mult"] * self.dim, self.dim),
                    nn.Dropout(p=config["dropout_proba"]),
                )
                for _ in range(config["n_blocks"])
            ]
        )

        self.norms1 = nn.ModuleList(
            [nn.LayerNorm(self.dim) for _ in range(config["n_blocks"])]
        )
        self.norms2 = nn.ModuleList(
            [nn.LayerNorm(self.dim) for _ in range(config["n_blocks"])]
        )

        if _DEVICE == "cuda":
            k = get_kernel("kernels-community/flash-attn3")
            self.flash = k.flash_attn_func

    def _new_block(self):
        return nn.ModuleDict(
            {
                "wq": nn.Linear(self.dim, self.n_heads * self.h_dim),
                "wk": nn.Linear(self.dim, self.n_heads * self.h_dim),
                "wv": nn.Linear(self.dim, self.n_heads * self.h_dim),
                "wo": nn.Linear(self.n_heads * self.h_dim, self.dim, bias=False),
            }
        )

    def forward(self, x: Tensor):
        x = pe(x)
        B, N, _ = x.shape
        for i, block in enumerate(self.blocks):
            normed = self.norms1[i](x)
            q = block["wq"](normed).view(B, N, self.n_heads, self.h_dim)
            k = block["wk"](normed).view(B, N, self.n_heads, self.h_dim)
            v = block["wv"](normed).view(B, N, self.n_heads, self.h_dim)

            if _DEVICE == "cuda":
                out = self.flash(q, k, v, softmax_scale=None, causal=False)[0]
            else:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                out = attn(q, k, v, dim=self.h_dim).transpose(1, 2)

            out = out.reshape(B, N, -1)
            x = block["wo"](out) + x
            x = self.ffn[i](self.norms2[i](x)) + x

        return x

class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        last_dim = config["height"] * config["width"]
        self.dim = config["dim"]

        self.model = nn.Sequential(
            nn.Linear(self.dim, self.dim * 5),
            nn.GELU(),
            nn.Linear(self.dim * 5, self.dim * 5),
            nn.GELU(),
            nn.Linear(self.dim * 5, last_dim),
        )

    def forward(self, x: Tensor):
        if x.dim() == 2:
            return self.model(x)

        B, N, D = x.shape
        out = self.model(x.view(B * N, D)).view(B, N, -1)

        return out  # (B, N, H*W)

class WorldModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = VisionEncoder(config)
        self.transformer = MHAttention(config)
        self.decoder = Decoder(config)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config["dim"]))

    def forward(self, x: Tensor):
        x = self.encoder(x)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        cls = x[:, 0, :]

        return self.decoder(cls)
