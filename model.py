"""
Full implementation of our Latent World Model
1. Encoder: A single layer CNN
2. Transformer:
    - Positional Encoding (TODO: try RoPE)
    - GELU activation function (TODO: try SiLU)
3. Decoder: A simple MLP
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from kernels import get_kernel

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

if _DEVICE == "cuda":
    _flash_attn = get_kernel("kernels-community/flash-attn3")
    _flash_attn_func = _flash_attn.flash_attn_func

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

    def _new_block(self):
        return nn.ModuleDict(
            {
                "wq": nn.Linear(self.dim, self.n_heads * self.h_dim),
                "wk": nn.Linear(self.dim, self.n_heads * self.h_dim),
                "wv": nn.Linear(self.dim, self.n_heads * self.h_dim),
                "wo": nn.Linear(self.n_heads * self.h_dim, self.dim, bias=False),
            }
        )

    def attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, N, H, D = q.shape
        if _DEVICE == "cuda":
            # Flash Attention requires float16 or bfloat16
            # Convert to float16 if in float32
            orig_dtype = q.dtype
            if orig_dtype == torch.float32:
                q = q.half()
                k = k.half()
                v = v.half()
            out = _flash_attn_func(q, k, v, causal=False, return_attn_probs=False)
            # Convert back to original dtype if needed
            if orig_dtype == torch.float32:
                out = out.float()
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = attn(q, k, v, self.h_dim).transpose(1, 2)
        return out.reshape(B, N, -1)

    def forward(self, x: Tensor) -> Tensor:
        x = pe(x)
        B, N, _ = x.shape

        for i, block in enumerate(self.blocks):
            normed = self.norms1[i](x)
            q = block["wq"](normed).view(B, N, self.n_heads, self.h_dim)
            k = block["wk"](normed).view(B, N, self.n_heads, self.h_dim)
            v = block["wv"](normed).view(B, N, self.n_heads, self.h_dim)

            out = self.attn(q, k, v)
            x = block["wo"](out) + x
            x = self.ffn[i](self.norms2[i](x)) + x

        return x

class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        patch_dim = config["patch_size"] * config["patch_size"] * 3

        self.model = nn.Sequential(
            nn.Linear(self.dim, self.dim * 5),
            nn.GELU(),
            nn.Linear(self.dim * 5, self.dim * 5),
            nn.GELU(),
            nn.Linear(self.dim * 5, patch_dim),
        )

    def forward(self, x: Tensor):
        B, N, D = x.shape
        out = self.model(x.view(B * N, D))
        return out.view(B, N, -1)

# (B, N, patch_size*patch_size*3) -> (B, 3, H, W)
def fold_patches(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    B = x.size(0)
    n_h = height // patch_size
    n_w = width // patch_size

    # (B, N, 3, p, p) -> (B, n_h, n_w, 3, p, p)
    x = x.view(B, n_h, n_w, 3, patch_size, patch_size)

    # (B, 3, H, W)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, 3, height, width)

class WorldModelConfig(PretrainedConfig):
    model_type = "world_model"

    def __init__(
        self,
        height: int = 240,
        width: int = 320,
        patch_size: int = 16,
        dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 3,
        ffn_mult: int = 3,
        dropout_proba: float = 0.1,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.ffn_mult = ffn_mult
        self.dropout_proba = dropout_proba
        super().__init__(**kwargs)

class WorldModel(PreTrainedModel):
    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        mh_config = {
            "dim": config.dim,
            "n_heads": config.n_heads,
            "n_blocks": config.n_blocks,
            "ffn_mult": config.ffn_mult,
            "dropout_proba": config.dropout_proba,
        }
        self.encoder = VisionEncoder(
            {
                **mh_config,
                "height": config.height,
                "width": config.width,
                "patch_size": config.patch_size,
            }
        )
        self.transformer = MHAttention(mh_config)
        self.decoder = Decoder(
            {
                **mh_config,
                "height": config.height,
                "width": config.width,
                "patch_size": config.patch_size,
            }
        )

        self.action_embedding = nn.Linear(9, config.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))

    def forward(self, x: Tensor, action: Tensor):
        # Handle both (B, C, H, W) and (B, M, C, H, W) inputs
        if x.dim() == 5:
            batch_size, M, C, H, W = x.shape
            x = x.reshape(batch_size * M, C, H, W)
            action = action.reshape(batch_size * M, -1)
            is_sequence = True
        else:
            batch_size = x.size(0)
            M = 1
            is_sequence = False

        x = self.encoder(x)

        token_batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(token_batch_size, -1, -1)
        action_tokens = self.action_embedding(action.float()).unsqueeze(1)

        # adding action tokens like this maybe hard to infer dynamics
        x = torch.cat([cls_tokens, action_tokens, x], dim=1)
        x = self.transformer(x)

        patches = x[:, 0, :]
        n_patches = (self.config.height // self.config.patch_size) * (
            self.config.width // self.config.patch_size
        )
        decoded = self.decoder(
            patches.unsqueeze(1).expand(-1, n_patches, -1).contiguous()
        )

        if is_sequence:
            patches = patches.view(batch_size, M, self.config.dim)[:, -1]
            decoded = decoded.view(batch_size, M, n_patches, -1)[:, -1]

        return patches, fold_patches(
            decoded,
            self.config.height,
            self.config.width,
            self.config.patch_size,
        )
