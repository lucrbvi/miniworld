import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, out_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Policy(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        dim = config.dim
        hidden = dim * 2
        self.in_proj = nn.Linear(dim * 6, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = MLP(dim, hidden_dim=hidden, dropout=config.dropout_proba)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = MLP(dim, hidden_dim=hidden, dropout=config.dropout_proba)
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, start: Tensor, current: Tensor, target: Tensor) -> Tensor:
        x = torch.cat([start, current, target, current - start, target - current, current * target], dim=-1)
        x = self.in_proj(x)
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return self.out_proj(x).squeeze(-1)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * ffn_mult, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_mask = None
        if self.causal:
            n = x.size(1)
            attn_mask = torch.triu(
                torch.full((n, n), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x

class TransformerStack(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_blocks: int, ffn_mult: int, dropout: float, causal: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ffn_mult, dropout, causal) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class Projector(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * hidden_mult, dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x))

class ViTEncoder(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        self.patch_size = config.patch_size
        self.grid_h = config.height // config.patch_size
        self.grid_w = config.width // config.patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.patchify = nn.Conv2d(3, config.dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, config.dim) * 0.02)
        self.blocks = TransformerStack(config.dim, config.n_heads, config.n_blocks, config.ffn_mult, config.dropout_proba, causal=False)
        self.projector = Projector(config.dim)

    def forward(self, frames: Tensor) -> Tensor:
        patches = self.patchify(frames).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patches.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos
        return self.projector(self.blocks(tokens))

class Predictor(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        self.action_proj = nn.Linear(config.action_dim, config.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, config.dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, config.max_seq_len, 1, config.dim) * 0.02)
        self.kind_pos = nn.Parameter(torch.randn(1, 1, 2, config.dim) * 0.02)
        self.blocks = TransformerStack(config.dim, config.n_heads, config.n_blocks, config.ffn_mult, config.dropout_proba, causal=config.causal)
        self.projector = Projector(config.dim)

    def forward(self, tokens: Tensor, actions: Tensor) -> Tensor:
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        b, t, n, d = tokens.shape
        if t > self.time_pos.size(1):
            raise ValueError(f"Sequence length {t} > max_seq_len={self.time_pos.size(1)}")

        actions = actions.to(dtype=tokens.dtype)
        action = self.action_proj(actions).unsqueeze(2)
        image_tokens = tokens + action + self.time_pos[:, :t] + self.kind_pos[:, :, 1:2]
        cls = self.cls_token.expand(b, t, 1, d) + self.time_pos[:, :t] + self.kind_pos[:, :, 0:1]
        seq = torch.cat([cls, image_tokens], dim=2).reshape(b, t * (n + 1), d)
        seq = self.projector(self.blocks(seq)).view(b, t, n + 1, d)
        return seq[:, :, :n]

class Decoder(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        self.patch_size = config.patch_size
        self.grid_h = config.height // config.patch_size
        self.grid_w = config.width // config.patch_size
        self.n_patches = self.grid_h * self.grid_w
        patch_dim = config.patch_size * config.patch_size * 3
        hidden = config.dim * config.decoder_hidden_mult

        self.pos = nn.Parameter(torch.randn(1, self.n_patches, config.dim) * 0.02)
        self.from_state = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.SiLU(),
        )
        self.to_patch = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, patch_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() == 2:
            tokens = self.from_state(tokens).unsqueeze(1).expand(-1, self.n_patches, -1)
        if tokens.dim() == 4:
            b, t, n, d = tokens.shape
            tokens = tokens.reshape(b * t, n, d)
        if tokens.size(1) == self.n_patches + 1:
            tokens = tokens[:, 1:]
        if tokens.size(1) != self.n_patches:
            raise ValueError(f"Decoder expected {self.n_patches} patch tokens, got {tokens.size(1)}")

        patches = self.to_patch(tokens + self.pos.to(dtype=tokens.dtype))
        p = self.patch_size
        img = patches.view(tokens.size(0), self.grid_h, self.grid_w, 3, p, p)
        img = img.permute(0, 3, 1, 4, 2, 5).contiguous()
        return torch.sigmoid(img.view(tokens.size(0), 3, self.grid_h * p, self.grid_w * p))

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
        decoder_hidden_mult: int = 4,
        decoder_noise_std: float = 0.05,
        decoder_pred_token_ratio: float = 0.5,
        decoder_curriculum_end: float = 0.8,
        ffn_mult: int = 3,
        dropout_proba: float = 0.1,
        causal: bool = True,
        action_dim: int = 9,
        max_seq_len: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.decoder_hidden_mult = decoder_hidden_mult
        self.decoder_noise_std = decoder_noise_std
        self.decoder_pred_token_ratio = decoder_pred_token_ratio
        self.decoder_curriculum_end = decoder_curriculum_end
        self.ffn_mult = ffn_mult
        self.dropout_proba = dropout_proba
        self.causal = causal
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len

class WorldModel(PreTrainedModel):
    config_class = WorldModelConfig
    all_tied_weights_keys = {}

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        assert config.height % config.patch_size == 0
        assert config.width % config.patch_size == 0
        self.n_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.encoder = ViTEncoder(config)
        self.predictor = Predictor(config)
        self.decoder = Decoder(config)

    def encode(self, frames: Tensor, return_tokens: bool = False):
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        b, t, c, h, w = frames.shape
        frames = frames.reshape(b * t, c, h, w)
        if frames.is_cuda:
            frames = frames.contiguous(memory_format=torch.channels_last)
        tokens = self.encoder(frames).view(b, t, self.n_patches + 1, self.config.dim)
        states = tokens[:, :, 0]
        return (states, tokens) if return_tokens else states

    def predict(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.predictor(states, actions)

    def forward(self, frames: Tensor, actions: Tensor, decode: bool = False):
        if not decode:
            _, tokens = self.encode(frames, return_tokens=True)
            return self.predict(tokens, actions)
        _, tokens = self.encode(frames, return_tokens=True)
        pred_tokens = self.predict(tokens, actions)
        return pred_tokens[:, :, 0], self.decoder(pred_tokens)
