import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, PretrainedConfig

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, out_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Projector(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * hidden_mult, dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x))

class TransformerStack(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_blocks: int, ffn_mult: int, dropout: float, causal: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                MLP(dim, dim * ffn_mult, dim, dropout),
            ]))
        self.norm = nn.LayerNorm(dim)
        self.causal = causal
        self.gradient_checkpointing = False

    def _run_block(self, norm1, attn, norm2, mlp, x: Tensor) -> Tensor:
        attn_mask = None
        if self.causal:
            n = x.size(1)
            attn_mask = torch.triu(torch.full((n, n), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        h = norm1(x)
        x = x + attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + mlp(norm2(x))

    def forward(self, x: Tensor) -> Tensor:
        for norm1, attn, norm2, mlp in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(self._run_block, norm1, attn, norm2, mlp, x, use_reentrant=False)
            else:
                x = self._run_block(norm1, attn, norm2, mlp, x)
        return self.norm(x)

class ActionPolicy(nn.Module):
    def __init__(self, dim: int, action_dim: int = 9, n_heads: int = 4, n_blocks: int = 2, ffn_mult: int = 3, dropout: float = 0.1, max_seq_len: int = 64):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.time_pos = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.blocks = TransformerStack(dim, n_heads, n_blocks, ffn_mult, dropout, causal=True)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, action_dim),
        )

    def forward(self, states: Tensor) -> Tensor:
        if states.dim() == 2:
            states = states.unsqueeze(1)
        t = states.size(1)
        if t > self.time_pos.size(1):
            raise ValueError(f"ActionPolicy got sequence length {t} > max_seq_len {self.time_pos.size(1)}")
        return self.head(self.blocks(states + self.time_pos[:, :t]))

class ViTEncoder(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        self.patch_size = config.patch_size
        self.n_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.patchify = nn.Conv2d(3, config.dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, config.dim) * 0.02)
        self.blocks = TransformerStack(config.dim, config.n_heads, config.n_blocks, config.ffn_mult, config.dropout_proba)
        self.projector = Projector(config.dim)

    def forward(self, frames: Tensor) -> Tensor:
        patches = self.patchify(frames).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(patches.size(0), -1, -1), patches], dim=1) + self.pos
        return self.projector(self.blocks(tokens))

class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, dim * ffn_mult, dim, dropout)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        self.gradient_checkpointing = False

    def _run(self, x: Tensor, cond: Tensor, attn_mask: Tensor | None) -> Tensor:
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = self.mod(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)

    def forward(self, x: Tensor, cond: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._run, x, cond, attn_mask, use_reentrant=False)
        return self._run(x, cond, attn_mask)

class Predictor(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        self.action_proj = nn.Linear(config.action_dim, config.dim)
        self.time_pos = nn.Parameter(torch.randn(1, config.max_seq_len, 1, config.dim) * 0.02)
        self.blocks = nn.ModuleList([
            AdaLNBlock(config.dim, config.n_heads, config.ffn_mult, config.dropout_proba)
            for _ in range(config.n_blocks)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.projector = Projector(config.dim)
        self.causal = config.causal

    def forward(self, tokens: Tensor, actions: Tensor) -> Tensor:
        if tokens.dim() == 3: tokens = tokens.unsqueeze(1)
        if actions.dim() == 2: actions = actions.unsqueeze(1)
        b, t, n, d = tokens.shape
        cap = self.time_pos.size(1)
        if t > cap:
            raise ValueError(
                f"Sequence length {t} exceeds model capacity ({cap}). "
                f"This means the training loop built a context longer than the Predictor's time_pos size. "
                f"Check that the data's context_len and the model's max_seq_len are consistent."
            )
        action_emb = self.action_proj(actions.to(dtype=tokens.dtype))
        cond = action_emb.unsqueeze(2).expand(-1, -1, n, -1).reshape(b, t * n, d)
        x = (tokens + self.time_pos[:, :t]).reshape(b, t * n, d)
        attn_mask = self._block_causal_mask(t, n, x.device, x.dtype) if self.causal else None
        for block in self.blocks:
            x = block(x, cond, attn_mask=attn_mask)
        return self.projector(self.norm(x)).view(b, t, n, d)

    @staticmethod
    def _block_causal_mask(t: int, n: int, device, dtype) -> Tensor:
        frame_idx = torch.arange(t * n, device=device) // n
        return torch.where(frame_idx[None, :] > frame_idx[:, None], float("-inf"), 0.0).to(dtype=dtype)

class TokenTransition(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_blocks: int = 2, ffn_mult: int = 3, dropout: float = 0.0):
        super().__init__()
        self.blocks = TransformerStack(dim, n_heads, n_blocks, ffn_mult, dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        lead = tokens.shape[:-2]
        n, d = tokens.shape[-2:]
        out = self.blocks(tokens.reshape(-1, n, d))
        return out.reshape(*lead, n, d)

class Decoder(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        p = config.patch_size
        gh = config.height // p
        gw = config.width // p
        self.grid_h = gh
        self.grid_w = gw
        self.n_patches = gh * gw
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, config.dim) * 0.02)
        self.blocks = TransformerStack(config.dim, config.n_heads, max(1, config.decoder_n_blocks), config.ffn_mult, config.dropout_proba)
        up_c = getattr(config, "decoder_up_width", 64)
        self.to_grid = nn.Conv2d(config.dim, up_c, 1)
        self.up = nn.Sequential(
            nn.Conv2d(up_c, up_c * 4, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(up_c, up_c * 4, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(up_c, 3 * 25, 3, padding=1), nn.PixelShuffle(5),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.size(-2) != self.n_patches + 1:
            raise ValueError(f"Decoder needs CLS + {self.n_patches} patches, got {tokens.size(-2)} tokens")
        lead = tokens.shape[:-2]
        n, d = tokens.shape[-2:]
        tokens = tokens.reshape(-1, n, d)
        x = self.blocks(tokens + self.pos.to(dtype=tokens.dtype))[:, 1:]
        x = x.permute(0, 2, 1).reshape(-1, d, self.grid_h, self.grid_w)
        out = torch.sigmoid(self.up(self.to_grid(x)))
        return out.reshape(*lead, *out.shape[1:])

class WorldModelConfig(PretrainedConfig):
    model_type = "world_model"

    def __init__(
        self,
        height: int = 240, width: int = 320, patch_size: int = 16, dim: int = 256, n_heads: int = 4,
        n_blocks: int = 3, decoder_hidden_mult: int = 4, decoder_n_blocks: int = 4,
        decoder_noise_std: float = 0.05, decoder_pred_token_ratio: float = 0.98,
        decoder_curriculum_end: float = 0.85, transition_n_blocks: int = 2, ffn_mult: int = 3,
        dropout_proba: float = 0.1, causal: bool = True, action_dim: int = 9, max_seq_len: int = 64,
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
        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_noise_std = decoder_noise_std
        self.decoder_pred_token_ratio = decoder_pred_token_ratio
        self.decoder_curriculum_end = decoder_curriculum_end
        self.transition_n_blocks = transition_n_blocks
        self.ffn_mult = ffn_mult
        self.dropout_proba = dropout_proba
        self.causal = causal
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len

class WorldModel(PreTrainedModel):
    config_class = WorldModelConfig
    all_tied_weights_keys = {}
    supports_gradient_checkpointing = True

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.n_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.encoder = ViTEncoder(config)
        self.predictor = Predictor(config)
        self.transition = TokenTransition(config.dim, config.n_heads, config.transition_n_blocks, config.ffn_mult, config.dropout_proba)
        self.decoder = Decoder(config)
        self._sync_max_seq_len()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        for module in self.modules():
            if isinstance(module, (TransformerStack, AdaLNBlock)):
                module.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        for module in self.modules():
            if isinstance(module, (TransformerStack, AdaLNBlock)):
                module.gradient_checkpointing = False

    def _sync_max_seq_len(self) -> None:
        actual = self.predictor.time_pos.size(1)
        if self.config.max_seq_len != actual:
            self.config.max_seq_len = actual

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model._sync_max_seq_len()
        return model

    def encode(self, frames: Tensor, return_tokens: bool = False):
        if frames.dim() == 4: frames = frames.unsqueeze(1)
        b, t, c, h, w = frames.shape
        frames = frames.reshape(b * t, c, h, w)
        if frames.is_cuda: frames = frames.contiguous(memory_format=torch.channels_last)
        tokens = self.encoder(frames).view(b, t, self.n_patches + 1, self.config.dim)
        states = tokens[:, :, 0]
        return (states, tokens) if return_tokens else states

    def predict(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.predictor(states, actions)

    def forward(self, frames: Tensor, actions: Tensor, decode: bool = False):
        _, tokens = self.encode(frames, return_tokens=True)
        if not decode: return self.predict(tokens, actions)
        pred_tokens = self.predict(tokens, actions)
        return pred_tokens[:, :, 0], self.decoder(self.transition(pred_tokens[:, -1]))
