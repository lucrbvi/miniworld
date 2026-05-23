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
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Projector(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.BatchNorm1d(dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        return self.net(x.reshape(-1, x.shape[-1])).reshape(*lead, x.shape[-1])

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

    def forward(self, x: Tensor) -> Tensor:
        for norm1, attn, norm2, mlp in self.blocks:
            attn_mask = None
            if self.causal:
                n = x.size(1)
                attn_mask = torch.triu(torch.full((n, n), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
            h = norm1(x)
            x = x + attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
            x = x + mlp(norm2(x))
        return self.norm(x)

class ActionPolicy(nn.Module):
    def __init__(self, dim: int, action_dim: int = 9, n_heads: int = 4, n_blocks: int = 2, ffn_mult: int = 3, dropout: float = 0.1, max_seq_len: int = 64):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim
        self.time_pos = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.action_embed = nn.Linear(action_dim, dim, bias=False)
        nn.init.normal_(self.action_embed.weight, std=0.02)
        self.blocks = TransformerStack(dim, n_heads, n_blocks, ffn_mult, dropout, causal=True)
        head_dim = 3 + 3 + 3 + 3  # move(3) + strafe(3) + turn(3) + binary(3)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, head_dim),
        )

    def forward(self, states: Tensor, past_actions: Tensor | None = None) -> Tensor:
        if states.dim() == 2:
            states = states.unsqueeze(1)
        t = states.size(1)
        if t > self.time_pos.size(1):
            raise ValueError(f"ActionPolicy got sequence length {t} > max_seq_len {self.time_pos.size(1)}")

        x = states
        if past_actions is not None and past_actions.numel() > 0:
            a_emb = self.action_embed(past_actions.to(dtype=states.dtype))
            n = min(a_emb.size(1), t - 1)
            if n > 0:
                x = x.clone()
                x[:, 1:1 + n] = x[:, 1:1 + n] + a_emb[:, :n]

        return self.head(self.blocks(x + self.time_pos[:, :t]))

    @staticmethod
    def logits_to_binary(logits: Tensor) -> Tensor:
        move = logits[..., 0:3].argmax(dim=-1)
        strafe = logits[..., 3:6].argmax(dim=-1)
        turn = logits[..., 6:9].argmax(dim=-1)
        attack = (torch.sigmoid(logits[..., 9]) > 0.5).long()
        use = (torch.sigmoid(logits[..., 10]) > 0.5).long()
        speed = (torch.sigmoid(logits[..., 11]) > 0.5).long()
        actions = torch.zeros(*logits.shape[:-1], 9, device=logits.device, dtype=torch.long)
        actions[..., 0] = (move == 0).long()
        actions[..., 1] = (move == 1).long()
        actions[..., 2] = (strafe == 0).long()
        actions[..., 3] = (strafe == 1).long()
        actions[..., 4] = (turn == 0).long()
        actions[..., 5] = (turn == 1).long()
        actions[..., 6] = attack
        actions[..., 7] = use
        actions[..., 8] = speed
        return actions

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

    def forward(self, x: Tensor, cond: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = self.mod(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)

class Predictor(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        self.action_proj = nn.Linear(config.action_dim, config.dim)
        self.time_pos = nn.Parameter(torch.randn(1, config.max_seq_len, config.dim) * 0.02)
        self.blocks = nn.ModuleList([
            AdaLNBlock(config.dim, config.n_heads, config.ffn_mult, config.dropout_proba)
            for _ in range(config.n_blocks)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.projector = Projector(config.dim)
        self.causal = config.causal

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        b, t, d = states.shape
        cap = self.time_pos.size(1)
        if t > cap:
            raise ValueError(
                f"Sequence length {t} exceeds model capacity ({cap}). "
                f"Check that the data's context_len and the model's max_seq_len are consistent."
            )
        x = states + self.time_pos[:, :t]
        cond = self.action_proj(actions.to(dtype=states.dtype))
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.full((t, t), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
            )
        for block in self.blocks:
            x = block(x, cond, attn_mask=attn_mask)
        return self.projector(self.norm(x))

class Decoder(nn.Module):
    def __init__(self, config: "WorldModelConfig"):
        super().__init__()
        p = config.patch_size
        gh = config.height // p
        gw = config.width // p
        self.grid_h = gh
        self.grid_w = gw
        self.n_patches = gh * gw
        up_c = getattr(config, "decoder_up_width", 64)
        self._up_c = up_c

        self.attn_block = AdaLNBlock(config.dim, config.n_heads, config.ffn_mult, config.dropout_proba)
        self.proj = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, up_c),
            nn.SiLU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(up_c, up_c * 4, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(up_c, up_c * 4, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(up_c, 3 * 25,   3, padding=1), nn.PixelShuffle(5),
        )

    def forward(self, cls_next: Tensor, patches_prev: Tensor) -> Tensor:
        lead = cls_next.shape[:-1]
        cls_flat = cls_next.reshape(-1, cls_next.shape[-1])
        patches_flat = patches_prev.reshape(-1, self.n_patches, patches_prev.shape[-1])
        B = cls_flat.shape[0]

        cond = cls_flat.unsqueeze(1).expand(-1, self.n_patches, -1)
        x = self.attn_block(patches_flat, cond)
        spatial = self.proj(x).transpose(1, 2).reshape(B, self._up_c, self.grid_h, self.grid_w)
        out = torch.sigmoid(self.up(spatial))
        return out.reshape(*lead, *out.shape[1:])

class WorldModelConfig(PretrainedConfig):
    model_type = "world_model"

    def __init__(
        self,
        height: int = 240, width: int = 320, patch_size: int = 16, dim: int = 384, n_heads: int = 6,
        n_blocks: int = 3, decoder_hidden_mult: int = 4, decoder_n_blocks: int = 4,
        decoder_noise_std: float = 0.05, decoder_pred_token_ratio: float = 0.98,
        decoder_curriculum_end: float = 0.85, ffn_mult: int = 3,
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
        self.n_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.encoder = ViTEncoder(config)
        self.predictor = Predictor(config)
        self.decoder = Decoder(config)
        self._sync_max_seq_len()

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

    def decode(self, cls_next: Tensor, patches_prev: Tensor) -> Tensor:
        return self.decoder(cls_next, patches_prev)

    def forward(self, frames: Tensor, actions: Tensor):
        states = self.encode(frames)
        return self.predict(states, actions)
