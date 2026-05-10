import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig


class Projector(nn.Module):
    """Projection head used as the LeJEPA embedding space."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.BatchNorm1d(dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        return self.net(x.reshape(-1, shape[-1])).view(shape)


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim = config["dim"]
        self.patch_size = config["patch_size"]
        self.grid_h = config["height"] // self.patch_size
        self.grid_w = config["width"] // self.patch_size
        n_queries = self.grid_h * self.grid_w
        patch_dim = self.patch_size * self.patch_size * 3

        self.queries = nn.Parameter(torch.randn(n_queries, dim))
        self.action_embedding = nn.Linear(9, dim)
        self.cross = nn.MultiheadAttention(dim, num_heads=config.get("n_heads", 4), batch_first=True)
        self.norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.get("n_heads", 4),
            dim_feedforward=config.get("ffn_mult", 3) * dim,
            dropout=config.get("dropout_proba", 0.1),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=1)
        self.to_patch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, patch_dim),
        )

    def forward(self, tokens: Tensor, actions: Tensor) -> Tensor:
        if actions.size(1) == tokens.size(1) - 1:
            pad = actions.new_zeros(actions.size(0), 1, actions.size(2))
            actions = torch.cat([pad, actions], dim=1)

        x = tokens + self.action_embedding(actions.float()).unsqueeze(2)
        B = x.size(0)
        x = x.reshape(B, -1, x.size(-1))
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross(q, x, x, need_weights=False)
        out = self.blocks(self.norm(out + q))
        patches = torch.sigmoid(self.to_patch(out))
        return fold_patches(
            patches,
            self.grid_h * self.patch_size,
            self.grid_w * self.patch_size,
            self.patch_size,
        )


def fold_patches(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    B = x.size(0)
    n_h = height // patch_size
    n_w = width // patch_size
    x = x.view(B, n_h, n_w, 3, patch_size, patch_size)
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
        causal: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.ffn_mult = ffn_mult
        self.dropout_proba = dropout_proba
        self.causal = causal


class WorldModel(PreTrainedModel):
    config_class = WorldModelConfig

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        assert config.height % config.patch_size == 0
        assert config.width % config.patch_size == 0

        self.n_patches = (config.height // config.patch_size) * (
            config.width // config.patch_size
        )

        self.patchify = nn.Conv2d(
            3,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, self.n_patches + 1, config.dim) * 0.02
        )

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_mult * config.dim,
            dropout=config.dropout_proba,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(spatial_layer, num_layers=config.n_blocks)
        self.encoder_projector = Projector(config.dim)

        self.action_embedding = nn.Linear(9, config.dim)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_mult * config.dim,
            dropout=config.dropout_proba,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(temporal_layer, num_layers=config.n_blocks)
        self.predictor_projector = Projector(config.dim)

        self.decoder = Decoder(config.to_dict())

    def encode(self, frames: Tensor, return_tokens: bool = False):
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)

        batch_size, seq_len, channels, height, width = frames.shape
        frames = frames.reshape(batch_size * seq_len, channels, height, width)
        if frames.is_cuda:
            frames = frames.contiguous(memory_format=torch.channels_last)

        patches = self.patchify(frames).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patches.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.spatial_pos
        tokens = self.encoder(tokens)
        tokens = self.encoder_projector(tokens)

        tokens = tokens.view(batch_size, seq_len, self.n_patches + 1, self.config.dim)
        states = tokens[:, :, 0]
        if not return_tokens:
            return states
        return states, tokens

    def predict(self, states: Tensor, actions: Tensor) -> Tensor:
        token_input = states.dim() == 4
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        if token_input:
            B, T, N, D = states.shape
            action = self.action_embedding(actions.float()).unsqueeze(2)
            tokens = (states + action).permute(0, 2, 1, 3).reshape(B * N, T, D)
        else:
            tokens = states + self.action_embedding(actions.float())

        mask = None
        if self.config.causal:
            seq_len = tokens.size(1)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
                diagonal=1,
            )

        pred_states = self.predictor_projector(self.predictor(tokens, mask=mask))
        if token_input:
            pred_states = pred_states.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return pred_states

    def forward(self, frames: Tensor, actions: Tensor, decode: bool = False):
        if not decode:
            states = self.encode(frames)
            return self.predict(states, actions)

        _, tokens = self.encode(frames, return_tokens=True)
        pred_tokens = self.predict(tokens, actions)
        decoder_tokens = torch.cat([tokens, pred_tokens[:, -1:]], dim=1)
        return pred_tokens[:, :, 0], self.decoder(decoder_tokens, actions)
