import lejepa
import numpy as np
import torch
import torch.nn as nn
from datasets import (
    Dataset as HFDataset,
)
from datasets import (
    load_dataset,
)
from transformers import Trainer, TrainingArguments

import wandb
from model import WorldModel, WorldModelConfig

class WMDataset(torch.utils.data.Dataset):
    """Dataset for World Model that stacks M frames and actions as context."""

    def __init__(self, hf_dataset: HFDataset, context_len: int = 16, rollout_len: int = 2):
        self.context_len = context_len
        self.rollout_len = rollout_len
        self.frames = []
        self.actions = []
        self.episodes = []

        for sample in hf_dataset:
            frame = sample["frame"]
            if hasattr(frame, "convert"):
                frame = np.array(frame)
            if frame.shape[-1] == 3:
                frame = np.transpose(frame, (2, 0, 1))
            self.frames.append(frame)
            self.actions.append(np.array(sample["action"]))
            self.episodes.append(sample.get("video_idx", 0))

        self.valid_indices = [
            i
            for i in range(len(self.frames) - context_len - rollout_len + 1)
            if self.episodes[i] == self.episodes[i + context_len + rollout_len - 1]
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        frames = np.stack(self.frames[start : start + self.context_len])
        actions = np.stack(self.actions[start : start + self.context_len])
        future_start = start + self.context_len
        future_frames = np.stack(
            self.frames[future_start : future_start + self.rollout_len]
        )
        future_actions = np.stack(
            self.actions[future_start : future_start + max(self.rollout_len - 1, 0)]
        )
        next_frame = future_frames[0]
        return {
            "frames": torch.from_numpy(frames).float() / 255.0,
            "next_frame": torch.from_numpy(next_frame).float() / 255.0,
            "future_frames": torch.from_numpy(future_frames).float() / 255.0,
            "future_actions": torch.from_numpy(future_actions).float(),
            "actions": torch.from_numpy(actions).float(),
        }

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

# Create and apply a binary mask to let n_visible masks visible on multiple frames
# (it generate a new random mask for each frames)
def make_mask(B, M, H, W, n_visible, device, patch_size=16):
    ph, pw = H // patch_size, W // patch_size
    idx = torch.stack(
        [torch.randperm(ph * pw, device=device)[:n_visible] for _ in range(B * M)]
    )
    mask = torch.zeros(B * M, ph * pw, device=device).scatter_(1, idx, 1.0)
    mask = mask.view(B, M, ph, pw).unsqueeze(2)
    return mask.repeat_interleave(patch_size, -2).repeat_interleave(
        patch_size, -1
    )  # (B, M, 1, H, W)

class WMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=self.univariate_test, num_slices=1024
        )

    def rollout_step(self, model, latent, action):
        action_token = model.action_embedding(action.float()).unsqueeze(1)
        x = torch.cat([latent.unsqueeze(1), action_token], dim=1)
        return model.transformer(x)[:, 0]

    # We are training the world model and the decoder at the same time
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frames = inputs["frames"]  # [B, M, C, H, W]
        next_frame = inputs["next_frame"]
        future_frames = inputs["future_frames"]
        future_actions = inputs["future_actions"]
        actions = inputs["actions"]

        B, M, C, H, W = frames.shape
        device = frames.device

        n_views = 6
        lambda_sigreg = 0.05
        lambda_pixel = 0.1
        lambda_rollout = 0.25

        views = [
            frames
            * make_mask(
                B,
                M,
                H,
                W,
                n_visible=6,
                device=device,
            )
            for _ in range(n_views)
        ]

        x_views = torch.cat(views, dim=0).contiguous()  # [n_views * B, M, C, H, W]
        actions_views = actions.repeat(n_views, *([1] * (actions.ndim - 1)))

        pred_next_emb, pixel_pred = model(x_views, actions_views)

        K = pred_next_emb.shape[-1]
        pred_next_emb = pred_next_emb.view(n_views, B, K)  # [V, B, K]

        with torch.no_grad():
            target_next_emb = model.encoder(next_frame).mean(dim=1)
            target_rollout_emb = model.encoder(future_frames[:, 1]).mean(dim=1)

        latent_pred_loss = (pred_next_emb - target_next_emb[None]).square().mean()

        rollout_action = future_actions[:, 0].repeat(n_views, 1)
        rollout_pred = self.rollout_step(
            model,
            pred_next_emb.reshape(n_views * B, K),
            rollout_action,
        )
        rollout_pred = rollout_pred.view(n_views, B, K)
        rollout_loss = nn.functional.l1_loss(
            rollout_pred,
            target_rollout_emb[None].expand_as(rollout_pred),
        )

        sigreg_pred = torch.stack(
            [self.sigreg_loss_fn(pred_next_emb[v]) for v in range(n_views)]
        ).mean()
        sigreg_target = self.sigreg_loss_fn(target_next_emb)
        sigreg = 0.5 * (sigreg_pred + sigreg_target)

        pixel_target = next_frame.repeat(n_views, *([1] * (next_frame.ndim - 1)))
        pixel_loss = nn.functional.mse_loss(pixel_pred, pixel_target)
        embedding_loss = (
            1.0 - lambda_sigreg
        ) * latent_pred_loss + lambda_sigreg * sigreg
        loss = (
            (1.0 - lambda_pixel) * embedding_loss
            + lambda_pixel * pixel_loss
            + lambda_rollout * rollout_loss
        )

        if return_outputs:
            return loss, {
                "pred_next_emb": pred_next_emb,
                "target_next_emb": target_next_emb,
                "pixel_pred": pixel_pred,
                "latent_pred_loss": latent_pred_loss.detach(),
                "rollout_loss": rollout_loss.detach(),
                "sigreg": sigreg.detach(),
                "sigreg_pred": sigreg_pred.detach(),
                "sigreg_target": sigreg_target.detach(),
                "pixel_loss": pixel_loss.detach(),
            }

        return loss

def train(config: WorldModelConfig, context_len: int = 16):
    print(f"Device: {_DEVICE} | Config: {config.to_dict()}")

    ds = load_dataset("lucrbrtv/doom-e1-internet-gameplay", split="train").shuffle(
        seed=42
    )
    ds = ds.train_test_split(test_size=0.1, shuffle=True)

    train_dataset = WMDataset(ds["train"], context_len=context_len)
    eval_dataset = WMDataset(ds["test"], context_len=context_len)

    print(
        f"Train sequences: {len(train_dataset)} | Eval sequences: {len(eval_dataset)}"
    )

    model = WorldModel(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    wandb.init(project="miniworld-wm")

    trainer = WMTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=60,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=1e-4,
            warmup_steps=500,
            weight_decay=0.05,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=800,
            logging_dir="./checkpoints/logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=1.0,
            bf16=_DEVICE == "cuda",
            gradient_accumulation_steps=2,
            dataloader_num_workers=8,
            dataloader_prefetch_factor=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            remove_unused_columns=False,
            report_to=["wandb"],
            push_to_hub=True,
            hub_model_id="doom-world-model",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    train(
        WorldModelConfig(
            height=240,
            width=320,
            patch_size=16,
            dim=384,
            n_heads=4,
            n_blocks=8,
            ffn_mult=3,
            dropout_proba=0.1,
        )
    )
