import lejepa
import math
import wandb
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset, load_dataset
from transformers import Trainer, TrainingArguments
from model import WorldModel, WorldModelConfig

class WMDataset(torch.utils.data.Dataset):
    """Dataset for World Model that stacks M frames and actions as context."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        context_len: int = 16,
        rollout_len: int = 2,
        sequence_stride: int | None = None,
        name: str = "dataset",
    ):
        self.hf_dataset = hf_dataset.select_columns(["frame", "action"])
        self.context_len = context_len
        self.rollout_len = rollout_len
        self.sequence_stride = sequence_stride or context_len
        print(
            f"Building {name} sequence index from {len(hf_dataset):,} frames...",
            flush=True,
        )

        window_len = context_len + rollout_len
        if "video_idx" in hf_dataset.column_names:
            episodes = hf_dataset.select_columns("video_idx").with_format("numpy")[:][
                "video_idx"
            ]
        else:
            episodes = np.zeros(len(hf_dataset), dtype=np.int64)

        if len(episodes) >= window_len:
            valid = episodes[: 1 - window_len] == episodes[window_len - 1 :]
            self.valid_indices = np.flatnonzero(valid)[:: self.sequence_stride].tolist()
        else:
            self.valid_indices = []

        print(
            f"Built {name} sequence index: {len(self.valid_indices):,} sequences "
            f"(stride={self.sequence_stride})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    @staticmethod
    def _frame_to_chw(frame):
        frame = np.asarray(frame)
        if frame.shape[-1] == 3:
            frame = np.transpose(frame, (2, 0, 1))
        return frame

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.context_len + self.rollout_len
        samples = self.hf_dataset[start:end]

        all_frames = [self._frame_to_chw(frame) for frame in samples["frame"]]
        all_actions = [np.asarray(action) for action in samples["action"]]

        frames = np.stack(all_frames[: self.context_len])
        actions = np.stack(all_actions[: self.context_len])
        future_offset = self.context_len
        future_frames = np.stack(
            all_frames[future_offset : future_offset + self.rollout_len]
        )
        future_actions = np.stack(
            all_actions[future_offset : future_offset + max(self.rollout_len - 1, 0)]
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
        cls_token = model.cls_token.expand(latent.size(0), -1, -1)
        action_token = model.action_embedding(action.float()).unsqueeze(1)
        x = torch.cat([latent.unsqueeze(1) + action_token, cls_token], dim=1)
        return model.transformer(x)[:, -1]

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    # We are training the world model and the decoder at the same time
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frames = inputs["frames"]  # [B, M, C, H, W]
        next_frame = inputs["next_frame"]
        future_frames = inputs["future_frames"]
        future_actions = inputs["future_actions"]
        actions = inputs["actions"]

        B, M, C, H, W = frames.shape
        device = frames.device
        self.sigreg_loss_fn = self.sigreg_loss_fn.to(device)

        n_views = 3 # 6 by default, but is slowing training down
        lambda_sigreg = 0.05
        lambda_pixel = 0.5
        lambda_masked = 0.25
        lambda_rollout = 0.25

        pred_next_emb, pixel_pred = model(frames, actions)

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
        masked_pred_next_emb, _ = model(x_views, actions_views)

        K = masked_pred_next_emb.shape[-1]
        masked_pred_next_emb = masked_pred_next_emb.view(n_views, B, K)  # [V, B, K]

        target_next_emb = model.encoder(next_frame).mean(dim=1)
        target_rollout_emb = model.encoder(future_frames[:, 1]).mean(dim=1)

        latent_pred_loss = (pred_next_emb - target_next_emb).square().mean()
        masked_latent_pred_loss = (
            masked_pred_next_emb - target_next_emb[None]
        ).square().mean()

        rollout_action = future_actions[:, 0]
        rollout_pred = self.rollout_step(
            model,
            pred_next_emb,
            rollout_action,
        )
        rollout_loss = nn.functional.l1_loss(
            rollout_pred,
            target_rollout_emb,
        )

        sigreg_pred = 0.5 * (
            self.sigreg_loss_fn(pred_next_emb)
            + torch.stack(
                [self.sigreg_loss_fn(masked_pred_next_emb[v]) for v in range(n_views)]
            ).mean()
        )
        sigreg_target = self.sigreg_loss_fn(target_next_emb)
        sigreg = 0.5 * (sigreg_pred + sigreg_target)

        pixel_loss = nn.functional.mse_loss(pixel_pred, next_frame)
        embedding_loss = (
            1.0 - lambda_sigreg
        ) * (
            latent_pred_loss + lambda_masked * masked_latent_pred_loss
        ) + lambda_sigreg * sigreg
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
                "masked_latent_pred_loss": masked_latent_pred_loss.detach(),
                "rollout_loss": rollout_loss.detach(),
                "sigreg": sigreg.detach(),
                "sigreg_pred": sigreg_pred.detach(),
                "sigreg_target": sigreg_target.detach(),
                "pixel_loss": pixel_loss.detach(),
            }

        return loss

def train(
    config: WorldModelConfig,
    context_len: int = 16,
    sequence_stride: int | None = None,
    max_eval_sequences: int = 2048,
):
    print(f"Device: {_DEVICE} | Config: {config.to_dict()}", flush=True)

    print("Loading dataset...", flush=True)
    ds = load_dataset("lucrbrtv/doom-e1-internet-gameplay", split="train")
    print(f"Loaded dataset: {len(ds):,} frames", flush=True)

    print("Splitting train/eval...", flush=True)
    ds = ds.train_test_split(test_size=0.1, shuffle=False)
    print(
        f"Split sizes: train={len(ds['train']):,} | eval={len(ds['test']):,}",
        flush=True,
    )

    train_dataset = WMDataset(
        ds["train"],
        context_len=context_len,
        sequence_stride=sequence_stride,
        name="train",
    )
    eval_dataset = WMDataset(
        ds["test"],
        context_len=context_len,
        sequence_stride=sequence_stride,
        name="eval",
    )
    eval_dataset = torch.utils.data.Subset(
        eval_dataset,
        range(min(max_eval_sequences, len(eval_dataset))),
    )

    print(
        f"Train sequences: {len(train_dataset)} | Eval sequences: {len(eval_dataset)}",
        flush=True,
    )

    model = WorldModel(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    print("Initializing wandb...", flush=True)
    wandb.init(project="miniworld-wm")

    training_args = TrainingArguments(
        max_steps=1000,
        output_dir="./checkpoints",
        num_train_epochs=60,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.05,
        logging_steps=20,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_dir="./checkpoints/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        bf16=_DEVICE == "cuda",
        gradient_accumulation_steps=4,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        push_to_hub=True,
        hub_model_id="doom-world-model",
    )

    train_batches_per_epoch = math.ceil(
        len(train_dataset) / training_args.per_device_train_batch_size
    )
    steps_per_epoch = math.ceil(
        train_batches_per_epoch / training_args.gradient_accumulation_steps
    )
    total_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else steps_per_epoch * math.ceil(training_args.num_train_epochs)
    )
    eval_batches = math.ceil(len(eval_dataset) / training_args.per_device_eval_batch_size)
    print(
        "Training plan: "
        f"{steps_per_epoch:,} steps/epoch | "
        f"{total_steps:,} total steps | "
        f"{total_steps // training_args.eval_steps:,} evals "
        f"({eval_batches:,} batches/eval) | "
        f"{total_steps // training_args.save_steps:,} saves",
        flush=True,
    )
    print("Starting training...", flush=True)

    trainer = WMTrainer(
        model=model,
        args=training_args,
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
            causal=True,
        )
    )
