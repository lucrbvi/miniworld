import lejepa
import math
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset, load_dataset
from transformers import Trainer, TrainingArguments
from model import WorldModel, WorldModelConfig

class WMDataset(torch.utils.data.Dataset):
    """Dataset for latent world-model prediction, decoding, and rollout."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        context_len: int = 16,
        rollout_len: int = 4,
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
        if self.rollout_len > 1:
            future_actions = np.stack(
                all_actions[future_offset : future_offset + self.rollout_len - 1]
            )
        else:
            future_actions = np.zeros((0, *actions.shape[1:]), dtype=actions.dtype)

        return {
            "frames": torch.from_numpy(frames).float() / 255.0,
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

class WMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=self.univariate_test, num_slices=1024
        )
        self._sigreg_device = None

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return value.detach().float().cpu().item()
        return float(value)

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

    def rollout_latents(self, model, context_emb, actions, future_actions, horizon):
        preds = []
        rollout_context = context_emb

        for step in range(horizon):
            if step == 0:
                rollout_actions = actions
            else:
                rollout_actions = torch.cat(
                    [actions[:, step:], future_actions[:, :step]],
                    dim=1,
                )

            pred = model.predict_latent(rollout_context, rollout_actions)
            preds.append(pred)
            rollout_context = torch.cat(
                [rollout_context[:, 1:], pred.detach().unsqueeze(1)],
                dim=1,
            )

        return torch.stack(preds, dim=1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frames = inputs["frames"]  # [B, M, C, H, W]
        future_frames = inputs["future_frames"]  # [B, R, C, H, W]
        future_actions = inputs["future_actions"]  # [B, R - 1, A]
        actions = inputs["actions"]

        device = frames.device
        if self._sigreg_device != device:
            self.sigreg_loss_fn = self.sigreg_loss_fn.to(device)
            self._sigreg_device = device

        lambda_sigreg = 0.1
        lambda_decoder = 1.0
        max_lambda_rollout = 0.25
        rollout_warmup_steps = 500
        rollout_weight = max_lambda_rollout * min(
            1.0,
            self.state.global_step / rollout_warmup_steps,
        )

        context_emb = model.encode_sequence(frames)
        target_future_emb = model.encode_sequence(future_frames)
        target_next_emb = target_future_emb[:, 0]

        pred_next_emb = model.predict_latent(context_emb, actions)

        rollout_pred_emb = self.rollout_latents(
            model,
            context_emb.detach(),
            actions,
            future_actions,
            horizon=future_frames.size(1),
        )

        pred_loss = (pred_next_emb - target_next_emb).square().mean()
        encoder_emb = torch.cat(
            [context_emb.flatten(0, 1), target_future_emb.flatten(0, 1)],
            dim=0,
        )
        sigreg = self.sigreg_loss_fn(encoder_emb)

        decoded_next = model.decode_latent(target_next_emb.detach())
        decoder_loss = F.mse_loss(decoded_next, future_frames[:, 0])

        if rollout_pred_emb.size(1) > 1:
            rollout_loss = F.mse_loss(
                rollout_pred_emb[:, 1:],
                target_future_emb[:, 1:].detach(),
            )
        else:
            rollout_loss = pred_loss.new_zeros(())

        phase_a_loss = pred_loss + lambda_sigreg * sigreg
        phase_b_loss = lambda_decoder * decoder_loss
        phase_c_loss = rollout_weight * rollout_loss
        loss = phase_a_loss + phase_b_loss + phase_c_loss

        with torch.no_grad():
            z_std = encoder_emb.std(dim=0)
            self.log(
                {
                    "loss_total": self._scalar(loss),
                    "latent_loss": self._scalar(phase_a_loss),
                    "pred_loss": self._scalar(pred_loss),
                    "sigreg": self._scalar(sigreg),
                    "decoder_loss": self._scalar(decoder_loss),
                    "rollout_loss": self._scalar(rollout_loss),
                    "rollout_weight": self._scalar(rollout_weight),
                    "z_std_mean": self._scalar(z_std.mean()),
                    "z_std_min": self._scalar(z_std.min()),
                    "pred_std": self._scalar(pred_next_emb.std(dim=0).mean()),
                    "decoded_std": self._scalar(decoded_next.std(dim=(0, 2, 3)).mean()),
                }
            )

        if return_outputs:
            return loss, {
                "pred_next_emb": pred_next_emb,
                "target_next_emb": target_next_emb,
                "pred_loss": pred_loss.detach(),
                "sigreg": sigreg.detach(),
                "decoder_loss": decoder_loss.detach(),
                "rollout_loss": rollout_loss.detach(),
            }

        return loss

def train(
    config: WorldModelConfig,
    context_len: int = 16,
    rollout_len: int = 4,
    sequence_stride: int | None = 1,
    max_eval_sequences: int = 2048,
):
    if _DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")

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
        rollout_len=rollout_len,
        sequence_stride=sequence_stride,
        name="train",
    )
    eval_dataset = WMDataset(
        ds["test"],
        context_len=context_len,
        rollout_len=rollout_len,
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
        logging_steps=1,
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
        torch_compile=True,
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
