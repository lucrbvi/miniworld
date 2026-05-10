import json
import math
import os

import lejepa
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset as HFDataset, load_dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from model import WorldModel, WorldModelConfig


def find_last_checkpoint(path: str) -> str | None:
    return get_last_checkpoint(path) if os.path.isdir(path) else None

class WMDataset(torch.utils.data.Dataset):
    """Contiguous frame windows for next-latent prediction."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        context_len: int,
        sequence_stride: int = 1,
        name: str = "dataset",
    ):
        self.context_len = context_len
        self.window_len = context_len + 1
        self.sequence_stride = sequence_stride

        print(
            f"Building {name} sequence index from {len(hf_dataset):,} frames...",
            flush=True,
        )
        episodes = self._episode_ids(hf_dataset)
        self.valid_indices = self._valid_window_starts(episodes)
        self.hf_dataset = hf_dataset.select_columns(["frame", "action"]).with_format("numpy")

        print(
            f"Built {name} sequence index: {len(self.valid_indices):,} sequences "
            f"(stride={self.sequence_stride})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.valid_indices[idx]
        samples = self.hf_dataset[start : start + self.window_len]
        frames = self._frames_to_nchw(samples["frame"])
        actions = np.asarray(samples["action"])

        return {
            "frames": torch.from_numpy(frames[: self.context_len]),
            "target_frame": torch.from_numpy(frames[self.context_len]),
            "actions": torch.from_numpy(actions[: self.context_len]).float(),
        }

    def _valid_window_starts(self, episodes: np.ndarray) -> list[int]:
        if len(episodes) < self.window_len:
            return []

        same_episode = episodes[: 1 - self.window_len] == episodes[self.window_len - 1 :]
        return np.flatnonzero(same_episode)[:: self.sequence_stride].tolist()

    @staticmethod
    def _episode_ids(hf_dataset: HFDataset) -> np.ndarray:
        if "video_idx" not in hf_dataset.column_names:
            return np.zeros(len(hf_dataset), dtype=np.int64)

        return hf_dataset.select_columns("video_idx").with_format("numpy")[:]["video_idx"]

    @staticmethod
    def _frame_to_chw(frame) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.shape[-1] == 3:
            frame = np.transpose(frame, (2, 0, 1))
        return frame

    @staticmethod
    def _frames_to_nchw(frames) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.dtype == object:
            return np.stack([WMDataset._frame_to_chw(frame) for frame in frames])
        if frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 3, 1, 2))
        return np.ascontiguousarray(frames)

class DecoderTrainer(Trainer):
    @staticmethod
    def scalar(value: torch.Tensor) -> float:
        return value.detach().float().cpu().item()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frames = inputs["frames"].float() / 255.0
        target_frame = inputs["target_frame"].float() / 255.0
        actions = inputs["actions"]

        with torch.no_grad():
            observations = torch.cat([frames, target_frame.unsqueeze(1)], dim=1)
            _, tokens = model.encode(observations, return_tokens=True)
            pred_tokens = model.predict(tokens[:, :-1], actions)
            pred_sequence = torch.cat([tokens[:, :-1], pred_tokens[:, -1:]], dim=1)

        true_recon = model.decoder(tokens.detach(), actions)
        pred_recon = model.decoder(pred_sequence.detach(), actions)
        loss = F.l1_loss(true_recon, target_frame) + F.l1_loss(pred_recon, target_frame)

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            self.log({"decoder_loss": self.scalar(loss)})

        if return_outputs:
            return loss, {"true_recon": true_recon.detach(), "pred_recon": pred_recon.detach()}
        return loss

class WMTrainer(Trainer):
    def __init__(self, *args, sigreg_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigreg_weight = sigreg_weight
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=17),
            num_slices=1024,
        )
        self._sigreg_device = None

    @staticmethod
    def scalar(value: torch.Tensor) -> float:
        return value.detach().float().cpu().item()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frames = inputs["frames"].float() / 255.0
        target_frame = (inputs["target_frame"].float() / 255.0).unsqueeze(1)
        actions = inputs["actions"]

        if self._sigreg_device != frames.device:
            self.sigreg = self.sigreg.to(frames.device)
            self._sigreg_device = frames.device

        observations = torch.cat([frames, target_frame], dim=1)
        embeddings, tokens = model.encode(observations, return_tokens=True)
        pred_tokens = model.predict(tokens[:, :-1], actions)
        target_tokens = tokens[:, 1:]

        sigreg_loss = torch.stack(
            [self.sigreg(embeddings[:, t]) for t in range(embeddings.size(1))]
        ).mean()
        pred_loss = F.mse_loss(pred_tokens, target_tokens)
        loss = pred_loss + self.sigreg_weight * sigreg_loss

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            flat_z = embeddings.flatten(0, 1)
            z_std = flat_z.std(dim=0)
            pred_z = pred_tokens[:, :, 0]
            target_z = embeddings[:, 1:]
            self.log(
                {
                    "loss_total": self.scalar(loss),
                    "pred_loss": self.scalar(pred_loss),
                    "sigreg": self.scalar(sigreg_loss),
                    "z_std_mean": self.scalar(z_std.mean()),
                    "z_std_min": self.scalar(z_std.min()),
                    "z_norm_mean": self.scalar(flat_z.norm(dim=-1).mean()),
                    "pred_std": self.scalar(pred_z.std(dim=(0, 1)).mean()),
                    "target_std": self.scalar(target_z.std(dim=(0, 1)).mean()),
                }
            )

        if return_outputs:
            return loss, {
                "pred_tokens": pred_tokens.detach(),
                "target_tokens": target_tokens.detach(),
            }
        return loss

def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train(
    config: WorldModelConfig,
    context_len: int = 16,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
    decoder_only: bool = True,
):
    device = device_name()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device} | Config: {config.to_dict()}", flush=True)
    print("Loading dataset...", flush=True)
    ds = load_dataset("lucrbrtv/doom-e1-internet-gameplay", split="train")
    print(f"Loaded dataset: {len(ds):,} frames", flush=True)

    print("Splitting train/eval...", flush=True)
    split = ds.train_test_split(test_size=0.1, shuffle=False)
    print(
        f"Split sizes: train={len(split['train']):,} | eval={len(split['test']):,}",
        flush=True,
    )

    train_dataset = WMDataset(
        split["train"],
        context_len=context_len,
        sequence_stride=sequence_stride,
        name="train",
    )
    eval_dataset = WMDataset(
        split["test"],
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

    os.environ.setdefault("WANDB_PROJECT", "miniworld-wm")

    model = WorldModel(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    args = TrainingArguments(
        output_dir="./checkpoints/world-model",
        num_train_epochs=1,
        max_steps=-1,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=40,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=0,
        weight_decay=1e-3,
        max_grad_norm=1.0,
        bf16=device == "cuda",
        logging_steps=20,
        logging_first_step=False,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=False,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="world-model",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        torch_compile=True,
    )

    steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / args.per_device_train_batch_size)
        / args.gradient_accumulation_steps
    )
    total_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    eval_batches = math.ceil(len(eval_dataset) / args.per_device_eval_batch_size)
    print(
        "Training plan: "
        f"{steps_per_epoch:,} steps/epoch | "
        f"{args.num_train_epochs:g} epoch(s) | "
        f"~{total_steps:,} total steps | "
        f"{total_steps // args.eval_steps:,} evals ({eval_batches:,} batches/eval) | "
        f"{total_steps // args.save_steps:,} saves",
        flush=True,
    )

    last_checkpoint = find_last_checkpoint(args.output_dir) or find_last_checkpoint("./checkpoints")
    if last_checkpoint is not None:
        ckpt_config = os.path.join(last_checkpoint, "config.json")
        if os.path.isfile(ckpt_config):
            with open(ckpt_config) as f:
                old_config = json.load(f)
            for key in ("height", "width", "patch_size", "dim", "n_heads", "n_blocks", "ffn_mult"):
                if old_config.get(key) != getattr(config, key):
                    print(f"Ignoring incompatible checkpoint: {last_checkpoint}", flush=True)
                    last_checkpoint = None
                    break

    if decoder_only:
        if last_checkpoint is None:
            raise RuntimeError("No compatible world-model checkpoint found for decoder training")
        print(f"Loading world model from checkpoint: {last_checkpoint}", flush=True)
        from safetensors.torch import load_file
        model.load_state_dict(load_file(os.path.join(last_checkpoint, "model.safetensors"), device="cpu"))
    else:
        trainer = WMTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            sigreg_weight=0.2,
        )
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint: {last_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=last_checkpoint)
        wandb.finish()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    model.encoder.eval()
    model.encoder_projector.eval()
    model.predictor.eval()
    model.predictor_projector.eval()

    decoder_args = TrainingArguments(
        output_dir="./checkpoints/decoder",
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size * 3,
        per_device_eval_batch_size=args.per_device_eval_batch_size * 3,
        learning_rate=1e-4,
        weight_decay=1e-4,
        bf16=device == "cuda",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="decoder-probe",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        torch_compile=False,
    )
    decoder_checkpoint = find_last_checkpoint(decoder_args.output_dir)
    if decoder_checkpoint is not None:
        print(f"Resuming decoder from checkpoint: {decoder_checkpoint}", flush=True)
    DecoderTrainer(
        model=model,
        args=decoder_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    ).train(resume_from_checkpoint=decoder_checkpoint)

if __name__ == "__main__":
    train(
        context_len=10,
        config=
        WorldModelConfig(
            height=240,
            width=320,
            patch_size=16,
            dim=480,
            n_heads=8,
            n_blocks=3,
            ffn_mult=3,
            dropout_proba=0.1,
            causal=True,
        ),
    )
