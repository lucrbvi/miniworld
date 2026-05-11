import argparse
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
    def create_optimizer(self):
        if self.optimizer is None:
            low_lr = self.args.learning_rate * 0.1
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [p for p in self.model.decoder.parameters() if p.requires_grad], "lr": self.args.learning_rate},
                    {"params": [p for p in self.model.predictor.projector.parameters() if p.requires_grad], "lr": low_lr},
                    {"params": [p for p in self.model.predictor.blocks.blocks[-1].parameters() if p.requires_grad], "lr": low_lr},
                ],
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    @staticmethod
    def scalar(value: torch.Tensor) -> float:
        return value.detach().float().cpu().item()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        dtype = next(model.parameters()).dtype
        frames = inputs["frames"].to(dtype=dtype) / 255.0
        target_frame = inputs["target_frame"].to(dtype=dtype) / 255.0
        actions = inputs["actions"].to(dtype=dtype)

        with torch.no_grad():
            observations = torch.cat([frames, target_frame.unsqueeze(1)], dim=1)
            _, tokens = model.encode(observations, return_tokens=True)
        pred_tokens = model.predict(tokens[:, :-1], actions)
        pred_sequence = torch.cat([tokens[:, :-1], pred_tokens[:, -1:]], dim=1)

        true_recon = model.decoder(tokens.detach(), actions)
        pred_recon = model.decoder(pred_sequence, actions)
        loss = F.l1_loss(true_recon.float(), target_frame.float()) + F.l1_loss(pred_recon.float(), target_frame.float())

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
        dtype = next(model.parameters()).dtype
        frames = inputs["frames"].to(dtype=dtype) / 255.0
        target_frame = (inputs["target_frame"].to(dtype=dtype) / 255.0).unsqueeze(1)
        actions = inputs["actions"].to(dtype=dtype)

        if self._sigreg_device != frames.device:
            self.sigreg = self.sigreg.to(frames.device)
            self._sigreg_device = frames.device

        observations = torch.cat([frames, target_frame], dim=1)
        embeddings, tokens = model.encode(observations, return_tokens=True)
        pred_tokens = model.predict(tokens[:, :-1], actions)
        target_tokens = tokens[:, 1:]

        sigreg_loss = torch.stack(
            [self.sigreg(embeddings[:, t].float()) for t in range(embeddings.size(1))]
        ).mean()
        pred_loss = F.mse_loss(pred_tokens.float(), target_tokens.float())
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


def resolve_checkpoint(resume_from: str | None, output_dir: str) -> str | None:
    if resume_from is None or resume_from == "none":
        return None
    if resume_from == "auto":
        return find_last_checkpoint(output_dir)
    if not os.path.isdir(resume_from):
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
    return resume_from

def train(
    config: WorldModelConfig,
    mode: str = "wm",
    output_root: str = "./checkpoints",
    resume_from: str | None = "auto",
    wm_checkpoint: str | None = None,
    context_len: int = 16,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
):
    if mode not in {"wm", "decoder"}:
        raise ValueError("mode must be 'wm' or 'decoder'")
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
    os.makedirs(output_root, exist_ok=True)

    model = WorldModel(config).to(torch.bfloat16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    wm_output_dir = os.path.join(output_root, "world-model")
    decoder_output_dir = os.path.join(output_root, "decoder")

    args = TrainingArguments(
        output_dir=wm_output_dir,
        num_train_epochs=1,
        max_steps=-1,
        per_device_train_batch_size=60,
        per_device_eval_batch_size=60,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=0,
        weight_decay=1e-3,
        max_grad_norm=1.0,
        bf16=device == "cuda",
        logging_steps=20,
        logging_first_step=False,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
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

    last_checkpoint = resolve_checkpoint(resume_from, args.output_dir)
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

    if mode == "wm":
        trainer = WMTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            sigreg_weight=0.2,
        )
        if last_checkpoint is not None:
            print(f"Resuming world-model training from: {last_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.model.to(torch.bfloat16)
        trainer.save_model(args.output_dir)
        wandb.finish()
        return

    pretrained_checkpoint = wm_checkpoint or find_last_checkpoint(wm_output_dir)
    if pretrained_checkpoint is None:
        raise RuntimeError(
            "Decoder mode needs a pretrained world model. "
            "Pass --wm-checkpoint or train with --mode wm first."
        )
    print(f"Loading pretrained world model from: {pretrained_checkpoint}", flush=True)
    model = WorldModel.from_pretrained(pretrained_checkpoint, ignore_mismatched_sizes=True).to(torch.bfloat16)

    for param in model.parameters():
        param.requires_grad = False
    for module in (model.decoder, model.predictor.projector, model.predictor.blocks.blocks[-1]):
        for param in module.parameters():
            param.requires_grad = True

    model.encoder.eval()

    decoder_args = TrainingArguments(
        output_dir=decoder_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size * 2,
        per_device_eval_batch_size=args.per_device_eval_batch_size * 2,
        learning_rate=1e-4,
        weight_decay=1e-4,
        bf16=device == "cuda",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
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
    decoder_checkpoint = resolve_checkpoint(resume_from, decoder_args.output_dir)
    if decoder_checkpoint is not None:
        print(f"Resuming decoder from checkpoint: {decoder_checkpoint}", flush=True)
    trainer = DecoderTrainer(
        model=model,
        args=decoder_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=decoder_checkpoint)
    trainer.model.to(torch.bfloat16)
    trainer.save_model(decoder_args.output_dir)
    wandb.finish()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train miniworld world model or decoder probe.")
    parser.add_argument("--mode", choices=["wm", "decoder"], default="wm")
    parser.add_argument("--output-root", default=os.environ.get("MINIWORLD_CHECKPOINT_DIR", "./checkpoints"))
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a checkpoint directory")
    parser.add_argument("--wm-checkpoint", default=None, help="Pretrained WM checkpoint for --mode decoder")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--max-eval-sequences", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    train(
        mode=cli.mode,
        output_root=cli.output_root,
        resume_from=cli.resume_from,
        wm_checkpoint=cli.wm_checkpoint,
        context_len=cli.context_len,
        sequence_stride=cli.sequence_stride,
        max_eval_sequences=cli.max_eval_sequences,
        config=WorldModelConfig(
            height=240,
            width=320,
            patch_size=20,
            dim=380,
            n_heads=4,
            n_blocks=3,
            ffn_mult=3,
            dropout_proba=0.1,
            causal=True,
        ),
    )
