import argparse
import json
import math
import os

import lejepa
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset as HFDataset, load_dataset
from safetensors.torch import save_file
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from model import RewardModel, WorldModel, WorldModelConfig

WANDB_PROJECT = "miniworld-wm"

def setup_wandb() -> None:
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

def finish_wandb() -> None:
    if wandb.run is not None:
        wandb.finish()

def start_wandb_run(name: str, config: dict | None = None) -> None:
    finish_wandb()
    wandb.init(project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT), name=name, config=config)

def find_last_checkpoint(path: str) -> str | None:
    return get_last_checkpoint(path) if os.path.isdir(path) else None

class WMDataset(torch.utils.data.Dataset):
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

class SectionedWandbTrainer(Trainer):
    wandb_section: str = ""

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self.state.epoch is not None and "epoch" not in logs:
            logs["epoch"] = round(self.state.epoch, 4)

        if self.wandb_section:
            prefix = f"{self.wandb_section}/"
            logs = {
                key if key == "epoch" or key.startswith(prefix) else f"{prefix}{key}": value
                for key, value in logs.items()
            }

        if wandb.run is not None:
            wandb.log(logs, step=self.state.global_step)

        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

def scalar(value: torch.Tensor) -> float:
    return value.detach().float().cpu().item()

class DecoderTrainer(SectionedWandbTrainer):
    wandb_section = "decoder"

    def __init__(self, *args, lpips_weight: float = 0.1, rollout_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips_weight = lpips_weight
        self.rollout_weight = rollout_weight
        self.lpips_loss = lpips.LPIPS(net="alex").eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

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

        model.encoder.eval()
        model.predictor.eval()
        self.lpips_loss = self.lpips_loss.to(frames.device)

        with torch.no_grad():
            observations = torch.cat([frames, target_frame.unsqueeze(1)], dim=1)
            _, tokens = model.encode(observations, return_tokens=True)
            encoder_latents = tokens[:, -1]
            predicted_latents = model.predict(tokens[:, :-1], actions)[:, -1]
            rollout_latents = []
            rollout_tokens = tokens[:, :1]
            for t in range(actions.size(1)):
                pred_tokens = model.predict(rollout_tokens, actions[:, : t + 1])
                next_tokens = pred_tokens[:, -1]
                rollout_latents.append(next_tokens)
                rollout_tokens = torch.cat([rollout_tokens, next_tokens.unsqueeze(1)], dim=1)
            rollout_latents = torch.stack(rollout_latents, dim=1)
            if self.state.max_steps <= 0:
                progress = 0.0
            else:
                end_step = max(1.0, self.state.max_steps * model.config.decoder_curriculum_end)
                progress = min(1.0, self.state.global_step / end_step)
            pred_ratio = model.config.decoder_pred_token_ratio * progress
            use_pred = torch.rand(encoder_latents.size(0), 1, 1, device=encoder_latents.device) < pred_ratio
            decoder_latents = torch.where(use_pred, predicted_latents, encoder_latents)
            if model.config.decoder_noise_std > 0:
                decoder_latents = decoder_latents + torch.randn_like(decoder_latents) * model.config.decoder_noise_std
                rollout_latents = rollout_latents + torch.randn_like(rollout_latents) * model.config.decoder_noise_std

        recon = model.decoder(decoder_latents)
        rollout_recon = model.decoder(rollout_latents)
        rollout_target = observations[:, 1:].reshape(-1, *observations.shape[2:])
        l1_loss = F.l1_loss(recon.float(), target_frame.float())
        rollout_l1_loss = F.l1_loss(rollout_recon.float(), rollout_target.float())
        lpips_loss = self.lpips_loss(recon.float() * 2 - 1, target_frame.float() * 2 - 1).mean()
        rollout_lpips_loss = self.lpips_loss(rollout_recon.float() * 2 - 1, rollout_target.float() * 2 - 1).mean()
        loss = l1_loss + self.rollout_weight * rollout_l1_loss + self.lpips_weight * (lpips_loss + self.rollout_weight * rollout_lpips_loss)

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "loss": scalar(loss),
                    "l1": scalar(l1_loss),
                    "rollout_l1": scalar(rollout_l1_loss),
                    "lpips": scalar(lpips_loss),
                    "rollout_lpips": scalar(rollout_lpips_loss),
                    "pred_token_ratio": pred_ratio,
                    "used_pred_token_ratio": scalar(use_pred.float().mean()),
                    "latent_noise_std": model.config.decoder_noise_std,
                }
            )

        if return_outputs:
            return loss, {"recon": recon.detach()}
        return loss

class WMTrainer(SectionedWandbTrainer):
    wandb_section = "wm"

    def __init__(self, *args, sigreg_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigreg_weight = sigreg_weight
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=17),
            num_slices=1024,
        )
        self._sigreg_device = None

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
        rollout_preds = []
        rollout_tokens = tokens[:, :1]
        for t in range(actions.size(1)):
            next_preds = model.predict(rollout_tokens, actions[:, : t + 1])[:, -1]
            rollout_preds.append(next_preds)
            rollout_tokens = torch.cat([rollout_tokens, next_preds.unsqueeze(1)], dim=1)
        rollout_preds = torch.stack(rollout_preds, dim=1)

        sigreg_loss = torch.stack(
            [self.sigreg(embeddings[:, t].float()) for t in range(embeddings.size(1))]
        ).mean()
        pred_loss = F.mse_loss(pred_tokens[:, :, 0].float(), embeddings[:, 1:].float())
        token_pred_loss = F.mse_loss(pred_tokens.float(), tokens[:, 1:].float())
        rollout_loss = F.mse_loss(rollout_preds.float(), tokens[:, 1:].float())
        loss = pred_loss + token_pred_loss + rollout_loss + self.sigreg_weight * sigreg_loss

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            flat_z = embeddings.flatten(0, 1)
            z_std = flat_z.std(dim=0)
            pred_z = pred_tokens[:, :, 0]
            target_z = embeddings[:, 1:]
            self.log(
                {
                    "loss_total": scalar(loss),
                    "pred_loss": scalar(pred_loss),
                    "token_pred_loss": scalar(token_pred_loss),
                    "rollout_loss": scalar(rollout_loss),
                    "sigreg": scalar(sigreg_loss),
                    "z_std_mean": scalar(z_std.mean()),
                    "z_std_min": scalar(z_std.min()),
                    "z_norm_mean": scalar(flat_z.norm(dim=-1).mean()),
                    "pred_std": scalar(pred_z.std(dim=(0, 1)).mean()),
                    "target_std": scalar(target_z.std(dim=(0, 1)).mean()),
                    "pred_target_std_ratio": scalar(pred_z.std(dim=(0, 1)).mean() / target_z.std(dim=(0, 1)).mean().clamp_min(1e-6)),
                }
            )

        if return_outputs:
            return loss, {
                "pred_tokens": pred_tokens.detach(),
                "target_embeddings": embeddings[:, 1:].detach(),
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

class RewardModelTrainer(SectionedWandbTrainer):
    wandb_section = "reward_model"

    def __init__(self, wm_model: WorldModel, *args, rank_temperature: float = 0.25, reward_tau: float = 8.0, predicted_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_model = wm_model.eval()
        self.rank_temperature = rank_temperature
        self.reward_tau = reward_tau
        self.predicted_ratio = predicted_ratio
        for p in self.wm_model.parameters():
            p.requires_grad = False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        dtype = next(self.wm_model.parameters()).dtype
        frames = inputs["frames"].to(device=self.args.device, dtype=dtype) / 255.0
        target_frame = inputs["target_frame"].to(device=self.args.device, dtype=dtype).unsqueeze(1) / 255.0
        actions = inputs["actions"].to(device=self.args.device, dtype=dtype)

        with torch.no_grad():
            observations = torch.cat([frames, target_frame], dim=1)
            _, real_tokens = self.wm_model.encode(observations, return_tokens=True)
            pred_tokens = self.wm_model.predict(real_tokens[:, :-1], actions)
            use_pred = torch.rand(real_tokens.size(0), 1, 1, 1, device=real_tokens.device) < self.predicted_ratio
            candidate_tokens = torch.where(use_pred, pred_tokens, real_tokens[:, 1:])

        batch_size = real_tokens.size(0)
        steps = candidate_tokens.size(1)
        batch_idx = torch.arange(batch_size, device=real_tokens.device)
        i = torch.randint(0, steps, (batch_size,), device=real_tokens.device)
        j = torch.randint(0, steps, (batch_size,), device=real_tokens.device)
        current_i = candidate_tokens[batch_idx, i]
        current_j = candidate_tokens[batch_idx, j]
        start = real_tokens[:, 0]
        goal = real_tokens[:, -1]
        score_i = model(start.float(), current_i.float(), goal.float())
        score_j = model(start.float(), current_j.float(), goal.float())
        rank_logit = (score_i - score_j) / self.rank_temperature
        rank_label = (i > j).float()
        ranking_loss = F.binary_cross_entropy_with_logits(rank_logit, rank_label)

        flat_current = candidate_tokens.reshape(batch_size * steps, *candidate_tokens.shape[2:])
        flat_start = start[:, None].expand(-1, steps, -1, -1).reshape_as(flat_current)
        flat_goal = goal[:, None].expand(-1, steps, -1, -1).reshape_as(flat_current)
        flat_score = model(flat_start.float(), flat_current.float(), flat_goal.float())
        steps_to_goal = torch.arange(steps - 1, -1, -1, device=real_tokens.device, dtype=torch.float32)
        y = torch.exp(-steps_to_goal / self.reward_tau).repeat(batch_size)
        calibration_loss = F.binary_cross_entropy_with_logits(flat_score, y)

        wrong_goal = goal.roll(1, dims=0)
        wrong_score = model(start.float(), candidate_tokens[:, -1].float(), wrong_goal.float())
        negative_goal_loss = F.binary_cross_entropy_with_logits(wrong_score, torch.zeros_like(wrong_score))
        loss = ranking_loss + 0.05 * calibration_loss + 0.01 * negative_goal_loss

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            score = torch.cat([score_i, score_j, flat_score, wrong_score])
            self.log(
                {
                    "ranking_loss": scalar(ranking_loss),
                    "calibration_loss": scalar(calibration_loss),
                    "negative_goal_loss": scalar(negative_goal_loss),
                    "used_predicted_ratio": scalar(use_pred.float().mean()),
                    "score_mean": scalar(score.mean()),
                    "score_std": scalar(score.std()),
                    "prob_mean": scalar(torch.sigmoid(score).mean()),
                    "prob_std": scalar(torch.sigmoid(score).std()),
                }
            )

        if return_outputs:
            return loss, {"score_i": score_i.detach(), "score_j": score_j.detach(), "y": y.detach()}
        return loss

    def save_reward_model(self, output_dir: str | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_file(self.model.state_dict(), os.path.join(output_dir, "reward_model.safetensors"))
        with open(os.path.join(output_dir, "reward_model_config.json"), "w") as f:
            json.dump(self.wm_model.config.to_dict(), f, indent=2)
        print(f"Saved reward model to {output_dir} (reward_model.safetensors and reward_model_config.json)", flush=True)

def train(
    config: WorldModelConfig,
    mode: str = "all",
    output_root: str = "./checkpoints",
    resume_from: str | None = "auto",
    wm_checkpoint: str | None = None,
    context_len: int = 16,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
):
    if mode not in {"all", "wm", "decoder", "reward_model"}:
        raise ValueError("mode must be 'all', 'wm', 'decoder' or 'reward_model'")
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

    setup_wandb()
    os.makedirs(output_root, exist_ok=True)

    model = WorldModel(config).to(torch.bfloat16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    wm_output_dir = os.path.join(output_root, "world-model")
    decoder_output_dir = os.path.join(output_root, "decoder")
    reward_model_output_dir = os.path.join(output_root, "reward-model")

    args = TrainingArguments(
        output_dir=wm_output_dir,
        num_train_epochs=1,
        max_steps=-1,
        per_device_train_batch_size=65,
        per_device_eval_batch_size=65,
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
        report_to=[],
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

    if mode in {"all", "wm"}:
        trainer = WMTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            sigreg_weight=0.1,
        )
        if last_checkpoint is not None:
            print(f"Resuming world-model training from: {last_checkpoint}", flush=True)
        start_wandb_run("world-model", config=config.to_dict())
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.model.to(torch.bfloat16)
        trainer.save_model(args.output_dir)
        finish_wandb()
        if mode == "wm":
            return
        model = trainer.model
    else:
        pretrained_checkpoint = wm_checkpoint or find_last_checkpoint(wm_output_dir)
        if pretrained_checkpoint is None:
            raise RuntimeError(
                f"{mode.capitalize()} mode needs a pretrained world model. "
                "Pass --wm-checkpoint or train with --mode wm first."
            )
        print(f"Loading pretrained world model from: {pretrained_checkpoint}", flush=True)
        model = WorldModel.from_pretrained(pretrained_checkpoint, ignore_mismatched_sizes=True).to(device=device, dtype=torch.bfloat16)

    if mode == "reward_model":
        reward_model_args = TrainingArguments(
            output_dir=reward_model_output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=args.per_device_train_batch_size * 5,
            per_device_eval_batch_size=args.per_device_eval_batch_size * 5,
            learning_rate=1e-4,
            weight_decay=1e-4,
            max_grad_norm=1.0,
            bf16=device == "cuda",
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            dataloader_num_workers=16,
            dataloader_prefetch_factor=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            remove_unused_columns=False,
            report_to=[],
            run_name="reward-model",
            optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
            torch_compile=True,
        )
        reward_model_trainer = RewardModelTrainer(
            wm_model=model,
            model=RewardModel(model.config),
            args=reward_model_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        start_wandb_run("reward-model", config=config.to_dict())
        reward_model_trainer.train()
        reward_model_trainer.save_reward_model(reward_model_args.output_dir)
        finish_wandb()
        return

    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    model.eval()
    model.decoder.train()
    print("Frozen world-model encoder/predictor; training decoder only.", flush=True)

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
        report_to=[],
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
    start_wandb_run("decoder-probe", config=config.to_dict())
    trainer.train(resume_from_checkpoint=decoder_checkpoint)
    trainer.model.to(torch.bfloat16)
    trainer.save_model(decoder_args.output_dir)
    finish_wandb()
    if mode == "decoder":
        return

    reward_model_args = TrainingArguments(
        output_dir=reward_model_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size * 5,
        per_device_eval_batch_size=args.per_device_eval_batch_size * 5,
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        bf16=device == "cuda",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=[],
        run_name="reward-model",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        torch_compile=True,
    )
    reward_model_trainer = RewardModelTrainer(
        wm_model=model,
        model=RewardModel(model.config),
        args=reward_model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    start_wandb_run("reward-model", config=config.to_dict())
    reward_model_trainer.train()
    reward_model_trainer.save_reward_model(reward_model_args.output_dir)
    finish_wandb()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train miniworld world model, decoder probe, or reward model.")
    parser.add_argument("--mode", choices=["all", "wm", "decoder", "reward_model"], default="all")
    parser.add_argument("--output-root", default=os.environ.get("MINIWORLD_CHECKPOINT_DIR", "./checkpoints"))
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a checkpoint directory")
    parser.add_argument("--wm-checkpoint", default=None, help="Pretrained WM checkpoint for --mode decoder or reward_model")
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
