import argparse
import json
import math
import os

import lejepa
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset as HFDataset, load_dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from model import ActionPolicy, WorldModel, WorldModelConfig

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
    def __init__(self, hf_dataset: HFDataset, context_len: int, sequence_stride: int = 1, name: str = "dataset"):
        self.context_len = context_len
        self.window_len = context_len
        self.sequence_stride = sequence_stride
        episodes = self._episode_ids(hf_dataset)
        self.valid_indices = self._valid_window_starts(episodes)
        self.hf_dataset = hf_dataset.select_columns(["frame", "action"]).with_format("numpy")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.valid_indices[idx]
        samples = self.hf_dataset[start : start + self.window_len]
        frames = self._frames_to_nchw(samples["frame"])
        actions = np.asarray(samples["action"])
        return {
            "frames": torch.from_numpy(frames[: self.context_len]),
            "target_frame": torch.from_numpy(frames[self.context_len - 1]),
            "actions": torch.from_numpy(actions[: self.context_len - 1]).float(),
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

    def __init__(self, *args, lpips_weight: float = 0.2, noise_std: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips_weight = lpips_weight
        self.noise_std = noise_std
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
        B, T, C, H, W = frames.shape

        if self.model.training:
            frames = F.interpolate(frames.flatten(0, 1), size=(H + 32, W + 48), mode="bilinear").view(B, T, C, H + 32, W + 48)
            top = torch.randint(0, 33, (B,), device=frames.device)
            left = torch.randint(0, 49, (B,), device=frames.device)
            frames = torch.stack([frames[b, :, :, t:t + H, l:l + W] for b, (t, l) in enumerate(zip(top, left))])

        self.lpips_loss = self.lpips_loss.to(frames.device)

        with torch.no_grad():
            states, tokens = model.encode(frames, return_tokens=True)

        patches_prev = tokens[:, :-1, 1:].detach()
        cls_next = states[:, 1:]

        if self.noise_std > 0:
            cls_next = cls_next + torch.randn_like(cls_next) * self.noise_std

        cls_next_flat = cls_next.flatten(0, 1)
        patches_prev_flat = patches_prev.flatten(0, 1)
        targets = frames[:, 1:].flatten(0, 1)

        recon = model.decode(cls_next_flat, patches_prev_flat)

        l1_loss = F.l1_loss(recon.float(), targets.float())
        lpips_loss = self.lpips_loss(recon.float() * 2 - 1, targets.float() * 2 - 1).mean()
        loss = l1_loss + self.lpips_weight * lpips_loss

        gen_loss = None
        if self.model.training and self.noise_std > 0:
            gen_loss, disc_loss = self._gan_loss(recon.float(), targets.float())
            loss = loss + 0.1 * gen_loss

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            log_dict = {"loss": scalar(loss), "l1": scalar(l1_loss), "lpips": scalar(lpips_loss)}
            if gen_loss is not None:
                log_dict["gan"] = scalar(gen_loss)
                log_dict["discriminator_loss"] = scalar(disc_loss)
            self.log(log_dict)

        if return_outputs:
            return loss, {"recon": recon.detach()}
        return loss

    def _gan_loss(self, fake: torch.Tensor, real: torch.Tensor, r1_gamma: float = 10.0, r1_batch: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_disc"):
            self._disc = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
                nn.Conv2d(256, 1, 4, 1, 0),
            ).to(fake.device)
            self._disc_opt = torch.optim.Adam(self._disc.parameters(), lr=1e-4, betas=(0.5, 0.9))

        fake_det = fake.detach()
        real_pred = self._disc(real)
        fake_pred_det = self._disc(fake_det)
        disc_loss = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred_det).mean()

        r1_penalty = torch.tensor(0.0, device=fake.device)
        if r1_gamma > 0:
            n = min(real.size(0), r1_batch)
            idx = torch.randperm(real.size(0), device=real.device)[:n]
            real_sub = real[idx].detach().requires_grad_(True)
            real_pred_sub = self._disc(real_sub)
            grad_real, = torch.autograd.grad(
                outputs=real_pred_sub.sum(), inputs=real_sub, create_graph=True, only_inputs=True,
            )
            r1_penalty = 0.5 * r1_gamma * grad_real.pow(2).mean()

        self._disc_opt.zero_grad()
        (disc_loss + r1_penalty).backward()
        self._disc_opt.step()

        gen_loss = -self._disc(fake).mean()
        return gen_loss, disc_loss.detach()

class WMTrainer(SectionedWandbTrainer):
    wandb_section = "wm"

    def __init__(
        self,
        *args,
        sigreg_weight: float = 0.1,
        sigreg_n_samples: int = 1024,
        rollout_steps: int = 0,
        rollout_weight: float = 1.0,
        noise_std: float = 0.0,
        variance_penalty_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigreg_weight = sigreg_weight
        self.sigreg_n_samples = sigreg_n_samples
        self.rollout_steps = rollout_steps
        self.rollout_weight = rollout_weight
        self.noise_std = noise_std
        self.variance_penalty_weight = variance_penalty_weight
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=17), num_slices=1024,
        )
        self._sigreg_device = None

    def _current_rollout_steps(self) -> int:
        k = self.rollout_steps
        if k < 2 or self.state.max_steps <= 1:
            return k
        progress = self.state.global_step / self.state.max_steps
        ramp = min(1.0, progress / 0.5)
        return max(1, round(1 + (k - 1) * ramp))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def _rollout_loss(self, model, states: torch.Tensor, actions: torch.Tensor, k: int) -> torch.Tensor:
        T = states.size(1)
        if k < 2 or T <= k + 1:
            return torch.tensor(0.0, device=states.device)

        s = torch.randint(0, T - k, (1,)).item()
        history = states[:, :s + 1].detach()
        preds = []
        for _ in range(k):
            inp = history if not preds else torch.cat([history, torch.stack(preds, dim=1)], dim=1)
            pred = model.predict(inp, actions[:, :inp.size(1)])[:, -1]
            preds.append(pred)

        rollout_preds = torch.stack(preds, dim=1)
        rollout_targets = states[:, s + 1:s + 1 + k].detach()

        # Weight later steps more heavily
        weights = torch.arange(1, k + 1, dtype=torch.float32, device=states.device)
        weights = weights / weights.sum()
        per_step = F.mse_loss(rollout_preds.float(), rollout_targets.float(), reduction="none")
        return (per_step.mean(dim=-1).mean(dim=0) * weights).sum()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        dtype = next(model.parameters()).dtype
        frames = inputs["frames"].to(dtype=dtype) / 255.0
        actions = inputs["actions"].to(dtype=dtype)

        if self._sigreg_device != frames.device:
            self.sigreg = self.sigreg.to(frames.device)
            self._sigreg_device = frames.device

        T = frames.size(1)
        training = self.model.training

        states = model.encode(frames)

        pred_inputs = states[:, :-1]
        if training and self.noise_std > 0:
            pred_inputs = pred_inputs + torch.randn_like(pred_inputs) * self.noise_std

        pred_states = model.predict(pred_inputs, actions[:, :T - 1])
        target_states = states[:, 1:]

        pred_loss = F.mse_loss(pred_states.float(), target_states.float())

        pred_std = pred_states.detach().float().std(dim=-1).mean()
        target_std = target_states.detach().float().std(dim=-1).mean()
        variance_penalty = F.relu(target_std - pred_std)

        flat_states = states.flatten(0, 1).float()
        n = min(self.sigreg_n_samples, flat_states.size(0))
        perm = torch.randperm(flat_states.size(0), device=flat_states.device)[:n]
        sigreg_loss = self.sigreg(flat_states[perm])

        current_k = self._current_rollout_steps() if training else 0
        rollout_loss = self._rollout_loss(model, states, actions, current_k) if training else torch.tensor(0.0)

        loss = (
            pred_loss
            + self.sigreg_weight * sigreg_loss
            + self.variance_penalty_weight * variance_penalty
            + self.rollout_weight * rollout_loss
        )

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                flat_z = states.flatten(0, 1).float()
            log_dict = {
                "loss_total": scalar(loss),
                "pred_loss": scalar(pred_loss),
                "sigreg": scalar(sigreg_loss),
                "variance_penalty": scalar(variance_penalty),
                "z_std_mean": scalar(flat_z.std(dim=0).mean()),
                "z_std_min": scalar(flat_z.std(dim=0).min()),
                "z_norm_mean": scalar(flat_z.norm(dim=-1).mean()),
                "pred_std": scalar(pred_std),
                "target_std": scalar(target_std),
            }
            if self.rollout_steps >= 2:
                log_dict["rollout_loss"] = scalar(rollout_loss)
                log_dict["rollout_k"] = float(current_k)
            self.log(log_dict)

        if return_outputs:
            return loss, {"pred_states": pred_states.detach(), "target_states": target_states.detach()}
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

ACTION_NAMES = ["FWD", "BCK", "LEFT", "RIGHT", "TURN_L", "TURN_R", "ATTACK", "USE", "SPEED"]


def binary_to_grouped(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    move_target = torch.where(actions[..., 0] > 0.5, 0, torch.where(actions[..., 1] > 0.5, 1, 2))
    strafe_target = torch.where(actions[..., 2] > 0.5, 0, torch.where(actions[..., 3] > 0.5, 1, 2))
    turn_target = torch.where(actions[..., 4] > 0.5, 0, torch.where(actions[..., 5] > 0.5, 1, 2))
    binary_targets = actions[..., 6:9]
    return move_target, strafe_target, turn_target, binary_targets

def compute_grouped_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    move_t, strafe_t, turn_t, bin_t = binary_to_grouped(actions)
    loss_move = F.cross_entropy(logits[..., 0:3].flatten(0, -2), move_t.flatten())
    loss_strafe = F.cross_entropy(logits[..., 3:6].flatten(0, -2), strafe_t.flatten())
    loss_turn = F.cross_entropy(logits[..., 6:9].flatten(0, -2), turn_t.flatten())
    loss_bin = F.binary_cross_entropy_with_logits(logits[..., 9:12].flatten(0, -2), bin_t.flatten())
    return loss_move + loss_strafe + loss_turn + loss_bin

class ActionPolicyTrainer(SectionedWandbTrainer):
    wandb_section = "action_policy"

    def __init__(self, wm_model: WorldModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_model = wm_model.eval()
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
        actions = inputs["actions"].to(device=self.args.device, dtype=dtype)
        actions = actions[:, :frames.size(1)]

        with torch.no_grad():
            states = self.wm_model.encode(frames)

        if self._teacher_prob() < 1.0 and torch.rand(1).item() > self._teacher_prob():
            with torch.no_grad():
                logits_teacher = model(states, past_actions=actions)
                pred_binary = ActionPolicy.logits_to_binary(logits_teacher.detach()).float()
                past = pred_binary[:, :-1]
        else:
            past = actions

        if self._noise_prob() > 0:
            noise = torch.rand_like(past) < self._noise_prob()
            past = past.clone()
            past[noise] = 1 - past[noise]

        logits = model(states, past_actions=past)
        logits = logits[:, :-1]
        loss = compute_grouped_loss(logits, actions)

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                preds = ActionPolicy.logits_to_binary(logits.detach()).float()
                acc = (preds == actions).float().mean()
                f1_per_action = []
                for j in range(9):
                    tp = (preds[..., j] * actions[..., j]).sum().clamp_min(1)
                    fp = (preds[..., j] * (1 - actions[..., j])).sum()
                    fn = ((1 - preds[..., j]) * actions[..., j]).sum()
                    f1_per_action.append((2 * tp / (2 * tp + fp + fn)).item())
                f1 = sum(f1_per_action) / 9
            self.log({"loss": scalar(loss), "acc": scalar(acc), "f1": f1,
                      "teacher_prob": self._teacher_prob(), "noise_prob": self._noise_prob()})

        if return_outputs:
            return loss, {"logits": logits.detach()}
        return loss

    def _teacher_prob(self) -> float:
        if self.state.max_steps <= 1:
            return 1.0
        p = self.state.global_step / (self.state.max_steps - 1)
        return max(0.2, 1.0 - p * 0.8)

    def _noise_prob(self) -> float:
        if self.state.max_steps <= 1:
            return 0.0
        p = self.state.global_step / (self.state.max_steps - 1)
        return 0.05 * (1.0 - p)

def train(
    config: WorldModelConfig,
    mode: str = "all",
    output_root: str = "./checkpoints",
    resume_from: str | None = "auto",
    wm_checkpoint: str | None = None,
    context_len: int = 40,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
    dataloader_num_workers: int = 4,
    dataloader_prefetch_factor: int | None = 2,
    wm_epochs: int = 3,
    decoder_epochs: int = 5,
    action_policy_epochs: int = 5,
    decoder_noise_std: float = 0.15,
    rollout_steps: int = 3,
    rollout_weight: float = 1.0,
    predictor_noise_std: float = 0.05,
    variance_penalty_weight: float = 0.1,
):
    if mode not in {"all", "wm", "decoder", "action_policy"}:
        raise ValueError("mode must be 'all', 'wm', 'decoder' or 'action_policy'")
    device = device_name()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"Device: {device} | Dataset: {config.to_dict()['height']}x{config.to_dict()['width']} | dim={config.dim} | blocks={config.n_blocks} | heads={config.n_heads}", flush=True)
    ds = load_dataset("lucrbrtv/doom-e1-internet-gameplay", split="train")
    split = ds.train_test_split(test_size=0.1, shuffle=False)
    train_dataset = WMDataset(split["train"], context_len=context_len, sequence_stride=sequence_stride, name="train")
    eval_dataset = WMDataset(split["test"], context_len=context_len, sequence_stride=sequence_stride, name="eval")
    eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(max_eval_sequences, len(eval_dataset))))
    print(f"Data: {len(ds):,} frames | {len(train_dataset):,} train seqs | {len(eval_dataset):,} eval seqs | context={context_len} | stride={sequence_stride}", flush=True)
    setup_wandb()
    os.makedirs(output_root, exist_ok=True)
    model = WorldModel(config).to(torch.bfloat16)
    enc = sum(p.numel() for p in model.encoder.parameters())
    pred = sum(p.numel() for p in model.predictor.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Model: {enc+pred+dec:,} total | enc {enc:,} | pred {pred:,} | dec {dec:,}", flush=True)

    wm_output_dir = os.path.join(output_root, "world-model")
    decoder_output_dir = os.path.join(output_root, "decoder")
    action_policy_output_dir = os.path.join(output_root, "action-policy")

    args = TrainingArguments(
        output_dir=wm_output_dir,
        num_train_epochs=wm_epochs,
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
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=[],
        run_name="world-model",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        torch_compile=True,
    )

    steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / args.per_device_train_batch_size) / args.gradient_accumulation_steps
    )
    total_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    print(f"Plan: ~{total_steps:,} steps | {args.eval_steps} eval | {args.save_steps} save | bs={args.per_device_train_batch_size} | lr={args.learning_rate}", flush=True)

    last_checkpoint = resolve_checkpoint(resume_from, args.output_dir)
    if last_checkpoint is not None:
        ckpt_config = os.path.join(last_checkpoint, "config.json")
        if os.path.isfile(ckpt_config):
            with open(ckpt_config) as f:
                old_config = json.load(f)
            for key in ("height", "width", "patch_size", "dim", "n_heads", "n_blocks", "ffn_mult"):
                if old_config.get(key) != getattr(config, key):
                    print(f"Skip incompatible checkpoint: {last_checkpoint}", flush=True)
                    last_checkpoint = None
                    break

    if mode in {"all", "wm"}:
        trainer = WMTrainer(
            model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            sigreg_weight=0.01, rollout_steps=rollout_steps, rollout_weight=rollout_weight,
            noise_std=predictor_noise_std, variance_penalty_weight=variance_penalty_weight,
        )
        if last_checkpoint is not None:
            print(f"Resume: {last_checkpoint}", flush=True)
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
        print(f"Load: {pretrained_checkpoint}", flush=True)
        model = WorldModel.from_pretrained(pretrained_checkpoint, ignore_mismatched_sizes=True).to(
            device=device, dtype=torch.bfloat16
        )

    if mode == "action_policy":
        ap_args = TrainingArguments(
            output_dir=action_policy_output_dir,
            num_train_epochs=action_policy_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=1e-4,
            weight_decay=1e-4,
            max_grad_norm=1.0,
            bf16=device == "cuda",
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            remove_unused_columns=False,
            report_to=[],
            run_name="action-policy",
            optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
            torch_compile=True,
        )
        policy = ActionPolicy(
            dim=model.config.dim,
            n_heads=model.config.n_heads,
            ffn_mult=model.config.ffn_mult,
            max_seq_len=max(context_len, model.config.max_seq_len),
        )
        policy_params = sum(p.numel() for p in policy.parameters())
        print(f"ActionPolicy: {policy_params:,} params", flush=True)
        ap_trainer = ActionPolicyTrainer(
            wm_model=model, model=policy, args=ap_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
        )
        start_wandb_run("action-policy", config=config.to_dict())
        ap_trainer.train()
        if hasattr(ap_trainer.model, "_orig_mod"):
            ap_trainer.model = ap_trainer.model._orig_mod
        ap_trainer.save_model(ap_args.output_dir)
        finish_wandb()
        return

    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.eval()
    model.decoder.train()
    if device == "cuda":
        model.decoder = torch.compile(model.decoder)
    dt_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Phase: training decoder ({dt_params:,} params, torch.compile={device=='cuda'})", flush=True)

    decoder_args = TrainingArguments(
        output_dir=decoder_output_dir,
        num_train_epochs=decoder_epochs,
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
        print(f"Resume decoder: {decoder_checkpoint}", flush=True)
    trainer = DecoderTrainer(model=model, args=decoder_args, train_dataset=train_dataset, eval_dataset=eval_dataset, noise_std=decoder_noise_std)
    start_wandb_run("decoder-probe", config=config.to_dict())
    trainer.train(resume_from_checkpoint=decoder_checkpoint)
    if hasattr(trainer.model.decoder, "_orig_mod"):
        trainer.model.decoder = trainer.model.decoder._orig_mod
    trainer.model.to(torch.bfloat16)
    trainer.save_model(decoder_args.output_dir)
    finish_wandb()
    if mode == "decoder":
        return
    ap_args = TrainingArguments(
        output_dir=action_policy_output_dir,
        num_train_epochs=action_policy_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        bf16=device == "cuda",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=[],
        run_name="action-policy",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        torch_compile=True,
    )
    ap_trainer = ActionPolicyTrainer(
        wm_model=model, model=ActionPolicy(model.config.dim), args=ap_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
    )
    start_wandb_run("action-policy", config=config.to_dict())
    ap_trainer.train()
    ap_trainer.save_model(ap_args.output_dir)
    finish_wandb()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train miniworld world model, decoder probe, or action policy.")
    parser.add_argument("--mode", choices=["all", "wm", "decoder", "action_policy"], default="all")
    parser.add_argument("--output-root", default=os.environ.get("MINIWORLD_CHECKPOINT_DIR", "./checkpoints"))
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a checkpoint directory")
    parser.add_argument("--wm-checkpoint", default=None, help="Pretrained WM checkpoint for --mode decoder or action_policy")
    parser.add_argument("--context-len", type=int, default=40)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--max-eval-sequences", type=int, default=512)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--wm-epochs", type=int, default=3)
    parser.add_argument("--decoder-epochs", type=int, default=5)
    parser.add_argument("--action-policy-epochs", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=3, help="Max rollout depth for WM training (curriculum grows 1->k over first 50%% of training)")
    parser.add_argument("--rollout-weight", type=float, default=1.0, help="Weight of the rollout loss")
    parser.add_argument("--predictor-noise-std", type=float, default=0.05, help="Gaussian noise std on predictor input states during WM training")
    parser.add_argument("--variance-penalty-weight", type=float, default=0.1, help="Weight of the pred_std < target_std penalty")
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
        dataloader_num_workers=cli.dataloader_num_workers,
        dataloader_prefetch_factor=cli.dataloader_prefetch_factor,
        config=WorldModelConfig(
            height=240, width=320, patch_size=20, dim=408, n_heads=6,
        ),
        wm_epochs=cli.wm_epochs,
        decoder_epochs=cli.decoder_epochs,
        action_policy_epochs=cli.action_policy_epochs,
        rollout_steps=cli.rollout_steps,
        rollout_weight=cli.rollout_weight,
        predictor_noise_std=cli.predictor_noise_std,
        variance_penalty_weight=cli.variance_penalty_weight,
    )
