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
        self.window_len = context_len + 1
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

    def __init__(self, *args, lpips_weight: float = 0.1, rollout_decode_steps: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips_weight = lpips_weight
        self.rollout_decode_steps = rollout_decode_steps
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
            b, t, n_p1, d = tokens.shape

            if self.state.max_steps <= 0:
                progress = 0.0
            else:
                end_step = max(1.0, self.state.max_steps * model.config.decoder_curriculum_end)
                progress = min(1.0, self.state.global_step / end_step)
            pred_ratio = model.config.decoder_pred_token_ratio * progress

            steps = min(t - 1, model.predictor.time_pos.size(1) - 1)
            latents = torch.empty(b, steps, n_p1, d, dtype=tokens.dtype, device=tokens.device)
            context = tokens[:, :1].contiguous()
            for s in range(steps):
                pred = model.predict(context, actions[:, :s + 1])
                next_pred = pred[:, -1]
                latents[:, s] = next_pred
                use_teacher = torch.rand(b, 1, 1, 1, device=next_pred.device) >= pred_ratio
                next_token = torch.where(use_teacher, tokens[:, s + 1:s + 2], next_pred.unsqueeze(1))
                assert next_token.shape == (b, 1, n_p1, d), f"next_token shape {next_token.shape} != {(b, 1, n_p1, d)} (silent broadcast?)"
                context = torch.cat([context, next_token], dim=1)
                assert context.size(1) == s + 2, f"context grew to {context.size(1)} at step {s} (expected {s + 2})"

            if self.rollout_decode_steps > 0 and steps > self.rollout_decode_steps:
                step_idx = torch.randperm(steps, device=latents.device)[:self.rollout_decode_steps].sort().values
                latents = latents[:, step_idx]
                targets = observations[:, 1:][:, step_idx]
            else:
                targets = observations[:, 1:]

            latents = model.transition(latents.reshape(-1, n_p1, d)).reshape(b, -1, n_p1, d)

            if model.config.decoder_noise_std > 0:
                latents = latents + torch.randn_like(latents) * model.config.decoder_noise_std

        recon = model.decoder(latents.reshape(-1, n_p1, d))
        targets = targets.reshape(-1, *observations.shape[2:])
        assert recon.shape == targets.shape, f"recon {recon.shape} vs targets {targets.shape}"
        l1_loss = F.l1_loss(recon.float(), targets.float())
        lpips_loss = self.lpips_loss(recon.float() * 2 - 1, targets.float() * 2 - 1).mean()
        loss = l1_loss + self.lpips_weight * lpips_loss

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss": scalar(loss),
                "l1": scalar(l1_loss),
                "lpips": scalar(lpips_loss),
                "pred_token_ratio": pred_ratio,
                "decode_steps": latents.size(1),
                "latent_noise_std": model.config.decoder_noise_std,
            })

        if return_outputs:
            return loss, {"recon": recon.detach()}
        return loss

class WMTrainer(SectionedWandbTrainer):
    wandb_section = "wm"

    def __init__(self, *args, sigreg_weight: float = 0.1, rollout_steps: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigreg_weight = sigreg_weight
        self.rollout_steps = rollout_steps
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=17), num_slices=1024,
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

        max_pred_len = min(tokens.size(1) - 1, model.predictor.time_pos.size(1))
        pred_tokens = model.predict(tokens[:, :max_pred_len], actions[:, :max_pred_len])
        target_tokens = tokens[:, 1:1 + max_pred_len]

        rollout_len = min(self.rollout_steps, actions.size(1), model.predictor.time_pos.size(1) - 1)
        rollout_preds = []
        rollout_tokens = tokens[:, :1]
        for t in range(rollout_len):
            next_preds = model.predict(rollout_tokens, actions[:, : t + 1])[:, -1]
            rollout_preds.append(next_preds)
            rollout_tokens = torch.cat([rollout_tokens, next_preds.unsqueeze(1)], dim=1)
        rollout_preds = torch.stack(rollout_preds, dim=1)

        sigreg_loss = torch.stack([
            self.sigreg(tokens[:, t].reshape(-1, tokens.size(-1)).float()) for t in range(tokens.size(1))
        ]).mean()
        pred_loss = F.mse_loss(pred_tokens.float(), target_tokens.float())
        rollout_loss = F.mse_loss(rollout_preds.float(), tokens[:, 1 : 1 + rollout_len].float())
        loss = pred_loss + rollout_loss + self.sigreg_weight * sigreg_loss

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            flat_z = tokens.flatten(0, 2)
            self.log({
                "loss_total": scalar(loss),
                "pred_loss": scalar(pred_loss),
                "rollout_loss": scalar(rollout_loss),
                "sigreg": scalar(sigreg_loss),
                "z_std_mean": scalar(flat_z.std(dim=0).mean()),
                "z_std_min": scalar(flat_z.std(dim=0).min()),
                "z_norm_mean": scalar(flat_z.norm(dim=-1).mean()),
                "patch_spatial_std": scalar(tokens[:, :, 1:].std(dim=2).mean()),
                "pred_std": scalar(pred_tokens.flatten(0, 2).std(dim=0).mean()),
                "target_std": scalar(target_tokens.flatten(0, 2).std(dim=0).mean()),
                "pred_target_std_ratio": scalar(pred_tokens.flatten(0, 2).std(dim=0).mean() / target_tokens.flatten(0, 2).std(dim=0).mean().clamp_min(1e-6)),
            })

        if return_outputs:
            return loss, {"pred_tokens": pred_tokens.detach(), "target_embeddings": embeddings[:, 1:].detach()}
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

        with torch.no_grad():
            states = self.wm_model.encode(frames)

        logits = model(states)
        loss = F.binary_cross_entropy_with_logits(logits, actions)

        if self.state.global_step == 0 or self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                acc = ((logits > 0).float() == actions).float().mean()
            self.log({"loss": scalar(loss), "acc": scalar(acc)})

        if return_outputs:
            return loss, {"logits": logits.detach()}
        return loss

def train(
    config: WorldModelConfig,
    mode: str = "all",
    output_root: str = "./checkpoints",
    resume_from: str | None = "auto",
    wm_checkpoint: str | None = None,
    context_len: int = 16,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
    dataloader_num_workers: int = 4,
    dataloader_prefetch_factor: int | None = 2,
):
    if mode not in {"all", "wm", "decoder", "action_policy"}:
        raise ValueError("mode must be 'all', 'wm', 'decoder' or 'action_policy'")
    device = device_name()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    trans = sum(p.numel() for p in model.transition.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Model: {enc+pred+trans+dec:,} total | enc {enc:,} | pred {pred:,} | dec+trans {dec+trans:,}", flush=True)

    wm_output_dir = os.path.join(output_root, "world-model")
    decoder_output_dir = os.path.join(output_root, "decoder")
    action_policy_output_dir = os.path.join(output_root, "action-policy")

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
    eval_batches = math.ceil(len(eval_dataset) / args.per_device_eval_batch_size)
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
            model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, sigreg_weight=0.1,
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
            num_train_epochs=1,
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
        return

    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.transition.requires_grad_(True)
    model.eval()
    model.decoder.train()
    model.transition.train()
    if device == "cuda":
        model.decoder = torch.compile(model.decoder)
        model.transition = torch.compile(model.transition)
    dt_params = sum(p.numel() for p in model.decoder.parameters()) + sum(p.numel() for p in model.transition.parameters())
    print(f"Phase: training decoder + transition ({dt_params:,} params, torch.compile={device=='cuda'})", flush=True)

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
        print(f"Resume decoder: {decoder_checkpoint}", flush=True)
    trainer = DecoderTrainer(model=model, args=decoder_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    start_wandb_run("decoder-probe", config=config.to_dict())
    trainer.train(resume_from_checkpoint=decoder_checkpoint)
    trainer.model.to(torch.bfloat16)
    trainer.save_model(decoder_args.output_dir)
    finish_wandb()
    if mode == "decoder":
        return
    ap_args = TrainingArguments(
        output_dir=action_policy_output_dir,
        num_train_epochs=1,
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
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--max-eval-sequences", type=int, default=512)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
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
            height=240, width=320, patch_size=20, dim=380, n_heads=4, n_blocks=3,
            ffn_mult=3, dropout_proba=0.1, causal=True,
        ),
    )
