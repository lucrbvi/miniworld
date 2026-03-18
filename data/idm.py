"""
Inverse Dynamics Model (IDM) inspired by the VPT paper from OpenAI
Paper: Video PreTraining (VPT) [Baker et al., 2022] https://arxiv.org/abs/2206.11795
Dataset: https://huggingface.co/datasets/lucrbrtv/doom-e1-gameplay
"""

import os
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
import random

from torch import Tensor
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MHAttention

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

class IDMConfig(PretrainedConfig):
    model_type = "idm"

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
        context_len: int = 16,
        n_buttons: int = 9,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.ffn_mult = ffn_mult
        self.dropout_proba = dropout_proba
        self.context_len = context_len
        self.n_buttons = n_buttons
        super().__init__(**kwargs)

class IDM(PreTrainedModel):
    """Inverse Dynamics Model inspired by OpenAI VPT."""

    config_class = IDMConfig

    def __init__(self, config: IDMConfig):
        super().__init__(config)
        dim = config.dim

        self.initial_conv = nn.Conv3d(
            3, 128, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0)
        )
        self.patch_embed = nn.Conv2d(
            128, dim, kernel_size=config.patch_size, stride=config.patch_size
        )

        mh_config = {
            "dim": dim,
            "n_heads": config.n_heads,
            "n_blocks": config.n_blocks,
            "ffn_mult": config.ffn_mult,
            "dropout_proba": config.dropout_proba,
        }
        self.image_transformer = MHAttention(mh_config)
        self.frame_dense = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.GELU(),
        )
        self.temporal = MHAttention(mh_config)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(config.dropout_proba),
            nn.Linear(dim, config.n_buttons),
        )

    def forward(self, frames: Tensor, labels: Tensor | None = None) -> Tensor:
        B, T, C, H, W = frames.shape
        x = frames.to(torch.float32) / 255.0
        x = self.initial_conv(x.permute(0, 2, 1, 3, 4))
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, 128, H, W)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        patches = self.image_transformer(patches)
        frame_emb = self.frame_dense(patches.mean(dim=1)).view(B, T, -1)
        return self.head(self.temporal(frame_emb))

def preprocess_frame(frame) -> np.ndarray | None:
    if isinstance(frame, dict):
        frame = frame.get("array", frame.get("image"))
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    if not isinstance(frame, np.ndarray):
        return None
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = np.transpose(frame, (2, 0, 1)) # HWC → CHW
    return frame

def preprocess_dataset(
    hf_dataset: HFDataset,
    context_len: int = 16,
    max_samples: int | None = None,
    cache_dir: str = "/tmp/idm_cache",
) -> tuple["IDMDataset", np.ndarray]:
    os.makedirs(cache_dir, exist_ok=True)

    frames_path = os.path.join(cache_dir, "frames.npy")
    actions_path = os.path.join(cache_dir, "actions.npy")
    max_frames = min(max_samples, len(hf_dataset)) if max_samples else len(hf_dataset)

    frames_mm: np.memmap | None = None
    actions_arr: np.ndarray | None = None
    episodes: list[str] = []
    count = 0

    for sample in tqdm(hf_dataset, desc="Preprocessing", total=max_frames):
        if max_samples and count >= max_samples:
            break

        frame = preprocess_frame(sample["frame"])
        if frame is None:
            continue

        # Lazy memmap init: we need the first frame to know C, H, W
        if frames_mm is None:
            C, H, W = frame.shape
            n_buttons = len(sample["action"])
            frames_mm = np.memmap(
                frames_path,
                dtype=np.uint8,
                mode="w+",
                shape=(max_frames, C, H, W),
            )
            actions_arr = np.empty((max_frames, n_buttons), dtype=np.float32)

        frames_mm[count] = frame
        actions_arr[count] = sample["action"]
        episodes.append(sample["episode"])
        count += 1

    if frames_mm is None:
        raise RuntimeError("No valid frames found in dataset.")

    frames_mm.flush()
    del frames_mm

    with open(frames_path, "r+b") as f:
        f.truncate(count * C * H * W)

    actions_arr = actions_arr[:count]
    np.save(actions_path, actions_arr)

    valid_indices = [
        i
        for i in range(count - context_len)
        if episodes[i] == episodes[i + context_len - 1]
    ]
    print(f"{len(valid_indices)} valid sequences (context_len={context_len})")

    dataset = IDMDataset(frames_path, actions_arr, valid_indices, context_len)
    return dataset, actions_arr[np.array(valid_indices)]

def compute_pos_weights(actions: np.ndarray, n_buttons: int = 9) -> Tensor:
    flat = actions.reshape(-1, n_buttons)
    n_pos = flat.sum(axis=0).clip(min=1)
    n_neg = (len(flat) - flat.sum(axis=0)).clip(min=1)
    weights = n_neg / n_pos
    print(f"pos_weight: {np.round(weights, 1)}")
    return torch.from_numpy(weights.astype(np.float32))

class IDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames_path: str,
        actions_arr: np.ndarray,
        valid_indices: list[int],
        context_len: int,
    ):
        n_frames, *chw = actions_arr.shape[0], *self._infer_chw(frames_path, actions_arr.shape[0])
        self.frames_mm = np.memmap(
            frames_path,
            dtype=np.uint8,
            mode="r",
            shape=(n_frames, *chw),
        )
        self.actions_arr = actions_arr
        self.valid_indices = valid_indices
        self.context_len = context_len

    @staticmethod
    def _infer_chw(frames_path: str, n_frames: int) -> tuple[int, int, int]:
        C, H, W = 3, 240, 320
        expected = n_frames * C * H * W
        actual = os.path.getsize(frames_path)
        assert actual == expected, f"File size mismatch: {actual} != {expected}"
        return C, H, W

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        frames = torch.from_numpy(self.frames_mm[start:start + self.context_len].copy()).contiguous()
        labels = torch.from_numpy(self.actions_arr[start:start + self.context_len].copy()).contiguous()

        # Simulate video compression artifacts (JPEG/H.264)
        if random.random() > 0.5:
            noise = torch.randn_like(frames.float()) * 2.5
            frames = (frames.float() + noise).clamp(0, 255).byte()

        # Simulate 30fps by skipping frames (doubling motion magnitude)
        if random.random() > 0.3:
            T = frames.size(0)
            idx_30 = torch.arange(0, T, 2)[:T//2]
            frames = frames[idx_30].repeat_interleave(2, dim=0)[:T]
            labels = labels[idx_30].repeat_interleave(2, dim=0)[:T]

        return {"frames": frames, "labels": labels}

class IDMTrainer(Trainer):
    def __init__(self, *args, pos_weight: Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = model(inputs["frames"])
        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None)(
            logits[:, :-1], inputs["labels"][:, :-1]
        )
        return (loss, {"logits": logits}) if return_outputs else loss

def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = map(torch.from_numpy, eval_pred)
    # :-1 matches forward() — last frame has no action target
    logits, labels = logits[:, :-1], labels[:, :-1]
    preds = (torch.sigmoid(logits) > 0.5).float()

    BUTTONS = ["FWD","BCK","LEFT","RIGHT","TURN_L","TURN_R","ATTACK","USE","SPEED"]
    metrics = {
        "accuracy": (preds == labels).float().mean().item(),
        "loss": nn.BCEWithLogitsLoss()(logits, labels).item(),
        "whole_frame_accuracy": (preds == labels).all(dim=-1).float().mean().item(),
    }

    for i, name in enumerate(BUTTONS):
        tp = (preds[..., i] * labels[..., i]).sum()
        fp = (preds[..., i] * (1 - labels[..., i])).sum()
        fn = ((1 - preds[..., i]) * labels[..., i]).sum()
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
        metrics[f"f1_{name}"] = f1

    return metrics

def train(
    output_dir: str = "./checkpoints",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 64,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 1e-4,
    warmup_steps: float = 200,
    weight_decay: float = 0.05,
    logging_steps: int = 10,
    eval_steps: int = 250,
    save_steps: int = 500,
    max_grad_norm: float = 1.0,
    config: IDMConfig | None = None,
    device: str = _DEVICE,
    max_eval_samples: int | None = None,
):
    config = config or IDMConfig()
    print(f"Device: {device} | Config: {config.to_dict()}")

    ds = load_dataset("lucrbrtv/doom-e1-gameplay", split="train").shuffle(seed=42)
    split = ds.train_test_split(test_size=0.1, shuffle=True)

    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

    train_dataset, train_actions = preprocess_dataset(
        split["train"], config.context_len, cache_dir="./data/cache/idm/train"
    )
    eval_dataset, _ = preprocess_dataset(
        split["test"],
        config.context_len,
        max_samples=max_eval_samples,
        cache_dir="./data/cache/idm/eval",
    )

    print(f"Train sequences: {len(train_dataset)} | Eval sequences: {len(eval_dataset)}")

    model = IDM(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    wandb.init(project="miniworld-idm")

    trainer = IDMTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            logging_dir=f"{output_dir}/logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=max_grad_norm,
            bf16=device == "cuda",
            gradient_accumulation_steps=2,
            dataloader_num_workers=8,
            dataloader_prefetch_factor=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            remove_unused_columns=False,
            report_to=["wandb"],
            push_to_hub=True,
            hub_model_id="idm"
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        pos_weight=compute_pos_weights(train_actions, config.n_buttons),
    )

    trainer.train()

    return model

def load_idm(
    model_path: str,
    config: IDMConfig | None = None,
    device: str = _DEVICE,
) -> IDM:
    config = config or IDMConfig()

    if os.path.isdir(model_path):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path)
    elif model_path.endswith(".safetensors"):
        model = IDM(config)
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported format: {model_path}")

    return model.to("cpu").eval()

# need to be called ONE TIME after training and before labelisation
def calibrate_thresholds(model: IDM, dataset, device: str = _DEVICE, n_buttons: int = 9) -> list[float]:
    """Calibrate per-button thresholds on validation set for optimal F1."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            labels = batch["labels"]
            logits = model(frames)[:, :-1]
            all_logits.append(logits.cpu())
            all_labels.append(labels[:, :-1])

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    thresholds = []
    BUTTONS = ["FWD","BCK","LEFT","RIGHT","TURN_L","TURN_R","ATTACK","USE","SPEED"]

    for i in range(n_buttons):
        best_t, best_f1 = 0.5, 0.0
        for t in torch.linspace(0.1, 0.9, 80):
            preds = (torch.sigmoid(logits[..., i]) > t).float()
            tp = (preds * labels[..., i]).sum()
            fp = (preds * (1 - labels[..., i])).sum()
            fn = ((1 - preds) * labels[..., i]).sum()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            if f1 > best_f1:
                best_f1, best_t = f1, t.item()
        thresholds.append(best_t)
        print(f"{BUTTONS[i]}: threshold={best_t:.3f}, F1={best_f1:.3f}")

    return thresholds

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDM Training / Inference")
    parser.add_argument("--run", action="store_true", help="Load weights and run inference instead of training")
    parser.add_argument("--weights", type=str, default="checkpoints/model.safetensors", help="Path to weights file for inference")
    args = parser.parse_args()

    config = IDMConfig(
        height=240,
        width=320,
        patch_size=16,
        dim=384, # head_size must be divisible by 8
        n_heads=16,
        n_blocks=4,
        ffn_mult=3,
        dropout_proba=0.15,
        context_len=16,
        n_buttons=9,
    )

    if args.run:
        m = load_idm(args.weights, config=config)
        ds = load_dataset("lucrbrtv/doom-e1-gameplay", split="train")
        train_dataset, _ = preprocess_dataset(ds, config.context_len, cache_dir="./data/cache/idm")
        frames = train_dataset[210].get("frames").unsqueeze(0)
        labels = train_dataset[210].get("labels").unsqueeze(0)
        logits = m(frames)
        loss = nn.BCEWithLogitsLoss()(logits[:, :-1], labels[:, :-1])
        print(f"Loss: {loss.item():.4f}")
    else:
        train(
            num_train_epochs=30,
            per_device_train_batch_size=40,
            per_device_eval_batch_size=40,
            learning_rate=5e-5,
            config=config,
            output_dir="./checkpoints"
        )
