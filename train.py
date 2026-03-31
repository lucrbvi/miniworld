import torch
import torch.nn as nn
import wandb
import lejepa
import numpy as np

from torch import Tensor
from datasets import (
    load_dataset,
    Dataset as HFDataset,
)
from model import WorldModel, WorldModelConfig
from transformers import Trainer, TrainingArguments

class WMDataset(torch.utils.data.Dataset):
    """Dataset for World Model that stacks M frames and actions as context."""

    def __init__(self, hf_dataset: HFDataset, context_len: int = 16):
        self.context_len = context_len
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
            for i in range(len(self.frames) - context_len)
            if self.episodes[i] == self.episodes[i + context_len - 1]
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        frames = np.stack(self.frames[start : start + self.context_len])
        actions = np.stack(self.actions[start : start + self.context_len])
        next_frame = self.frames[start + self.context_len]
        return {
            "frames": torch.from_numpy(frames).float() / 255.0,
            "next_frame": torch.from_numpy(next_frame).float() / 255.0,
            "actions": torch.from_numpy(actions).float(),
        }

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

LAMBDA = 0.1 # Sigreg
LAMBDA2 = 0.3 # latent / pixels

# Create and apply a binary mask to let n_visible masks visible on multiple frames
# (it generate a new random mask for each frames)
def make_mask(B, M, H, W, n_visible, device, patch_size=16):
    ph, pw = H // patch_size, W // patch_size
    idx = torch.stack([torch.randperm(ph * pw, device=device)[:n_visible] for _ in range(B * M)])
    mask = torch.zeros(B * M, ph * pw, device=device).scatter_(1, idx, 1.0)
    mask = mask.view(B, M, ph, pw).unsqueeze(2)
    return mask.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1) # (B, M, 1, H, W)

class WMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=self.univariate_test, num_slices=1024
        )

    # We are training the world model and the decoder at the same time
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        frames = inputs["frames"]
        next = inputs["next_frame"]
        actions = inputs["actions"]
        B, M, C, H, W = frames.shape

        global_views = [
            frames * make_mask(B, M, H, W, n_visible=2, device=frames.device)
        ]
        all_views = [frames * make_mask(B, M, H, W, n_visible=6, device=frames.device)]

        # Don't know if it breaks LeJEPA to add actions w/ masked images
        g_emb, _ = model(torch.cat(global_views), actions)
        a_emb, _ = model(torch.cat(all_views), actions)
        _, pixel_pred = model(frames, actions)

        K = g_emb.shape[-1]

        centers = g_emb.view(-1, B, K).mean(0)
        a_emb = a_emb.view(-1, B, K)
        sim = (centers - a_emb).square().mean()

        sigreg = self.sigreg_loss_fn(a_emb.view(-1, K))
        pixels = nn.functional.mse_loss(pixel_pred, next)
        loss = (1 - LAMBDA2) * ((1 - LAMBDA) * sim + LAMBDA * sigreg) + pixels * LAMBDA2

        return (loss, {"g_embeddings": g_emb, "a_embeddings": a_emb, "pixel_pred": pixel_pred}) if return_outputs else loss

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
            logging_dir=f"./checkpoints/logs",
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
        # TODO: add compute_metrics for accuracy
    )

    trainer.train()

if __name__ == "__main__":
    train(WorldModelConfig(
        height=240,
        width=320,
        patch_size=16,
        dim=384,
        n_heads=4,
        n_blocks=8,
        ffn_mult=3,
        dropout_proba=0.1,
    ))
