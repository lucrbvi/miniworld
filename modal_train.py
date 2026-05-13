import os
import runpy
import sys

import modal

APP_DIR = "/root/miniworld"
CHECKPOINT_DIR = f"{APP_DIR}/checkpoints"
HF_CACHE_DIR = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(".")
    .add_local_file("train.py", f"{APP_DIR}/train.py")
    .add_local_file("model.py", f"{APP_DIR}/model.py")
)

checkpoint_volume = modal.Volume.from_name("miniworld-training-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("miniworld-huggingface-cache", create_if_missing=True)

app = modal.App("miniworld-training", image=image)


@app.function(
    gpu="A10",
    cpu=8,
    timeout=24 * 60 * 60,
    volumes={
        CHECKPOINT_DIR: checkpoint_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_training(
    mode: str = "wm",
    resume_from: str = "auto",
    wm_checkpoint: str | None = None,
    context_len: int = 10,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
):
    os.chdir(APP_DIR)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE_DIR}/datasets"
    os.environ["MINIWORLD_CHECKPOINT_DIR"] = CHECKPOINT_DIR
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, APP_DIR)

    argv = [
        "train.py",
        "--mode", mode,
        "--output-root", CHECKPOINT_DIR,
        "--resume-from", resume_from,
        "--context-len", str(context_len),
        "--sequence-stride", str(sequence_stride),
        "--max-eval-sequences", str(max_eval_sequences),
    ]
    if wm_checkpoint:
        argv += ["--wm-checkpoint", wm_checkpoint]

    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path("train.py", run_name="__main__")
    finally:
        sys.argv = old_argv
        checkpoint_volume.commit()
        hf_cache_volume.commit()


@app.local_entrypoint()
def main(
    mode: str = "wm",
    resume_from: str = "auto",
    wm_checkpoint: str | None = None,
    context_len: int = 10,
    sequence_stride: int = 1,
    max_eval_sequences: int = 2048,
):
    """
    Examples:
      modal run modal_train.py --mode wm
      modal run modal_train.py --mode decoder
      modal run modal_train.py --mode decoder --wm-checkpoint /root/miniworld/checkpoints/world-model/checkpoint-4000
      modal run modal_train.py --mode wm --resume-from none
    """
    run_training.remote(
        mode=mode,
        resume_from=resume_from,
        wm_checkpoint=wm_checkpoint,
        context_len=context_len,
        sequence_stride=sequence_stride,
        max_eval_sequences=max_eval_sequences,
    )
