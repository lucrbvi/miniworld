#!/usr/bin/env python3
"""Download pretrained weights from Hugging Face Hub.

Downloads the world model to ./checkpoints/world-model/ so that
dream.py finds it out of the box with default arguments.

Usage:
    uv run download_weights.py              # download everything available
    uv run download_weights.py --help       # see all options
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "lucrbrtv/doom-world-model"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

WM_FILES = [
    "model.safetensors",
    "config.json",
]

AP_FILES = [
    "action-policy.safetensors",
]

def _download_file(repo_id: str, filename: str, dest: Path) -> bool:
    """Download a single file from the Hub. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install it with: uv add huggingface_hub")
        return False

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=None)
    except Exception as exc:
        print(f"  ✗ {filename} — {exc}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✓ {filename}  ({size_mb:.1f} MB)")
    return True

def main() -> None:
    parser = argparse.ArgumentParser(description="Download miniworld weights from Hugging Face")
    parser.add_argument(
        "--repo", default=REPO_ID,
        help=f"Hugging Face repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--dir", default=None,
        help=f"Target directory (default: {CHECKPOINTS_DIR})",
    )
    parser.add_argument(
        "--no-world-model", action="store_false", dest="world_model",
        help="Skip world model download",
    )
    parser.add_argument(
        "--no-action-policy", action="store_false", dest="action_policy",
        help="Skip action policy download",
    )
    parser.set_defaults(world_model=True, action_policy=True)
    args = parser.parse_args()

    base = Path(args.dir or CHECKPOINTS_DIR)

    print(f"Downloading from {args.repo} → {base}/\n")

    downloaded = 0
    missing = 0

    if args.world_model:
        dest_dir = base / "world-model"
        print(f"[world-model] -> {dest_dir}/")
        for fname in WM_FILES:
            if _download_file(args.repo, fname, dest_dir / fname):
                downloaded += 1
            else:
                missing += 1
        print()

    if args.action_policy:
        dest_dir = base / "action-policy"
        print(f"[action-policy] -> {dest_dir}/")
        for fname in AP_FILES:
            if _download_file(args.repo, fname, dest_dir / fname):
                downloaded += 1
            else:
                missing += 1
        print()

    if missing > 0 and downloaded == 0:
        print("Nothing downloaded. Check the repo name and your internet connection.", file=sys.stderr)
        sys.exit(1)
    elif missing > 0:
        print(f"Done: {downloaded} files downloaded, {missing} not available (optional components).")
    else:
        print(f"Done: {downloaded} files downloaded. Ready to go!")

if __name__ == "__main__":
    main()
