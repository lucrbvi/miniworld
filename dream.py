import argparse
import io
import os
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyray as rl
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers.trainer import load_sharded_checkpoint

from model import WorldModel, WorldModelConfig

KEY = rl.KeyboardKey
MOUSE = rl.MouseButton

ACTION_NAMES = [
    "FWD",
    "BCK",
    "LEFT",
    "RIGHT",
    "TURN_L",
    "TURN_R",
    "ATTACK",
    "USE",
    "SPEED",
]

def config_for_checkpoint(model_path: str, state_dict: dict[str, torch.Tensor]) -> WorldModelConfig:
    checkpoint_dir = os.path.dirname(model_path)
    if checkpoint_dir and os.path.isfile(os.path.join(checkpoint_dir, "config.json")):
        return WorldModelConfig.from_pretrained(checkpoint_dir)

    config = WorldModelConfig()
    cls_token = state_dict.get("cls_token")
    patchify = state_dict.get("patchify.weight")
    spatial_pos = state_dict.get("spatial_pos")
    first_ffn = state_dict.get("encoder.layers.0.linear1.weight")

    if cls_token is not None:
        config.dim = cls_token.shape[-1]
    if patchify is not None:
        config.dim = patchify.shape[0]
        config.patch_size = patchify.shape[-1]
    if first_ffn is not None and config.dim:
        config.ffn_mult = first_ffn.shape[0] // config.dim

    if spatial_pos is not None:
        n_patches = spatial_pos.shape[1] - 1
        default_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        if n_patches != default_patches:
            config.height, config.width = infer_image_size(n_patches, config.patch_size)

    block_ids = set()
    for key in state_dict:
        match = re.match(r"(?:encoder|predictor)\.layers\.(\d+)\.", key)
        if match:
            block_ids.add(int(match.group(1)))
    if block_ids:
        config.n_blocks = max(block_ids) + 1

    config.causal = True
    return config

def infer_image_size(n_patches: int, patch_size: int) -> tuple[int, int]:
    best_h, best_w = 1, n_patches
    target_ratio = 4 / 3
    for h in range(1, int(n_patches ** 0.5) + 1):
        if n_patches % h == 0:
            w = n_patches // h
            if abs((w / h) - target_ratio) < abs((best_w / best_h) - target_ratio):
                best_h, best_w = h, w
    return best_h * patch_size, best_w * patch_size

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_state_dict_checked(model: WorldModel, state_dict: dict[str, torch.Tensor], source: str) -> None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("decoder.")}
        result = model.load_state_dict(filtered, strict=False)
        optional_prefixes = ("decoder.", "policy.")
        bad_missing = [k for k in result.missing_keys if not k.startswith(optional_prefixes)]
        bad_unexpected = [k for k in result.unexpected_keys if not k.startswith(optional_prefixes)]
        if bad_missing or bad_unexpected:
            sample_keys = ", ".join(list(state_dict.keys())[:8])
            raise RuntimeError(
                f"Checkpoint {source!r} is not compatible with the WorldModel defined in model.py. "
                f"First checkpoint keys: {sample_keys}"
            ) from exc

def load_world_model(model_path: str, device: str) -> WorldModel:
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        torch_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.isfile(config_path):
            config = WorldModelConfig.from_pretrained(model_path)
            model = WorldModel(config)
            if os.path.isfile(safetensors_path):
                load_state_dict_checked(model, load_file(safetensors_path, device="cpu"), safetensors_path)
            elif os.path.isfile(torch_path):
                load_state_dict_checked(model, torch.load(torch_path, map_location="cpu"), torch_path)
            else:
                load_sharded_checkpoint(model, model_path, strict=True, prefer_safe=True)
        else:
            candidates = sorted(Path(model_path).glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(
                    f"No config.json/model.safetensors found in {model_path!r}. "
                    "Pass a Trainer checkpoint dir or a .safetensors file."
                )
            state_path = str(candidates[-1])
            state_dict = load_file(state_path, device="cpu")
            model = WorldModel(config_for_checkpoint(state_path, state_dict))
            load_state_dict_checked(model, state_dict, state_path)
    elif model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device="cpu")
        config = config_for_checkpoint(model_path, state_dict)
        model = WorldModel(config)
        load_state_dict_checked(model, state_dict, model_path)
    else:
        raise ValueError("model_path must be a checkpoint directory or a .safetensors file")

    return model.to(device).eval()

def load_frame(path: str, height: int, width: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)

def frame_to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    chw = np.transpose(frame, (2, 0, 1)).copy()
    return torch.from_numpy(chw).to(device=device, dtype=torch.float32) / 255.0

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    frame = tensor.mul(255).byte().cpu().numpy()
    return np.ascontiguousarray(np.transpose(frame, (1, 2, 0)))

def build_action() -> list[float]:
    action = [0.0] * len(ACTION_NAMES)

    if rl.is_key_down(KEY.KEY_W) or rl.is_key_down(KEY.KEY_UP):
        action[0] = 1.0
    if rl.is_key_down(KEY.KEY_S) or rl.is_key_down(KEY.KEY_DOWN):
        action[1] = 1.0
    if rl.is_key_down(KEY.KEY_A):
        action[2] = 1.0
    if rl.is_key_down(KEY.KEY_D):
        action[3] = 1.0
    if rl.is_key_down(KEY.KEY_Q) or rl.is_key_down(KEY.KEY_LEFT):
        action[4] = 1.0
    if rl.is_key_down(KEY.KEY_E) or rl.is_key_down(KEY.KEY_RIGHT):
        action[5] = 1.0
    if rl.is_mouse_button_down(MOUSE.MOUSE_BUTTON_LEFT) or rl.is_key_down(KEY.KEY_SPACE):
        action[6] = 1.0
    if rl.is_key_down(KEY.KEY_F) or rl.is_key_down(KEY.KEY_ENTER):
        action[7] = 1.0
    if rl.is_key_down(KEY.KEY_LEFT_SHIFT) or rl.is_key_down(KEY.KEY_RIGHT_SHIFT):
        action[8] = 1.0

    return action

def make_texture(frame: np.ndarray) -> rl.Texture:
    data = io.BytesIO()
    Image.fromarray(frame).save(data, format="BMP")
    raw = data.getvalue()
    ray_image = rl.load_image_from_memory(".bmp", raw, len(raw))
    texture = rl.load_texture_from_image(ray_image)
    rl.unload_image(ray_image)
    return texture

def replace_texture(texture: rl.Texture, frame: np.ndarray) -> rl.Texture:
    rl.unload_texture(texture)
    return make_texture(frame)

def autocast_context(device: str, enabled: bool, is_half: bool = False):
    if enabled and device == "cuda" and not is_half:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

@torch.inference_mode()
def encode_frame(model: WorldModel, frame: torch.Tensor, device: str, amp: bool) -> torch.Tensor:
    with autocast_context(device, amp, frame.dtype == torch.float16):
        _, tokens = model.encode(frame.unsqueeze(0).unsqueeze(0), return_tokens=True)
    return tokens[:, 0]

@torch.inference_mode()
def predict_next_frame(
    model: WorldModel,
    token_history: torch.Tensor,
    actions: list[list[float]] | torch.Tensor,
    max_context_len: int,
    device: str,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]] | torch.Tensor]:
    if isinstance(actions, list):
        actions = actions[-token_history.size(1) :]
        action_tensor = torch.tensor(actions, device=device, dtype=token_history.dtype).unsqueeze(0)
    else:
        action_tensor = actions
    with autocast_context(device, amp, token_history.dtype == torch.float16):
        next_tokens = model.predict(token_history, action_tensor)[:, -1]
        pixel_pred = model.decoder(next_tokens.unsqueeze(1), action_tensor[:, -1:], context_tokens=token_history[:, -1:])[0]

    next_tok = next_tokens.unsqueeze(1)
    if token_history.size(1) < max_context_len:
        token_history = torch.cat([token_history, next_tok], dim=1)
    else:
        token_history = torch.cat([token_history[:, 1:], next_tok], dim=1)
        actions = actions[-max_context_len + 1 :]
    return pixel_pred, token_history, actions

def draw_ui(action: list[float], fps: float, generated_count: int) -> None:
    active = [name for name, value in zip(ACTION_NAMES, action) if value]
    rl.draw_rectangle(0, 0, 320, 88, rl.fade(rl.BLACK, 0.65))
    rl.draw_text("World Model", 10, 8, 18, rl.RAYWHITE)
    rl.draw_text(f"Frames generated: {generated_count}", 10, 30, 16, rl.LIGHTGRAY)
    rl.draw_text(f"Inputs: {', '.join(active) if active else 'none'}", 10, 52, 16, rl.LIGHTGRAY)
    rl.draw_text(f"Target FPS: {fps:g}", 10, 70, 14, rl.GRAY)

@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    device = args.device or get_device()
    model = load_world_model(args.model, device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if args.compile:
        model = torch.compile(model)
    if device != "cpu" and not args.fp32:
        model = model.half()
    config = model.config
    frame = load_frame(args.frame, config.height, config.width)
    frame_tensor = frame_to_tensor(frame, device).to(dtype=next(model.parameters()).dtype)

    token_history = encode_frame(model, frame_tensor, device, args.amp).unsqueeze(1)
    actions = []

    rl.init_window(config.width * args.scale, config.height * args.scale, "miniworld")
    rl.set_target_fps(60)
    texture = make_texture(frame)

    generated_count = 0
    last_step_time = rl.get_time()
    frame_interval = 1.0 / args.fps

    try:
        while not rl.window_should_close():
            action = build_action()
            now = rl.get_time()

            if now - last_step_time >= frame_interval:
                actions.append(action)
                next_frame, token_history, actions = predict_next_frame(
                    model,
                    token_history,
                    actions,
                    args.context_len,
                    device,
                    args.amp,
                )
                frame = tensor_to_frame(next_frame)
                texture = replace_texture(texture, frame)
                generated_count += 1
                last_step_time = now

            rl.begin_drawing()
            rl.clear_background(rl.BLACK)
            rl.draw_texture_ex(texture, rl.Vector2(0, 0), 0.0, args.scale, rl.WHITE)
            draw_ui(action, args.fps, generated_count)
            rl.end_drawing()
    finally:
        rl.unload_texture(texture)
        rl.close_window()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play with a trained DOOM world model through a raylib UI.")
    parser.add_argument("--model", default="./checkpoints", help="Checkpoint directory or .safetensors file")
    parser.add_argument("--frame", help="Initial DOOM frame image")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--fps", type=float, default=1.0, help="World-model prediction rate")
    parser.add_argument("--scale", type=int, default=2, help="Window scale factor")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA bf16 autocast")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model")
    parser.add_argument("--fp32", action="store_true", help="Use float32 precision instead of float16")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    if args.context_len < 1:
        raise ValueError("--context-len must be >= 1")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.frame is None:
        args.frame = input("Path to an initial DOOM frame: ").strip()
    if not Path(args.frame).is_file():
        raise FileNotFoundError(args.frame)
    return args

if __name__ == "__main__":
    run(parse_args())
