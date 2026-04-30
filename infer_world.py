import argparse
import io
import os
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
    first_ffn = state_dict.get("transformer.ffn.0.0.weight")

    if cls_token is not None:
        config.dim = cls_token.shape[-1]
    if first_ffn is not None and config.dim:
        config.ffn_mult = first_ffn.shape[0] // config.dim

    block_ids = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("transformer.blocks.") and key.split(".")[2].isdigit()
    }
    if block_ids:
        config.n_blocks = max(block_ids) + 1

    return config

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_world_model(model_path: str, device: str) -> WorldModel:
    if os.path.isdir(model_path):
        config = WorldModelConfig.from_pretrained(model_path)
        model = WorldModel(config)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        torch_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.isfile(safetensors_path):
            model.load_state_dict(load_file(safetensors_path, device="cpu"))
        elif os.path.isfile(torch_path):
            model.load_state_dict(torch.load(torch_path, map_location="cpu"))
        else:
            load_sharded_checkpoint(model, model_path, strict=True, prefer_safe=True)
    elif model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device="cpu")
        config = config_for_checkpoint(model_path, state_dict)
        model = WorldModel(config)
        model.load_state_dict(state_dict)
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
    frame = tensor.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    return np.transpose(frame, (1, 2, 0)).copy()

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
    Image.fromarray(frame).save(data, format="PNG")
    png = data.getvalue()
    ray_image = rl.load_image_from_memory(".png", png, len(png))
    texture = rl.load_texture_from_image(ray_image)
    rl.unload_image(ray_image)
    return texture

def replace_texture(texture: rl.Texture, frame: np.ndarray) -> rl.Texture:
    rl.unload_texture(texture)
    return make_texture(frame)

@torch.inference_mode()
def predict_next_frame(
    model: WorldModel,
    frames: list[torch.Tensor],
    actions: list[list[float]],
    device: str,
) -> torch.Tensor:
    x = torch.stack(frames, dim=0).unsqueeze(0)
    action = torch.tensor(actions, device=device, dtype=torch.float32).unsqueeze(0)
    _, pixel_pred = model(x, action)
    return pixel_pred[0]

def draw_ui(action: list[float], fps: float, generated_count: int) -> None:
    active = [name for name, value in zip(ACTION_NAMES, action) if value]
    rl.draw_rectangle(0, 0, 320, 88, rl.fade(rl.BLACK, 0.65))
    rl.draw_text("World Model", 10, 8, 18, rl.RAYWHITE)
    rl.draw_text(f"Frames generated: {generated_count}", 10, 30, 16, rl.LIGHTGRAY)
    rl.draw_text(f"Inputs: {', '.join(active) if active else 'none'}", 10, 52, 16, rl.LIGHTGRAY)
    rl.draw_text(f"Target FPS: {fps:g}", 10, 70, 14, rl.GRAY)

def run(args: argparse.Namespace) -> None:
    device = args.device or get_device()
    model = load_world_model(args.model, device)
    config = model.config
    frame = load_frame(args.frame, config.height, config.width)

    frames = [frame_to_tensor(frame, device) for _ in range(args.context_len)]
    actions = [[0.0] * len(ACTION_NAMES) for _ in range(args.context_len)]

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
                next_frame = predict_next_frame(model, frames, actions, device)
                frame = tensor_to_frame(next_frame)
                frames = [*frames[1:], frame_to_tensor(frame, device)]
                actions = [*actions[1:], action]
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
    parser = argparse.ArgumentParser(
        description="Play with a trained DOOM world model through a raylib UI."
    )
    parser.add_argument("--model", default="./checkpoints", help="Checkpoint directory or .safetensors file")
    parser.add_argument("--frame", help="Initial DOOM frame image")
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0, help="World-model prediction rate")
    parser.add_argument("--scale", type=int, default=2, help="Window scale factor")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()

    if args.frame is None:
        args.frame = input("Path to an initial DOOM frame: ").strip()
    if not Path(args.frame).is_file():
        raise FileNotFoundError(args.frame)
    return args

if __name__ == "__main__":
    run(parse_args())
