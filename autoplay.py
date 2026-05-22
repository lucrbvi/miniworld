import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import vizdoom as vzd
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file

from dream import ACTION_NAMES, get_device, load_world_model
from model import ActionPolicy, WorldModel

BUTTONS = [
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK, vzd.Button.USE, vzd.Button.SPEED,
]

def load_action_policy(path: str, config, device: str) -> ActionPolicy:
    p = Path(path)
    if p.is_dir():
        ap_path = next((p / name for name in ["model.safetensors", "pytorch_model.bin"] if (p / name).is_file()), None)
        if ap_path is None:
            raise FileNotFoundError(f"ActionPolicy not found in {path!r}")
    elif p.is_file():
        ap_path = p
    else:
        raise FileNotFoundError(f"ActionPolicy not found: {path!r}")
    sd = load_file(ap_path, device="cpu") if ap_path.suffix == ".safetensors" else torch.load(ap_path, map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    max_seq_len = sd["time_pos"].shape[1] if "time_pos" in sd else 64
    ap = ActionPolicy(config.dim, n_heads=config.n_heads, ffn_mult=config.ffn_mult, max_seq_len=max_seq_len)
    ap.load_state_dict(sd)
    return ap.to(device=device).eval()

@torch.inference_mode()
def encode_frame(model: WorldModel, frame: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    x = torch.from_numpy(np.transpose(frame, (2, 0, 1)).copy()).to(device=device, dtype=dtype) / 255.0
    return model.encode(x[None, None])[:, 0]

def run(args: argparse.Namespace) -> None:
    load_dotenv()
    args.wad = args.wad or os.getenv("DOOM_WAD_PATH")
    device = args.device or get_device()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if device == "cuda" and not args.fp32 else torch.float32
    model = load_world_model(args.model, device).to(dtype=dtype).eval()
    ap = load_action_policy(args.action_policy, model.config, device).to(dtype=dtype)

    game = vzd.DoomGame()
    if args.wad:
        game.set_doom_game_path(args.wad)
    if args.map:
        game.set_doom_map(args.map)
    for button in BUTTONS:
        game.add_available_button(button)
    game.set_screen_resolution(
        vzd.ScreenResolution.RES_1280X960 if args.scale >= 4
        else vzd.ScreenResolution.RES_1024X768 if args.scale >= 3
        else vzd.ScreenResolution.RES_640X480 if args.scale >= 2
        else vzd.ScreenResolution.RES_320X240
    )
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(not args.headless)
    game.set_sound_enabled(not args.headless)
    game.set_render_hud(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args(f"+skill {args.skill}")
    game.init()

    video = None
    history: torch.Tensor | None = None
    past_actions: torch.Tensor | None = None
    max_ctx = ap.max_seq_len
    try:
        game.new_episode()
        step = 0
        while not game.is_episode_finished() and step < args.max_steps:
            state = game.get_state()
            raw = state.screen_buffer.copy()

            h, w = model.config.height, model.config.width
            if raw.shape[:2] != (h, w):
                frame = np.asarray(Image.fromarray(raw).resize((w, h), Image.Resampling.BILINEAR))
            else:
                frame = raw

            cls_token = encode_frame(model, frame, device, dtype)
            history = cls_token.unsqueeze(1) if history is None else torch.cat([history, cls_token.unsqueeze(1)], dim=1)[:, -max_ctx:]

            logits = ap(history, past_actions=past_actions)[:, -1]
            action = ActionPolicy.logits_to_binary(logits / args.temperature)[0].cpu().tolist()
            if args.epsilon > 0 and torch.rand(1).item() < args.epsilon:
                action = [0.0] * 9
                action[torch.randint(0, 9, (1,)).item()] = 1.0

            action_t = torch.tensor([action], device=device, dtype=dtype).unsqueeze(0)
            past_actions = action_t if past_actions is None else torch.cat([past_actions, action_t], dim=1)[:, -(max_ctx - 1):]

            active = [name for name, val in zip(ACTION_NAMES, action) if val]
            overlay = f"step={step} action={' '.join(active) if active else 'none'}"
            cv2.putText(raw, overlay, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
            cv2.imshow("miniworld - Autoplay", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            if args.record:
                if video is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    h_raw, w_raw = raw.shape[:2]
                    video = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.record_fps, (w_raw, h_raw))
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                for _ in range(args.frame_skip):
                    video.write(bgr)

            game.make_action(action, args.frame_skip)

            if args.log_every and step % args.log_every == 0:
                print(f"step={step:05d} action={' '.join(active) if active else 'none'}", flush=True)

            step += 1
    finally:
        cv2.destroyAllWindows()
        if video is not None:
            video.release()
            print(f"video saved to {args.record}")
        game.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play DOOM with a world model and learned action policy.")
    parser.add_argument("--model", default="./checkpoints/world-model")
    parser.add_argument("--action-policy", default="./checkpoints/action-policy")
    parser.add_argument("--wad", default=None)
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--skill", type=int, default=1)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sigmoid/softmax (lower = more deterministic)")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Random action probability to break loops")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--record", default=None)
    parser.add_argument("--record-fps", type=float, default=35.0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.wad is not None and not Path(args.wad).is_file():
        raise FileNotFoundError(args.wad)
    return args

if __name__ == "__main__":
    run(parse_args())
