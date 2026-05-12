import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file

from dream import ACTION_NAMES, get_device, load_frame, load_world_model
from model import Policy, WorldModel

BUTTONS = [
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK, vzd.Button.USE, vzd.Button.SPEED,
]

def autocast(device: str, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled and device == "cuda" else nullcontext()

def encode(model: WorldModel, frame: np.ndarray, device: str, amp: bool) -> torch.Tensor:
    x = torch.from_numpy(np.transpose(frame, (2, 0, 1)).copy()).to(device=device, dtype=torch.float32) / 255.0
    with torch.no_grad(), autocast(device, amp):
        _, tokens = model.encode(x[None, None], return_tokens=True)
    return tokens[:, 0].float().detach().clone()

def plan(model: WorldModel, policy: Policy | None, token_history: torch.Tensor, action_history: torch.Tensor, target: torch.Tensor, previous_logits: torch.Tensor | None, args: argparse.Namespace, device: str) -> tuple[list[int], torch.Tensor, float]:
    logits = torch.randn(args.horizon, len(ACTION_NAMES), device=device) if previous_logits is None else torch.cat([previous_logits[1:].detach(), torch.randn(1, len(ACTION_NAMES), device=device)])
    logits.requires_grad_(True)
    opt = torch.optim.AdamW([logits], lr=args.lr)
    loss_value = 0.0
    for _ in range(args.iters):
        opt.zero_grad(set_to_none=True)
        actions, tokens, imagined = torch.sigmoid(logits), token_history, []
        with autocast(device, args.amp):
            for t in range(args.horizon):
                act = torch.cat([action_history[:, :-1], actions[: t + 1].unsqueeze(0)], dim=1)
                pred = model.predict(tokens, act.to(tokens.dtype))[:, -1]
                imagined.append(pred[:, 0])
                tokens = torch.cat([tokens, pred.unsqueeze(1)], dim=1)
            imagined = torch.stack(imagined, dim=1)
            current, goal = token_history[:, -1, 0], target[:, 0].unsqueeze(1).expand_as(imagined)
            reward = -F.mse_loss(imagined.float(), goal.float())
            if policy is not None and args.policy_weight:
                reward = reward + args.policy_weight * policy(imagined.reshape(-1, imagined.size(-1)), goal.reshape(-1, goal.size(-1))).mean().float()
            loss = -reward - args.away_weight * F.mse_loss(imagined[:, -1].float(), current.float())
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu())
    a = torch.sigmoid(logits[0]).detach().cpu().tolist()
    action = [0] * len(ACTION_NAMES)
    for i, j in ((0, 1), (2, 3), (4, 5)):
        if max(a[i], a[j]) >= args.threshold:
            action[i if a[i] >= a[j] else j] = 1
    action[6:] = [int(a[i] >= args.threshold) for i in range(6, 9)]
    return action, logits.detach().clone(), loss_value

def run(args: argparse.Namespace) -> None:
    load_dotenv()
    args.wad = args.wad or os.getenv("DOOM_WAD_PATH")
    device = args.device or get_device()
    model = load_world_model(args.model, device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    target = encode(model, load_frame(args.target_frame, cfg.height, cfg.width), device, args.amp)
    policy = None
    if args.policy != "none":
        policy = model.policy
        if args.policy != "embedded":
            if os.path.isdir(args.policy):
                safe, pt, full = (os.path.join(args.policy, n) for n in ("policy.safetensors", "policy.pt", "model.safetensors"))
                if os.path.isfile(safe):
                    policy.load_state_dict(load_file(safe, device="cpu"))
                elif os.path.isfile(pt):
                    policy.load_state_dict(torch.load(pt, map_location="cpu"))
                elif os.path.isfile(full):
                    policy.load_state_dict(WorldModel.from_pretrained(args.policy).policy.state_dict())
                else:
                    print(f"Policy introuvable dans {args.policy!r}; autoplay sans policy.", flush=True)
                    policy = None
            elif os.path.isfile(args.policy):
                policy.load_state_dict(load_file(args.policy, device="cpu") if args.policy.endswith(".safetensors") else torch.load(args.policy, map_location="cpu"))
            else:
                print(f"Policy introuvable: {args.policy!r}; autoplay sans policy.", flush=True)
                policy = None
        if policy is not None:
            policy.to(device).eval()
            for p in policy.parameters():
                p.requires_grad_(False)
    game = vzd.DoomGame()
    if args.wad:
        game.set_doom_game_path(args.wad)
    if args.map:
        game.set_doom_map(args.map)
    for button in BUTTONS:
        game.add_available_button(button)
    game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960 if args.scale >= 4 else vzd.ScreenResolution.RES_1024X768 if args.scale >= 3 else vzd.ScreenResolution.RES_640X480 if args.scale >= 2 else vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(not args.headless)
    game.set_sound_enabled(not args.headless)
    game.set_render_hud(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args(f"+skill {args.skill}")
    game.init()
    last_plan = token_history = video = None
    action_history: list[list[int]] = []
    try:
        game.new_episode()
        step = 0
        while not game.is_episode_finished() and step < args.max_steps:
            state = game.get_state()
            if state is None:
                game.make_action([0] * len(ACTION_NAMES), args.frame_skip)
                continue
            raw = state.screen_buffer
            if args.record:
                if video is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = raw.shape[:2]
                    video = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.record_fps, (w, h))
                    if not video.isOpened():
                        raise RuntimeError(f"Could not open video writer: {args.record}")
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                for _ in range(args.frame_skip):
                    video.write(bgr)
            frame = raw if raw.shape[:2] == (cfg.height, cfg.width) else np.asarray(Image.fromarray(raw).resize((cfg.width, cfg.height), Image.Resampling.BILINEAR))
            current = encode(model, frame, device, args.amp)
            token_history = current.unsqueeze(1) if token_history is None else torch.cat([token_history, current.unsqueeze(1)], dim=1)
            action_history.append([0] * len(ACTION_NAMES))
            token_history, action_history = token_history[:, -args.context_len:], action_history[-token_history.size(1):]
            action_tensor = torch.tensor(action_history, device=device, dtype=torch.float32).unsqueeze(0)
            action, last_plan, loss = plan(model, policy, token_history, action_tensor, target, last_plan, args, device)
            action_history[-1] = action
            game.make_action(action, args.frame_skip)
            if args.log_every and step % args.log_every == 0:
                active = [name for name, value in zip(ACTION_NAMES, action) if value]
                print(f"step={step:05d} loss={loss:.4f} action={active or ['none']}", flush=True)
            step += 1
    finally:
        if video is not None:
            video.release()
            print(f"video saved to {args.record}")
        game.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal gradient-based VizDOOM autoplay with a world model.")
    parser.add_argument("--model", default="./checkpoints/world-model")
    parser.add_argument("--target-frame", required=True)
    parser.add_argument("--wad", default=None, help="Defaults to DOOM_WAD_PATH from .env")
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--skill", type=int, default=1)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--away-weight", type=float, default=0.2, help="Encourage predicted latent to move away from current latent")
    parser.add_argument("--policy", default="./checkpoints/policy", help="Policy dir/.safetensors/.pt, 'embedded', or 'none'")
    parser.add_argument("--policy-weight", type=float, default=0.6, help="Weight of learned policy/value score in Grad-MPC objective")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--record", default=None, help="Save gameplay to an .mp4 file")
    parser.add_argument("--record-fps", type=float, default=35.0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--headless", action="store_true")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    if not Path(args.target_frame).is_file():
        raise FileNotFoundError(args.target_frame)
    if args.wad is not None and not Path(args.wad).is_file():
        raise FileNotFoundError(args.wad)
    if min(args.context_len, args.horizon, args.iters, args.frame_skip, args.scale) < 1:
        raise ValueError("context-len, horizon, iters, frame-skip and scale must be >= 1")
    return args

if __name__ == "__main__":
    run(parse_args())
