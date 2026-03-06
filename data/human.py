"""
Used to record a human playing DOOM on the first episode.

https://huggingface.co/datasets/lucrbrtv/doom-e1-gameplay

Attention: This script assume your are using the DOOM (1993) WAD file.
If not you should change the map names to make it work.

*If you want to use DOOM WAD file you must buy the game (I recommend GOG)*
"""

import os
import time

import pygame
import vizdoom as vzd
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from datasets import Image as HFImage
from dotenv import load_dotenv
from PIL import Image as PILImage

load_dotenv()

wad = os.getenv("DOOM_WAD_PATH")
SKILL = 1

RESOLUTION = vzd.ScreenResolution.RES_320X240

game = vzd.DoomGame()
if wad:
    game.set_doom_game_path(wad)

BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,
    vzd.Button.SPEED,
]
for b in BUTTONS:
    game.add_available_button(b)

game.set_screen_resolution(RESOLUTION)
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_window_visible(True)
game.set_mode(vzd.Mode.PLAYER)
game.set_sound_enabled(True)
game.add_game_args(
    f"+skill {SKILL} +snd_samplerate 44100 +snd_efx 0 +set snd_backend openal"
)
game.set_render_hud(True)
game.init()

N_BUTTONS = game.get_available_buttons_size()

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No controller detected.")

joystick = pygame.joystick.Joystick(0)
print(f"Controller : {joystick.get_name()}")

DEADZONE = 0.2

PS_AXIS = {"lx": 0, "ly": 1, "rx": 2, "ry": 3, "l2": 4, "r2": 5}
PS_BTN = {
    "cross": 0,
    "circle": 1,
    "square": 2,
    "triangle": 3,
    "l1": 4,
    "r1": 5,
    "l2": 6,
    "r2": 7,
    "share": 8,
    "options": 9,
    "l3": 10,
    "r3": 11,
}

def axis(name: str) -> float:
    val = joystick.get_axis(PS_AXIS[name])
    return val if abs(val) > DEADZONE else 0.0

def btn(name: str) -> bool:
    return bool(joystick.get_button(PS_BTN[name]))

def build_action() -> list[int]:
    action = [0] * N_BUTTONS
    ly, lx, rx, r2 = axis("ly"), axis("lx"), axis("rx"), axis("r2")

    if ly < 0:
        action[0] = 1
    elif ly > 0:
        action[1] = 1
    if lx < 0:
        action[2] = 1
    elif lx > 0:
        action[3] = 1
    if rx < 0:
        action[4] = 1
    elif rx > 0:
        action[5] = 1
    if btn("cross") or r2 > 0.5:
        action[6] = 1
    if btn("circle"):
        action[7] = 1
    if btn("l1") or btn("r1"):
        action[8] = 1

    return action

EPISODE_1_MAPS = [
    "E1M1",
    "E1M2",
    "E1M3",
    "E1M4",
    "E1M5",
    "E1M6",
    "E1M7",
    "E1M8",
    "E1M9",
]

SHARDS_DIR = "gameplay_shards"
os.makedirs(SHARDS_DIR, exist_ok=True)

features = Features(
    {
        "episode": Value("string"),
        "step": Value("int32"),
        "frame": HFImage(),
        "action": Sequence(Value("int32")),
    }
)

for map_name in EPISODE_1_MAPS:
    game.set_doom_map(map_name)
    game.new_episode()
    step = 0

    records: dict[str, list] = {"episode": [], "step": [], "frame": [], "action": []}

    print(f"\n--- {map_name} | Skill {SKILL} ---")

    while not game.is_episode_finished():
        pygame.event.pump()

        state = game.get_state()
        action = build_action()
        game.make_action(action)

        pil_frame = PILImage.fromarray(state.screen_buffer)

        records["episode"].append(map_name)
        records["step"].append(step)
        records["frame"].append(pil_frame)
        records["action"].append(action)

        step += 1
        time.sleep(1 / 60)

    # flush on disk to not saturate RAM
    shard_path = os.path.join(SHARDS_DIR, map_name)
    shard = Dataset.from_dict(records, features=features)
    shard.save_to_disk(shard_path)

    print(f"{map_name} — {step} steps — shard saved → {shard_path}")

game.close()

print("\nConcatenate shards..")
shards = [
    Dataset.load_from_disk(os.path.join(SHARDS_DIR, m))
    for m in EPISODE_1_MAPS
    if os.path.exists(os.path.join(SHARDS_DIR, m))
]
full_dataset = concatenate_datasets(shards)
full_dataset.save_to_disk("gameplay_dataset")

print(f"Dataset: {len(full_dataset)} steps on {len(shards)} maps")
