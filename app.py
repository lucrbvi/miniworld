"""
HuggingFace Space Doom World Model interactive demo.
Deploy alongside model.py in the Space repo.
"""

import os

import functools

import gradio as gr
import numpy as np
import torch
from PIL import Image

from model import WorldModel

try:
    import spaces
    HAS_ZERO_GPU = True
except ImportError:
    # Running locally without ZeroGPU — create a no-op decorator
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(fn):
            return fn
    HAS_ZERO_GPU = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO = "lucrbrtv/doom-worldmodel"  # HF Hub repo with the weights
MAX_CONTEXT = 8   # sliding window length fed to the predictor
HEIGHT, WIDTH = 240, 320

ACTIONS: dict[str, list[float]] = {
    "Forward":      [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Back":         [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Strafe Left":  [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "Strafe Right": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "Turn Left":    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "Turn Right":   [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Attack":       [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "Use":          [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Idle":         [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model — loaded once, lives on CPU between ZeroGPU calls
# ---------------------------------------------------------------------------
_model: WorldModel | None = None


def get_model() -> WorldModel:
    global _model
    if _model is None:
        _model = WorldModel.from_pretrained(MODEL_REPO).half().eval()
        if not HAS_ZERO_GPU:
            _model = _model.to(DEVICE)
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """HWC uint8 numpy → CHW float16 CUDA tensor in [0, 1]."""
    chw = np.ascontiguousarray(np.transpose(frame.astype(np.float32), (2, 0, 1)))
    return torch.from_numpy(chw).to(device=DEVICE, dtype=torch.float16) / 255.0


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """CHW float tensor in [0, 1] → PIL Image."""
    arr = t.mul(255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

@spaces.GPU
def initialize(init_image: np.ndarray | None):
    """Encode the initial frame and return the starting state."""
    if init_image is None:
        return gr.update(), None, [], 0

    m = get_model().to(DEVICE)

    pil = Image.fromarray(init_image).convert("RGB").resize((WIDTH, HEIGHT), Image.BILINEAR)
    frame = np.array(pil)
    ft = _frame_to_tensor(frame)

    with torch.inference_mode():
        _, tokens = m.encode(ft.unsqueeze(0).unsqueeze(0), return_tokens=True)

    # (1, 1, n_patches+1, dim) — store on CPU between calls
    token_history = tokens.cpu()

    return pil, token_history, [], 0, "Steps: 0"


@spaces.GPU
def step(action_name: str, token_history, actions_list: list, step_count: int):
    """Run one world-model prediction step for the given action."""
    if token_history is None:
        return gr.update(), None, [], 0

    m = get_model().to(DEVICE)
    action = ACTIONS[action_name]
    actions_list = actions_list + [action]

    token_history = token_history.to(device=DEVICE, dtype=torch.float16)
    t = token_history.size(1)

    # Build action tensor aligned with token history length
    action_slice = actions_list[-t:]
    action_tensor = torch.tensor(action_slice, device=DEVICE, dtype=torch.float16).unsqueeze(0)

    with torch.inference_mode():
        predicted_tokens = m.predict(token_history, action_tensor)[:, -1]   # (1, n+1, d)
        pixel_pred = m.decoder(m.transition(predicted_tokens))[0]           # (C, H, W)

    next_tok = predicted_tokens.unsqueeze(1)  # (1, 1, n+1, d)
    if t < MAX_CONTEXT:
        token_history = torch.cat([token_history, next_tok], dim=1)
    else:
        token_history = torch.cat([token_history[:, 1:], next_tok], dim=1)
        actions_list = actions_list[-MAX_CONTEXT:]

    new_count = step_count + 1
    return _tensor_to_pil(pixel_pred), token_history.cpu(), actions_list, new_count, f"Steps: {new_count}"


# ---------------------------------------------------------------------------
# Keyboard JS — maps WASD / arrows / space to action buttons
# ---------------------------------------------------------------------------
_JS = """
() => {
    const KEY_MAP = {
        'w': 'btn_fwd',       'arrowup': 'btn_fwd',
        's': 'btn_back',      'arrowdown': 'btn_back',
        'a': 'btn_sleft',     'd': 'btn_sright',
        'q': 'btn_tleft',     'arrowleft': 'btn_tleft',
        'e': 'btn_tright',    'arrowright': 'btn_tright',
        ' ': 'btn_attack',
        'f': 'btn_use',
        'i': 'btn_idle',
    };
    document.addEventListener('keydown', function(ev) {
        if (['INPUT','TEXTAREA','SELECT'].includes(ev.target.tagName)) return;
        const id = KEY_MAP[ev.key.toLowerCase()];
        if (!id) return;
        ev.preventDefault();
        const el = document.getElementById(id);
        if (el) { const b = el.querySelector('button'); if (b) b.click(); }
    });
}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
#frame-display { border-radius: 8px; overflow: hidden; }
.action-btn button { font-size: 15px; padding: 10px 0; width: 100%; }
.primary-action button { background: #e53935 !important; color: white !important; }
"""

with gr.Blocks(title="Doom World Model", css=CSS, js=_JS) as demo:
    # ---- session state ----
    token_state  = gr.State(value=None)
    actions_state = gr.State(value=[])
    step_count   = gr.State(value=0)

    gr.Markdown(
        "# Doom World Model\n"
        "Upload a Doom screenshot, hit **Initialize**, then use the buttons "
        "(or keyboard shortcuts) to explore the world model's imagination.\n\n"
        "> **Keys:** W/↑ Forward · S/↓ Back · A Strafe L · D Strafe R · "
        "Q/← Turn L · E/→ Turn R · Space Attack · F Use · I Idle"
    )

    with gr.Row():
        # ---- main display ----
        with gr.Column(scale=3):
            frame_out = gr.Image(
                label="World Model Output",
                type="pil",
                interactive=False,
                elem_id="frame-display",
                height=480,
            )
            steps_out = gr.Markdown("Steps: 0")

        # ---- controls ----
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Initialize")
            init_img = gr.Image(label="Initial frame", type="numpy", height=150)
            init_btn = gr.Button("Initialize", variant="primary")

            gr.Markdown("### Actions")
            with gr.Row():
                b_tleft  = gr.Button("↺ Turn L",  elem_id="btn_tleft",  elem_classes="action-btn")
                b_fwd    = gr.Button("↑ Forward",  elem_id="btn_fwd",    elem_classes=["action-btn", "primary-action"])
                b_tright = gr.Button("↻ Turn R",   elem_id="btn_tright", elem_classes="action-btn")
            with gr.Row():
                b_sleft  = gr.Button("← Strafe",   elem_id="btn_sleft",  elem_classes="action-btn")
                b_back   = gr.Button("↓ Back",      elem_id="btn_back",   elem_classes="action-btn")
                b_sright = gr.Button("Strafe →",    elem_id="btn_sright", elem_classes="action-btn")
            with gr.Row():
                b_attack = gr.Button("🔫 Attack",   elem_id="btn_attack", elem_classes=["action-btn", "primary-action"])
                b_use    = gr.Button("🚪 Use",       elem_id="btn_use",    elem_classes="action-btn")
                b_idle   = gr.Button("Idle",         elem_id="btn_idle",   elem_classes="action-btn")

    # ---- init handler ----
    init_btn.click(
        fn=initialize,
        inputs=[init_img],
        outputs=[frame_out, token_state, actions_state, step_count, steps_out],
    )

    # ---- action handlers ----
    _STEP_INPUTS  = [token_state, actions_state, step_count]
    _STEP_OUTPUTS = [frame_out, token_state, actions_state, step_count, steps_out]

    for btn, action_name in [
        (b_fwd,    "Forward"),
        (b_back,   "Back"),
        (b_sleft,  "Strafe Left"),
        (b_sright, "Strafe Right"),
        (b_tleft,  "Turn Left"),
        (b_tright, "Turn Right"),
        (b_attack, "Attack"),
        (b_use,    "Use"),
        (b_idle,   "Idle"),
    ]:
        btn.click(
            fn=functools.partial(step, action_name),
            inputs=_STEP_INPUTS,
            outputs=_STEP_OUTPUTS,
        )

if __name__ == "__main__":
    demo.launch()
