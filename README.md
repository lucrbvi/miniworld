# miniworld

A small *Latent World Model* of [DOOM (1993)](https://en.wikipedia.org/wiki/Doom_(1993_video_game)). The model predicts the next frame's latent tokens from past frames and actions, then decodes them to pixels. It runs fast enough that you can play inside it in real time.

## Install

```sh
uv sync
```

## Download pretrained weights

```sh
uv run download_weights.py
```

This pulls the world model (and action policy, if available) from the
[Hugging Face Hub](https://huggingface.co/lucrbrtv/doom-world-model) into
`./checkpoints/world-model/` and `./checkpoints/action-policy/`.

## Train

```sh
uv run train.py --mode wm              # encoder + predictor
uv run train.py --mode decoder         # pixel decoder on a frozen WM
uv run train.py --mode action_policy   # forward dynamics action policy
uv run train.py --mode all             # all three, in order
```

Training data is downloaded automatically from Hugging Face. Checkpoints are written to `./checkpoints/world-model`, `./checkpoints/decoder` and `./checkpoints/action-policy`, and resume from the latest one when you rerun.

The `decoder` and `action_policy` modes need a trained world model to start from. Either run `--mode wm` first (or `--mode all`), or pass `--wm-checkpoint <path>` to point at one checkpoint explicitly.

## Play inside the world model

Drive the world model with your keyboard and play a pseudo-DOOM game.
First download the weights (see above), then run:

```sh
uv run dream.py
```

## Let the policy play

The learned action policy plays in the real DOOM engine while you watch.

```sh
uv run autoplay.py
```

This uses `./checkpoints/world-model` and `./checkpoints/action-policy` by default.

Pass `--wad path/to/doom.wad` to play the real game (for example `DOOM.WAD`). Without it, VizDoom falls back to its bundled FreeDoom resources, which work but look slightly off and have different level names. You can also add `--record out.mp4` to save a video, or `--headless` to skip the window.

## Architecture

The encoder is a ViT that maps frames to patch tokens of shape `[B, T, n_patches+1, D]`. The predictor is a causal transformer over time, conditioned on actions, trained in latent space with MSE, short rollouts and the LeJEPA SIGReg objective. The decoder is a simple convolutionnal neural networks with Pixel Shuffling to generate pixels. The action policy is a MLP that predict the next action vector.

## References

* [LeWorldModel (Maes et al., 2026)](https://arxiv.org/abs/2603.19312), a good general recipe to follow.
* [Next Embedding Prediction Makes World Models Stronger (Bredis et al., 2026)](https://arxiv.org/abs/2603.02765), demonstration of a good latent world model.
* [LeJEPA (Balestriero & LeCun, 2025)](https://www.arxiv.org/abs/2511.08544), the SIGReg objective.
* [V-JEPA 2 (Assran et al., 2025)](https://www.arxiv.org/abs/2506.09985), action conditioned rollouts.
* [VPT (Baker et al., 2022)](https://arxiv.org/abs/2206.11795), inverse dynamics from unlabeled gameplay.
* [ViZDoom (Kempka et al., 2016)](https://arxiv.org/abs/1605.02097), the DOOM gym used for data and `autoplay`.
