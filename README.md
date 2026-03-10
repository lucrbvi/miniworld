# miniworld

A repository to train and improve a small *Latent World Model* to simulate [DOOM (1993)](https://en.wikipedia.org/wiki/Doom_(1993_video_game)). You can play the game inside the model in real-time.

Everything in the repo is open-source (under the MIT License).

You can learn more about techical details in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Install

```sh
uv sync
```

or

```sh
pip install -r requirements.txt
```

Dependencies:
* [pytorch](https://pytorch.org)
* [numpy](https://numpy.org)
* [vizdoom](https://vizdoom.farama.org/) (to generate the training data)
* [raylib](https://www.raylib.com/) (to show the generated frames and listen to users inputs)
* [datasets](https://huggingface.co/docs/datasets/index) (to store and use our training data)
* [pillow](https://python-pillow.github.io/) (to handle images and videos)
* [pandas](https://pandas.pydata.org/) (to handle various data)
* [safetensors](https://huggingface.co/docs/safetensors/index) (to store our weights)
* [transformers](https://huggingface.co/docs/transformers/index) (to easily train our model)
* [kernels](https://huggingface.co/docs/kernels/index) (to speed up our model)

## Quick start

You can directly train the current best model on our data by doing `uv run train.py` or `python3 train.py` (it will download the training data from huggingface).

After that you can run `uv run play.py ./weights` to load your train world model and play DOOM!

## But I just want to play DOOM

Run `uv run play.py` or `python3 play.py`, this will download a pre-trained version of the world model, so you can play DOOM inside it.

## Contribute

You can contribute by submitting in a PR a new loss record in Model Architecture or Training Data. Other PRs might be closed.

## Academic Sources

Here is some relevant academic papers to learn more about latent world models:
- [Next Embedding Prediction Makes World Models Stronger [Bredis et al, 2026]](https://arxiv.org/abs/2603.02765): we train our world model entirely in latent representations, which is faster than pixel space
- [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics [Balestriero & LeCun, 2025]](https://www.arxiv.org/abs/2511.08544): a theorical paper, but this is the framework that we use to make self-supervised learning much more efficient and easy to do
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning [Assran et al., 2025]](https://www.arxiv.org/abs/2506.09985): we use the same method as theirs to train the world model using rollouts (you can see it in part 3 "V-JEPA 2-AC: Learning an Action-Conditioned World Model")

## Citations

Here are all the papers and scientific projects that we are using for miniworld:
- [ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning [Kempka et al., 2016]](https://arxiv.org/abs/1605.02097)
- [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics [Balestriero & LeCun, 2025]](https://www.arxiv.org/abs/2511.08544)
- [Muon: An optimizer for hidden layers in neural networks [Jordan et al., 2024]](https://kellerjordan.github.io/posts/muon/)
- [Attention Is All You Need [Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)
