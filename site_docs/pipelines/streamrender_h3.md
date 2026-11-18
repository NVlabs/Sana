# StreamRender-H3

StreamRender-H3 is an end-to-end runtime that connects a controllable Three.js game engine to a resident, multi-GPU H3 streaming renderer and browser video feedback. Semantic engine frames are encoded with a streaming TAE, rendered by H3 with two denoising evaluations per chunk, and decoded causally. Code and full docs: [`models/streamrender_h3`](https://github.com/NVlabs/Sana/tree/sol-engine/models/streamrender_h3).

## Runtime profile

- Engine input and render output: 1344 x 768, 24 fps.
- 2 denoising evaluations, CFG=1; no fixed final video duration.
- One active interactive session per worker allocation.
- Local launch on 8 GPUs in one NVLink domain; the Slurm profile requests two nodes x four GPUs.

## Entry points

| Entry | Role |
|---|---|
| [README](https://github.com/NVlabs/Sana/tree/sol-engine/models/streamrender_h3) | Assets, environment, local and Slurm launch |
| [Reactor runtime](https://github.com/NVlabs/Sana/tree/sol-engine/models/streamrender_h3/reactor) | Browser demo for recorded semantic videos, references and prompts |

## Launch

```bash
cd models/streamrender_h3
GPUS=8 PYTHON_BIN=python bash scripts/launch_local.sh \
  --assets configs/assets.local.json
```

Then open `http://127.0.0.1:5191`. Weights: [Efficient-Large-Model/StreamRender-H3](https://huggingface.co/Efficient-Large-Model/StreamRender-H3).

## Project page

- [StreamRender-H3](https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/)
