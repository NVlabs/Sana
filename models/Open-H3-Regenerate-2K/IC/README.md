# Open-H3-Regenerate-2K-IC

Regenerates a low-resolution MiniMax-H3 draft (768P, 1344×768) at 1080P, 2K (2560×1440) or 4K (3840×2160) and keeps
the draft's faces, lip sync, lettering and layout. The draft stays in the context of MiniMax-H3's Ref2VA transformer
(`transformer_ref`) as a reference video; an in-context (IC) LoRA on that transformer generates the high-resolution
version of the same clip.

[Project page](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/#ic) ·
[Weights](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC)

```
draft ──► H3 VAE latent ──┬──────────────────────────────────────────────► reference video ──┐
                          └──► latent upscaler ──► noised to σ 0.835 ──► target video ───────┴──► 3 steps ──► fast H3 VAE decoder ──► 2560×1440
                                                                  (transformer_ref + HyperFlow 8-step + LMS Sharp + IC LoRA, CFG 1)
```

## Install

Linux, Python 3.11, PyTorch with CUDA, Diffusers from `main` (MiniMax-H3 pipelines), `ffmpeg` on `PATH`:

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, torch 2.11.0+cu130, diffusers 0.40.0.dev0, transformers 5.6.0, kernels 0.14.0 (FlashAttention-3
hub kernel on Hopper) and PyAV 16.1.0.

## Weights

```bash
python download.py --out ./weights
```

| Folder | Content |
|---|---|
| `lora/` | IC LoRA (rank 384, 312 linears of `transformer_ref`), HyperFlow 8-step LoRA, LMS Sharp v1.0 LoRA |
| `latent_upsampler/` | 3D-conv latent upscaler (start point of `up3` / `up4`) |
| `vae_fast/` | fast H3 VAE decoder v2 (non-causal, 24 layers), distilled from the H3 VAE decoder as in StreamRender-H3 |
| `assets/` | silent audio latent (the audio rows of the Ref2VA sequence) |
| `base_model/` | MiniMax-H3 Diffusers components, unmodified copy of revision `42ed227`: `transformer_ref`, VAE, Qwen3-VL text encoder, tokenizer, processor, scheduler (`--base-from-minimax` fetches them from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) instead) |
| `sharp/` | LMS Sharp's `[1025, 8]` `adaln_t_table`, read by byte range from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6) |

## Run

```bash
torchrun --nproc_per_node 8 infer.py --weights ./weights --input draft.mp4 --output draft_2k.mp4 \
    --caption "the MiniMax-H3 prompt the draft was generated from"

# several clips in one run (each model is loaded once): one {"input", "output", "caption"} per line
torchrun --nproc_per_node 8 infer.py --weights ./weights --jobs jobs.jsonl

# 4K (3840x2160)
torchrun --nproc_per_node 8 infer.py ... --resolution 4k

# 8 HyperFlow steps from noise instead of the upscaled draft
torchrun --nproc_per_node 8 infer.py ... --sampler hf8

# a 384P draft (672x384) -> 768P
torchrun --nproc_per_node 8 infer.py ... --resolution 768p --lq 384
```

| Option | Values |
|---|---|
| `--resolution` | `2k` 2560×1440 (default) · `1080p` 1920×1080 · `4k` 3840×2160 · `768p` 1344×768 (with `--lq 384`, `512` or `576`) |
| `--sampler` | `up3` (default) · `up4` · `hf8`; a comma-separated list runs each and adds the sampler name to the output name |
| `--sharp` | LMS Sharp strength, default 1.0 (`config.json`); 0 turns it off |
| `--vae` | `fast` (default): fast H3 VAE decoder · `standard`: released H3 VAE decoder (FP32, tiled) |
| `--seed` | VAE posterior sample and sampling noise (default 0) |
| `--save-latent` | also writes `<output>.latent.pt` (normalized latent) |

The transformer is sharded over the GPUs (FSDP) with Ulysses context parallelism, so the world size must divide its
56 heads: 1, 2, 4, 7 or 8. Rank 0 encodes every draft (H3 VAE, Qwen3-VL) before the transformer is loaded; the
outputs are decoded in parallel over the ranks. The fast decoder runs its FlashAttention-4 / Triton / CUDA-graph
engine on Blackwell and plain PyTorch elsewhere; a clip too large for the plain engine on the GPU is decoded by the
released H3 VAE.

**Input.** The draft is read at 24 fps, 124 frames (at least 113 real frames; the tail repeats the last frame),
cover-resized and center-cropped to 1344×768 (or the `--lq` size). The output has the draft's length and audio.

**Prompt.** Pass the MiniMax-H3 prompt that generated the draft as `--caption`, unchanged. It may also be empty.

## Samplers

All samplers run CFG 1 on the HyperFlow 8-step sigma grid (video shift 12):

| `--sampler` | Start | Sigmas |
|---|---|---|
| `up3` (default) | latent upscaler(draft), noised to σ 0.835 | 0.834942, 0.696852, 0.468753, 0 |
| `up4` | latent upscaler(draft), noised to σ 0.923 | 0.923077, 0.834942, 0.696852, 0.468753, 0 |
| `hf8` | noise | 1, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0 |

The start state is `(1 - σ₀) · upscaled + σ₀ · noise`. The reference rows hold `0.999 · draft latent + 0.001 · noise`
at timestep `max(1 - σ, 0.999)`, the audio rows a fixed silent latent at timestep 1.

## LoRAs

`infer.py` merges the HyperFlow 8-step LoRA and then LMS Sharp (at `--sharp`) into `transformer_ref` at load time and
applies the IC LoRA unmerged (`alpha = rank = 384`, scale 1). For runtimes that load one LoRA per model,

```bash
python merge_lora.py --weights ./weights
```

writes the IC LoRA and Sharp as one LoRA (`lora/open_h3_regenerate_2k_ic_merged_lora.safetensors`, factors
concatenated along the rank axis, so it equals the two adapters exactly) plus Sharp's curve branches
(`lora/sharp_curve_branches.safetensors`); HyperFlow is applied before it, separately.

## Code

| File | Content |
|---|---|
| `infer.py` | encode → load → sample → decode for one clip or a jobs file |
| `download.py` | fetches the weights above |
| `merge_lora.py` | one LoRA from the IC LoRA and LMS Sharp |
| `ref2va_ic/media.py` | draft reading (24 fps, LQ canvas) and video writing with the draft's audio |
| `ref2va_ic/encode.py` | H3 VAE encode / decode, Qwen3-VL prompt embeddings with the draft as `<Video 1>` |
| `ref2va_ic/layout.py` | packed Ref2VA sequence `[context, reference video, silent audio, target video]` and row timesteps |
| `ref2va_ic/hyperflow.py`, `sharp.py`, `lora.py` | the three adapters |
| `ref2va_ic/cp.py` | Ulysses context parallelism and the per-step transformer call |
| `ref2va_ic/upscale.py`, `upscaler_node.py` | the author latent upscaler outside ComfyUI (`upscaler_node.py` unchanged, MIT) |
| `ref2va_ic/fast_vae/` | fast H3 VAE decoder: the modules of `../open_h3/fast_vae/`, with the v2 weights (`vae_fast/fast_h3_vae_v2_24l.safetensors`) and an automatic engine choice |

## Acknowledgements

- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe): Ref2VA
  transformer, VAE and text encoder.
- [HyperFlow](https://github.com/Video-Rebirth/hyperflow) ([adapter](https://huggingface.co/videorebirth/hyperflow/tree/5bde94d7de3e78dafa2abc43ee2f280a1cb244e8)):
  the 8-step sigma grid and LoRA.
- [LMS Sharp](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI) by Alissonerdx.
- [H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) by LBH-123-AI.
- [StreamRender-H3](https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/): the fast H3 VAE decoder.
- [Open-H3-Regenerate-2K](../), [SoL-Refiner](https://nvlabs.github.io/Sana/Sol-Refiner/) and
  [SoL-Engine](https://nvlabs.github.io/Sana/Sol-Engine/).

## License

The weights follow the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)
(see the weights repository's `LICENSE` and `NOTICE`). `ref2va_ic/upscaler_node.py` is MIT
(`ref2va_ic/LICENSE-upscaler-node.txt`). Open-H3-Regenerate-2K-IC is not an official MiniMax product.
Powered by MiniMax H3.
