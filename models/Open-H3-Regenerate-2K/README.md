# Open-H3-Regenerate-2K

An open replacement for the closed H3-Regenerate-2K API. A MiniMax-H3 768P draft (1344×768, 121 frames, 24 fps) is
upscaled in latent space and regenerated at **2560×1440** with **three** denoising steps (`--steps 2` for two).

[Project page](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/) ·
[Weights](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)

```
768P H3 latent ──► latent upscaler (whole latent, 2×) ──► 3-step refine (MiniMax-H3 + merged LoRA) ──► VAE ──► 2560×1440
```

The video is decoded by the fast H3 VAE decoder of [StreamRender-H3](https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/)
(non-causal, 24 layers, distilled from the released H3 VAE) by default with the two SoL-Engine forwards;
`--vae standard` uses the released H3 VAE decoder.

## Forwards

Three forwards share the same pipeline (`--fwd`):

| `--fwd` | What runs |
|---|---|
| `sol_engine` (default) | `sol_engine_fwd`: strided cuDNN + `torch.compile`, Sol-Attn sparse attention, MXFP8 GEMMs with fused SwiGLU quantization, resident stage weights, one CUDA graph per step; refine step 0 dense, steps 1–2 sparse (tau 1.0) |
| `sol_engine_fast` | `sol_engine_fwd` with every refine step sparse (tau 1.75): the fastest |
| `standard` | `standard_fwd`: Diffusers MiniMax-H3 blocks, dense cuDNN attention, BF16 weights; the unaccelerated reference |

Latent upscale + 3-step refine of one 5 s clip on one GB200, warm, median of 3 runs, VAE decode not included:

| `--fwd` | Upscale + refine | Speedup |
|---|---:|---:|
| `standard` | 73.11 s | 1.00× |
| `sol_engine` (default) | 40.61 s | 1.80× |
| `sol_engine_fast` | 20.46 s | 3.57× |

Speed rises from `standard` to `sol_engine` to `sol_engine_fast`, while closeness to the `standard` output on fine
detail goes the other way: `sol_engine` keeps the first refine step dense and stays nearer to `standard`;
`sol_engine_fast` trades a little of that for another 2× in speed. The two presets differ only in the refine stage
(`SPARSE` in `open_h3/sol_engine_fwd.py`); the HyperFlow draft stage is the same.

## Speed of `sol_engine_fast` on one B200 (GB200)

One 5 s clip, 768P latent → 2560×1440 video, warm, median of 3 runs, one GPU:

| | Latent upscale | Refine (3 steps) | VAE decode | Total |
|---|---:|---:|---:|---:|
| `standard_fwd`, released VAE | 1.01 s | 71.27 s | 14.99 s | **87.26 s** |
| `sol_engine_fast`, released VAE | 1.21 s | 19.65 s | 16.48 s | 37.39 s (2.3×) |
| `sol_engine_fast`, fast H3 VAE decoder | 1.05 s | 19.75 s | 1.58 s | **22.45 s** (3.9×) |

Total = median of upscale + refine plus median of VAE decode, so the columns need not add up exactly. With the fast
decoder, two other prompts took 22.41 s and 22.23 s; its first call for a new latent shape builds the engine and
records a CUDA graph (39 s). Per module, latent upscale + 3-step refine (each row adds one module to the row above):

| Configuration | Upscale | Refine (3 steps) | Total | Speedup |
|---|---:|---:|---:|---:|
| `standard_fwd` | 1.06 s | 72.05 s | **73.11 s** | 1.00× |
| + strided cuDNN, `torch.compile` | 1.11 s | 65.98 s | 67.08 s | 1.09× |
| + Sol-Attn sparse attention | 1.08 s | 25.04 s | 26.11 s | 2.80× |
| + MXFP8 linear layers | 1.12 s | 21.64 s | 22.77 s | 3.21× |
| + SwiGLU fused into MXFP8 quantization | 1.16 s | 21.33 s | 22.49 s | 3.25× |
| + CUDA graphs = `sol_engine_fast` | 0.89 s | 19.57 s | **20.46 s** | **3.57×** |

Not included in either table: the 768P draft, model loading, text encoding and MP4 writing. With HyperFlow and the
refine stage sharing one transformer, switching the LoRA set costs 1.2 s per request in `standard_fwd`; resident
stage weights make it free in `sol_engine_fwd`. Repeats: 73.70 s for `standard_fwd` on another GPU, 20.37 s for
`sol_engine_fast` on another prompt.

## Install

Linux, Python 3.11, PyTorch 2.11 + CUDA 13.0:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -m pip install -e ../../techniques/sparse_backends        # Sol-Attn, needed by sol_engine_fwd
```

## Weights

```bash
python download.py --weights ./weights --hyperflow
```

| Folder | Content |
|---|---|
| `lora/` | merged LoRA (refiner + LightX2V Turbo + LMS Sharp, exact), Sharp curve branches, `adaln_t_table.bin` |
| `latent_upsampler/` | author 3D-conv latent upscaler |
| `vae_fast/` | fast H3 VAE decoder (non-causal, 24 layers) from StreamRender-H3 |
| `base_model/` | MiniMax-H3 Diffusers components, unmodified copy of revision `6818f6c` (`--base-from-minimax` fetches them from the official repo instead) |
| `hyperflow/` | optional HyperFlow 8-step adapter, for prompt → 768P |

The merged LoRA concatenates the three adapters along the rank axis with their strengths folded in, so it equals the
separate adapters exactly (relative error 2.2e-8 over all 312 linears). `tools/merge_adapters.py` rebuilds it.

## Run

```bash
# prompt -> HyperFlow 768P draft -> 2560x1440
python infer.py --weights ./weights --hyperflow --prompt "A red fox trots through fresh snow ..." --output fox.mp4

# a draft video: encoded with the H3 VAE at 1344x768 (default) or 1024x576 (--input-size 576p)
python infer.py --weights ./weights --input-video draft.mp4 --prompt "..." --output out.mp4

# an existing normalized H3 latent, torch.save({"video_latents": z}), z of shape [24, 37, 48, 84] (768P) or [24, 37, 36, 64] (576P)
python infer.py --weights ./weights --input-latents native.pt --prompt "..." --output out.mp4

# the fastest forward
python infer.py ... --fwd sol_engine_fast

# unaccelerated forward and the released VAE decoder
python infer.py ... --fwd standard

# the released VAE decoder with a sol_engine forward
python infer.py ... --vae standard

# other output sizes: 768p (refine only), 1080p, 2k (default), 4k (two 2x upscales via 2K)
python infer.py ... --resolution 4k

# a different start noise; the steps follow the same shift-6 rule
python infer.py ... --start-sigma 0.8

# the project page's resolution ladder: refine the 768P draft at 768P, upscale that result and refine it again, then refine
# once more at the same size (a latent already on the --resolution grid is not upscaled)
python infer.py ... --fwd standard --resolution 768p --start-sigma 0.7 --save-latents --output dance-768p.mp4
python infer.py ... --fwd standard --input-latents dance-768p.latents.pt --resolution 4k --start-sigma 0.7 --save-latents --output dance-4k-pass1.mp4
python infer.py ... --fwd standard --input-latents dance-4k-pass1.latents.pt --resolution 4k --start-sigma 0.6 --output dance-4k.mp4

# two refine steps from a lower start noise (0.7) instead of three
python infer.py ... --steps 2
```

With `--input-video`, the draft is resampled to 24 fps, scaled to cover the input size (bilinear) and center-cropped,
cut or padded (last frame held) to 121 frames plus 3 copies of the last frame (124 frames = 37 latent frames), then
encoded by the FP32 H3 VAE with ImageNet pixel normalization; the posterior is sampled with `--vae-seed` (default 42).
Use 768P for H3 drafts and 576P for drafts from other video models, as in our comparisons. The VAE is loaded and freed
before the transformer.

`--warmup N --runs M` repeats the request; per-stage seconds (`upscale`, `merge_refine`, `refine3` or `refine2`,
`vae_decode`, and `upscale+refine3` / `upscale+refine2`) are printed and written to `<output>.json`. The first
`sol_engine_fwd` requests compile kernels and capture CUDA graphs; measure after warm-up. A change of prompt length
re-captures the graphs.

Sampling: Euler, CFG 1, initial state `(1 - σ₀) · upscaled + σ₀ · noise`, model time `1 − σ`, fixed encoded silence
for the audio stream. Sigmas (`open_h3/sigmas.py`):

| `--steps` | Sigmas | Upscale + refine (`sol_engine_fast`, one GB200, warm median) |
|---|---|---:|
| 3 (default) | `0.909375, 0.857143, 0.666667, 0` | 20.46 s (37.39 s with the released VAE decode) |
| 2 | `0.7, 0.494118, 0` | 14.1 s (30.1 s with the released VAE decode) |

The fast H3 VAE decoder takes 1.6 s for either.

The 2-step grid follows the refiner's original shift-6 rule (even steps in unshifted σ, then `6σ / (1 + 5σ)`; from
0.909375 it gives the original 4-step grid) started at 0.7, so less of the upscaled draft is re-noised.
On our check prompts it is about 30% faster but visibly softer on fine texture (fur, moss) than 3 steps,
which stays the default.

## Notes

- `sol_engine_fwd` is lossy (sparse attention, MXFP8); the timing table's quality was checked by eye on sample
  prompts. Same-seed PSNR is not a usable metric for this model: rounding-level changes move the 3-step output a lot.
- This code was validated on one GB200 against the measured reference runs (three separate adapters): the 768P
  draft and the upscaled latent are bit-identical; the refined latent differs by 1.4% (relative L2) for
  `standard_fwd` and 5–12% for `sol_engine_fwd`, the latter within the spread between reference configurations
  that differ only by kernel rounding. This check used the `sol_engine_fast` setting, which took 20.2–20.6 s for
  upscale + refine.
- `--input-video` gives the same latent, bit for bit, as the reference preprocessing the comparison inputs were made
  with (checked on a Wan 2.1 draft at 576P), and was run end to end on an H3 768P draft.
- `--steps 2` was run on the same GPU type and inputs (both forwards); `--steps 3` output is bit-identical before
  and after the option was added. Upscale + 2-step refine took 14.1–14.6 s with `sol_engine_fast`.
- The fast H3 VAE decoder is lossy relative to the released decoder: on a refined fox latent, frame PSNR between
  the two decoders is 30.0 dB; its deployment engine matches its plain PyTorch decoder (`engine="ref"`) at 53.3 dB.
  It needs FlashAttention-4 and quack (`requirements.txt`); its GEMMs run without autotuning.
- One GPU needs about 150 GB for the SoL-Engine forwards with CUDA graphs (B200 / GB200). Text encoding loads the
  32B Qwen3-VL text encoder once before the DiT and frees it.

## Acknowledgements

- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/6818f6c32d12b210915e44ad56a4228c2608f160): base model, VAE and text encoder.
- [H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  ([weights](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/tree/3f941d5d182014dd5c0a5e16330420ee2d4aa0c6)):
  `open_h3/upscaler_node.py` is its inference node, loaded outside ComfyUI by `open_h3/upscaler.py`.
- [LightX2V FL2VA Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/3ec17a324ced54151364f24f8b5fb6bf7e26414f)
  and [LMS Sharp](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/tree/849fb3d4f434f2e9b4a4d4f7253471bf5e6719f9),
  merged into the LoRA; Sharp's `adaln_t_table` is read from
  [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6).
- [HyperFlow](https://github.com/Video-Rebirth/hyperflow) ([adapter](https://huggingface.co/videorebirth/hyperflow/tree/5bde94d7de3e78dafa2abc43ee2f280a1cb244e8)):
  `open_h3/hyperflow/`, the optional 768P draft generator.
- [StreamRender-H3](https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/): the fast H3 VAE decoder in `open_h3/fast_vae/`.
- [SoL-Refiner](https://nvlabs.github.io/Sana/Sol-Refiner/), [SoL-Engine](https://nvlabs.github.io/Sana/Sol-Engine/) and
  [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/).

## Validation

```bash
python -m unittest discover -s tests -v
```
