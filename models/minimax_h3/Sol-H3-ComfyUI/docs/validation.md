# Validation · RTX 5090 · 2026-10-13

All three profiles generated the same eagle-over-an-alpine-lake prompt with
seed 42 on one RTX 5090. Single-stage H3, four updates, 1344×768, 121 frames,
24 fps, 5.042 seconds, with native stereo audio. BF16 and fast-VAE profiles also
completed the full browser-submitted ComfyUI graph and SaveVideo node without
browser errors.

| Profile | Request → MP4 | Video decoder stage | Session loading + warmup |
|---|---:|---:|---:|
| BF16 + original VAE | 103.89 s | 17.10 s | 233.41 s |
| FP8 + original VAE | 85.62 s | 17.96 s | 297.32 s |
| FP8 + noncausal VAE | 79.76 s | 10.38 s | 217.69 s |

Each entry is one formal request, not a dataset mean. The request clock includes
fresh Qwen conditioning, denoising, video/audio decode and MP4 writing. Model
loading, warmup and the additional ComfyUI SaveVideo operation are outside that
clock. The environment already had compilation caches. BF16 is the four-update
comparison arm, not an undistilled full-step baseline. The distilled noncausal
decoder may change quality; no quality equivalence is claimed.

## Original-H3 baseline (release comparison)

`Original H3 50-step baseline — RTX 5090` runs MiniMax-H3 without the FastH3
adapter on FastVideo's official 50-point `MiniMaxH3Scheduler` grid (49 DiT
forwards; guidance-distilled, CFG 1.0, as in `basic_minimax_h3_t2v.py`). It
shares the NVFP4 Qwen encoder, the original H3 VAE, CPU offload, FlashAttention
4 and regional compile with the Sol-H3 profiles, so the comparison isolates the
Sol-H3 DiT recipe. The original BF16 Qwen-32B encoder was not used because its
weights were not on the test host; this is stated wherever the baseline appears.

Release measurement, 2026-10-13: the eagle prompt above, seed 42, submitted
through the real ComfyUI UI (release tree, headless browser on the GPU host).
One unrecorded warmup request first; the table is the following formal request
with the model already loaded. Both GPUs are RTX 5090 in the same host; each
was idle (≤6 MiB) when admitted.

| Profile | GPU | Request → MP4 | Denoise | Video decode | Mux | DiT forwards |
|---|---|---:|---:|---:|---:|---:|
| Original H3 50-step baseline (BF16) | 0 | 988.95 s | 958.38 s | 16.93 s | 4.48 s | 49 |
| Sol-H3 FP8 + noncausal VAE | 1 | 79.07 s | 54.20 s | 10.55 s | 4.79 s | 4 |

Speedup 12.5×. Startup and warmup (outside these clocks) were 1027.7 s for the
baseline session. Single formal request per profile, not a distribution. The
baseline's own warmup request measured 989.12 s, consistent with the formal one.
The noncausal decoder is distilled and can change image quality; the baseline
uses the NVFP4 Qwen encoder as stated above.

## Evidence

- Measured receipts, with machine paths omitted:
  [BF16](benchmarks/bf16.json), [FP8](benchmarks/fp8.json),
  [noncausal VAE](benchmarks/fast.json). Each includes the source receipt hash.
- [Generated clips side by side](assets/comparison.mp4): silent comparison.
- [Actual browser recording](assets/comfyui-demo.mp4): frames sampled about
  every two seconds, encoded at original timestamps, with no speedup.
- [Completed workflow screenshot](assets/comfyui.png).

These GPU results were obtained before the package layout reorganization.
The reorganization preserves every runtime source/config byte; CPU checks cover
the relocated plugin, subprocess bridge and optional-decoder import path.
No new GPU performance result is inferred from those checks.

## Observed environment

Linux, Python 3.12.3, PyTorch 2.12.0+cu130, Triton 3.7.0, Transformers 5.12.1,
ComfyUI frontend 1.47.11. The optional decoder uses the previously delivered
Diffusers source and QuACK 0.5.3 overlay; the original runtime's QuACK 0.6.3
installation is unchanged. These observations are not a clean-install lockfile.

Other resolutions, durations, new-machine installation and B200 API serving
have not been validated by this integration.

## HyperFlow · 8 steps through Sol-H3 Studio (2026-10-14)

Same eagle prompt and seed 42 as the original-H3 baseline, submitted through the real
Sol-H3 Studio in ComfyUI 0.30 (release tree, headless browser on the GPU host), one
unrecorded warmup and one recorded formal request, RTX 5090 (GPU 0, idle when admitted).

| | Request → MP4 | HyperFlow pipeline | Qwen (NVFP4) | DiT forwards |
|---|---:|---:|---:|---:|
| Warmup (cold, model load 67.1 s) | 271.58 s | 190.01 s | 0.50 s | 8 |
| **Formal (model loaded)** | **193.59 s** | 184.24 s | 0.49 s | 8 |

Versus the original-H3 50-step baseline (988.95 s): **5.1×**. HyperFlow uses the official
HyperFlow blocks with the unmerged PEFT LoRA (videorebirth/hyperflow@5bde94d7,
sha256 9297f450…df447) on Diffusers 0.40.0; the 62 GB BF16 Qwen3-VL is replaced by the
NVFP4 conditioning the other profiles use, as in HyperFlow's own SplitPipeline; the DiT
streams one block group at a time. Page errors: none. The recording shows the prompt, the
encode stage, steps 1/8 to 8/8 and playback. An earlier standalone probe measured 177 s
because it excluded conditioning and worker overhead; the Studio number above is the claim.

Showcase first-frame clips in the film were rendered with the same engine on an H100
(promotional content, not a timing claim), from Qwen-Image-2.1 keyframes conditioned by the
official BF16 MiniMax-H3 conditioner.

## HyperFlow · 8 steps fast (2026-10-16)

Profile `Sol-H3 HyperFlow · 8 steps fast — RTX 5090`, the same eagle prompt and seed 42,
submitted to ComfyUI 0.30's `/prompt` API with the Studio's graph
(`scripts/hyperflow/e2e_comfy.py`). One warmup request (cold: startup and warmup 179.3 s),
then three formal requests with the model loaded. Second RTX 5090 host (32 GB, host RAM
capped at 90 GB), PyTorch 2.12.0+cu130, Triton 3.7.0, SageAttention 2.2.0 built for SM120.

| Formal request | Request → MP4 | Qwen (resident, NVFP4) | Denoise, 8 steps | Fast VAE decode | MP4 write |
|---|---:|---:|---:|---:|---:|
| 1 | 65.86 s | 3.26 s | 42 s | 8.94 s | 9.67 s |
| 2 | 67.31 s | 3.45 s | 42 s | 8.92 s | 10.10 s |
| 3 | **67.20 s** | 3.85 s | 42 s | 8.95 s | 9.72 s |

Median **67.2 s**, 14.7× the original-H3 50-step baseline (988.95 s, first host). The
request clock includes fresh Qwen conditioning, denoising, video and audio decode and MP4
writing. In earlier sessions with the same profile the MP4 write took 3.7 s (request → MP4
59.3–59.5 s); the x264 encode runs on the host CPU and varies with host load.

What the profile changes, each measured as an e2e step on this host (median of three):

| Step | Request → MP4 |
|---|---:|
| FP8 DiT + Sol-Attn sparse attention + resident Qwen | 125.1 s |
| + exact AdaLN table, SageAttention 2++ for dense steps, fast noncausal VAE | 78.8 s |
| + top-level DiT parameters kept on the card (no blocking copies per forward) | 68.2 s |
| + regional `torch.compile` of each block | 59.5 s (this run: 67.2 s, MP4 write) |

Quality. The AdaLN table is bit-identical to the official module (16/16 forward sets,
max abs difference 0). Keeping top-level parameters on the card does not change results
(45.9–52.9 dB against the previous configuration; same-configuration repeats 43.8–59.4 dB).
FP8, SageAttention and compilation are not bit-identical: with an 8-step sampler any numeric
change moves the clip to a different but equally plausible trajectory (about 20 dB PSNR
against the uncompiled run, also for an exact-vs-inexact FP8 rounding change). The fast
VAE alone measures 38.2 dB mean against the original VAE. Quality of the final profile was
judged visually on nine first-frame clips against the H100 BF16 reference: same composition,
subject and motion in all nine, differences in pose and timing, no artefacts. No metric
equivalence is claimed.

## Studio v2 on the second RTX 5090 host (2026-10-16)

Every request below went through ComfyUI's `/prompt` API with the Studio's graph
(`scripts/hyperflow/e2e_comfy.py`), seed 42, model loaded. Clips: `scripts/fetch_assets.py gallery`.

### One stage (8-step LoRA, Sol-H3 acceleration)

| Request | Resolution | Request → MP4 |
|---|---|---:|
| Text, eagle prompt | 768p (1344×768) | 59.5 s, 59.7 s (later sessions; MP4 write 3.8 s) |
| Text, eagle prompt | 480p (864×480) | 22.0 s, 22.0 s |
| First frame, castle | 768p | 74.5 s |
| First frame, castle | 480p | 35.0 s |
| Text, three gallery prompts | 768p | 66.3 – 69.2 s |

Text requests share the first-frame pipeline's components (one copy of the DiT in host RAM)
but run the official text-to-video blocks; the eagle clip matches the earlier separate-pipeline
output at 43.6 dB mean PSNR, within run-to-run variation (44–60 dB), i.e. the same trajectory.
480p is 864×480, the 16:9 canvas at a 480-pixel short edge (both axes multiples of 32).
The first request at a new resolution compiles that shape once (not counted above).

### Two stage (H3 draft + LTX refine)

NVlabs/Sana `sol-engine` `models/minimax_h3/Sol-H3-RTX5090` at 801f040, used unchanged through
`sol_h3_engine/two_stage_worker.py`; eagle prompt, seed 42, 1344×768, 121 frames with audio.

| Qwen lifetime | Request → MP4 (formal) | Qwen | Stage1 (H3 draft) | Stage2 (LTX refine, decode, mux) |
|---|---:|---:|---:|---:|
| Package default: loaded per request | 90.3, 91.4, 90.6, 90.5 s | 56 s load + 0.6 s | 19.7–20.1 s | 12.5–12.9 s |
| **Resident session** (`two_stage.resident_qwen`) | **35.5, 36.0, 35.3, 35.4 s** | 0.24 s | 21.1–22.2 s | 12.5–13.1 s |

Median **35.5 s** with a resident Qwen session, 27.9× the original-H3 baseline. The package
loads Qwen for every request and drops its file from the page cache afterwards; on this host
(90 GB RAM cap, stage1 + LTX resident ≈ 72 GB) the 15.7 GB checkpoint is reread from disk each
time. A resident session keeps it on the card, as the DGX Spark package does; the sampling
recipe is unchanged and the eagle output matches the per-request run at 39.3 dB mean PSNR.
Host RAM: stage1 peaks at 73 GB while loading BF16 and settles at 29 GB after FP8; with LTX
loaded the session holds 68–74 GB, no OOM. First session: 881 s load and warmup (LTX
autotune); later sessions reuse `TORCHINDUCTOR_CACHE_DIR`. Prompt-dependent: two gallery
prompts took 108 s per request with per-request Qwen (stage1 37 s).

DGX Spark two stage (released Sol-H3-Spark package, mean of three): 64.3 s.

### Reference images (one stage)

Two references (a portrait and a crystal-forest scene with a stag), 768p, seed 42: the clip
keeps the subject's look and places her in the second reference's scene with the stag.
Generation 97–108 s per request; references add conditioning rows to the sequence. Only one
FP8 DiT fits beside the rest in 90 GB of host RAM, so switching between reference and
text/first-frame requests swaps the checkpoint: 72 s into `transformer_ref`, 46 s back.
Consecutive reference requests do not swap (request → MP4 103.5 s).
