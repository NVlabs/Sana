# MiniMax-H3 baseline

Pristine upstream inference, no acceleration technique. This is the denominator every
`minimax_h3` speedup is reported against.

## What runs

`gpu_infer.py` drives the upstream **Modular Diffusers** integration of MiniMax-H3
(diffusers PR #14355, branch `minimax-h3`), vendored here as `diffusers_src/`. MiniMax-H3 is
integrated as blocks only — there is no `DiffusionPipeline` half — so `ModularPipeline` is the
whole surface. Nothing else wraps it: no server, no scheduler daemon, no sglang runtime.

Untouched from upstream: eager DiT, released BF16/FP32 precision policy, default attention
backend, no cache, no sparsity, no `torch.compile`, no quantization, no offload.

## Placement

Everything is resident on one GB200: the 33B Omni-Transformer (61.7 GB bf16), the Qwen3-VL-32B
conditioner (62.1 GB) and both VAEs — 134 GB of weights in 186 GB, with the rest left for the
packed sequence. Nothing offloads, nothing streams, so the measured latency is pure compute. This
is also the plainest upstream usage: a bare `pipe.to(device)`.

hsg's `QOSMinGRES` makes a whole 4-GPU node the smallest allocation, so the job takes the node
exclusively while the baseline itself uses one card.

### Why not the documented two-card split

Upstream documents putting the conditioner on a second card through a `device_map`, which is the
right move on 80 GB cards. It cannot work here, and the reason is structural:
`ModularPipeline._execution_device` walks `self.components` in insertion order and returns the
device of the first module carrying an accelerate hook. `text_encoder` is the *first* entry of
`MINIMAX_H3_COMPONENTS`, so a `device_map` on it makes the whole pipeline resolve to that card —
latents, timesteps and layout all get built there while the DiT stays on the other one. Every
modular block reads `components._execution_device`, so the blocks assume a single execution device
throughout. Job 5812896 died exactly there, in `MiniMaxH3LoopDenoiser`, on `cuda:1` vs `cuda:0`.

Making the split work needs a patch, and a baseline must not carry one. `--encoder-device` is kept
for smaller cards where the split is unavoidable and a patch is acceptable.

**There is no context parallelism here, and that is a property of upstream, not a choice.**
`MiniMaxH3Transformer3DModel` declares no `_cp_plan`, so diffusers cannot shard the packed
sequence across cards — even though `MiniMaxH3AttnProcessor` already carries `_parallel_config`
and forwards it to `dispatch_attention_fn`. Wiring Ulysses across the 4 cards is optimization
line #1 and is measured against this file.

## Workload

Matches the published SGLang MiniMax-H3 benchmark cell, so the numbers are comparable to the
vendor's own table:

| | |
|---|---|
| task | `t2va` (text → video + stereo audio) |
| resolution | 1344×768 |
| frames | 124 @ 24 fps (5.167 s) |
| denoising steps | 50 |
| flow shift (video / audio) | 12.0 / 3.0 — from the released scheduler configs |
| prompt | `prompts/t2va_example_1.json`, the official reproducible-768p T2VA H3-Context-IR output |
| seed | 0 |
| requests | 1 warmup + 1 measured |

The checkpoint is guidance-distilled, so there is no CFG: every step is exactly one forward pass
and there is no `guidance_scale` to set.

## Timing

`request_s` is the authoritative number — one `pipe(...)` call with both models resident, CUDA
synchronized on both sides. The sub-intervals are real device time from CUDA events wrapped around
the components themselves, not host intervals: `denoise_gpu_s` sums the DiT evaluations,
`video_decode_gpu_s` / `audio_decode_gpu_s` the two VAE decodes, `encode_gpu_s` the conditioner.
They do not sum to `request_s` — packing, layout and scheduler work sit between them on the host.

Instrumentation is attached only after warmup, so autotuning never lands in the numbers.

## Checkpoint

The released repository is *not* in the diffusers layout, so it is converted once by
`../convert_h3.sbatch` (the conversion script ships inside `diffusers_src/scripts/`):

- only the **FL2VA** half is converted — it serves `t2va` and `fl2va`, and `MiniMaxH3Blocks` never
  touches `transformer_ref/`
- `text_encoder`, `tokenizer` and `processor` are used as released and symlinked rather than
  copied, so the 62 GB conditioner is stored once
- the transformer is streamed shard by shard, peak RSS ≈ one shard (~4.9 GiB)

## Files

| File | Role |
|---|---|
| `gpu_infer.py` | The baseline itself: placement, request loop, CUDA-event timing, JSON result. |
| `scripts/run_minimax_h3_gpu.sh` | Registered runtime entrypoint, env-driven, `OUT_DIR` injected by the launcher. |
| `diffusers_src/` | Vendored diffusers @ `minimax-h3`, pinned in `SOURCE_SNAPSHOT.json`. |

Job scripts are not tracked here: their account, partition and QoS are
site-specific and useless elsewhere. `docs/simple-launch.md` shows the four-line
wrapper to write for your own scheduler.
