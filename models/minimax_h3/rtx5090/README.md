# MiniMax-H3 on one RTX 5090

Single-card BF16 inference for the released MiniMax-H3 FL2VA checkpoint. The
33B DiT, Qwen3-VL conditioner, and VAEs do not fit together in 32 GiB, so this
runtime uses SGLang's layerwise component offload and keeps one component on
the GPU at a time.

## What runs

`gpu_infer.py` calls SGLang's local `DiffGenerator`; there is no HTTP server or
client in the benchmark path. The dense profile uses the pinned SGLang
MiniMax-H3 implementation unchanged. Sparse profiles register the local
`model.py` through `ModelRegistry`, and full-opt registers the local pipeline
stage subclass in `pipeline.py`.

No installed SGLang file is edited, copied, or shadowed. `registration.py`
checks the upstream source hashes before installing process-local registry
entries, so changing the pinned SGLang checkout fails before model loading.

## Placement

All profiles use one RTX 5090 and the released BF16 FL2VA weights. The DiT,
text encoder, and VAEs use layerwise CPU offload. The DiT prefetches one layer
and keeps no layer permanently resident. This is layer/component offload, not
expert offload.

## Profiles

| Profile | Attention | Cache | Compile | Video VAE |
|---|---|---|---|---|
| `dense` | pinned SGLang dense backend | off | off | layerwise offload |
| `sol` | SM120 Sol-Attn | off | off | layerwise offload |
| `fullopt` | SM120 Sol-Attn | TeaCache 0.10 | regional | resident for decode |

Sol-Attn uses `tau=1.0`, diagonal thresholding, no token reorder, the first 10
denoising steps and first two DiT blocks dense, exact text K/V sinks, and dense
text-query rows. The dynamic attention backend runs outside Dynamo; projections,
QK-norm/RoPE, residuals, and MLPs remain eligible for regional compile. Dense
and Sol profiles use SGLang's native fused QK-norm/RoPE path when its capability
check passes. Full-opt uses the upstream compile-compatible QK-norm/RoPE branch
inside the regional graph.

Full-opt keeps the same attention policy, adds TeaCache, and materializes the
complete BF16 video VAE only after denoising. The VAE returns to layerwise
offload after decode.

## Workload

The registered candidates run one warmup followed by one measured request:

| | |
|---|---|
| task | `t2va` |
| resolution | 1344x768 |
| duration | 5 seconds at 24 fps |
| denoising steps | 50 |
| flow shift (video / audio) | 12.0 / 3.0 |
| prompt | `../prompts/t2va_example_1.json` |
| weights | released FL2VA BF16 |

Sparse profiles use a complete same-shape warmup so Sol-Attn autotuning and
regional compilation are excluded from the measured request. The authoritative
latency and peak allocation are written to `benchmark.json` from SGLang's
request metrics.

## Files

| File | Role |
|---|---|
| `gpu_infer.py` | Offline entrypoint, warmup, measured request, and benchmark JSON. |
| `model.py` | MiniMax-H3 DiT with Sol-Attn routing and TeaCache block reuse. |
| `adapter.py` | Packed attention policy, text sink, and route-density logging. |
| `teacache.py` | Request-local TeaCache controller. |
| `pipeline.py` | Compile/offload fix and post-denoise full-VAE residency. |
| `registration.py` | Source verification and process-local SGLang registration. |
| `scripts/run_minimax_h3_gpu.sh` | Environment and registered runtime launcher. |

## Running

Install this repository's sparse backend in the same environment as the pinned
SGLang checkout, then provide the checkpoint and Python paths:

```bash
export H3_MODEL_PATH=/path/to/MiniMax-H3/FL2VA
export PYTHON_BIN=/path/to/venv/bin/python

python scripts/launch_candidate.py candidates/minimax_h3_rtx5090_dense.toml \
  --mode local --env H3_MODEL_PATH="$H3_MODEL_PATH" --env PYTHON_BIN="$PYTHON_BIN"
python scripts/launch_candidate.py candidates/minimax_h3_rtx5090_sol.toml \
  --mode local --env H3_MODEL_PATH="$H3_MODEL_PATH" --env PYTHON_BIN="$PYTHON_BIN"
python scripts/launch_candidate.py candidates/minimax_h3_rtx5090_fullopt.toml \
  --mode local --env H3_MODEL_PATH="$H3_MODEL_PATH" --env PYTHON_BIN="$PYTHON_BIN"
```

Each run writes `out.mp4`, `benchmark.json`, and the launcher log under its
output directory. Set `H3_PROMPT` or `H3_PROMPT_FILE` to change the prompt
without changing the candidate policy.

## Environment

The measured runtime uses PyTorch `2.11.0+cu130`, Triton `3.6.0`, and SGLang
commit `6fa3f9df11c8bdbc0e3b4ddc87a3d873343aca72`. The SM120 Sol-Attn extension
also requires the matching CUDA/CuTe dependencies. The selected environment's
`bin` directory or `${H3_ROOT}/tools/ffmpeg-static` must provide `ffmpeg` and
`ffprobe`; use `H3_FFMPEG_BIN_DIR` for another location. The launcher adds the
selected directory to `PATH` and checks both tools before loading weights. Do
not change the GPU driver as part of environment setup.
