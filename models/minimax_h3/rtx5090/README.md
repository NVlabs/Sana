# MiniMax-H3 on one RTX 5090

This directory is the RTX 5090 hardware port of MiniMax-H3. It is a sibling of
`../gb200/` and `../gb10/`; the early `gb200/optimized/` name predates the
hardware split and should not be copied as a new top-level model variant.

The port runs the released FL2VA model in BF16 on one 32 GiB RTX 5090. It does
not replace model weights or modify the driver, system Python, or the checked-out
SGLang source. Source changes are installed in a SHA-guarded Python overlay.

## Profiles

| Profile | Attention | Cache | Compile | Video VAE |
| --- | --- | --- | --- | --- |
| `dense` | SGLang dense backend | off | off | layerwise offload |
| `sol` | SM120 Sol-Attn | off | off | layerwise offload |
| `fullopt` | SM120 Sol-Attn | TeaCache 0.10 | regional | full BF16 residency after denoise |

All three profiles use SGLang's fast fused QK-norm/RoPE path when its runtime
capability check passes. All three also use the same single-rank placement:
DiT, text encoder, and VAE are layerwise-offloaded, with one prefetched DiT
layer and zero resident DiT layers. This is component/layer offload, not expert
offload.

The Sol-Attn policy is fixed across `sol` and `fullopt`:

- `tau=1.0`, `diag` threshold, no token reorder;
- first 10 denoise forwards and first 2 DiT blocks dense;
- valid text K/V rows form an exact sink;
- valid text query rows are recomputed densely;
- the real-QKV correctness gate is enabled;
- one complete 50-step, same-shape request warms compilation and autotuning;
  only the following request is measured.

## Source contract

The patches target SGLang commit
`6fa3f9df11c8bdbc0e3b4ddc87a3d873343aca72`. The launcher verifies the three
upstream source hashes before constructing an overlay, so a changed SGLang
checkout fails closed instead of silently applying stale model glue.

The measured environment was:

- one RTX 5090, SM120, 32 GiB;
- Python virtual environment with PyTorch `2.11.0+cu130` and Triton `3.6.0`;
- NVIDIA CuTe DSL / CUTLASS Python and `cuda-python` for the SM120 kernel;
- 188 GiB host RAM;
- released `MiniMaxAI/MiniMax-H3` FL2VA layout at `model/FL2VA`.

Install this repository's Sol-Attn package into the same environment:

```bash
uv pip install --python "$H3_ROOT/.venv/bin/python" \
  -e techniques/sparse_backends
```

The SGLang console command must come from the pinned checkout's environment.
Do not update the GPU driver as part of this setup.

## Run

The checked-in candidates record the original host paths. On another machine,
override `H3_ROOT`, `H3_SGLANG_ROOT`, `H3_MODEL_PATH`, and `PYTHON_BIN`.

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_rtx5090_dense.toml \
  --mode local

python scripts/launch_candidate.py \
  candidates/minimax_h3_rtx5090_sol.toml \
  --mode local

python scripts/launch_candidate.py \
  candidates/minimax_h3_rtx5090_fullopt.toml \
  --mode local
```

The runtime can also be called directly:

```bash
H3_ROOT=/path/to/h3-runtime \
H3_SGLANG_ROOT=/path/to/sglang \
H3_MODEL_PATH=/path/to/FL2VA \
H3_RTX5090_PROFILE=fullopt \
OUT_DIR=/path/to/output \
bash models/minimax_h3/rtx5090/scripts/run_minimax_h3_gpu.sh
```

Each output contains the warmup and measured videos, API metadata,
`server.log`, one-second `resources.csv`, Sol-Attn/TeaCache events, and
`summary.json`. Quote `final_status.inference_time_s` and
`final_status.peak_memory_mb` from SGLang's API metadata; client wall time and
the `nvidia-smi` process sample are retained as separate diagnostics.

## Measured BF16 results

Official 1344x768, 5-second, 50-step T2VA requests were run through one
persistent server per profile. The three-prompt suite used aligned prompts and
seeds for dense and full-opt.

| Prompt | Dense E2E | Full-opt E2E | Speedup | Dense peak | Full-opt peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flamenco dancer | 1047.791 s | 271.129 s | 3.864x | 16,584 MB | 22,388 MB |
| Garage mechanics | 1051.134 s | 237.622 s | 4.424x | 16,568 MB | 22,380 MB |
| Official starship | 1066.377 s | 243.895 s | 4.372x | 17,132 MB | 22,390 MB |
| Mean | 1055.101 s | 250.882 s | **4.206x** | 16,761 MB | 22,386 MB |

On the seed-1101 single case, dense was 1045.409 seconds and Sol-Attn-only was
781.783 seconds, a 1.337x E2E speedup. The complete raw values and provenance
are in `results/bf16_5090_20260805.json`.

## Files

- `adapter.py`: packed H3 self-attention routing, text sink, dense text queries,
  request/step tracking, density logging, and correctness gate.
- `teacache.py`: residual reuse controller used only by `fullopt`.
- `patches/minimax_h3.py`: minimal DiT hooks for the adapter and TeaCache.
- `patches/denoising.py`: preserves user-requested DiT offload across compile
  warmup.
- `patches/minimax_h3_decoding.py`: makes the complete BF16 video VAE resident
  only after denoising, then restores layerwise offload.
- `scripts/launch_server.sh`: validates sources, builds the overlay, and starts
  the single-rank server.
- `scripts/run_minimax_h3_gpu.sh`: scoped server lifecycle, warmup, measurement,
  resource sampling, and summary generation.

The runtime intentionally contains no alternative checkpoint loader. Its
contract is the released BF16 FL2VA model only.
