# MiniMax-H3 on H100

This self-contained runtime runs the released BF16 FL2VA checkpoint at
1344x768, 124 frames, and 50 steps on four H100-80GB GPUs. It uses SGLang FSDP
inference plus Ulysses-4, with no offload and no `torch.compile`.

`gpu_infer.py` calls SGLang's process-local `DiffGenerator`, `registration.py`
verifies the pinned upstream source before registering `model.py`, and the
installed SGLang checkout is never patched or modified. This directory does
not import implementation or policy code from another hardware runtime.

## Profiles

| Profile | Attention | Cache |
|---|---|---|
| `dense` | Official SGLang dense backend | None |
| `quality` | Sol-Attn, tau 0.5, diag, 15 dense steps | EasyCache 0.30, retain 10, max hit 1 |
| `balanced` | Sol-Attn, tau 1.0, diag, 10 dense steps | EasyCache 0.30, retain 6, max hit 2 |
| `aggressive` | Sol-Attn, tau 1.0, diag, 10 dense steps | FirstBlockCache 0.08 |
| `fullopt_exact` | Sol-Attn, tau 1.0, exact, 10 dense steps | FirstBlockCache 0.08 |

Every sparse profile leaves the first two transformer layers dense. The whole
multimodal prefix is an exact KV sink, its query rows are recomputed densely,
and token reordering is disabled. This runtime requires SM90 and selects the
`cute_sm90` Sol-Attn backend. Profile definitions are locked in `profiles.py`;
a conflicting environment override fails instead of changing a run silently.

## Environment

The candidate manifests pin:

- `lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86`
- PyTorch `2.11.0+cu130`
- Triton `3.6.0`
- MiniMax-H3 revision `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`

Set `H3_STORAGE_ROOT` to a shared path containing this checkout. Set
`H3_MODEL_PATH` to a local MiniMax-H3 root or FL2VA directory for offline
clusters; otherwise the Hugging Face repository in the manifest is used. The
runner supports `pyxis`, `singularity`/`apptainer`, and `none`.

## Run

```bash
python scripts/launch_candidate.py \
  candidates/minimax_h3_h100_fullopt_exact.toml \
  --mode sbatch --confirm-submit \
  --env H3_STORAGE_ROOT=/shared/path/Sana
```

Select `dense`, `quality`, `balanced`, `aggressive`, or `fullopt_exact` in the
candidate filename. Site-specific Slurm account and partition are omitted.
Each run writes only `out.mp4`, `benchmark.json`, and `run.log` under its output
folder. `benchmark.json` uses SGLang's inference time and peak-memory fields and
also records the selected backend, warmup routing density, and cache reuse.

## Files

- `model.py`: pinned SGLang MiniMax-H3 model with attention/cache hook points.
- `adapter.py`: H100 packed-sequence Sol-Attn policy and full-prefix sink.
- `easycache.py`, `first_block_cache.py`: collective cache controllers.
- `registration.py`: source verification and process-local model registration.
- `gpu_infer.py`: warmup plus measured offline generation.
- `scripts/run_minimax_h3_gpu.sh`: native/container launcher.
