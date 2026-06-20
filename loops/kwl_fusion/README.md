# Dimension: kwl_fusion - kernel and operator optimization

A search dimension for kernel and operator fusion experiments. Native subagents
should read `search_space/05_kernel_fusion.md`, then inspect and modify target-model
hot inference paths directly in their isolated worktree.

## What it searches

`efficiency/transforms/kwl_fusions.py` registers **`kwl_fusions`**, a
build-time diagnostic helper that can emit `SGLANG_HQ_KWL_*` build flags. This
helper can be reused, replaced, or ignored by subagents. `dimension.toml`
records search axes only; it does not define a fixed flag bundle to copy.

KWL means guarded implementation optimization. Candidate mechanisms include
GEMM epilogues, norm/modulation fusion, attention-adjacent dense fusion, compile
or CUDA graph capture, layout/copy elimination, launch batching, stream overlap,
decode/postprocess fusion, backend selection, and quality-gated approximate
kernel/backend paths.

## Exploration Mode

Do not wait for a predeclared kernel-fusion seam. Inspect actual hot operator
chains, module construction, and compile behavior, then implement the candidate
where it is easiest to prove OFF identity and warm-speed benefit.

KWL is treated as operator-only: it must not change scheduler, step count,
prompting, guidance, LoRA state, resolution, frame count, token set, attention
semantics, cache/prune semantics, or quantization policy. OFF is expected to be
identity; ON may be non-bit-exact because fused or approximate kernels can
change floating-point behavior, so side-by-side visual review is required for
retention and speed-target selection.

KWL uses the same fixed-budget frontier rule as other dimensions. A candidate is
retained when latency improves, peak memory improves, aligned quality improves,
or reliable numeric stability improves. Bit-exactness is useful ranking and risk
metadata, not the default promotion requirement.

Required preflight before the first runnable candidate:

- hot-path evidence for the operator chain being fused;
- launch count, memory traffic, dtype, shape, and backend availability;
- cold/warm compile or graph-capture state when applicable;
- OFF identity proof for the guarded baseline path;
- fallback behavior for unsupported shapes, dtypes, or backend failures.

## Independent test

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/kwl_fusion/test_kwl_fusion.py
```

CPU-only; validates the transform through `efficiency.compose`, checks the full
KWL env bundle and checks a subset/ablation bundle.

## Run it in the search

```bash
python search/search.py --model <id>
```

The search reports this dimension and its loop contract. See `acceptance.md` for
frontier retention, final speed-target selection, and structured-negative
proposal requirements.
