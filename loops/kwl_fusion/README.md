# Dimension: kwl_fusion - kernel and operator optimization

A search dimension for kernel and operator fusion experiments. Native subagents
should read `search_space/05_kernel_fusion.md`, then inspect and modify target-model
hot inference paths directly in their isolated worktree.

## What it searches

`efficiency/transforms/kwl_fusions.py` registers **`kwl_fusions`**, a
build-time diagnostic helper that can emit `SGLANG_HQ_KWL_*` build flags. This
helper can be used for historical context, but env flags alone are not valid
KWL candidates. `dimension.toml` records search axes only; it does not define a
fixed flag bundle to copy.

KWL means guarded implementation optimization. Candidate mechanisms include
GEMM epilogues, norm/modulation fusion, attention-adjacent dense fusion,
layout/copy elimination, launch batching, stream overlap, decode/postprocess
fusion, custom kernels, and quality-gated approximate operator paths.
Framework backend selection, SDPA backend swaps, FlashAttention/FlashInfer
dispatch switches, and env-flag-only bundles are not valid startup candidates.
If a previous local status or journal contains backend-selection work, mark it
stale/cancelled instead of resuming it.

## Exploration Mode

Do not wait for a predeclared kernel-fusion seam. Inspect actual hot operator
chains, module construction, and compile behavior, then implement the candidate
where it is easiest to prove OFF identity and warm-speed benefit.

Prefer lossless kernel-level work first: fused pointwise chains around existing
GEMMs and norms, launch batching, layout/copy/allocation removal, custom
epilogues, and static metadata/workspace reuse. Run a module-level or
DiT-block-level warm paired microbench before any full denoising/video
generation run. OFF baseline and ON candidate timing must run in the same
process, same Slurm allocation/GPU, same warmed cache state, same tensors, and
same dtype.
Only after local module/kernel candidates are exhausted should the agent try
stable-region compile, regional compile, or CUDA graph capture.

KWL is treated as operator-only: it must not change scheduler, step count,
prompting, guidance, LoRA state, resolution, frame count, token set, attention
semantics, cache/prune semantics, or quantization policy. OFF is expected to be
identity; ON may be non-bit-exact because fused or approximate kernels can
change floating-point behavior, so side-by-side visual review is required for
retention and speed-target selection. If final visual validation fails after a
passing microbench, assume a kernel/module bug first rather than harmless
numeric drift.
Full denoising is a visual sanity and gross-regression check for KWL speed; do
not judge sub-percent KWL speedups from one candidate full run against a
historical canonical baseline. Use the warm paired DiT/module median and the
expected full-contribution estimate as the primary speed evidence.

KWL uses the same fixed-budget frontier rule as other dimensions. A candidate is
retained when latency improves, peak memory improves, aligned quality improves,
or reliable numeric stability improves. Bit-exactness is useful ranking and risk
metadata, not the default promotion requirement.

Required preflight before the first runnable candidate:

- hot-path evidence for the operator chain being fused;
- launch count, memory traffic, dtype, shape, and kernel availability;
- microbench command, shape, paired OFF/ON median/p25/p75/min/max warm latency,
  tensor diff, expected full contribution, and JSON result;
- cold/warm compile or graph-capture state only after module-local candidates
  are exhausted;
- OFF identity proof for the guarded baseline path;
- fallback behavior for unsupported shapes, dtypes, or backend failures.

For `hunyuan_diffusers`, concrete DiT candidates include attention Q/K
projection output + QK RMSNorm + RoPE, packed latent/text QKV projections,
single-stream `cat(attn, mlp) -> proj_out -> gate -> residual`, dual-stream
attention and FFN gate/residual epilogues, `LayerNorm -> scale/shift`, attention
output split/projection, final output projection/layout, and static mask or
RoPE/layout descriptor construction.

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
