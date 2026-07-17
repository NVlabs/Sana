# Sol-Attn: colmask PISA2 for B200

This branch is the compact release of the `colmask` BF16 PISA2 forward backend
for NVIDIA B200 / SM100. The complete optimization history remains on the
project's history branch; this tree carries the promoted source, its import
closure, the public wrapper, tests, and release evidence.

## Result

- Correctness: 45/45.
- Versus Triton PISA2: GM 0.7462, 45/45 wins, worst 0.7987.
- Versus Triton PISA0: GM 1.0003 (statistical parity), 30/45 wins, worst
  1.1696 at `T32768-B4-H12-d0.05`.
- Fixed point: 0.714 ms.

Ratios are candidate/reference latency ratios. The kernel resources are
REG168, zero spill, TMEM256, six warps, and two CTA/SM. Its routing schedule is
logical G256 over physical N128 tiles. The backend is a single kernel with
online routing and no fast paths.

- Kernel SHA256: `261d1d0e71fc6b907948eb9547adabe9c0c00932db318f8a32932f628b4b1f3e`
- Runner SHA256: `dc6529fd79ac66d3d724b29e2a0c766465ac96f89086d8c4928c532d893201ae`

## Mechanism lineage

The promoted mechanism follows this lineage:

1. G256-parity baseline.
2. C25 packed f32x2 float math.
3. C29 per-column additive `0/-inf` masks replacing per-element select chains.

C29 constructs the column masks once for each physical score tile and reuses
them across owner rows. It preserves the routing decisions, online recurrence,
phase graph, and launch topology of its parent.

## Public API

```python
from sol_attn import make_pisa2_sm100

run = make_pisa2_sm100(
    T,
    q,
    k,
    v,
    kc,
    vc,
    global_threshold,
    128**-0.5,
)
output = run()
lse = run.lse
```

Inputs use contiguous BHTD BF16 layout with head dimension 128. `kc` and `vc`
are block summaries, `global_threshold` is FP32, and this release is
non-causal. The public signature is unchanged from the previous compact
release.

The public wrapper is intentionally thin. The evidence-bound kernel and runner
remain byte-for-byte unchanged under `kernels/pisa2_sm100/` and
`experiments/pisa2/`; the exact G256 parent kernel/runner and matching support
modules are retained as their import closure.

## Numerical Notes

The additive mask has a signed-zero edge: in IEEE floating-point arithmetic,
`x + 0.0` may flip `-0.0` to `+0.0`. The verified scope is bitwise-identical
route/O/LSE against the parent on all tested points, plus 45/45 reference-limit
correctness on the canonical matrix. No broader bitwise-equivalence claim is
made beyond that verified scope.

## Evidence

- `evidence/full45/full45-summary.json` and `per-case-results.csv` contain the
  canonical 45-case correctness and profiler-free three-leg timing results.
- `evidence/promotion/v1-full-result.json` contains the fixed-point,
  bitwise-parent, SASS, and resource promotion gate.
- `evidence/matched-ncu/PENDING.md` records that the claude55 five-leg
  collection will be added before tagging.

Performance claims use unprofiled timing. Nsight Compute evidence is reserved
for bottleneck attribution and is not part of the current release commit.

## Environment

The evidence was collected on GB200/B200 SM100 with Python 3.12, CUDA 12.8,
PyTorch 2.11.0+cu128, Triton 3.7.0, cuda-python, and the project's CUTLASS CuTe
DSL environment. Install the pip-level dependencies with:

```bash
python -m pip install -r requirements.txt
```

CUTLASS/CuTe DSL must also be importable in the environment.

## Wan 2.1 integration experiment

The `integrations/wan/` package contains the minimum adapter, Sparse-VideoGen
patch, smoke test, and Slurm runners used to exercise this kernel inside Wan
2.1 T2V 14B. The canonical 1280x720, 81-frame experiment completed on SM100
with a measured route density of 0.15004073 and passed the real-QKV
CuTeDSL-versus-Triton correctness gate.

See [`docs/experiments/wan-b200-sm100-colmask.md`](docs/experiments/wan-b200-sm100-colmask.md)
for the exact workload, reproduction commands, results, and limitations.

## Validate the release package

The release contracts do not require a GPU:

```bash
PYTHONPATH=. python3 -m py_compile \
  kernels/pisa2_sm100/native_bf16_claude49_g256_colmask_fwd.py \
  experiments/pisa2/native_bf16_claude50_colmask_full45_runner.py \
  sol_attn/pisa2_sm100.py \
  sol_attn/__init__.py
PYTHONPATH=. python3 -c 'import sol_attn'
python3 -m pytest tests/
shasum -a 256 --check SHA256SUMS
```

## Matched NCU (promotion-grade, cu128)

Five-leg matched NCU at the fixed point (colmask / packedsel parent /
G256 baseline / Triton PISA2 / fixed Triton PISA0), same GPU, same
prepared inputs, canonical `torch 2.11.0+cu128 / triton 3.7.0` runtime
(signature-gated).  Evidence in `evidence/matched-ncu/`; archive SHA256
`f50517b31b4b9b9a25333101be926ef2c15022ed73084b448d5a096d1a630d2f`.

- Dynamic SASS instructions: 303.5M (colmask) vs 349.5M (parent) vs
  366.2M (baseline) vs 255.2M (PISA0) — the select complex collapses
  from 28.66M SEL to 0.37M under the C29 column-mask mechanism.
- Tensor pipe active: 34.7% (colmask) vs 31.1% (baseline) vs 36.6%
  (PISA0).
- An unprofiled same-GPU ABBA canary in the same job reproduces the
  promotion ratio (0.9262 observed vs 0.9267 established) with bitwise
  colmask/parent output+LSE agreement.

### Environment sensitivity

The release claims are verified under `cu128`.  Under a drifted
`torch 2.11.0+cu130 / CUDA 13.0` runtime the current colmask lowering
compiles measurably worse (~10% slower than its parent at the fixed
point, order inverted); those runs are quarantined and are not release
evidence.  See `evidence/matched-ncu/RELEASE-ENVIRONMENT-CAVEAT.md`.
