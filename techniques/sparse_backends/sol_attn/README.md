# Sol-Attn: CuTe DSL PISA2 for B200

This branch is the compact release of the promoted BF16 PISA2 forward kernel
for NVIDIA B200 / SM100. The complete optimization lineage is available on
branch `history/sm100-pisa2-cutedsl-optimization`.

## Result

The release backend is
`lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128`.

- Kernel SHA256: `75dccb78d282c7741eeb3833ea09af15543438cfe2d207ade9af72d07400feb5`
- Runner SHA256: `840ccdee2c8852f580d835357f14e52896052528bf5049ec6f206aa0cbe6aeed`
- Correctness: 45/45
- Unprofiled wins over Triton PISA2: 45/45
- Overall candidate/Triton latency geometric mean: `0.8296319471`
- Fixed `T16384-B1-H32-density5%`: `0.8460793318`
- Worst point: `0.8864503171`

Every T, B/H, and density subgroup geometric mean is below 1.0. Every point is
below the 1.03 promotion cap.

The kernel uses logical G512, physical N128, TMEM256, six warps, and two CTA/SM.
It replaces lane-serial lowbit compaction with owner-local selected-lane prefix
scatter. There is no shape- or density-specific fast path.

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
are block summaries, `global_threshold` is FP32, and the release is non-causal.

The public wrapper is intentionally thin. The evidence-bound kernel and runner
remain byte-for-byte unchanged under `kernels/pisa2_sm100/` and
`experiments/pisa2/`.

## Environment

The promoted evidence was collected on GB200/B200 SM100 with Python 3.12,
CUDA 12.8, PyTorch 2.11.0+cu128, Triton 3.7.0, cuda-python, and the CUTLASS
CuTe DSL Python environment used by the project. Install the pip-level
dependencies with:

```bash
python -m pip install -r requirements.txt
```

CUTLASS/CuTe DSL must also be importable in the environment.

## Validate the release package

The host-side release and compaction contracts do not require a GPU:

```bash
python -m pytest \
  tests/test_release_contract.py \
  tests/test_g512_cursor_ballotscatter_compaction_contract.py
```

The full45 device harnesses require one visible SM100 GPU per process. Generate
a source manifest for this pruned release tree, then run the correctness and
timing modules with the promoted backend:

```bash
find . -type f -not -path './.git/*' -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 > SOURCE_SHA256SUMS

python -m experiments.pisa2.check_b200_lean6_routeidx_full45_correctness \
  --help
python -m experiments.pisa2.benchmark_b200_lean6_routeidx_vs_triton_pisa2_full45 \
  --help
python -m experiments.pisa2.summarize_b200_lean6_routeidx_vs_triton_pisa2_full45 \
  --help
```

Use candidate backend
`lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128` and
`evidence/full45/edge-receipt.json`. The historical edge receipt binds the exact
kernel and evidence. A fresh run must bind its rows to the newly generated
release-tree source manifest; it must not claim the historical 1,376-file
manifest identity.

## Evidence

- `evidence/full45/`: immutable 45 correctness rows, 45 profiler-free timing
  rows, strict gates, digest, edge receipt, and verified evidence archive.
- `evidence/matched-ncu/`: 12 same-GB200 independent candidate/parent/Triton
  reports with full metrics, PM sampling, source/SASS, bindings, and archive
  verification.

Performance promotion is based on unprofiled ABBA/full45 timing. Nsight Compute
is retained for bottleneck attribution only.

Final full45 evidence archive SHA256:
`e1d8bf5601c2122bc99cdfdab790a0ce002fdab502216fe874047c4f63ca44c4`.

Matched-NCU archive SHA256:
`f08ef10c86612e787fa0600ba8b69d82d282d769497fd3f6d9fbe190aa108497`.
