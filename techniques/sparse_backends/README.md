<h1 align="center">Sol-Attn</h1>

<h4 align="center">
  Accelerating Video Generation Inference via On-the-Fly Attention Sparsification
</h4>

<p align="center">
  <a href="./sol_attn/"><img src="https://img.shields.io/badge/💻_Code-Sol--Attn-76b900?style=flat-square" alt="Code"/></a>
  <a href="https://arxiv.org/abs/2607.24027"><img src="https://img.shields.io/badge/📄_arXiv-2607.24027-b31b1b?style=flat-square" alt="Paper"/></a>
  <a href="https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/sparse/"><img src="https://img.shields.io/badge/📖_Docs-Sol--Engine-blue?style=flat-square" alt="Docs"/></a>
  <a href="../../README.md#-license"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square" alt="License"/></a>
</p>

---

## Introduction

Sol-Attn is a training-free sparse attention method for accelerating image
and video generation. It performs dynamic block routing during online softmax
and reuses proxy scores to approximate unselected blocks, avoiding a
materialized routing map while preserving visual quality. CuTe DSL kernels
support SM90, SM100, and SM120; SM80 and SM89 use Triton kernels.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.10
- CUDA ≥ 12.8
- Triton ≥ 3.6

The optimized CuTe paths additionally require:

- NVIDIA CuTe DSL / CUTLASS Python ≥ 4.5
- `cuda-python`

The released kernels are forward-only and require contiguous BF16 Q/K/V
tensors in BTHD layout with head dimension 128.

| GPU | Compute capability | CuTe backend | `kv_splits` |
|---|---:|---|---|
| H100 | SM90 | warp-group MMA/TMA | 1, 2, or 4 |
| B200 | SM100 | tcgen05/TMEM | 1 |
| RTX 5090 | SM120 | warp MMA/TMA | 1 |

## Installation

From the repository root:

```bash
python -m pip install -e techniques/sparse_backends
```

Backend dependencies are imported lazily, and kernel compilation happens on
the first eligible call for a given configuration.

## Backend dispatch

The public `sol_attn(...)` API selects the implementation from `q.device`:

| GPU architecture | Example GPU | Preferred backend | Fallback |
|---|---|---|---|
| SM90 | NVIDIA H100 | CuTe DSL SM90 kernel | Triton |
| SM100 | NVIDIA B200 | CuTe DSL SM100 kernel | Triton |
| SM120 | RTX 5090 | CuTe DSL SM120 kernel | Triton |
| SM80 / SM89 | A100 / RTX 4090 | Triton | — |
| Other NVIDIA CC ≥ 8.0 | Architecture dependent | Triton | — |

CuTe DSL and `cuda-python` are optional at runtime. When either cannot be
imported, the same public API falls back to Triton.

## Usage

For kernel and library integrations, `sol_attn(...)` is the public API. It
validates the tensor contract and automatically selects the CuTe or Triton
backend for the input device.

### Core API

```python
from sol_attn import sol_attn

out = sol_attn(
    q,  # Query: contiguous BF16 CUDA tensor in [batch, tokens, heads, 128].
    k,  # Key: same shape, dtype, layout, and device as q.
    v,  # Value: same shape, dtype, layout, and device as q.
    tau=1.0,  # Higher values select fewer KV blocks for exact attention.
    thresh_type="exact",  # Use the full-covariance routing threshold.
)
# out is contiguous [batch, tokens, heads, 128] BF16 on q.device.
```

### Exact KV sink

An exact sink keeps every KV block overlapping a contiguous token range exact
for all queries. This is useful for text tokens in joint image/video-text
attention.

| Text placement | `sink_start` | `sink_tokens` |
|---|---:|---:|
| Prefix | `0` | Number of valid text tokens |
| Suffix | Omit | Number of valid text tokens |
| Interior range | First text-token index | Number of valid text tokens |

For example, keep a text prefix exact:

```python
out = sol_attn(
    q,
    k,
    v,
    tau=1.0,
    sink_start=0,                  # Text begins before the image/video tokens.
    sink_tokens=valid_text_tokens, # All overlapping 64-token KV blocks are exact.
)
```

For a text suffix, omit `sink_start`:

```python
out = sol_attn(
    q,
    k,
    v,
    tau=1.0,
    sink_tokens=valid_text_tokens,
)
```

Every query attends all KV blocks overlapping the sink range exactly.
Exactness is applied at 64-token KV-block granularity. The sink does not
make text query rows dense: an MMDiT integration should still use dense
attention for valid text queries and Sol-Attn for image/video queries.

### Long-sequence tuning on H100

The SM90 CuTe kernel supports `kv_splits=1`, `2`, and `4`. Splitting KV can
improve utilization for very long sequences; the best value depends on shape
and realized sparsity.

```python
out = sol_attn(q, k, v, tau=1.0, kv_splits=4)
```

B200, RTX 5090, and Triton currently use `kv_splits=1`.

## Sol-Engine integration

Sol-Engine exposes two integration factories. They preserve the model's
original attention function as a dense fallback and route eligible calls to
the public `sol_attn(...)` kernel API.

| Entry point | Integration |
|---|---|
| `make_sol_attn_dispatch(...)` | Ordinary self-attention |
| `make_mmdit_sol_attn_dispatch(...)` | Joint MMDiT attention with exact text K/V and dense text-query rows |

`kv_splits="auto"` is provided by the integration layer. It selects split 4
for SM90 CuTe sequences of at least 65,536 tokens and split 1 otherwise. The
core kernel API accepts integer `kv_splits` values only.

Sol-Engine adapters fall back to dense attention when a call is ineligible or
a backend fails. Set `SOL_ATTN_STRICT=1` during validation to raise the original
error instead.

## Triton research implementation

The Triton implementation can also be selected explicitly for portability,
kernel studies, and comparisons against the architecture-specialized CuTe
implementations:

```python
from sol_attn.triton_ref import sol_attn as triton_sol_attn

out = triton_sol_attn(
    q,
    k,
    v,
    tau=1.0,
    thresh_type="exact",
)
```

This explicit entry point bypasses automatic backend selection. It keeps the
same tensor, threshold, and sink semantics as the public `sol_attn(...)` API,
but currently uses a single KV split.

## Citation

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```
