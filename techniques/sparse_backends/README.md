<h1 align="center">Sol-Attn</h1>

<h4 align="center">
  Efficient sparse attention kernels for video diffusion transformers
</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2607.24027">Paper</a> |
  <a href="./sol_attn/">Code</a>
</p>

---

## Introduction

Sol-Attn is a training-free sparse attention method for accelerating image
and video generation. It performs dynamic block routing during online softmax
and reuses proxy scores to approximate unselected blocks, avoiding a
materialized routing map while preserving visual quality. This release
includes CuTe DSL implementations for NVIDIA Hopper and Blackwell GPUs.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.10
- CUDA ≥ 12.8
- Triton ≥ 3.6
- NVIDIA CuTe DSL / CUTLASS Python ≥ 4.5
- `cuda-python`

The released kernels are forward-only and require contiguous BF16 Q/K/V
tensors in BTHD layout with head dimension 128.

## Installation

From the repository root:

```bash
python -m pip install -e techniques/sparse_backends/sol_attn
```

CuTe DSL is imported lazily, so kernel compilation happens on the first
eligible call for a given configuration.

## Usage

### Core kernel API

```python
from sol_attn import sol_attn

out = sol_attn(
    q,  # BTHD
    k,  # BTHD
    v,  # BTHD
    tau=1.0,
    thresh_type="exact",
)
```

### Exact KV sink

Any contiguous valid KV range can be kept exact. For a text prefix:

```python
out = sol_attn(
    q,
    k,
    v,
    tau=1.0,
    sink_start=0,
    sink_tokens=valid_text_tokens,
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
change query routing: an MMDiT integration should still compute valid text
query rows with dense attention and use Sol-Attn for image/video query rows.

### Split KV on H100

H100 supports `kv_splits=1`, `2`, and `4`; B200 currently uses
`kv_splits=1`.

```python
out = sol_attn(q, k, v, tau=1.0, kv_splits=4)
```

Split 4 is a useful starting point for very long H100 sequences, while the
best setting depends on sequence length, batch size, head count, and realized
sparsity.

## Model integration helpers

The integration module provides dense fallback, architecture-aware split
selection, Diffusers dispatch hooks, MMDiT padding handling, and lightweight
runtime counters:

```python
from techniques.sparse_backends.sol_attn_backend import (
    sol_attn_attention,
    sol_attn_hunyuan,
)

# Ordinary self-attention, BTHD.
out = sol_attn_attention(
    q,
    k,
    v,
    tau=1.0,
    thresh_type="exact",
    kv_splits="auto",
)

# MMDiT joint attention, BHSD with [video, padded-text] order.
out = sol_attn_hunyuan(
    q,
    k,
    v,
    video_len=video_tokens,
    key_valid=key_padding_mask,
    tau=1.0,
    thresh_type="exact",
    kv_splits="auto",
)
```

The MMDiT helper crops right padding, passes valid text KV blocks as an exact
sink, replaces valid text-query rows with dense SDPA, and leaves padded query
rows zero.

Diffusers integrations can use:

- `make_sol_attn_dispatch(...)` for ordinary self-attention.
- `make_hunyuan_sol_attn_dispatch(...)` for joint MMDiT attention.
- `sol_attn_begin_forward()` as a transformer pre-hook.
- `reset_sol_attn_state()` after an untimed warmup.
- `get_sol_attn_stats()` to verify dispatch and kernel calls.

`kv_splits="auto"` is provided by the integration layer. It selects split 4
for SM90 sequences of at least 65,536 tokens and split 1 otherwise. The core
kernel API accepts integer `kv_splits` values only.

Set `SOL_ATTN_STRICT=1` during validation to raise kernel or integration
errors instead of silently falling back to dense attention.

## Citation

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```
