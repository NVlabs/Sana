# Sol-Attn

[Paper](https://arxiv.org/abs/2607.24027) ·
[Code](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn)

Sol-Attn is a training-free sparse attention method for accelerating image and
video generation. It performs dynamic block routing during online softmax and
reuses proxy scores to approximate unselected blocks, avoiding a materialized
routing map while preserving visual quality.

## How it works

Sol-Attn combines routing, sparse computation, and approximation correction in
one online-softmax pass:

- it computes a lightweight proxy score for each key/value block;
- blocks that pass the on-the-fly threshold are evaluated exactly;
- unselected blocks reuse their proxy scores to approximate their contribution
  instead of being discarded.

This produces a dynamic block budget without materializing a full routing map.

## Kernel support

| GPU | Architecture | Execution |
|---|---|---|
| NVIDIA RTX 4090 | SM89 | CuTe DSL |
| NVIDIA H100 | SM90 | CuTe DSL, including split-KV execution |
| NVIDIA B200 / GB200 | SM100 | CuTe DSL |
| NVIDIA RTX 5090 | SM120 | CuTe DSL |
| NVIDIA A100 | SM80 | Triton reference |
| other supported architectures | — | Triton reference |

An architecture with no CuTe kernel falls back to the Triton reference, which
is correct but is not what the published speedups measure. `benchmark.json`
records the backend a run selected.

The released kernels are forward-only and require contiguous BF16 Q/K/V tensors
in BTHD layout with head dimension 128.

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.10
- CUDA ≥ 12.8
- Triton ≥ 3.6
- NVIDIA CuTe DSL / CUTLASS Python
- `cuda-python`

Install from the repository root:

```bash
python -m pip install -e techniques/sparse_backends
```

The CuTe DSL version has to match what the kernels were built against. A
mismatch fails at compile time — `module 'cutlass.cute.nvgpu' has no attribute
'OperandMajorMode'` is the SM100 symptom — rather than falling back, so that a
dense run is never reported as a sparse one.

## Core API

```python
from sol_attn import sol_attn

out = sol_attn(
    q,  # Queries: contiguous BF16 CUDA tensor of shape [B, T, H, 128].
    k,  # Keys: contiguous BF16 CUDA tensor of shape [B, T, H, 128].
    v,  # Values: contiguous BF16 CUDA tensor of shape [B, T, H, 128].
    tau=1.0,  # Threshold coefficient; larger values route fewer KV blocks exactly.
    thresh_type="exact",  # Use full covariance for the routing threshold.
)
# out: attention output [B, T, H, 128], with the same dtype/device as q.
```

CuTe DSL is imported lazily, so the first eligible call compiles the matching
kernel for the input device.

## Exact KV sink

Any contiguous valid key/value range can be kept exact. For a text prefix:

```python
out = sol_attn(q, k, v, tau=1.0, sink_start=0, sink_tokens=valid_text_tokens)
```

For a text suffix, omit `sink_start`:

```python
out = sol_attn(q, k, v, tau=1.0, sink_tokens=valid_text_tokens)
```

Exactness is applied at 64-token KV-block granularity. In an MMDiT
integration, the sink keeps valid text K/V blocks exact; valid text-query rows
still use dense attention, while Sol-Attn serves image or video query rows.

## Split KV on H100

H100 supports `kv_splits=1`, `2`, and `4`; B200, RTX 4090, and RTX 5090
currently use `kv_splits=1`.

```python
out = sol_attn(q, k, v, tau=1.0, kv_splits=4)
```

The model integration layer can select split 4 automatically for SM90 sequences
of at least 65,536 tokens and split 1 otherwise.

## Sol-Engine integration

Sol-Engine provides:

- ordinary self-attention dispatch through `sol_attn_attention`;
- MMDiT dispatch through `sol_attn_hunyuan`;
- dense fallback and architecture-aware split selection;
- right-padding handling, exact text K/V sinks, and dense text queries;
- lightweight runtime counters for end-to-end validation.

Set `SOL_ATTN_STRICT=1` during validation to raise kernel or integration errors
instead of silently falling back to dense attention.

## Citation

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```
