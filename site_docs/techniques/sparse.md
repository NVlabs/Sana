# Sparse attention

Sparse attention targets model-level redundancy in long video token sequences. Many spatiotemporal attention interactions contribute little to the final output, so structured sparse patterns can reduce attention cost.

The paper lists sparse attention as a model-level family. The sparse-attention agent chooses the backend, sparsified layers, sparsity pattern, and compensation policy for the deployment instance.

## In Sol-Engine

LTX-2.3 uses PISA-style sparse video self-attention in selected stage-2 refinement work. It is combined with cache, fusion, NVFP4, and token pruning in the full optimization stack.

Implemented entries:

- `python/sglang/multimodal_gen/runtime/efficiency/transforms/sparse_attention.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_video_gen_2_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_linear_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/block_sparse_attn.py`

## Methods

| Method | Open-source status | Integration | Role in the design space |
|---|---|---|---|
| [PISA](sparse/pisa.md) | open-source | local LTX runtime | piecewise sparse attention selected for the LTX-2.3 full optimization stack |
| [SpargeAttention](sparse/spargeattention.md) | open-source | external adapter | training-free block-wise sparse attention family |
| [Sparse VideoGen](sparse/sparse_videogen.md) | open-source | external adapter | spatial-temporal sparse execution with online profiling |
| [Sparse VideoGen2](sparse/sparse_videogen2.md) | open-source | local backend plus SVG adapter | semantic-aware sparse-token identification and GPU layout |

## Practical notes

- Sparse settings should be validated visually.
- The value of sparse attention depends on sequence length and stage placement.
- Report both denoise time and end-to-end time; decode and offload overhead can hide attention savings.
