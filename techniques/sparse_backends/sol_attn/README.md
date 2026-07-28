# Sol-Attn

Sol-Attn provides fused forward kernels for NVIDIA H100 (SM90) and B200
(SM100). The Python entry point accepts the same contiguous BTHD layout on
both architectures and selects the backend from the input tensor's device.

```python
import torch
from sol_attn import sol_attn

q = torch.randn(1, 16384, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

o = sol_attn(q, k, v)
o_exact = sol_attn(q, k, v, thresh_type="exact")
```

`q`, `k`, and `v` have shape `[batch, tokens, heads, 128]`; the output has the
same shape and dtype. `scale=None` uses `1 / sqrt(128)`. `tau` defaults to
`1.0` and controls the route threshold. The current release implements
noncausal forward attention for equal Q/K/V sequence lengths.
`thresh_type="diag"` keeps the diagonal variance approximation used by the
original kernel. `thresh_type="exact"` selects the fused full-covariance
threshold.

## KV sink range

MMDiT models can keep any contiguous valid text K/V range exact. For a text
prefix:

```python
o = sol_attn(q, k, v, sink_start=0, sink_tokens=valid_text_tokens)
```

For a text suffix, omit `sink_start`:

```python
o = sol_attn(q, k, v, sink_tokens=valid_text_tokens)
```

Every query attends all KV blocks overlapping
`[sink_start, sink_start + sink_tokens)` exactly. Exactness is block-granular,
so a 64-token KV block spanning an image/text boundary is exact in full.
`sink_tokens` does not change query routing; MMDiT integrations should compute
valid text-query rows with dense attention and use the Sol-Attn output for
image-query rows. `sink_tokens=0` preserves the original behavior. H100 and
B200 use the same sink API and N64 block semantics.

## Split KV on H100

`kv_splits` is optional and defaults to `1`:

```python
o = sol_attn(q, k, v, tau=1.0, kv_splits=4)
```

H100 supports `kv_splits=1`, `2`, and `4`. In the measured BH32 long-sequence
workload, `kv_splits=4` is the recommended starting point at 128K tokens and
70–90% sparsity. Split 2 is useful for intermediate sequence lengths, but the
best choice depends on batch size, head count, and density. B200 currently
uses `kv_splits=1`.

## Triton reference

The BF16 Triton version is kept separately as a readable reference:

```python
from sol_attn.triton_ref import sol_attn as triton_sol_attn

o_ref = triton_sol_attn(q, k, v, scale=None, tau=1.0)
```

The top-level package exports only the CuTe `sol_attn` function.

## Layout

```text
sol_attn/
  interface.py       public BTHD API and architecture dispatch
  preprocess.py      shared K/V reductions and route threshold
  common/            shared selector and layout primitives
  sm90/
    kernel.py        frozen Hopper recipe
    mainloop.py      route/approximate/exact mainloop
    exact.py         exact-block stream
    split_combine.py split-KV merge
  sm100/
    kernel.py        Blackwell entry
    mainloop.py      route/approximate/exact mainloop
    softmax.py       online-softmax helpers
    tmem.py          TMEM helpers
  triton_ref/        Triton reference
```

## Requirements

- Python 3.10 or newer
- PyTorch with CUDA 12.8
- Triton 3.6 or newer
- NVIDIA CuTe DSL / CUTLASS Python 4.5
- `cuda-python`

The kernels are forward-only; autograd is not provided in this release.
FlashAttention-derived files retain their upstream notices in
`THIRD_PARTY_NOTICES.md`.
