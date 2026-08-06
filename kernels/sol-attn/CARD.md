---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}tags:
- kernels
- cuda
- attention
- triton
- cute-dsl
---

# Sol-Attn

Sol-Attn accelerates image and video generation with on-the-fly attention
sparsification. It automatically dispatches to CuTe DSL kernels on SM90,
SM100, and SM120, and to Triton on SM80 and SM89 or when CuTe DSL is not
available.

## Usage

```python
from kernels import get_kernel

kernel = get_kernel(
    "{{ repo_id }}",
    version={{ version }},
    trust_remote_code=True,
)

out = kernel.sol_attn(
    q,  # Contiguous BF16 [batch, tokens, heads, 128].
    k,  # Same shape, dtype, layout, and device as q.
    v,  # Same shape, dtype, layout, and device as q.
    tau=1.0,
    thresh_type="exact",
)
```

The released implementation is noncausal and forward-only. An optional exact
KV sink is available through `sink_start` and `sink_tokens`.

## Backends

| Architecture | Example GPU | Backend |
|---|---|---|
| SM90 | H100 | CuTe DSL |
| SM100 | GB200 | CuTe DSL |
| SM120 | RTX 5090 | CuTe DSL |
| SM80 / SM89 | A100 / RTX 4090 | Triton |

## Paper and source

- [Paper](https://arxiv.org/abs/2607.24027)
- [Source](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn)
- [Documentation](https://nvlabs.github.io/Sana/Sol-Engine/docs/techniques/sparse/)
