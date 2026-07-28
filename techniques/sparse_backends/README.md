# Sol-Attn backend

`techniques/sparse_backends/sol_attn/` is a clean vendor of
[`hp-l33/Sol-Attn`](https://github.com/hp-l33/Sol-Attn) at source commit
`d7ffb1e25999a8a753b679ee7923d7c3f861da6b`. It contains the public SM90 and
SM100 CuTe DSL kernels. Sana keeps one adapter and one public kernel API for
both architectures.

## Public kernel API

Add the vendor root to `PYTHONPATH`, then use the upstream API unchanged:

```python
from sol_attn import sol_attn

# q, k, v: contiguous BF16 [batch, tokens, heads, 128]
out = sol_attn(
    q,
    k,
    v,
    tau=1.0,
    thresh_type="diag",
    kv_splits=1,
)
```

H100 supports `kv_splits=1`, `2`, or `4`; B200 uses `kv_splits=1`.

For a text prefix:

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

For a text suffix:

```python
out = sol_attn(
    q,
    k,
    v,
    tau=1.0,
    sink_tokens=valid_text_tokens,
)
```

The sink is exact at 64-token KV-block granularity. It does not make text
query rows dense by itself.

## Sana model adapters

Use `techniques.sparse_backends.sol_attn_backend` when integrating a model:

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
    thresh_type="diag",
    kv_splits="auto",
)

# HunyuanVideo, BHSD with native [video, padded-text] order.
out = sol_attn_hunyuan(
    q,
    k,
    v,
    video_len=video_tokens,
    key_valid=key_padding_mask,
    tau=1.0,
    thresh_type="diag",
    kv_splits="auto",
)
```

The Hunyuan adapter crops padding, passes every valid text KV block as an exact
sink, replaces valid text-query rows with dense SDPA, and leaves padded query
rows zero. This is the required MMDiT contract: text K/V is exact and text Q is
dense; only image/video query rows use Sol-Attn approximation.

Diffusers integrations can use:

- `make_sol_attn_dispatch(...)` for ordinary self-attention.
- `make_hunyuan_sol_attn_dispatch(...)` for Hunyuan joint attention.
- `sol_attn_begin_forward()` as a transformer pre-hook.
- `reset_sol_attn_state()` after an untimed warmup generation.
- `get_sol_attn_stats()` to verify dispatch/kernel calls in an E2E run.

The shipped candidates use these settings:

```text
*_SOL_ATTN=1
*_SOL_TAU=1.0
*_SOL_THRESH_TYPE=diag
*_SOL_KV_SPLITS=auto
SOL_ATTN_STRICT=1  # recommended for validation; disables silent fallback
```

`auto` is an integration convenience: it selects split 4 for SM90 sequences
of at least 65,536 tokens and split 1 otherwise. The upstream kernel API still
accepts only integer `kv_splits`.
