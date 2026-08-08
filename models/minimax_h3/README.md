# MiniMax-H3

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a 33B audio-video Omni-DiT: 50 dense
blocks, hidden size 5376, 56 heads x 128, guidance-distilled to a single denoising branch. It packs
text, conditioning video, audio and target video into **one** sequence and attends over all of it —
38,247 rows at 1344x768 / 124 frames, of which 37,296 are the target video and 951 are the text +
audio prefix.

That shape decides the whole acceleration line. There is no CFG branch to elide and no cross
attention to cache, so the only parallel axis is the sequence itself, and attention is 70% of the
hot path.

## Hardware runtimes

MiniMax-H3 implementations are grouped by the hardware they were adapted and
measured on:

- `gb200/`: the original resident baseline and the multi-GPU optimized line.
  The nested `optimized/` name is historical; it is the GB200 implementation.
- `gb10/`: the single-GB10 constrained-memory port.
- `rtx5090/`: the single-RTX-5090 BF16 SGLang port with layerwise offload,
  SM120 Sol-Attn, TeaCache, regional compile, and post-denoise VAE residency.

Do not mix runtime files across these directories. Candidate manifests select
the matching hardware directory explicitly.

## Acceleration line

8xGB200 (two nodes in one NVL72 rack), 1344x768, 124 frames, 50 steps. Times are the **hot path** —
denoise plus video decode — which excludes a fixed ~2.1 s of text/audio encoding, packing,
scheduling and output assembly that does not move with any of these techniques.

| Stage | Candidate | Hot path | vs previous | Cumulative |
|---|---|---:|---:|---:|
| diffusers, 8 GPU | `minimax_h3_baseline` | 27.21 s | — | 1.00x |
| + kernel line | `minimax_h3_kernel_only` | 19.51 s | 1.39x | **1.39x** |
| + Sol-Attn | `minimax_h3_kernel_sol` | 17.74 s | 1.10x | **1.53x** |
| + FirstBlockCache | `minimax_h3_fullopt` | 6.88 s | 2.58x | **3.97x** |

Peak memory falls from 144,474 MiB to 120,763 MiB along the way, which is what makes the 8-GPU
configuration fit without offload.

### 1. Context parallelism — the packed sequence, not the batch

`cp_plan.py` shards `hidden_states` at the block-stack entry and gathers at `proj_out`. 38,247 rows
do not divide by 8, so the split is `tensor_split` and every collective carries explicit per-rank
sizes; an equal-split `all_to_all_single` overruns the short rank's receive buffer, and does it
quietly, because the caching allocator usually has slack.

Cross-node scaling is 96% (CP=4 43.4 s -> CP=8 22.5 s). The 8-GPU cross-node all-to-all measures
503 GB/s against 486 GB/s for the 4-GPU intra-node one — NCCL's MNNVL path puts both inside the
rack's NVLink domain, so the second node is not a step down.

### 2. Kernel line (lossless) — 1.39x

Four changes, none of which alter the arithmetic of the denoising step:

- **One collective instead of three.** The checkpoint stores a fused QKV matrix; diffusers splits it
  into `to_q`/`to_k`/`to_v` and then pays three permuted copies and three all-to-alls.
  `ulysses_custom.py` keeps it packed.
- **Two Triton relayout kernels** (`relayout.py`) replacing `torch.stack` + a 5-D
  `permute().contiguous()`, which walked the whole QKV buffer twice before a byte left the GPU.
- **AdaLN precompute** (`adaln.py`): the modulation table is built once and the projection weights
  are freed. Bit-identical, and it returns 23.9 GB.
- **VAE decode batching** (`vae_shard.py`): each rank's four tiles go through the decoder as one
  batch instead of four launches. 1.93x on the decode (1.160 s -> 0.602 s), bit-identical.

### 3. Sol-Attn — 1.10x

[Sol-Attn](../../techniques/sparse_backends/) runs on the packed self-attention **inside** the
Ulysses exchange. After the all-to-all each rank holds the entire sequence for its own heads, which
is the only point in the model where a sequence-level operator is well defined under context
parallelism — installing it as a diffusers attention processor would apply it to a shard of the
rows, which is a different and wrong operator.

The policy is the released one: `tau=1.0`, `diag` threshold, no reordering, the 951-row prefix
passed as an exact KV sink with its own query rows recomputed densely, and the first 10 steps and
first 2 blocks left dense. Measured route density is 0.225.

H3's video tail is already a contiguous grid-ordered block, so unlike Wan it needs no Morton
reordering at all — neither the per-call kind nor the global block-stack kind.

### 4. FirstBlockCache — 2.58x

`cache_line.py` at threshold 0.08. The skip decision is an all-reduced global scalar, so all ranks
skip together; a per-shard decision would desynchronize the collectives.

Sol-Attn and the cache are not independent: the cache deletes ~69% of block-stack calls, and every
deleted call is one Sol-Attn would have accelerated. Sol-Attn keeps 27.9% of its standalone benefit
under the cache, which tracks the 30.8% of sparse calls that survive. Ordering the two the other way
gives the same endpoint by construction (1.10x then 2.37x, against 2.44x then 1.06x).

## Running it

```bash
python scripts/launch_candidate.py --candidate candidates/minimax_h3_fullopt.toml
```

Every technique is an independent env switch (`H3_KERNEL`, `H3_SOL_ATTN`, `H3_CACHE_THRESHOLD`,
`H3_SHARD_VAE`, `H3_VAE_COMPILE`), so two runs differing in one variable are a clean A/B. The
candidates above pin the combinations that were measured.

`SOL_ATTN_STRICT=1` is set by the launcher: a sparse configuration that quietly fell back to dense
would be a dense measurement wearing a sparse label, and the driver also asserts, per request, that
the kernel was reached at all.

## Quality

The kernel line is lossless — context parallelism, the sharded and batched decode, the fusions and
the AdaLN precompute are all bit-identical or exact reassociations. Sol-Attn and FirstBlockCache are
approximations, and `H3_VAE_COMPILE` reassociates. H3 emits audio as well as video, and the two do
not degrade together: a visual metric alone rates this model's approximations too highly.
