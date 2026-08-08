# MiniMax-H3 on one GB10 — FP8, no context parallelism

A third measured runtime beside `baseline/` (one GB200, BF16, pristine) and `optimized/`
(8×GB200, Ulysses). It exists because the interesting constraint is different: not how to
shard 33B across a rack, but what survives when there is one card and the released weights do
not fit on it.

The rows in `models/minimax_h3.toml [gb10]` are not comparable to `[baseline]` row for
row — different hardware, different checkpoint, a tenth of the canvas. What is comparable is
the shape: same timing scope, same fields, same recorded prompt.

## What had to change to fit

GB10 has 121 GiB of **unified** LPDDR5X shared with the host. The released BF16 weights alone
are 134 GiB. Two substitutions close the gap, and neither changes the pipeline's structure:

| | released | here | note |
|---|---|---|---|
| DiT | 61.7 GiB BF16 | 19.5 GiB FP8 | ComfyUI `pruned_fp8_scaled`, rank-8 AdaLN |
| conditioner | 62 GiB BF16 | 33 GiB FP8 | `Qwen/Qwen3-VL-32B-Instruct-FP8`, same model |

The FP8 DiT is a substitution, not an optimisation. Measured against the released BF16 model
on the same recorded step: **13,428 ms against 13,417 ms** — 0.08%, three alternating processes
each, spread under 1%. So every ratio below is also a ratio against the released model.

## What was measured

DiT forward on a recorded step, 832×480, round-robin interleaved inside one process:

| | ms | cumulative | pixels |
|---|---|---|---|
| bf16, released model | 13,417 | — | — |
| baseline (this port, no fusions) | 13,576 | 1.00× | reference |
| + fused QKV | 12,875 | 1.05× | EXACT |
| + Triton quantiser | 11,524 | 1.18× | EXACT |
| + fused AdaLN | 10,709 | 1.27× | EXACT |
| + fused RoPE | 9,913 | 1.37× | EXACT |
| + fused SwiGLU | 9,790 | **1.39×** | EXACT |
| + Sol-Attn τ=1 | 8,157 | 1.66× | cos 0.9865 |

`EXACT` means bit-identical to the eager path, checked by recomputing a recorded step and
comparing, not asserted. That is the whole reason the lossless tier needs no quality argument:
its speedup can be taken without one.

The fixed remainder is 31.8 s, of which the video decode is 29.4 s (92.6%). Batching its tiles
is 1.47× at PSNR 71.2 dB. Retiling is faster and was rejected: an untiled height reaches 2.29×
at PSNR 23.1 dB, with 60% of pixels visibly different.

## What is settled, and what is not

**Settled, and it deleted code.** Sol-Attn needs neither a bypass nor a fork here. The
snapshot this port began from refused SM121 outright and had no way to ask for a KV sink, so
it carried a copy of the Triton reference with the 951-row sink hardcoded. The current head
routes SM121 to Triton on its own and takes `sink_tokens`, and the released path reproduces
the copy's output to six decimal places at the same speed. The copy's other policy — handing
the prefix's query rows to flash — turned out to cost 7.8% for +0.0005 of cosine and went with
it. 484 lines of divergence, none of it needed.

**Settled.** FP8 quantisation on its own buys nothing (13,428 vs 13,417); the 1.40× the fused
path reaches is near its ceiling, because attention is 36% of the DiT's FLOPs and FP8 does not
touch it — `1/(0.36 + 0.64/1.88) = 1.43×`. Flash attention already runs at 93–96% of the
16-bit GEMM ceiling, so there is no lossless headroom left in attention either.

**Not settled, and load-bearing.** How Sol-Attn and the cache compose. The headline 4.27× is a
derivation, not a reading — the cache's standalone 2.7× applied to what is now the 8,157 ms
row, so even the derivation is stale by 5%. The two
full requests that were run end to end point the other way: with the cache on, Sol-Attn was
worth 1.09× on one prompt and 0.96× on the other. Running
`minimax_h3_gb10_fullopt` against `minimax_h3_gb10_cache` on the recorded step settles it.

**A quality finding that should not be buried.** On a low-motion, strongly-semantic prompt
(two mechanics, an open engine bay, dialogue), the cache alone stays on the reference's scene
while cache + Sol-Attn lands on a *different* scene — different car, closed bay. Both were run
at the same seed. PSNR is useless for saying so: cache-only scores 19.2 dB on that prompt
because the framing shifts, while being obviously the same shot. This is why
`minimax_h3_gb10_cache` exists as a candidate in its own right.

## Layout

```
gpu_infer.py         entrypoint; every acceleration env-gated, one per candidate manifest
relayout.py          ComfyUI key layout, QKV/SwiGLU questions, checkpoint reader
fusions.py           four Triton kernels + the activation quantisers they replace
fusion_install.py    installing them onto a built model, and the Sol-Attn dispatch
fp8_linear.py        Fp8Linear, honouring the checkpoint's per-layer comfy_quant scheme
adaln.py             the rank-8 AdaLN factorisation and the re-point onto it
build.py             assemble MiniMaxH3Transformer3DModel from the pruned checkpoint
cache_line.py        FirstBlockCache, minus the collective-decision fix one card cannot need
vae_shard.py         batched tile decode
```

The benchmarks and the bit-exactness checks the numbers above came from are not carried here,
matching how the other model directories are laid out — the runtime is what ships, and the
measurements live in the profile and in `runs/`. What those scripts established is kept where
it is load-bearing: the `div_rn` correction in `fusions.py`, the QKV layout question in
`relayout.py`, and the rank-8 error in `adaln.py` all carry their own numbers.

## Running

```bash
OUT_DIR=runs/gb10_fullopt \
  scripts/launch_candidate.py candidates/minimax_h3_gb10_fullopt.toml
```

By hand, without the harness — `paths.py` resolves everything from its own location, and the
env vars below are only needed when the checkpoints live outside the repository:

```bash
export HF_HOME=/path/to/checkpoints H3_DIFFUSERS_SRC=/path/to/diffusers/src
python gpu_infer.py --height 480 --width 832 --steps 50
```

## Things that will bite

**The pinned versions are not advisory.** Installing torchvision separately once pulled torch
2.11.0 → 2.13.0 and Triton 3.6 → 3.7. Nothing failed; every bit-exactness claim above would
simply have stopped meaning anything.

**Memory is one pool.** 85 GiB of weights on this part does not OOM — it pages, and the machine
stopped answering ssh for an hour. `earlyoom` is installed and did not fire, because it watches
`MemAvailable` and `MemAvailable` counts reclaimable page cache. Cap the job with a cgroup, or
keep one model resident at a time.

**Freeing a model does not return its memory to the kernel.** Measuring a second model in the
same process after freeing the first gave 20,746 ms for a configuration that takes 9,553 ms
from a clean start — a plausible-looking number produced entirely by the previous model.
Measure one model per process.
