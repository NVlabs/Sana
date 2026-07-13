# REPORT — Hot-inference latency optimization for LingBot-Video (4×GB200)

## Objective
Minimize **hot-inference latency** for a single request against an already-warm server
(both 30B DiTs resident). Metric (from `LINGBOT_PHASE_TIMING=1`):

    hot_total = base_denoise + refiner_vae_encode + refiner_denoise + vae_decode

Model loading (`base_load`, `refiner_load`, FSDP setup, broadcast) is **excluded** (cold-start,
out of scope). Constraint: exactly 4×GB200, one NVLink node; only parallelism/communication for
MoE(FFN) and Attention; no step/res/scheduler/seed/quant/cache/sparse changes. Correctness by
implementation audit + a valid-non-degenerate-video sanity check; numerical drift is acceptable.

## Result — one decisive win
The refiner denoise (263s, 70% of hot) was **attention-compute-bound**. Ulysses CP4 already splits
the 16 heads optimally (4 heads/rank) and the all-to-all volume (~2 GB/layer over 900 GB/s NVLink
≈ ms) is negligible — so the lever is the **attention kernel**, not the collective. The stack ships
FA2 2.8.3 (via the `flash_attn_interface` shim) which runs a **suboptimal kernel on sm_100
(Blackwell)**, while **cuDNN 9.15** (present in torch 2.10+cu13) has a fast Blackwell flash kernel.

Swapping the refiner attention to a cuDNN-backed SDPA path cut refiner_denoise **2.12×** and hot
latency **1.81×**.

## Frontier table (hot inference, warm compute; 4×GB200)
| config | base_denoise (s/step) | refiner_vae_enc | refiner_denoise (s/step) | vae_decode | hot_total | peak mem | correctness | status |
|--------|----------------------|-----------------|--------------------------|-----------|-----------|----------|-------------|--------|
| **c5: CP4+FSDP+batch_cfg + `LINGBOT_ATTN_KERNEL=cudnn`** | **61.96 (1.55)** | **20.52** | **123.88 (15.49)** | **1.54** | **207.9** | **113.5 GB** | audited + valid video (std 52.9) | **BEST** |
| golden: CP4+FSDP+batch_cfg (FA2) | 89.1 (2.16) | 19.3 | 263.2 (31.2) | ~4 | 375.6 | 117 GB | ref | dominated |

Speedup: **hot 1.81× (−167.7 s)**; refiner_denoise **2.12× (−139.3 s)**; base_denoise 1.44×
(cuDNN rejects the small 480p shape → falls back to torch FLASH, still beats the FA2 shim 89→62 s).
Peak memory is *lower* (cuDNN uses less scratch than FA2 varlen).

### Microbench that drove the decision (per-rank refiner attention shape, 1 GPU)
| kernel | ms/attn | over 48 layers |
|--------|---------|----------------|
| FA2 2.8.3 varlen (current) | 546.8 | 26.25 s |
| **SDPA cuDNN** | **191.2** | **9.18 s** |
| SDPA torch-flash | 602.6 | 28.92 s (slower) |
| priority-list [cuDNN,flash,eff] | 182.4 | 8.76 s (picks cuDNN) |

Correctness: FA2 vs cuDNN max_abs_diff 1.2e-4, rel 5.3e-3 (bf16 reduction-order drift, acceptable).

## Best config — exact reproduction
```
sbatch agent_opt/candidates/c5_cudnn_attn.sbatch
```
Identical to the golden `slurm/accel_cp4_refiner.sbatch` (CP4 Ulysses + FSDP + batch_cfg, 4×GB200)
plus two env vars:
```
export LINGBOT_PHASE_TIMING=1        # metric logging (lossless no-op)
export LINGBOT_ATTN_KERNEL=cudnn     # NEW: cuDNN SDPA attention path
```
CLI (unchanged): `--context_parallel_degree 4 --context_parallel_ulysses_anything
--enable_fsdp_inference --batch_cfg --refiner_batch_cfg --reuse_condition_features`
(480p base 40 steps → 1080p refiner 8 steps, seed 42).

## Implementation (code change)
`lingbot_video/transformer_lingbot_video.py`:
- New `_cudnn_varlen_attention(q,k,v,cu_seqlens)`: splits the Ulysses-gathered packed sequence by
  `cu_seqlens_kv` into per-segment (batch_cfg cond/uncond + zero-pad tail) blocks and runs each as a
  dense non-causal `F.scaled_dot_product_attention` under
  `sdpa_kernel([CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION], set_priority=True)`.
  Mathematically identical to the FA2 `flash_attn_varlen_func` block-diagonal call (no cross-segment
  leakage, no dropped/duplicated tokens); only the kernel/reduction order differs. All three
  fallbacks are memory-efficient (no S×S score matrix → no OOM at 250K tokens); the priority list
  keeps cuDNN for the large refiner sequence and falls back cleanly for shapes cuDNN rejects
  (e.g. the 480p base).
- Gated by `LINGBOT_ATTN_KERNEL` (default `fa2`). The OFF/FA2 path is byte-for-byte unchanged.
- Branch added in both the Ulysses CP path (base+refiner) and the non-CP packed path.

Correctness: (a) first-principles audit above, and (b) sanity check — exit 0, output video shape
(121, 1088, 1920, 3) uint8, range 0–255, mean 180.2, std 52.9 → valid, non-degenerate.

## Rejected / declined hypotheses (with signatures)
| hypothesis | signature / reason |
|------------|--------------------|
| Ring attention / Ulysses×Ring hybrid | Ulysses a2a is ~2 GB/layer over 900 GB/s NVLink ≈ ms — negligible vs the multi-second kernel. Changing the collective / overlapping P2P cannot recover time not spent on comm. |
| Overlapped / dedup EP, expert-TP | FFN is now only ~40% of the refiner step; prior RESULTS: EP is 2.8× slower at 1080p (activation a2a volume dominates). EP/TP redistribute FFN work without cutting compute (CP already shards by sequence) → cannot beat CP+FSDP. |
| cuDNN-only backend (first attempt, job 4505022) | Crashed: "No available kernel" on the 480p base shape (cuDNN rejects some shapes, no fallback). Fixed by priority list + `.contiguous()`. |
| FSDP-only (no CP) | prior: 0 speedup — FSDP shards memory, not compute. |
| CP4+EP4 naive | prior: refiner 88 s/step — EP adds a2a without cutting attention. |
| Model-load / broadcast / sharded-load overlap (c2/c3/c4) | Cold-start only; zero effect on hot inference (excluded objective). |

## Remaining untried hypotheses (low expected value)
- Refiner attention is near the hardware roofline (~4.9 s ideal bf16 vs 8.7 s measured ≈ 56% of
  peak); little kernel headroom remains without lossy FP8/sparse attention (out of scope).
- refiner_vae_encode (20.5 s, ~10% of hot) is now the third-largest term but is neither MoE nor
  attention → out of the allowed scope. If widened, sharding the 1080p VAE encode is the next lever.
- Batching the two batch_cfg segments into one SDPA call — only when cond/uncond text lengths are
  equal; negligible expected gain (same total work).

## Files
- Candidate: `agent_opt/candidates/c5_cudnn_attn.sbatch`
- Code: `lingbot_video/transformer_lingbot_video.py` (`_cudnn_varlen_attention`, `_attn_kernel`)
- Microbench: `slurm/bench_attn_kernel.py` + `slurm/bench_attn_kernel.sbatch`
- Run: `outputs/c5_cudnn_20260710_072019/` (log `slurm/logs/lbv-c5cudnn_4506351.out`)
