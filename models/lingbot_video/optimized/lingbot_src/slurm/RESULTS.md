# LingBot-Video — Deployment Baseline (GB200)

Hardware: 4× NVIDIA GB200 / node, 186 GB HBM3e each, sm_10.0 (Blackwell), CUDA 13, driver 580.
Env: conda `lingbot-video` (cloned from hunyuanvideo15) — torch 2.10.0+cu130 (`torch._grouped_mm` ✓),
diffusers 0.39.0, transformers 5.8.1, peft 0.19.1.
Model: MoE 30B-A3B (128 experts, top-8, 1 shared, 48 layers, hidden 2048) + separate 30B refiner
+ Qwen3-VL text encoder (~4-5B) + Wan VAE. FlowUniPC scheduler, bidirectional (non-causal) DiT.

Task: T2V, 480p (480×832), 5 s = 121 frames, seed 42, grouped_mm bf16, SDPA attention.

## Baselines (measured on GB200)

| ID | Config | GPUs | Base denoise | Refiner/step | Refiner total | End-to-end | Peak GPU mem | Status |
|----|--------|------|--------------|--------------|---------------|-----------|--------------|--------|
| B1 | base-only, 40 steps @480p, seq CFG | 1 | 40×7.45s=298s | — | — | **6:48** | — | ✅ |
| B2 (1-GPU, naive) | base + refiner(8 steps @1080p) | 1 | 299s ok | — | — | — | **OOM @184GB** | ❌ both DiTs resident (~120GB) |
| **B2-offload** | base→CPU before refiner | **1** | 299s | 130.9s | 17:27 | **26:28** | **157 GB** | ✅ |
| B2-FSDP4 | FSDP shards both DiTs | 4 | 299s | 133.2s | 17:45 | **28:40** | 143 GB/GPU | ✅ |

Both refined outputs are byte-identical (1,234,295 B) — offload changes nothing but memory.

### Decisive result
- **1-GPU offload (26:28) BEATS 4-GPU FSDP (28:40)** and uses 1 GPU instead of 4.
  FSDP-only is memory relief, NOT parallel speedup (same 131 s/step; comms make it slightly slower).
  → For single-request latency, multi-GPU FSDP is pure waste. Real multi-GPU speedup needs CP/TP/EP.
- **Refiner @1080p = 131 s/step = 17.6× the base step (7.45s)**; it is ~66% of end-to-end.
  1080p full attention (~250K tokens, O(n²), currently SDPA — no FA3) is THE bottleneck to attack.
- Peak 157 GB single-card (offload) → full two-stage fits one GB200 with ~27 GB headroom.

## Key findings (relevant to acceleration)

1. **Single-GPU fits the 480p base easily, but the 1080p refiner OOMs on one GB200.**
   The runner preloads the refiner (`_maybe_preload_refiner`) BEFORE running base, so during the
   refiner stage BOTH 30B DiTs are resident (~120 GB weights) + text encoder + 1080p activations
   → >184 GB. This is the real reason the official multi-GPU scripts exist — memory, not just latency.
   Fix: FSDP shards both DiTs across GPUs (120 GB → 30 GB/GPU on 4).

2. **`batch_cfg` and context-parallel (CP) both hard-require FlashAttention-3**
   (`flash_attn_interface.flash_attn_varlen_func`, transformer_lingbot_video.py:262/301).
   - B=1 (sequential CFG) → diffusers `dispatch_attention_fn` (SDPA) — works without FA3.
   - B>1 (`packed_batch`) or CP → FA3 varlen path.
   So FA3 is the linchpin for the two biggest speed levers. Not installed yet (no Blackwell/ARM wheel found).

3. Denoising cost is ~7.45 s/step at 480p with sequential CFG (2 forward passes/step → ~3.7 s/forward).

## Environment gotchas (all resolved)
- `hf_transfer` hangs silently on this network → use plain hub downloader / `wget -c`.
- `decord` has no aarch64 wheel → imageio-backed shim at `slurm/shims/decord.py` (refiner only).
- Qwen3-VL text encoder defaults to `flash_attention_3` → set `LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa`.

## Acceleration results

FA2 reused from nunchaku_blackwell (py3.11 clone `lingbot-fa2`) via a shim that maps
`flash_attn_interface.flash_attn_varlen_func` → FA2 2.8.3 (no FA3 build needed). This unlocks
`batch_cfg` and context parallel.

| Config | GPUs | Base/step | Refiner/step | End-to-end | Peak mem | vs 1-GPU offload |
|--------|------|-----------|--------------|-----------|----------|------------------|
| B2-offload (baseline) | 1 | 7.45s | 131s | 26:28 | 157 GB | 1.0× |
| B2-FSDP4 | 4 | 7.45s | 133s | 28:40 | 143 GB | 0.92× (slower) |
| **CP4 + FSDP + batch_cfg** | 4 | ~5s | **31.2s** | **9:32** | 117 GB | **2.8×** |

- **CP4 refiner = 31.2 s/step vs 131 s/step baseline → 4.2× (super-linear; attention is O(n²)).**
- Same 4 GPUs, CP4 is 3× faster than FSDP4 → CP shards the attention compute, FSDP only shards memory.
- `batch_cfg` alone at 480p base: 7.45→7.06 s/step (~5% only — base already saturates the GPU at B=1;
  batch_cfg mainly cuts kernel-launch overhead. It matters more where the GPU is under-utilized.)

## Expert parallelism (EP) — implemented, verified, benchmarked

Implemented EP for the MoE FFN (shard grouped experts across ranks; all-to-all dispatch
tokens → owning rank, local grouped_mm, all-to-all back, combine by router score).
Composes with CP on the SAME 4 ranks (each rank: 1/4 sequence + 1/4 experts). Router
weights stay full. Correctness: EP output == full-local MoE, max_abs_diff = 0 (4-rank test).

| Config | Base/step | Refiner/step | End-to-end | Peak mem |
|--------|-----------|--------------|-----------|----------|
| CP4 + FSDP | ~5s | **31.2s** | **9:32** | 117 GB |
| CP4 + EP4 (no FSDP) | 5.16s | **88.3s** | 20:47 | 160 GB |

**EP is 2.8× SLOWER on the refiner and uses more memory here.** Why:
- EP doesn't touch the attention bottleneck (CP already handles it); it only reshuffles the
  MoE FFN, adding all-to-all comm without reducing compute (CP already sharded FFN by sequence).
- At 1080p the activation all-to-all (≈2 GB/layer, ×2 passes) exceeds FSDP's weight all-gather
  (≈1.2 GB/layer). Token volume dominates at high resolution.
- (Impl caveat: current EP replicates each token per expert-assignment (8×) instead of dedup
  per destination rank; an optimized EP would be faster but still not beat CP+FSDP for this
  attention-bound single-request workload.)

**Takeaway: for single-request 1080p latency, CP is the right axis; EP hurts.**
EP wins in other regimes: high-throughput/large-batch (FFN-bound, amortized all-to-all),
memory-bound experts, or scaling beyond CP's head limit (num_heads=16 → CP≤16).

## MoE weight handling under CP4 (FSDP vs replication vs EP) — decisive

All CP4 + batch_cfg; only the MoE weight handling differs:

| MoE handling | Base/step | Refiner/step | End-to-end | Peak mem |
|--------------|-----------|--------------|-----------|----------|
| **FSDP** | 2.16s | 31.2s | **9:32** | **117 GB** |
| Replication + base offload | 2.16s | 31.2s | 12:02 | 148 GB |
| EP4 (unoptimized) | 5.16s | 88.3s | 20:47 | 160 GB |

- **FSDP all-gather is fully hidden by prefetch/overlap**: FSDP and replication have *identical*
  per-step times (base 2.16=2.16, refiner 31.2=31.2). Removing the gather (replication) buys nothing.
- Replication is actually WORSE end-to-end (offload 60GB base↔CPU overhead) and uses more memory.
  → FSDP is strictly better here (same speed, less memory, no offload dance).
- **The refiner is 31 s/step under ALL FFN-weight schemes** → it is attention-bound; how experts are
  sharded is irrelevant to the bottleneck. To speed the refiner, attack ATTENTION (CP8, fewer refiner
  steps, lower refiner res, faster attention kernel), not the FFN weight layout.
- (Correction: an earlier note listed CP4+FSDP base as ~5s; measured value is 2.16s. The 5.16s was EP4.)

## HOT-inference optimization (agent-driven; model resident, load excluded)

Metric redefined to hot-inference latency (warm server, one request): base_denoise +
refiner_vae + refiner_denoise + vae_decode; model loading EXCLUDED. Autonomous agent
(agent_opt/) searched lossless 4-GPU parallelism; decisive win:

| config | base_denoise | refiner_denoise | hot_total | peak | vs golden |
|--------|--------------|-----------------|-----------|------|-----------|
| golden CP4+FSDP+batch_cfg (FA2) | 89.1s (2.16/step) | 263.2s (31.2/step) | 375.6s | 117 GB | 1.0× |
| **+ LINGBOT_ATTN_KERNEL=cudnn** | 62.0s (1.55/step) | **123.9s (15.49/step)** | **207.9s** | 113.5 GB | **1.81×** |

- Root cause: the FA2 2.8.3 shim runs a **suboptimal kernel on Blackwell sm_100**; torch's
  **cuDNN 9.15 has a fast Blackwell flash kernel** (microbench: cuDNN 191 ms vs FA2 547 ms/attn).
- Fix: cuDNN-backed SDPA over the Ulysses-gathered packed sequence, split by cu_seqlens into
  per-segment dense SDPA (mathematically identical block-diagonal attention). Priority-list
  fallback [cuDNN, flash, efficient] handles shapes cuDNN rejects (480p base). OFF path unchanged.
- The Ulysses all-to-all is ~ms (negligible), so the lever was the kernel, not the collective →
  Ring/EP/TP rejected by the data. Refiner attention now ~56% of hardware peak (near roofline).
- Repro: `sbatch agent_opt/config/c5_cudnn_attn.sbatch`. Full report: `agent_opt/REPORT.md`.

## MoE megafusion (torch.compile) + overlap attempt

MoE block internal breakdown (1080p, 126K tokens/rank) — the matmul is NOT the cost:
| part | ms | % |
|------|----|----|
| router | 3.88 | 7.8% |
| reorder | 2.06 | 4.2% |
| grouped_mm (matmul) | 19.5 | 39.2% |
| **restore (scatter)** | **23.1** | **46.5%** |
| shared_expert | 0.87 | 1.8% |
| FULL eager | 49.7 | 100% |
→ "glue" (router+reorder+restore) = 58.5%; the restore scatter (memory-bound) alone is 46.5%.

**Approach 1 — megafusion (`LINGBOT_COMPILE_MOE=1`, runner._maybe_compile_moe): WORKS.**
torch.compile fuses the pointwise glue (restore scatter+weighted-sum, router, silu-gate, shared)
into fused Triton kernels + autotunes the matmul.
- MoE block: 49.7 → 23.4 ms = **2.13×**.
- End-to-end base denoise (1-GPU 480p, steady-state): 4.29 → 3.08 s/step = **1.39× (-28%)**.
- Correctness: max_abs_diff 4.3e-3 (bf16). One-time compile cost excluded (warm server).

**Approach 2 — cross-request stream overlap: NEGATIVE (1.01×).** Attention saturates all SMs,
so a concurrent FFN stream has no spare capacity (greedy CUDA scheduling → serialize). Would need
SM partitioning (MPS/green-contexts), but carving SMs from the attention bottleneck is a net wash.
Same root cause as "FFN too small to disaggregate": the bottleneck stage fills the GPU.

Takeaway: attack the MoE's memory-bound GLUE (fuse it as epilogues), not the matmul or overlap.

## Acceleration axes still to measure (task 3)
- [ ] Install FlashAttention-3 for Blackwell → unlock `batch_cfg` (halve sequential CFG) + CP.
- [ ] MoE backend: grouped_mm → sglang_triton → **sglang_triton_fp8** (native FP8 on GB200).
- [ ] Context-parallel CP2/CP4 (single node) for latency scaling (needs FA3).
- [ ] Steps / VAE dtype / torch.compile.
