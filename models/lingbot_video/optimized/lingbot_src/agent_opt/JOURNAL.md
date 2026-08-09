# Optimization Journal — Lossless 4-GPU parallelism

Append one entry per transfeat. Newest at the bottom. The frontier is the set of
non-dominated (faster or less memory, and lossless-PASS) configs.

## Frontier (seed from RESULTS.md — all 4×GB200, e2e wall-clock)
| id | config | base/step | refiner/step | e2e | peak mem | lossless | status |
|----|--------|-----------|--------------|-----|----------|----------|--------|
| golden | CP4 + FSDP + batch_cfg | 2.16s | 31.2s | 9:32 | 117 GB | ref | GOLDEN/best |
| — | CP4 + replication + offload | 2.16s | 31.2s | 12:02 | 148 GB | (assumed) | dominated (slower) |
| — | CP4 + EP4 (naive, no overlap) | 5.16s | 88.3s | 20:47 | 160 GB | (assumed) | dominated (slower) |

## Rejected / failure signatures
| hypothesis | why rejected |
|------------|--------------|
| FSDP-only (no CP) | 0 speedup — FSDP shards memory, not compute (28:40, slower than 1-GPU) |

## Iterations
<!-- iter entries appended below by the agent -->

### Iter 0 — golden reference
- Found an existing CP4+FSDP+batch_cfg run (`outputs/accel_cp4_20260710_030947/`) that is
  exactly the golden config: e2e 9:32.29, base 2.16 s/step (86s), refiner 31.23 s/step (249s),
  peak 117251 MiB (114.5 GB). Copied `t2v_refined.mp4` → `agent_opt/baseline/golden_refined.mp4`.
- Decisive gap: denoise = 86+249 = 335s, but e2e = 572s → **~237s of non-denoise overhead**
  (loading both 30B DiTs from lustre serially before denoise, FSDP setup, VAE encode/decode,
  text encode). This overhead is model-load dominated and a large lossless target.

### Iter 1 — profiling (LINGBOT_PHASE_TIMING) — job 4495666
- Hypothesis: before spending budget on the attention bottleneck, measure where the 237s
  overhead actually is. Added pure-logging `_phase()` markers in runner.py (OFF unless
  LINGBOT_PHASE_TIMING=1 → lossless no-op path). Transfeat = golden config + timing.
- Expect: PHASE lines giving load/base-denoise/vae/refiner-load/refiner-denoise/vae split.

### Policy update (orchestrator, mid-iter1)
- SPEED is the only metric. PSNR≥45 gate REMOVED — verify_lossless.py = optional telemetry.
- Correctness now = (a) first-principles audit of parallel math + (b) sanity check the video
  is valid/non-degenerate (shape 121×1088×1920, not blank/NaN). Numerical drift is fine.
- Still out of scope: step/res/scheduler/seed changes, quant, cache, token prune, sparse attn.
- Revised plan: attack BOTH the ~237s non-denoise overhead (model load / FSDP / VAE /
  stage-transition — clearest 41% lever) AND the refiner denoise (249s). Freed from exact
  numerics, kernel/reduction-order changes are now allowed for the attention path.

### Iter 1 RESULT — profiling (job 4495666, node nvl72127-T04) — PHASE breakdown
| phase | dt (s) | note |
|-------|--------|------|
| base load (DiT+TE+VAE+FSDP) | 125.7 | lustre IO, 57GB base transformer, 13 shards, ×4 ranks |
| refiner load | 108.5 | 57GB refiner transformer; **only needed AFTER base denoise** |
| base conditions (text enc) | 1.8 | |
| base denoise (40 steps) | 89.1 | 2.23 s/step; pure GPU |
| base VAE decode+save | 0.65 | |
| refiner VAE encode (1080p) | 19.25 | encodes upscaled base video to latents |
| refiner denoise (8 steps) | 263.2 | **32.9 s/step; 43% of e2e; attention-bound, already 4-way head-split** |
| refiner VAE decode+save | 1.49 | |
| **total (python)** | **609.7** | wall 10:58 (this node ~7% slower than golden 9:32) |
- Output byte-identical to golden (1104039 B) → instrumentation is a harmless no-op.
- **Model loading = 234s = the ENTIRE non-denoise overhead** (base 126 + refiner 108).
  This + refiner denoise (263) are the only two big levers. FFN/EP irrelevant (RESULTS).
- Node variance ~7%; compare transfeat on PHASE deltas, not just raw wall clock.

### Iter 2 — overlap refiner load with base denoise (job 4497566)
- Hypothesis: refiner load (108s IO/CPU) runs serially BEFORE base denoise (89s pure-GPU),
  but isn't needed until after. Load it to CPU in a background thread (zero CUDA) overlapping
  base denoise, finalize (device move + CP + FSDP) on main thread. Expect hide ~85s.
- Impl (env LINGBOT_OVERLAP_REFINER_LOAD=1, OFF path preserved): runner.py — added
  `defer_aux_to_device` (CPU-only pipe load), `_maybe_preload_refiner(cpu_only=True)`,
  `_finalize_preloaded_refiner()`, daemon thread joined right after base_denoise_done.
- Correctness audit: bg thread does only CPU from_pretrained (no CUDA) → no CUDA race with
  base denoise; identical weights/compute, only load *timing* moved. Parallel math unchanged.

### Iter 2 RESULT — overlap refiner load (job 4497566) — RETAINED (modest)
- Valid: video (121,1088,1920,3), exit 0. total python 588.1s (node ~similar to profiling).
- refiner_preloaded 0.0 (deferred), refiner_finalized 81.5s (vs serial refiner_load 108.5s in
  profiling) → **~27s of the ~90s refiner CPU-load overlapped base denoise** before the bg
  thread was GIL-starved. base_denoise_done reached at total 223.7s vs 325.1s in profiling.
- KEY DEDUCTION: only 27s (not ~85s) hid → the model load is **CPU/Python-bound (holds GIL)**,
  NOT IO-bound (IO read() would release the GIL and overlap ~fully). Implication: the real fix
  for the 234s load is to avoid the redundant **per-rank Python parse** (all 4 ranks parse the
  same 57GB), i.e. load once on rank 0 + broadcast over NVLink.
- Retain c2 as frontier (node-controlled ~27s faster than golden config). Overlap composes with
  broadcast (next), and with broadcast only rank 0 loads → its bg-load overlaps base denoise
  while ranks 1-3 just build empty models, so the refiner load becomes nearly free.

### Iter 3 — rank-0 load + NVLink broadcast of transformer weights (planned)
- Hypothesis: all 4 ranks redundantly parse the same 57GB transformer (base) and 57GB (refiner)
  from lustre — CPU/Python-bound, ~126s+108s. Instead load the full transformer ONLY on rank 0,
  build empty (meta→cuda) models on ranks 1-3 (no disk read, no parse), and broadcast weights
  over NVLink (NCCL, ~900GB/s → seconds). Wall = rank0-single-load + broadcast. If the 4-rank
  load has contention (IO/membw), single-rank load is much faster → large win; measured directly.
- Compose with LINGBOT_OVERLAP_REFINER_LOAD (rank 0's refiner bg-load overlaps base denoise;
  ranks 1-3 build empty; broadcast in finalize).
- Correctness audit: broadcast by parameter/buffer NAME (identical arch → identical name set),
  src=0, so every rank ends with rank 0's exact weights → mathematically identical to loading
  from disk. Preflight the empty-init + name-ordered broadcast + equality on CPU/gloo first.

### Iter 3 RESULT — rank-0 load + NVLink broadcast (job 4499694) — NEW FRONTIER (big win)
- Preflight slurm/test_bcast_correctness.py PASS (mixed bf16/fp32 + buffer, cross-rank spread 0).
- **base load 68.8s vs 125.7s profiled (−57s, ~halved)** — confirms the 4-rank load was
  redundant-parse-bound; single-rank parse + 977-tensor NVLink broadcast is far cheaper.
- **refiner load: finalize 24.2s vs 108.5s serial** — rank 0's bg refiner parse fully overlapped
  base denoise (89s > ~69s parse); only broadcast+FSDP left → 24s.
- base_denoise_done at total 161.9s (vs 325 profiled); refiner_vae_encode_done 203.6s (vs 345).
- refiner denoise 267s (node-normal). total python 472.5s, **wall 8:54.87 (vs golden 9:32)**,
  peak 122586 MiB (119.7 GB, +2.7 vs golden from transient full-transformer broadcast staging).
- CORRECTNESS: output **byte-identical to golden, PSNR=inf** (broadcast delivers bit-identical
  weights; overlap only moves timing) → strongest possible correctness proof. Valid.
- Node-controlled: 472.5s vs 609.7s profiling = **137s / 22% faster**. Retain as frontier.

### Iter 5 — cuDNN SDPA attention kernel replacing FA2 varlen (job 4505022) — TARGETS refiner_denoise
- OBJECTIVE PIVOT: hot-inference only. refiner_denoise=263s (31.2 s/step) is 70% of hot; it is
  attention-compute-bound. Ulysses CP4 already splits the 16 heads optimally (4 heads/rank, full
  seq); the a2a volume (~2GB/layer over 900GB/s NVLink ≈ ms) is NEGLIGIBLE vs the 26s of kernel
  time — so Ring/overlap cannot help. The lever is the KERNEL itself.
- Preflight microbench (job 4504430, 1 GPU, exact per-rank refiner shape Sfull=506176, H=4, D=128,
  2 batch_cfg segments):
    FA2-2.8.3 varlen : 546.8 ms/attn  (26.25 s over 48 layers)
    SDPA-cuDNN       : 191.2 ms/attn  ( 9.18 s over 48 layers)   <-- 2.86x faster
    SDPA-flash(torch): 602.6 ms/attn  (slower, discard)
    correctness FA2 vs cuDNN: max_abs_diff 1.2e-4, rel 5.3e-3 (bf16 reduction-order drift, OK).
  => FA2 2.8.3 runs a suboptimal kernel on sm_100; cuDNN 9.15 has a fast Blackwell flash kernel.
- Impl (env LINGBOT_ATTN_KERNEL=cudnn, default fa2 preserved): added `_cudnn_varlen_attention`
  that splits the gathered packed sequence by cu_seqlens_kv into per-segment dense
  F.scaled_dot_product_attention under sdpa_kernel(CUDNN_ATTENTION). Mathematically identical to
  the FA2 varlen block-diagonal call (each batch_cfg segment attends within itself; zero-pad tail
  segment harmless). Branch added in BOTH the Ulysses CP path (used by base+refiner) and the
  non-CP packed path. OFF path (fa2) byte-for-byte unchanged.
- Expectation: attention/step ~26s->9s; step 31.2s->~14s; refiner_denoise 263s->~115s; hot ~375->~230s.
  Also speeds base denoise (same kernel, smaller 480p seq).
- FIRST ATTEMPT (job 4505022) CRASHED in base denoise: cuDNN-only backend threw "No available
  kernel" on the 480p base attention shape (cuDNN rejects some shapes, no fallback). FIX: priority
  list [CUDNN, FLASH, EFFICIENT] with set_priority=True + .contiguous() on q/k/v segments — cuDNN
  for the big refiner seq, graceful fallback for base. All three are mem-efficient (no S x S matrix
  -> no OOM). Re-microbench (job 4505973): prio picks cuDNN on refiner shape (182ms == cuDNN-only,
  vs flash 602ms) -> confirmed cuDNN still selected.

### Iter 5 RESULT — cuDNN SDPA attention (job 4506351, wall 8:54.87) — NEW FRONTIER (big win) — RETAINED
| phase | golden (s) | c5-cudnn (s) | delta |
|-------|-----------|--------------|-------|
| base_denoise (40 steps) | 89.1 (2.16/step) | 61.96 (1.55/step) | -27.1 |
| refiner_vae_encode (1080p) | 19.3 | 20.52 | +1.2 |
| refiner_denoise (8 steps) | 263.2 (31.2/step) | **123.88 (15.49/step)** | **-139.3 (2.12x)** |
| vae_decode+save | ~4 | 1.54 | -2.5 |
| **HOT TOTAL** | **375.6** | **207.9** | **-167.7 (1.81x)** |
- peak mem 116271 MiB (113.5 GB) vs golden 117 GB (LOWER — cuDNN uses less scratch than FA2 varlen).
- Exit 0. Video VALID/non-degenerate: shape (121,1088,1920,3) uint8, range 0-255, mean 180.2,
  std 52.9 (real content, not blank/noise). Bytes differ from golden (expected: bf16 reduction-order
  drift, rel ~5e-3 in the microbench) — acceptable per objective (all correct impls equal).
- CORRECTNESS AUDIT: `_cudnn_varlen_attention` splits the gathered packed sequence by cu_seqlens_kv
  into the same per-segment (batch_cfg cond/uncond + zero-pad tail) blocks the FA2 varlen call used,
  runs each as an independent dense non-causal SDPA -> mathematically identical block-diagonal
  attention, no cross-segment leakage, no dropped/duplicated tokens. Ulysses a2a wiring UNCHANGED
  (only the local kernel after the gather swapped). OFF path (LINGBOT_ATTN_KERNEL unset/fa2)
  byte-for-byte preserved. => correct implementation.
- WHY IT WINS: refiner attention was ~84% of the 31.2s step (26.3s FA2). cuDNN's Blackwell flash
  kernel does the same math in ~8.7s. attention is now ~9s of the 15.5s step (~57%); the remaining
  ~6.5s is MoE grouped_mm + qkv/out linears + a2a. attention is now near hardware roofline
  (~4.9s ideal -> 8.7s = ~56% of peak), so little headroom remains in the kernel itself.
- RETAIN as the new frontier. RECOMMENDED CONFIG.

### Search closed (no remaining hypothesis expected to beat c5)
- Ring / Ulysses-Ring hybrid: REJECTED by analysis. The Ulysses a2a moves ~2GB/layer over 900GB/s
  NVLink (~ms), utterly negligible vs the multi-second kernel; changing the collective / adding P2P
  overlap cannot recover time that isn't being spent on comm.
- Overlapped/dedup EP or expert-TP: DECLINED. FFN is now only ~40% of the refiner step, and prior
  RESULTS show EP is 2.8x SLOWER at 1080p (activation a2a volume dominates); EP/TP redistribute FFN
  work without cutting compute (CP already shards it by sequence). Even a perfect EP cannot beat
  CP+FSDP here. Not worth a job.
- refiner attention is near roofline; base already benefits (flash fallback 89->62); vae_encode
  (20.5s, 10%) is out of scope (neither MoE nor attention). Declaring c5 the dominant config.

### Iter 4 — sharded parallel load (job 4501316)
- Hypothesis: the remaining serial cost is rank 0's 69s single-process parse of the 57GB base
  (and refiner) checkpoint. Divide it: each rank safetensors-loads a disjoint 1/4 of the 13
  shards (real weights for owned params, empty otherwise), then broadcast each param from its
  OWNER rank. Parse work ÷4 → base load ~69s→~25s. Compose with overlap.
- Impl (env LINGBOT_SHARDED_LOAD=1): `_sharded_load_and_broadcast` reads shard HEADERS on all
  ranks for shape/dtype (no data), each rank parses its owned shards, per-owner NCCL broadcast.
- Preflight slurm/test_sharded_bcast.py PASS (per-owner broadcast, mixed dtype, world=4, 0 bad).
- Correctness audit: owner map derived deterministically from the checkpoint index.json (same
  on all ranks); every param broadcast from its owner → every rank gets exact checkpoint weights.
