# Search Space: Lossless 4-GPU parallelism (MoE × Attention × placement × scheduling)

Method families and knobs only. This does NOT prescribe the winning combination —
discover it from the code, traces, and measured results. Constraint everywhere:
**exactly 4 GPUs, one NVLink node; output must stay lossless (PSNR ≥ 45 dB vs golden).**

## Hard facts that bound the space
- `num_attention_heads = 16` → Ulysses degree ≤ 16 (each rank needs ≥1 head). CP4 → 4 heads/rank.
- `num_experts = 128`, `top_k = 8`, `n_shared_experts = 1`, 48 layers, hidden 2048.
- Two DiTs (base 30B + refiner 30B) load together (~120 GB) unless sharded/offloaded.
- Refiner @1080p ≈ 250K tokens → attention (O(n²)) is the measured bottleneck (31 s/step
  under CP4; FFN weight scheme barely moves it). Base @480p ≈ 48K tokens, 2.16 s/step.
- Any 4-GPU factorization must multiply to 4: e.g. CP4, CP2×EP2, CP2×TP2, EP4, TP4,
  CP4 with FSDP/replication as an orthogonal memory choice.

## A. Attention parallelism (highest leverage — this is the refiner bottleneck)
- **Ulysses (context parallel)** — IMPLEMENTED. `--context_parallel_degree N
  --context_parallel_ulysses_anything`. Head-split all-to-all. Degree ≤ 16.
- **Ring attention** — NEEDS IMPL. Block-wise, P2P, sequence-split; not head-limited.
  Would let you scale attention parallelism beyond the head count and/or trade all-to-all
  for P2P ring sends. Compose with Ulysses (USP / unified SP): e.g. Ulysses within a
  head-group, Ring across the rest.
- **Ulysses × Ring hybrid** — NEEDS IMPL. On 4 GPUs, e.g. Ulysses2 × Ring2. Explore whether
  the hybrid reduces the all-to-all volume or improves overlap vs pure Ulysses4.
- **Attention kernel** — FA2 (via `slurm/shims/flash_attn_interface.py`) is the current
  varlen backend. Note whether the kernel or the collective dominates.

## B. MoE FFN parallelism (measured to barely affect the refiner bottleneck — but affects
   base + memory; may matter more if attention is sped up and FFN becomes the new floor)
- **Replication** — full experts resident per rank; 0 weight comm; needs memory (offload).
  `slurm/accel_cp4_replicate_offload.sbatch`.
- **FSDP** — `--enable_fsdp_inference`. Shards weights; per-layer all-gather is overlap-hidden
  (measured 0 per-step cost). Current default; lowest memory.
- **EP (expert parallel)** — IMPLEMENTED (`LINGBOT_MOE_EP=1`, `_run_ep_experts`).
  Correct (bit-exact) but the current impl has NO comm/compute overlap and replicates
  each token per expert-assignment (8×) → slow at 1080p. **Improving it is fair game:**
  dedup tokens per destination rank, chunk+pipeline the dispatch/compute/combine to overlap
  the all-to-all with grouped_mm, or shard experts at load instead of lazily.
- **Expert TP (tensor parallel within experts)** — NEEDS IMPL. Shard each expert's w1/w2/w3
  by the intermediate dim, all-reduce activations. Compose with CP.

## C. Weight / activation placement & memory
- **FSDP all-gather prefetch** (built into PyTorch FSDP2) — already overlaps.
- **CPU offload of the base DiT before the refiner** — `LINGBOT_OFFLOAD_BASE_BEFORE_REFINER=1`.
  Currently SERIAL (base fully done → whole DiT copied to CPU → refiner). The 60 GB copy is
  visible end-to-end overhead (~fraction of a minute). **Scheduling is fair game:**
  overlap the offload with the tail of base compute (pre-offload), and/or prefetch the
  refiner onto the freed memory while base finishes.
- **Layer-wise / streaming** weight residency for the refiner.

## D. Two-stage scheduling
- Serial (current).
- Overlapped stage transition: pre-offload base while its last layers compute; prefetch
  refiner in parallel; hide the ~stage-swap latency.
- (Disaggregating base and refiner onto disjoint GPU subsets breaks the "4 GPUs, one job"
  assumption for a single request — treat as out of scope unless it still totals 4 and
  stays lossless.)

## Knobs reference (CLI flags on scripts/inference.py, via torchrun)
- `--context_parallel_degree N`  (Ulysses degree)
- `--context_parallel_ulysses_anything`
- `--enable_fsdp_inference`
- `--batch_cfg --refiner_batch_cfg`  (needs FA2/FA3; B=2 packed attention path)
- `--reuse_condition_features`
- Env: `LINGBOT_MOE_EP`, `LINGBOT_OFFLOAD_BASE_BEFORE_REFINER`,
  `LINGBOT_MOE_EXPERT_BACKEND` (grouped_mm | sglang_triton | sglang_triton_fp8 — last two lossy),
  `PYTORCH_ALLOC_CONF=expandable_segments:True`.

## Frontier to beat (from RESULTS.md, 4×GB200, e2e wall-clock)
- CP4 + FSDP + batch_cfg : refiner 31.2 s/step, e2e **9:32**, peak 117 GB  ← current best (golden)
- CP4 + replication+offload : refiner 31.2, e2e 12:02, peak 148 GB (offload overhead)
- CP4 + EP4 (naive) : refiner 88.3, e2e 20:47, peak 160 GB (no overlap / 8× dup)

## Where to push (hypotheses, not prescriptions)
- Attack attention: Ring or Ulysses×Ring to change the collective; confirm the 16-head
  ceiling is/isn't binding at CP4; check overlap of the Ulysses all-to-all.
- Overlap the stage-transition offload (kill the 60 GB serial copy cost).
- Only if attention drops enough that FFN becomes the floor: an *overlapped* EP or expert-TP.
