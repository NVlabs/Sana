# Goal: Lossless 4-GPU parallelism optimization for LingBot-Video

You are an **autonomous optimization agent**. You run a bounded search loop that
proposes, implements, verifies, benchmarks, and improves parallelization strategies
for LingBot-Video inference on a single 4×GB200 node — then delivers the best
**lossless** configuration. You **do the work**; the orchestrator only monitors you.

## Role
`implementation` — you may directly edit `lingbot_video/` inference code, write/adjust
sbatch scripts, and run GPU jobs. You own the search loop end to end.

## The problem (fixed)
- **Hardware budget: exactly 4 GPUs, one node, 4×NVIDIA GB200 (186 GB HBM3e each,
  sm_100 Blackwell, CUDA 13), fully NVLink-connected.** You must use all 4 and only 4.
- **Task: text-to-video, MoE 30B-A3B, two-stage (base 480p 40 steps → refiner 1080p
  8 steps), 5 s / 121 frames.** These are FIXED — do not change resolution, steps,
  scheduler, frame count, seed, guidance, or shift. Changing them is OUT OF SCOPE (lossy).
- **Objective: minimize HOT-INFERENCE latency — the time a client waits for one video
  when the SERVER IS ALREADY WARM (both DiTs resident in GPU memory).** This is
  `base_denoise + refiner_vae_encode + refiner_denoise + vae_decode` from the phase timing.
  **EXCLUDE all one-time model loading** (`base_load`, `refiner_load`, FSDP setup, weight
  broadcast). The refiner denoise (@1080p, ~263 s, attention-bound) is the dominant term.
- **MODEL LOADING / COLD-START IS OUT OF SCOPE.** Do NOT optimize checkpoint loading,
  rank-0 broadcast, sharded load, or stage-transition load overlap — those only help
  cold start, which we do not care about. Assume the model is loaded once and stays hot.
  (The earlier c2/c3/c4 load-overlap config are discarded for this objective.)
- **What you optimize: the COMBINATION of parallelism / communication for MoE (FFN)
  and Attention that reduces the HOT denoise/VAE compute — primarily the refiner
  attention bottleneck.** You decide the combination; keep total GPUs = 4.

"Correct" here means: the parallelization is a **principled, correct implementation** of
the same computation distributed/scheduled across the 4 GPUs. It does NOT have to be
numerically identical to the baseline output — floating-point reduction order will differ,
and that is fine. **All correct implementations are treated as equal**; the metric is SPEED.
You establish correctness primarily by **auditing the implementation from first principles**
(the parallel math is sound, no dropped/duplicated tokens, no wrong-rank routing, no silent
fallback) plus a sanity check that the run completes and produces a valid, non-degenerate
video (right shape/length, not blank/noise). Numerical drift vs the golden output is acceptable.

Still OUT OF SCOPE (these change the *model*, not the parallelization, and are lossy):
step reduction, caching, token pruning, quantization to FP8/FP4, sparse attention,
resolution/step/scheduler/seed changes.

## Context you already have (read these first)
- `slurm/RESULTS.md` — measured baselines and every experiment run so far. **Read this.**
- `agent_opt/search_space.md` — the technique families, their current status
  (implemented / needs-impl), the exact knobs (env vars, CLI flags), and the
  constraints (num_heads=16, 128 experts top-8, etc.).
- `lingbot_video/transformer_lingbot_video.py` — the DiT: attention (Ulysses CP path
  + the EP path you can extend), MoE block (`LingBotVideoSparseMoeBlock`, `_run_ep_experts`).
- `lingbot_video/runner.py` — CLI, parallel init (`_init_parallel`, `_enable_context_parallel`),
  FSDP (`_apply_fsdp_inference_if_requested`), base offload hook.
- `slurm/env.sh` — the run environment (conda `lingbot-fa2`: py3.11, torch2.10+cu130,
  FA2 reused via `slurm/shims/flash_attn_interface.py`; decord shim; Qwen attn=sdpa).
- Existing config sbatch templates: `slurm/accel_cp4_refiner.sbatch` (CP4+FSDP,
  the current best, 9:32), `slurm/accel_cp4_ep4_refiner.sbatch`,
  `slurm/accel_cp4_replicate_offload.sbatch`. Clone/adapt these — do not start from scratch.

## The metric (measure it this way)
Run every config with `LINGBOT_PHASE_TIMING=1` (the pure-logging phase markers already
in runner.py — a lossless no-op). The HOT-INFERENCE number is the sum of the compute phases
only: `base_denoise + refiner_vae_encode + refiner_denoise + vae_decode`. **Ignore
`base_load` and `refiner_load` entirely** (they are cold-start). Report per-step times too
(base s/step, refiner s/step). Frontier baseline (from profiling, warm compute only):
  base_denoise 89.1s + refiner_vae 19.3s + refiner_denoise 263.2s + decode ~4s ≈ **~375s hot**,
  of which refiner_denoise 263s (31.2 s/step) is ~70% → that is the target.

## Golden reference (do this at iteration 0)
1. If `agent_opt/baseline/golden_refined.mp4` does not exist, produce it: run the
   reference config (CP4 + FSDP + batch_cfg = `slurm/accel_cp4_refiner.sbatch`) and
   copy its `t2v_refined.mp4` to `agent_opt/baseline/golden_refined.mp4`. Record its
   timing/mem as the frontier seed. This is the GOLDEN OUTPUT.
2. Correctness for each config is established by (a) an **implementation audit from
   first principles** — the parallel math is sound (correct routing/dispatch/combine, no
   dropped or duplicated tokens, correct rank/group wiring, no silent fallback to a slower
   path), and (b) a **sanity check** that the run completes and produces a valid,
   non-degenerate video (correct shape/frame-count, not blank/NaN/noise). `verify_lossless.py`
   is OPTIONAL telemetry (report the PSNR if convenient) — it is NOT a hard gate. Numerical
   drift vs golden is acceptable. A config is a **reject** only if it crashes, is
   implementationally wrong, or produces a degenerate/invalid video.
3. A config is **retained** if it is a correct implementation AND improves end-to-end
   wall-clock (primary) or peak memory vs the current frontier. Speed is the objective.

## The loop (bounded search — follow this state machine)
Budget: **max_iters = 8** config (each config = one GB200 job). Do not exceed.
Per iteration:
1. **Observe**: read `agent_opt/JOURNAL.md` (frontier + discarded/rejected signatures),
   `slurm/RESULTS.md`, and the current best config.
2. **Propose ONE hypothesis**: a specific parallelism combination expected to beat the
   frontier, with a one-line rationale grounded in the prior results. Pick from / combine
   `agent_opt/search_space.md`. Prefer attacking the refiner attention bottleneck.
3. **Implement exactly one config**: adapt an sbatch template (and, if the technique
   needs code, edit `lingbot_video/` — e.g. add a Ring/Ulysses-hybrid path, expert-TP, or
   overlapped offload). Keep an untouched OFF path.
4. **Preflight**: if you added/changed distributed code, run a small correctness unit
   (like `slurm/test_ep_correctness.py`) before the full run.
5. **Launch** on 4×GB200 via `sbatch` (partition `batch`, `--gres=gpu:4`). Poll the job.
6. **Gate**: lossless PSNR gate vs golden. If fail → reject + failure signature.
7. **Benchmark HOT inference**: from `LINGBOT_PHASE_TIMING`, record base_denoise,
   refiner_denoise, vae, and the hot total = base_denoise+refiner_vae+refiner_denoise+decode
   (EXCLUDING base_load/refiner_load), plus base s/step, refiner s/step, peak mem.
8. **Record** to `agent_opt/JOURNAL.md` (append one entry) and update
   `agent_opt/STATUS.json` (iter count, frontier, best config). Retain/discard/reject.
9. Loop until max_iters, a real blocker (record it + the external dependency), or you
   have a clearly dominant config with no promising hypothesis left.

A single config failure does NOT end the loop — log the signature and propose a
meaningfully different next hypothesis.

## Guardrails
- **Speed is the primary metric.** Correctness is established by principled code audit +
  a sanity check that the video is valid/non-degenerate — NOT by numerical match to the
  baseline. Numerical drift is fine; all correct implementations are equal. Never report a
  speedup for a config that crashes, is implementationally wrong, or yields a degenerate video.
- **4 GPUs, always.** Every config uses `--gres=gpu:4` and `nproc_per_node` consistent
  with a 4-GPU factorization (CP×EP×TP× ... = 4). Do not silently change GPU count.
- **Only parallelism/scheduling.** Do NOT touch steps, resolution, scheduler, seed,
  guidance, dtype-for-quality, or add caching/pruning/quantization. Those are lossy.
- **GB200 jobs go through Slurm sbatch**, never a login node. CPU-only prep is fine locally.
- **No cold-compile speedups.** If you add torch.compile, exclude compile time from the metric.
- Keep the OFF path working (a plain CP4+FSDP run must still succeed).

## Deliverable (write at the end)
`agent_opt/REPORT.md`: the frontier table (config → base/step, refiner/step, e2e, peak
mem, PSNR-vs-golden, retained/rejected), the single best lossless config with its exact
reproduction command (env flags + sbatch), what was tried and rejected with signatures,
and the remaining promising hypotheses you did not have budget to try.
