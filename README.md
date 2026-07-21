# autovideo

> **Orchestrating end-to-end acceleration?** Start at [`agents/orchestrator-entry.md`](agents/orchestrator-entry.md) — the main-agent runbook (scan -> fan out per-dimension agents -> gate -> integrate -> deliver 3 tiers).

`autovideo` is a self-contained workspace for video-generation acceleration:
model runtimes, acceleration techniques, declarative experiment candidates,
launch/collect automation, quality gates, and agent orchestration protocols —
with no serving-framework dependency (the former SGLang engine submodule has
been removed; see [`snippets/sol-ltx-infer-reference.md`](snippets/sol-ltx-infer-reference.md)
for pointers into the engine repo).

## News

- **2026-07-21** — Merged [**SOL Attention**](techniques/sparse_backends/)
  (PISA-family block-sparse attention for video DiTs) and used it to optimize
  two models end to end:
  - [**HunyuanVideo-13B**](models/hunyuan_video/): **5.03×** hot generation
    speedup (856.1 s → 170.4 s, audited hot-vs-hot on one GB200) with
    torch.compile + TeaCache + [**SOL Attention v3**](techniques/sparse_backends/sol_attn_hunyuan_v3.py)
    (dense-text / sparse-video split, Morton3D reorder, dense first 3 steps) —
    [`candidates/hunyuan_video_full_v3.toml`](candidates/hunyuan_video_full_v3.toml).
  - [**Wan2.1-T2V-14B**](models/wan21_t2v_14b/): **3.48×** warm generation
    speedup (563.8 s → 161.8 s) with
    torch.compile + EasyCache + [**SOL Attention**](techniques/sparse_backends/sol_attn_backend.py)
    (colmask kernel, density 0.15) —
    [`candidates/wan21_14b_fullstack.toml`](candidates/wan21_14b_fullstack.toml).

  Side-by-side comparison videos: HF datasets
  `yitongl/hunyuanvideo-sol-comparison` and `yitongl/wan14b-sol-attention-comparison`.
  Reports: [`optimization_reports/`](optimization_reports/).

## Models & Speedups

Generation-time speedups (load excluded, warm/hot steady state, single GPU
unless noted). Each row links to the model runtime and its delivery report.

| Model | Full-OPT speedup | Optimization stack | Evidence |
| --- | --- | --- | --- |
| [**HunyuanVideo-13B**](models/hunyuan_video/) | **5.03×** (856.1 s → 170.4 s, hot, audited) | compile + TeaCache + [**SOL Attention v3**](techniques/sparse_backends/sol_attn_hunyuan_v3.py) (d=0.15, Morton, dense-first-3) | [`hunyuan_sol_splitscreen_rootcause_20260719.md`](optimization_reports/hunyuan_sol_splitscreen_rootcause_20260719.md) |
| [**Wan2.1-T2V-14B**](models/wan21_t2v_14b/) | **3.48×** (563.8 s → 161.8 s, warm) | compile + EasyCache + [**SOL Attention**](techniques/sparse_backends/sol_attn_backend.py) (colmask d=0.15, Morton, dense guard) | `runs/20260721-070957-wan21_14b_fullstack-show-chameleon` |
| [Wan2.2 TI2V-5B](models/wan22_ti2v_5b/) | 2.885× (70.25 s → 24.35 s) | lossless kernel 1.519× ∘ EasyCache | [`wan22_ti2v_5b.md`](optimization_reports/wan22_ti2v_5b.md) |
| [Wan2.2 T2V-A14B](models/wan22_t2v_a14b/) | 1.94× vs 4-GPU CP4 fair baseline (7.59× vs 1-GPU naive) | CP4 Ulysses + fused kernels + EasyCache(0.30) + PISA(0.10) | [`baseline_vs_optimized_latency_matrix_20260714.md`](optimization_reports/baseline_vs_optimized_latency_matrix_20260714.md) |
| [SANA-Video 5B](models/sana_video/) | 4.09× (62.65 s → 15.30 s) | kernel/cache 2.04× ∘ EasyCache ∘ PISA(0.10) | [`sana_video.md`](optimization_reports/sana_video.md) |
| [LingBot-Video](models/lingbot_video/) | 2.60× | kernel + phase-specific PISA | [`lingbot_video_full_pisa_20260713.md`](optimization_reports/lingbot_video_full_pisa_20260713.md) |
| Bernini T2V | 2.257× (128.99 s → 57.14 s, 4-GPU) | kernel 1.594× ∘ EasyCache | [`bernini_t2v.md`](optimization_reports/bernini_t2v.md) |

## SOL Attention

[**SOL Attention**](techniques/sparse_backends/) is the block-sparse attention
family used above. All backends run on GB200/SM100; kernels are vendored
unmodified and consumed through thin adapters.

| Component | What it is |
| --- | --- |
| [`sol_attn_backend.py`](techniques/sparse_backends/sol_attn_backend.py) | v1 — colmask CuTeDSL kernel path for pure-video self-attention ([**Wan**](models/wan21_t2v_14b/)): global-threshold column routing + Morton3D token reorder + dense guard (first layer / first N steps). |
| [`sol_attn_hunyuan_v2.py`](techniques/sparse_backends/sol_attn_hunyuan_v2.py) | v2 — joint `[video, text]` split for [**HunyuanVideo**](models/hunyuan_video/): sparse video×video (colmask) ⊕ exact dense text, merged with the kernel's per-query LSE (NaN-safe). |
| [`sol_attn_hunyuan_v3.py`](techniques/sparse_backends/sol_attn_hunyuan_v3.py) | v3 — aligned line-for-line with Sparse-VideoGen `pisa-bidirectional`: per-block top-k routing, centroid contribution for non-selected blocks (mass-conserving), exact text sink, optional Morton3D; passes a true `density→1.0 == dense SDPA` identity check. |
| [`pisa_hyvideo/`](techniques/sparse_backends/pisa_hyvideo/) | Vendored upstream PISA hyvideo Triton kernels (import-renamed only; see `VENDOR_NOTES.md`). |
| [`scripts/_hunyuan_sol_v3_correctness.py`](scripts/_hunyuan_sol_v3_correctness.py) | GPU correctness gates: merge-convention isolation (dense fake-kernel injection), kernel LSE sanity, identity at density 1.0. |

Two levers matter for quality: **dense first steps** (the denoising trajectory
forks in the earliest steps — fully-sparse early denoising decoheres multi-region
compositions; 3 exact steps recover LPIPS 0.65 → 0.15 on the stress prompt) and
**Morton3D reorder** (compact 3D blocks sharpen routing and centroids; upstream
w-fastest bit-lane order). Full story:
[`hunyuan_sol_v3_alignment_20260719.md`](optimization_reports/hunyuan_sol_v3_alignment_20260719.md),
[`hunyuan_multiprompt_20260719.md`](optimization_reports/hunyuan_multiprompt_20260719.md),
[`hunyuan_sol_splitscreen_rootcause_20260719.md`](optimization_reports/hunyuan_sol_splitscreen_rootcause_20260719.md).

## Quick Start

Create a baseline run bundle without submitting GPU work:

```bash
python3 scripts/launch_candidate.py candidates/baseline.toml --mode dry-run
```

Collect artifacts and write a report for a run bundle:

```bash
python3 scripts/collect_run.py runs/<run-id>
```

Render the Slurm wrapper without submitting GPU work:

```bash
python3 scripts/launch_candidate.py candidates/baseline.toml --mode sbatch
```

Submit the same candidate through Slurm:

```bash
python3 scripts/launch_candidate.py candidates/baseline.toml --mode sbatch --confirm-submit
```

Run it directly on the current node, only when that node is a suitable GPU node:

```bash
python3 scripts/launch_candidate.py candidates/baseline.toml --mode local
```

Compare per-frame quality (SSIM/PSNR/LPIPS) against a baseline run:

```bash
python scripts/video_quality_metrics.py <baseline_frames> <opt_frames> [...]
```

Install Symposium skills for project-local Codex use:

```bash
python3 tools/symposium/install_project_skills.py --target codex
python3 tools/symposium/probe_goal_mode.py --json
```

## Folder Layout

| Path | Purpose |
| --- | --- |
| `models/` | One directory per model: vendored `baseline/` + `optimized/` runners and a canonical `model.toml` contract (see [`docs/model-onboarding.md`](docs/model-onboarding.md)). Includes [**HunyuanVideo**](models/hunyuan_video/) and [**Wan2.1-14B**](models/wan21_t2v_14b/). |
| `techniques/` | Engine-independent acceleration implementations: [**SOL Attention**](techniques/sparse_backends/) sparse backends, cache methods (TeaCache/step cache), transforms (KWL fusion, NVFP4 FFN), registry/compose. |
| `candidates/` | Declarative manifests for baseline and acceleration candidates; one TOML = one reproducible run. |
| `agents/` | Prompt/runbook material for the top-level launch agent. |
| `docs/` | Orchestration design, folder layout, artifact contracts, [model onboarding](docs/model-onboarding.md), and [mechanism issues](docs/mechanism-issues.md). |
| `evals/` | Eval profiles (per-model official configs), metrics, and visual-judge rubrics. |
| `search/` · `search_space/` | Candidate assessment (`plan_eval.py`) and the six-dimension optimization-space docs. |
| `optimization_reports/` | Per-model delivery reports and speedup matrices. |
| `snippets/` | Small reference snippets, incl. pointers into the removed SGLang engine repo. |
| `loops/` · `workflow/` · `orchestration/` | Independent sub-loop/goal folders and per-dimension agent workflows. |
| `scripts/` | Launch/collect/quality tooling (`launch_candidate.py`, `collect_run.py`, `video_quality_metrics.py`, public-reference alignment probes). |
| `tools/symposium/` | Vendored Symposium skill pack plus adapters for preparing Codex interactive goal bundles. |
| `runs/` | Generated run bundles, logs, videos, frames, and reports. Ignored except for `runs/README.md`. |

## Candidate Lifecycle

1. Describe one candidate in `candidates/*.toml`.
2. Generate a run bundle with `scripts/launch_candidate.py`.
3. Launch through `local` or `sbatch`.
4. Collect `run.log`, `out.mp4`, timing files, extracted frames, and a report
   under the run directory with `scripts/collect_run.py`.
5. Gate quality against the model's baseline frames
   (`scripts/video_quality_metrics.py`, aligned LPIPS + visual judge).
6. Compare against the official target-model baseline before promoting a candidate.

Benchmark convention: runners emit `benchmark.json` with schema v2 flat keys
(`total_s`/`denoise_s`/`decode_s`, `timing_scope`, `warm_steady_state`); the
speedup metric is generation time excluding one-time load, measured hot after a
warmup pass with all technique clocks (SOL step clock, TeaCache controller)
cold-started for the timed pass.
