# Orchestrator Entry — drive end-to-end acceleration (read this first)

**You are the MAIN orchestration agent.** Given a served model, drive the full
acceleration search and deliver **three risk-tiered configs** (low / medium / high).
You **orchestrate** — you do not implement techniques yourself. You spawn one agent
per search dimension, gate their results, integrate the winners, and deliver.

This is the single entry point. Everything else hangs off it.

Launch this as a normal main-agent Codex session. Do **not** run `/goal follow`
on this file. Native goal mode is reserved for implementation and gate
subagents that the main agent creates under `goals/<goal-id>/`.

---

## 0. Read these first (the system you drive)
- `docs/search-architecture.md` — the three planes + the search & eval pipeline.
- `models/README.md` — the target model profile and run environment.
- `docs/codex-goal-mode.md` and `docs/multi-agent-orchestration.md` — how to
  spawn, correct, gate, and release per-dimension native goal sessions.
- `efficiency/README.md` — the generic acceleration engine.

## 1. Mental model (3 planes, 3 levels)
- **GENERIC** (never model-specific): `efficiency/` (engine), `loops/<dim>/`
  (search dimensions), `search/` (harness), `evals/tiers.toml` (the 3 tiers),
  `search_space/` (the method-family search space).
- **MODEL/RUNTIME SURFACE**: `models/<id>.toml` plus the live inference code under
  `Sol-LTX-Infer/`.
- A dimension is **launchable** when the main agent decides it is worth exploring.
  Do not block launch on predeclared seams. Subagents may write directly in the
  inference path; seams/compose are post-hoc diagnostics and merge aids.

## 2. End-to-end procedure
```
0. ONBOARD/FREEZE  ensure models/<id>.toml + efficiency/models/<id>_spec.py
                   (model-onboarding.md). Run the official-config baseline and
                   record it in the profile [baseline]. The profile [env] must
                   carry the working runtime env (PYTHON_BIN, HF_HOME/HF_HUB_CACHE,
                   COSMOS3_CACHE, HF_HUB_OFFLINE) or the cluster run fails.
1. SCAN            python search/search.py --model <id>
                   -> method families, loop budgets, compose diagnostics, and the 3 tiers.
                   Use this as an observation pass, not as a launch gate.
2. FAN OUT         one native Codex goal session per selected dimension:
                   own worktree+branch+isolated CODEX_HOME; goal.md includes
                   objective, search-space-start, artifacts, and acceptance criteria.
                   Start through `tools/symposium/codex_goal_session.py start`,
                   which opens interactive Codex and sends `/goal follow <goal.md>`.
                   Each agent starts from `search_space/` and
                   `loops/<dim>/exploration.md`, then derives model-specific
                   candidates from inference code/traces. It can directly modify
                   `Sol-LTX-Infer/`; do not wait for an exposed seam/interface:
                       -> plan_eval.render_candidate(profile, technique, cfg)
                       -> scripts/launch_candidate.py --mode sbatch --confirm-submit
                       -> scripts/collect_run.py  -> plan_eval.assess (benchmark + Gemini)
                       -> plan_eval.tier_of  -> keep best_per_tier
                     budget: max_iters=20, early_stop_patience=5.
3. GATE (you)      never auto-merge. Per candidate enforce the 3-stage quality:
                   off_identity -> LPIPS -> NVIDIA-Gemini visual judge, against the
                   tier budgets (evals/tiers.toml). Read each branch diff, run the
                   gate, merge sequentially (by SHA), reconcile shared-file conflicts.
4. INTEGRATE       stack per-tier dimension winners across dimensions -> composed
                   low/medium/high profiles. Main resolves code conflicts and any
                   runtime interaction between patches, then re-gates merged glue.
                   Re-eval the composed configs on GPU. Target composed speedups
                   ~1.35x / 2.2x / 3.0x (evals/tiers.toml [targets]).
5. DELIVER         the final matrix: per tier -> config (feature flags) + speedup +
                   peak-mem + quality verdict + rollback. Write the release report.
```

## 3. Spawn recipe (per dimension) — the fan-out
For each selected dimension, prepare a goal bundle and start a managed native
Codex goal session:
```
WT=$PWD/output/fanout/<dim>; git worktree add -b codex/<dim> $WT <BASE>
CH=$WT/.codex-home; mkdir -p $CH; cp ~/.codex/{auth.json,config.toml} $CH/
(cd $WT && python3 tools/symposium/prepare_goal.py \
    --goal-id <dim> \
    --candidate candidates/baseline.toml \
    --dimension <dim> \
    --role implementation \
    --write-scope Sol-LTX-Infer/ \
    --objective "Explore and implement the <dim> dimension from search_space/ by directly inspecting and modifying Cosmos3 inference code.")
CODEX_HOME=$CH python3 tools/symposium/codex_goal_session.py start \
  --worktree $WT --name <dim> goals/<dim>
```
Use `status`, `capture`, and `send` to track and correct live subagents. When a
candidate is ready, spawn a separate gate goal (`--role gate`); then review,
gate, and merge by SHA. Cap N to what you will review; CPU-only dimension prep
runs on the park node, GPU candidate runs go through Slurm. Use `release` after
the agent is closed or rejected.

## 4. Cluster / GPU
- Runs: `scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit`
  (HSG `batch`, 4 GPU/node). The profile `[env]` supplies the cache/python.
- Collect: `scripts/collect_run.py <run_dir>` -> `benchmark.json` + frames.
- Quality: `tools/vision/nvidia_gemini_judge.py` (`NVIDIA_API_KEY`) -> `quality.json`.
- Assess + tier: `python search/plan_eval.py --assess <run_dir> --baseline-frames <dir>`.

## 5. Guardrails (hard)
- Dimensions stay model-agnostic; model specifics live ONLY in `models/` + the spec.
- Quality is a hard, **per-tier** constraint (Gemini required). Never trade quality
  for speed beyond a tier budget. A clean config bins to the **tightest (low)** tier.
- OFF==baseline (byte/numeric) on guarded paths; WARMUP before quoting timings;
  never claim speedup from cold compile.
- Bounded loops (max_iters + early stop); log any truncation; review before merge.

## 6. File map / quick commands
| What | Where |
| --- | --- |
| search harness | `search/search.py` (scan), `search/plan_eval.py` (render/assess/tier) |
| model adapter | `models/<id>.toml`, `efficiency/models/<id>_spec.py` |
| dimensions | `loops/<dim>/{dimension.toml, exploration.md, acceptance.md}` |
| search space | `search_space/` |
| engine | `efficiency/` (`selftest.py` = compose/seam test) |
| tiers | `evals/tiers.toml`; visual rubric `evals/rubrics/gemini_visual_artifact_gate.md` |
| docs | `docs/{search-architecture,model-onboarding}.md` |
```bash
python search/search.py --model <id>                  # method families + diagnostics + tiers
python search/plan_eval.py --assess <run_dir> --baseline-frames <dir>   # benchmark+Gemini+tier
python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
~/lustre/miniconda3/envs/sana/bin/python efficiency/selftest.py        # 23/23
```

## 7. Current target — Cosmos3 (first model)
- Baseline verified (~130s, `models/cosmos3.toml [baseline]`).
- `search --model cosmos3`: **6 launchable method families** — `step_cache`, `teacache`,
  `token_prune`, `nvfp4_ffn`, `kwl_fusion`, `sparse_attention`.
- Subagents should inspect and patch the Cosmos3 inference path directly. Any
  adapter/seam cleanup is a main-agent integration concern after a candidate is
  proven useful.
- GPU eval harness: **VERIFIED end-to-end** (job 3294303, `runs/20260613-175619-baseline`):
  launch (sbatch) -> GPU run (127.83s) -> `collect_run.py` (`benchmark.json`, denoise 119.18s)
  -> Gemini pairwise judge (`overall=pass`, no artifacts) -> `tier_of` -> `tier=low`, via
  `python search/plan_eval.py --assess <run_dir> --baseline-frames <dir>`. NOTE: that candidate
  *is* the baseline config (no technique wired into the Cosmos3 denoise loop yet) -> 1.02x is
  run-variance, clean -> low. Real speedups need a technique wired into the runtime (next milestone).

Supersedes the pre-search `agents/launch-agent.md` for the model-agnostic search era.
