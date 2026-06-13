# Orchestrator Entry — drive end-to-end acceleration (read this first)

**You are the MAIN orchestration agent.** Given a served model, drive the full
acceleration search and deliver **three risk-tiered configs** (low / medium / high).
You **orchestrate** — you do not implement techniques yourself. You spawn one agent
per search dimension, gate their results, integrate the winners, and deliver.

This is the single entry point. Everything else hangs off it.

---

## 0. Read these first (the system you drive)
- `docs/search-architecture.md` — the three planes + the search & eval pipeline.
- `docs/model-onboarding.md` — how to expand a model's interface (seams) so more
  dimensions become eligible.
- `models/README.md` — the model adapter (profile + ModelSpec).
- The **`codex-goal-fanout`** skill — how to spawn + gate per-dimension agents.
- `efficiency/README.md` — the generic acceleration engine.

## 1. Mental model (3 planes, 3 levels)
- **GENERIC** (never model-specific): `efficiency/` (engine), `loops/<dim>/`
  (search dimensions), `search/` (harness), `evals/tiers.toml` (the 3 tiers),
  `reference/` (LTX-2.3 priors).
- **MODEL-SPECIFIC** (the only model surface): `models/<id>.toml` +
  `efficiency/models/<id>_spec.py`.
- A dimension is **eligible** if the model's spec declares its capability,
  **functional** if the model wires the hook, **quality-valid** if OFF==baseline
  and it passes a tier gate. (See model-onboarding.md "three levels of wired".)

## 2. End-to-end procedure
```
0. ONBOARD/FREEZE  ensure models/<id>.toml + efficiency/models/<id>_spec.py
                   (model-onboarding.md). Run the official-config baseline and
                   record it in the profile [baseline]. The profile [env] must
                   carry the working runtime env (PYTHON_BIN, HF_HOME/HF_HUB_CACHE,
                   COSMOS3_CACHE, HF_HUB_OFFLINE) or the cluster run fails.
1. SCAN            python search/search.py --model <id>
                   -> eligible dimensions + each model's [seam_status] + the 3 tiers.
                   Expand seams (model-onboarding.md) to grow eligibility where worth it.
2. FAN OUT         one codex /goal agent per ELIGIBLE dimension (codex-goal-fanout):
                   own worktree+branch+isolated CODEX_HOME; AGENT-GOAL.md = shared
                   rules + the dimension spec. Each runs that dimension's [loop]:
                     enumerate dimension.toml [search_space] (LTX [[seeds]] first)
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
                   low/medium/high profiles. compose() rejects seam conflicts
                   (exclusive seams). Re-eval the composed configs on GPU. Target
                   composed speedups ~1.35x / 2.2x / 3.0x (evals/tiers.toml [targets]).
5. DELIVER         the final matrix: per tier -> config (feature flags) + speedup +
                   peak-mem + quality verdict + rollback. Write the release report.
```

## 3. Spawn recipe (per dimension) — the fan-out
Per the `codex-goal-fanout` skill (verified on HSG). For each eligible dimension:
```
WT=output/fanout/<dim>; git worktree add -b codex/<dim> $WT <BASE>
CH=$WT/.codex-home; mkdir -p $CH; cp ~/.codex/{auth.json,config.toml} $CH/
cat COMMON.md loops/<dim>/dimension.toml loops/<dim>/acceptance.md > $WT/AGENT-GOAL.md
# tmux /goal (no symposium bridge on HSG): unique socket per agent, isolated CODEX_HOME
tmux -L cc-fan-<dim> new-session -d -s g
tmux -L cc-fan-<dim> send-keys -t g 'export CODEX_HOME=$CH; \
  codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen -m gpt-5.5 \
  -c model_reasoning_effort=xhigh -C $WT "Read AGENT-GOAL.md and run this dimension fully"' Enter
```
Monitor commits + `SUMMARY.md`; then **review + gate + merge by SHA** (you are the gate).
Cap N to what you will review; CPU-only dimension prep runs on the park node, GPU
candidate runs go through Slurm.

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
| dimensions | `loops/<dim>/{dimension.toml, acceptance.md, references.md}` |
| LTX-2.3 priors | `reference/<dim>/` |
| engine | `efficiency/` (`selftest.py` = compose/seam test) |
| tiers | `evals/tiers.toml`; visual rubric `evals/rubrics/gemini_visual_artifact_gate.md` |
| docs | `docs/{search-architecture,model-onboarding}.md` |
```bash
python search/search.py --model <id>                  # eligible dimensions + seam_status + tiers
python search/plan_eval.py --assess <run_dir> --baseline-frames <dir>   # benchmark+Gemini+tier
python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
~/lustre/miniconda3/envs/sana/bin/python efficiency/selftest.py        # 23/23
```

## 7. Current target — Cosmos3 (first model)
- Baseline verified (~130s, `models/cosmos3.toml [baseline]`).
- `search --model cosmos3`: **6 eligible dimensions** — `step_cache`, `teacache`,
  `token_prune`, `nvfp4_ffn`, `kwl_fusion`, `sparse_attention`.
- `seam_status`: `swappable_attention` declared (sparse eligible); `teacache_signal`
  + `prunable_segment` refine = TODO (model-onboarding.md) — wiring these makes those
  dimensions *functional*, not just eligible.
- GPU eval harness: **VERIFIED end-to-end** (job 3294303, `runs/20260613-175619-baseline`):
  launch (sbatch) -> GPU run (127.83s) -> `collect_run.py` (`benchmark.json`, denoise 119.18s)
  -> Gemini pairwise judge (`overall=pass`, no artifacts) -> `tier_of` -> `tier=low`, via
  `python search/plan_eval.py --assess <run_dir> --baseline-frames <dir>`. NOTE: that candidate
  *is* the baseline config (no technique wired into the Cosmos3 denoise loop yet) -> 1.02x is
  run-variance, clean -> low. Real speedups need a technique wired into the runtime (next milestone).

Supersedes the pre-search `agents/launch-agent.md` for the model-agnostic search era.
