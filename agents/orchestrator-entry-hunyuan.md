# Orchestrator Entry — HunyuanVideo diffusers acceleration

**You are the MAIN orchestration agent for the HunyuanVideo diffusers target.**
Drive the full acceleration search and deliver **three speed-target configs**
(low / medium / high) against the verified default Hunyuan baseline.

You **orchestrate**: spawn one agent per search dimension, gate their results,
integrate the winners, and deliver. Implementation agents may directly patch the
local `Hunyuan-Diffusers/` submodule; the root repo keeps orchestration metadata,
candidate manifests, launch bundles, and eval artifacts.

This file is the Hunyuan-specific entry point. The existing
`agents/orchestrator-entry.md` remains the Cosmos3 entry point.

Launch this as a normal main-agent Codex session. Do **not** run `/goal follow`
on this file. Native goal mode is reserved for implementation and gate
subagents created under `goals/<goal-id>/`.

---

## 0. Read these first
- `docs/search-architecture.md` — the three planes plus search/eval pipeline.
- `docs/fanout-loop-contract.md` — loop state machine, gate authority, failure
  signatures, and stopping rules.
- `models/hunyuan_diffusers.toml` — Hunyuan model profile, official config,
  Slurm/env defaults, and submodule commit.
- `candidates/hunyuan_diffusers_baseline.toml` — canonical baseline candidate.
- `docs/codex-goal-mode.md` and `docs/multi-agent-orchestration.md` — how to
  spawn, correct, gate, and release per-dimension native goal sessions.
- `efficiency/README.md` — reusable acceleration engine.

## 1. Mental model
- **GENERIC**: `efficiency/`, `loops/<dim>/`, `search/`, `evals/tiers.toml`,
  and `search_space/` remain model-agnostic.
- **MODEL/RUNTIME SURFACE**: `models/hunyuan_diffusers.toml` plus the live
  inference code under `Hunyuan-Diffusers/`.
- **BASELINE CANDIDATE**: `candidates/hunyuan_diffusers_baseline.toml`.
- A dimension is **launchable** when the main agent decides it is worth
  exploring. Do not wait for a perfect seam abstraction; subagents may patch the
  Hunyuan diffusers inference path directly, then the main agent handles
  integration cleanup after a candidate proves useful.

## 2. End-to-end procedure
```text
0. ONBOARD/FREEZE  ensure models/hunyuan_diffusers.toml and Hunyuan-Diffusers/
                   are current. Run the official-config baseline and record the
                   benchmark evidence. The profile [env] must carry the working
                   runtime env (PYTHON_BIN, HF_HOME/HF_HUB_CACHE, HUNYUAN_CACHE,
                   HF_HUB_OFFLINE, MODEL_REPO) or the cluster run fails.
1. SCAN            python3 search/search.py --model hunyuan_diffusers
                   -> method families, loop budgets, compose diagnostics, and
                   the 1.5x / 2.0x / 3.0x targets.
2. FAN OUT         one native Codex goal session per selected dimension:
                   create a fresh RUN_ID and keep worktrees under
                   output/fanout_runs/$RUN_ID/. Each goal gets its own root
                   worktree, Hunyuan-Diffusers submodule branch, CODEX_HOME,
                   run directories, and Slurm jobs.
                   Each agent starts from search_space/ and loops/<dim>/, then
                   runs a bounded loop:
                     observe -> hypothesize -> implement one candidate ->
                     preflight -> launch -> authoritative gate ->
                     retain/discard/reject and loop.
                   Candidate failure does not finish the dimension; record a
                   failure signature and continue unless max_iters, a real
                   blocker, or explicit orchestrator release applies.
3. GATE (you)      never auto-merge. Enforce the authoritative quality gate:
                   OFF identity where applicable -> aligned LPIPS -> aligned
                   pairwise NVIDIA-Gemini. Read each branch diff, run the gate,
                   merge sequentially by SHA, and reconcile conflicts.
4. INTEGRATE       mandatory fan-in stage after selected dimensions are terminal.
                   Start one clean integration goal that stacks eligible winners
                   into composed low/medium/high Hunyuan profiles, launches GPU
                   generation, and re-gates each merged profile. Failed composed
                   gates record interaction failures and loop.
5. DELIVER         final matrix: tier -> config/flags + speedup + peak memory +
                   quality verdict + rollback. Write the release report.
```

## 3. Spawn recipe — Hunyuan fan-out
For each selected dimension, prepare a goal bundle and start a managed native
Codex goal session:

```bash
RUN_ID=${RUN_ID:-$(date -u +fanout_hunyuan_%Y%m%dT%H%M%SZ)}
FANOUT_ROOT=$PWD/output/fanout_runs/$RUN_ID
WT=$FANOUT_ROOT/<dim>
BR=codex/${RUN_ID}-<dim>

git worktree add -b $BR $WT <BASE>
CH=$WT/.codex-home; mkdir -p $CH; cp ~/.codex/{auth.json,config.toml} $CH/

(cd $WT && python3 tools/symposium/prepare_goal.py \
    --clean-stale-records \
    --run-id $RUN_ID)

(cd $WT && python3 tools/symposium/prepare_goal.py \
    --goal-id <dim> \
    --candidate candidates/hunyuan_diffusers_baseline.toml \
    --dimension <dim> \
    --role implementation \
    --run-id $RUN_ID \
    --write-scope Hunyuan-Diffusers/ \
    --root-branch $BR \
    --submodule-branch ${BR}-hunyuan \
    --objective "Explore and implement the <dim> dimension from search_space/ by directly inspecting and modifying the HunyuanVideo diffusers inference code in Hunyuan-Diffusers/.")

CODEX_HOME=$CH python3 tools/symposium/codex_goal_session.py start \
  --worktree $WT --name ${RUN_ID}-<dim> goals/<dim>
```

Use `status`, `capture`, and `send` to track and correct live subagents. GPU
candidate runs go through Slurm; do not run the default Hunyuan baseline on a
login node.

If reusing a checkout, clean durable stale records and live tmux sessions:

```bash
python3 tools/symposium/prepare_goal.py --clean-stale-records --run-id "$RUN_ID"
tmux ls | rg "$RUN_ID" || true
python3 tools/symposium/codex_goal_session.py list
```

## 3.1 Spawn recipe — fan-in integration
After every selected fan-out dimension is terminal and no dimension has active
Slurm/collector/gate work, start exactly one integration goal:

```bash
RUN_ID=${RUN_ID:?reuse the same fanout run id}
FANOUT_ROOT=$PWD/output/fanout_runs/$RUN_ID
python3 tools/symposium/loop_control.py ensure-integration \
  --fanout-root "$FANOUT_ROOT" \
  --run-id "$RUN_ID" \
  --base <BASE>
```

Integration acceptance:

- low/medium/high each has a gated composed Hunyuan manifest and video, or an
  explicit blocker such as `no_eligible_profile`;
- every composed profile has benchmark artifacts, aligned LPIPS, and aligned
  pairwise Gemini verdict against the canonical Hunyuan baseline;
- target winners rank quality using Gemini severity/status and LPIPS together,
  then speed as tie-breaker;
- failed merges or quality regressions are recorded as interaction failures and
  looped, not treated as completion.

## 4. Cluster / GPU
- Launch:
  `python3 scripts/launch_candidate.py candidates/hunyuan_diffusers_baseline.toml --mode sbatch --confirm-submit`
- Scan:
  `python3 search/search.py --model hunyuan_diffusers`
- Collect:
  `python3 scripts/collect_run.py <run_dir>`
- Assess:
  `/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames <canonical-frames>`
- Quality authority: OFF identity where applicable, aligned LPIPS, and aligned
  pairwise NVIDIA-Gemini. Collector-only `quality.json` Gemini is telemetry.

The Slurm profile currently requests 4 GPUs per node because of cluster QoS, but
the vanilla diffusers baseline uses `NUM_GPUS=1` and `HUNYUAN_PLACEMENT=cuda`.
Treat multi-GPU execution as an optimization target, not as baseline behavior.

## 5. Guardrails
- Dimensions stay model-agnostic; Hunyuan-specific code lives in
  `Hunyuan-Diffusers/`, and Hunyuan-specific launch metadata lives in
  `models/hunyuan_diffusers.toml` plus candidate manifests.
- Do not change official benchmark parameters when reporting speedup:
  `1280x720`, `129` frames, `24` fps, `50` steps, seed `42`.
- Do not claim speedup from smoke runs, cold compile, warm cache drift, or
  reduced frame/resolution/step settings.
- A failed candidate gate means reject/log/loop, not finish.
- A successful candidate is retained in `frontier_candidates` when quality or
  speed improves, then the search continues until a stop condition applies.
- Every fan-out dimension defaults to `max_iters=40` and
  `early_stop_patience=0`.
- Runtime state is not prose-only: require
  `tools/symposium/loop_control.py init`, `record-candidate`, `decide-next`, and
  `validate-status` on every dimension loop.
- Kill duplicate collectors/jobs for the same run; never let two processes race
  to write the same quality artifacts.
- Fan-out terminal state is not global completion. The run completes only after
  the integration loop produces composed low/medium/high artifacts or explicit
  per-tier blockers.

## 6. File map / quick commands
| What | Where |
| --- | --- |
| Hunyuan profile | `models/hunyuan_diffusers.toml` |
| baseline candidate | `candidates/hunyuan_diffusers_baseline.toml` |
| runtime submodule | `Hunyuan-Diffusers/` |
| GPU entry script | `Hunyuan-Diffusers/scripts/run_hunyuan_diffusers_gpu.sh` |
| Python runner | `Hunyuan-Diffusers/hunyuan_diffusers/gpu_infer.py` |
| search harness | `search/search.py`, `search/plan_eval.py` |
| dimensions | `loops/<dim>/{exploration.md,acceptance.md}` |
| search space | `search_space/` |
| speed targets | `evals/tiers.toml` |

```bash
python3 search/search.py --model hunyuan_diffusers
python3 scripts/launch_candidate.py candidates/hunyuan_diffusers_baseline.toml --mode sbatch --confirm-submit
/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames <canonical-frames>
~/lustre/miniconda3/envs/sana/bin/python efficiency/selftest.py
```

## 7. Current target — HunyuanVideo diffusers
- Model: `hunyuanvideo-community/HunyuanVideo`.
- Profile: `models/hunyuan_diffusers.toml`.
- Baseline candidate: `candidates/hunyuan_diffusers_baseline.toml`.
- Runtime: local `Hunyuan-Diffusers/` submodule at
  `7638072479d7740f039bc9717273d4bdf0d2c787`.
- Checkpoint snapshot:
  `/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/e8c2aaa66fe3742a32c11a6766aecbf07c56e773`.
- `search --model hunyuan_diffusers`: **6 launchable method families** —
  `step_cache`, `teacache`, `token_prune`, `nvfp4_ffn`, `kwl_fusion`,
  `sparse_attention`.
- Canonical baseline verified on Slurm job `3467620`:
  `runs/20260620-163957-hunyuan_diffusers_baseline-gpu-default-video`.
  Output video:
  `outputs/out.mp4`.
- Verified default output:
  `1280x720`, `129` frames, `24` fps, duration `5.375s`, `50` steps.
- Baseline timing:
  `load_s=61.60`, `placement_s=17.44`, `generate_s=881.85`,
  `export_s=1.88`, `total_s=1007.47`.
- Baseline memory:
  max allocated `51.0 GiB`, max reserved `62.2 GiB`.

Real Hunyuan speedups must preserve the default output contract above and should
be compared against this canonical baseline, not against smoke runs.
