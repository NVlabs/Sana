# autovideo

> **Orchestrating end-to-end acceleration?** Start at [`agents/orchestrator-entry.md`](agents/orchestrator-entry.md) — the main-agent runbook (scan -> fan out per-dimension agents -> gate -> integrate -> deliver 3 tiers).


`autovideo` is the orchestration layer for video-generation acceleration work.
It keeps experiment planning, launch automation, run artifacts, and agent
protocols outside the implementation repo.

The implementation plane is the `Sol-LTX-Infer` submodule. That repo owns model
code, SGLang diffusion runtime changes, Slurm scripts, kernels, quantization,
and the Cosmos3/LTX reference implementations. This repo owns how candidates
are described, launched, compared, and summarized.

## M0/M1 Scope

M0 establishes the folder layout and contracts:

- `docs/` explains the orchestration model and artifact layout.
- `candidates/` stores candidate manifests.
- `agents/` stores the launch-agent operating protocol.
- `scripts/` stores orchestration utilities.
- `runs/` stores local or cluster run bundles and outputs. Runtime artifacts are
  intentionally ignored by git.

M1 adds a baseline candidate and a launcher that can prepare a run bundle, dry
run the baseline, or submit it to Slurm.

M1.6 adds the evaluation and sub-loop structure used by future independent
Codex goals:

- `evals/` defines promotion gates and the official video T2V eval profile.
- `snippets/` records reusable patterns from successful `Sol-LTX-Infer` work.
- `loops/` defines one mature sub-loop folder per acceleration line.

## Quick Start

Initialize the execution submodule if needed:

```bash
git submodule update --init --recursive
```

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

Install Symposium skills for project-local Codex use:

```bash
python3 tools/symposium/install_project_skills.py --target codex
python3 tools/symposium/probe_goal_mode.py --json
```

## Folder Layout

| Path | Purpose |
| --- | --- |
| `Sol-LTX-Infer/` | Execution submodule with SGLang diffusion code and model-specific acceleration implementation. |
| `candidates/` | Declarative manifests for baseline and acceleration candidates. |
| `agents/` | Prompt/runbook material for the top-level launch agent. |
| `docs/` | Orchestration design, folder layout, and artifact contracts. |
| `evals/` | Eval profiles, metrics, and visual-judge rubrics. |
| `snippets/` | Small reference snippets from successful branches and reports. |
| `loops/` | Independent sub-loop/goal folders for each acceleration family. |
| `scripts/` | Local orchestration scripts; these call into `Sol-LTX-Infer` instead of reimplementing it. |
| `tools/symposium/` | Vendored Symposium skill pack plus adapters for preparing Codex interactive goal bundles. |
| `runs/` | Generated run bundles, logs, videos, frames, and reports. Ignored except for `runs/README.md`. |

## Candidate Lifecycle

1. Describe one candidate in `candidates/*.toml`.
2. Generate a run bundle with `scripts/launch_candidate.py`.
3. Launch through `local` or `sbatch`.
4. Collect `run.log`, `out.mp4`, timing files, extracted frames, and a report
   under the run directory with `scripts/collect_run.py`.
5. Compare against the official target-model baseline before promoting a candidate.

M2-M5 should each become independent candidate goals that plug into this same
contract: sparse attention, step cache, token pruning, KWL fusion, NVFP4, and
eventual full-stack composition.
