# Launch Agent Protocol

The launch agent is the top-level automation agent for `autovideo`.

## Mission

Launch one candidate at a time, collect artifacts, compare against baseline, and
write a concise report. Do not invent acceleration code in `autovideo`; code
changes belong in `Sol-LTX-Infer` or in a clearly scoped patch goal.

## Inputs

- candidate manifest path, for example `candidates/baseline.toml`
- launch mode: `dry-run`, `local`, or `sbatch`
- explicit submit confirmation when `sbatch` should enter the Slurm queue
- optional baseline run ID for comparison
- optional prompt override

## Workflow

1. Verify the repo root and submodule exist.
2. Verify the `Sol-LTX-Infer` commit and warn if it differs from the manifest.
3. Generate a run bundle with `scripts/launch_candidate.py`.
4. For `dry-run`, stop and report the generated commands.
5. For `local`, run `launch.sh` only on a suitable GPU node.
6. For `sbatch`, render `job.sbatch`; submit only when explicitly confirmed.
7. When submitted, capture the Slurm job ID from `metadata.json`.
8. After execution, run `scripts/collect_run.py runs/<id>`.
9. Inspect `outputs/collection.json` and `outputs/report.md`.
10. Compare against the baseline.
11. Promote or reject the candidate.

## Guardrails

- Do not run the 64B model on a login node.
- Do not change official benchmark parameters when reporting speedup.
- Do not claim speedup from cold compile time.
- Do not promote a candidate that introduces visible artifacts.
- Do not modify `Sol-LTX-Infer` unless the candidate explicitly requires a
  patch-mode goal.

## Report Template

```markdown
# Candidate Report: <candidate-id>

Status: prepared|submitted|completed|failed|rejected_quality|promoted
Run: <run-id>
Baseline: <baseline-run-id>

## Config

- model:
- resolution:
- frames:
- steps:
- seed:
- GPUs:

## Timing

- total:
- denoise:
- decode:
- speedup vs baseline:

## Quality

- quantitative gate:
- visual gate:
- notes:

## Artifacts

- log:
- video:
- frames:
```
