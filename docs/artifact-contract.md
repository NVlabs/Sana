# Artifact Contract

Every candidate launch writes a self-contained run bundle under `runs/`.

## Required Files

| File | Producer | Purpose |
| --- | --- | --- |
| `metadata.json` | launcher/collector | Machine-readable run metadata: candidate ID, purpose, time, mode, repo paths, runtime Python, submodule commit, current status, and `status_history`. |
| `manifest.resolved.toml` | launcher | Original manifest plus resolved paths and run IDs. |
| `launch.sh` | launcher | Exact shell entrypoint used for local execution or Slurm payload. |
| `job.sbatch` | launcher | Slurm wrapper, even in dry-run mode. |
| `outputs/run.log` | execution | Full command log from the implementation repo run script. |
| `outputs/out.mp4` | execution | Generated video output. |
| `outputs/benchmark.json` | collector | Canonical timing summary with total, denoise, and decode seconds when available. |
| `outputs/frames/` | collector | Extracted review frames for visual inspection. |
| `outputs/quality.json` | collector | Frame metrics plus optional judge outputs or deferred reasons. |
| `outputs/risk_notes.md` | collector | Risk notes for the run; baseline runs use the no-risk baseline stub. |
| `outputs/patch_summary.md` | agent or collector | Human-readable result summary. |
| `outputs/collection.json` | collector | Machine-readable artifact/timing/status summary. |

## Status Values

Use these status labels in reports:

- `prepared`: run bundle exists but no GPU job has started.
- `submitted`: Slurm job was submitted.
- `running`: job is active.
- `completed`: job finished and expected artifacts exist.
- `failed`: command or job failed.
- `submission_failed`: Slurm submission failed before a job id was created.
- `canceled_by_orchestrator_release`: job was intentionally cancelled because the orchestrator released or dropped the dimension.
- `blocked`: prerequisite missing, such as weights, CUDA env, or Slurm access.
- `rejected_quality`: output exists but required quality evidence is missing or an exact/numeric hard gate failed.
- `promoted`: candidate/profile has speed evidence and Gemini+LPIPS quality evidence for a delivery target.

`metadata.json.status_history` must append every state transition instead of
overwriting the past. This is the source of truth for lifecycle events such as
submit, collect, failure, and orchestrator-release cancellation.

## Baseline Comparison

Candidate reports should include:

- baseline run ID
- candidate run ID
- official config checksum or parameter table
- wall-clock time
- denoise time when available
- VAE/decode time when available
- speedup ratio
- quality evidence result, including aligned LPIPS and aligned pairwise Gemini
- links to `out.mp4`, sampled frames, and logs

## Generated Paths

The launcher sets `OUT_DIR` to the run bundle's `outputs/` directory. The current
Cosmos3 baseline script writes `run.log` and `out.mp4` there.

Any future collector should keep derived files inside the same `outputs/`
directory instead of writing into `Sol-LTX-Infer/outputs/`.

Canonical derived artifact names are fixed: `benchmark.json`,
`quality.json`, `risk_notes.md`, `patch_summary.md`, and `collection.json`.
Do not introduce alternate filenames for the same roles.

## Collection

Collect a run with:

```bash
python3 scripts/collect_run.py runs/<run-id>
```

The collector:

- updates `metadata.json`
- writes `outputs/collection.json`
- writes `outputs/benchmark.json`
- writes `outputs/quality.json`
- writes `outputs/risk_notes.md`
- writes `outputs/patch_summary.md`
- extracts frames with `ffmpeg` for completed runs when available
- returns non-zero for `failed`, `blocked`, or `rejected_quality`
