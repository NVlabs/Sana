# Artifact Contract

Every candidate launch writes a self-contained run bundle under `runs/`.

## Required Files

| File | Producer | Purpose |
| --- | --- | --- |
| `metadata.json` | launcher | Machine-readable run metadata: candidate ID, time, mode, repo paths, and submodule commit. |
| `manifest.resolved.toml` | launcher | Original manifest plus resolved paths and run IDs. |
| `launch.sh` | launcher | Exact shell entrypoint used for local execution or Slurm payload. |
| `job.sbatch` | launcher | Slurm wrapper, even in dry-run mode. |
| `outputs/run.log` | execution | Full command log from the implementation repo run script. |
| `outputs/out.mp4` | execution | Generated video output. |
| `outputs/perf.json` | execution or collector | Timing summary when available. |
| `outputs/frames/` | collector | Extracted review frames for visual inspection. |
| `outputs/report.md` | agent or collector | Human-readable result summary. |
| `outputs/collection.json` | collector | Machine-readable artifact/timing/status summary. |

## Status Values

Use these status labels in reports:

- `prepared`: run bundle exists but no GPU job has started.
- `submitted`: Slurm job was submitted.
- `running`: job is active.
- `completed`: job finished and expected artifacts exist.
- `failed`: command or job failed.
- `blocked`: prerequisite missing, such as weights, CUDA env, or Slurm access.
- `rejected_quality`: output exists but quality gates failed.
- `promoted`: candidate passed speed and quality gates.

## Baseline Comparison

Candidate reports should include:

- baseline run ID
- candidate run ID
- official config checksum or parameter table
- wall-clock time
- denoise time when available
- VAE/decode time when available
- speedup ratio
- quality gate result
- links to `out.mp4`, sampled frames, and logs

## Generated Paths

The launcher sets `OUT_DIR` to the run bundle's `outputs/` directory. The current
Cosmos3 baseline script writes `run.log` and `out.mp4` there.

Any future collector should keep derived files inside the same `outputs/`
directory instead of writing into `Sol-LTX-Infer/outputs/`.

## Collection

Collect a run with:

```bash
python3 scripts/collect_run.py runs/<run-id>
```

The collector:

- updates `metadata.json`
- writes `outputs/collection.json`
- writes `outputs/report.md`
- extracts frames with `ffmpeg` for completed runs when available
- returns non-zero for `failed`, `blocked`, or `rejected_quality`
