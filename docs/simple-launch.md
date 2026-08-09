# Simple launch — one file, one command, no scheduler

```bash
python3 scripts/run.py models/minimax_h3/gb200/baseline.toml
```

That is the whole interface.

## Where configs live

Beside the code they launch, one directory per hardware target. There is no
top-level `configs/` directory: a launch config belongs to the implementation it
launches, so it ships and moves with it.

```text
models/minimax_h3/              the model
  gb200/                          the GB200 implementation
    baseline.toml                   launch config  ->  runs gb200/baseline/
    optimized.toml                  launch config  ->  runs gb200/optimized/
    run.sbatch                      optional Slurm wrapper for this target
    README.md                       what this target is and what it reaches
    baseline/   optimized/          the code
  h100/  a100/  gb10/  rtx5090/  other targets, same shape
  prompts/                       shared across targets
```

Adding a target means adding a directory with the same shape. Nothing outside it
has to learn about it — no registry, no central list.

## The format is one layer

```toml
name    = "minimax_h3_gb200_baseline"   # lowercase -> the launcher
runtime = "baseline"                    # the sibling dir this config launches
entry   = "scripts/run_minimax_h3_gpu.sh"
gpus    = 1

H3_MODEL_PATH = "/path/to/MiniMax-H3-diffusers"   # UPPERCASE -> exported as env
H3_STEPS      = "50"
```

No `[tables]`, no references to other files. The split is the whole schema:
**lowercase keys belong to the launcher** (`name`, `runtime`, `entry`, `out`,
`gpus`, `description`), **UPPERCASE keys are exported verbatim as environment
variables**. Environment variables are conventionally uppercase, so the two
namespaces cannot collide, and a lowercase key that is not a launcher key is a
hard error rather than a silently ignored line.

A consequence worth having: the format is parseable by anything that can split
on `=`. `scripts/run.py` carries a 12-line fallback parser and therefore needs no
`tomllib`, so it also runs on the older pythons on draco and cs.

## Two path bases, one rule each

| keys | resolved against | why |
| --- | --- | --- |
| `runtime`, `entry` | **this config's directory** | a config beside its arm says `runtime = "baseline"`; move the pair and both still point at each other |
| UPPERCASE values | **the repo root** | they are handed to a process whose cwd is the repo root, so `H3_PROMPT_FILE = "models/minimax_h3/prompts/t2va_example_1.json"` reads the way it looks |

Entry scripts locate themselves through `BASH_SOURCE` and ignore cwd, which is
what leaves cwd free to carry that second meaning.

## Slurm is not in the picture

`scripts/run.py` reads no `SLURM_*` variable and calls no `srun`/`sbatch`/
`squeue`. It runs the arm in the current process, on the current machine. That
makes all three of these the same command:

```bash
# no scheduler at all
python3 scripts/run.py models/minimax_h3/gb200/baseline.toml

# inside an interactive allocation
srun -A nvr_elm_llm -p batch --qos=interactive -N1 --gpus-per-node=4 -t 02:00:00 --pty bash
python3 scripts/run.py models/minimax_h3/gb200/baseline.toml

# as a batch job
sbatch models/minimax_h3/gb200/run.sbatch
```

`run.sbatch` is the only file that mentions Slurm, and its last line is that same
`python3 scripts/run.py ...`. A site that schedules differently replaces that one
file and changes nothing else.

## What it checks before it runs

`run.py` fails fast, with the named path, if the runtime dir, the entry script,
or `PYTHON_BIN` does not exist. That check is deliberate: on this stack a missing
entry script otherwise surfaces only after a GPU allocation, which is the most
expensive moment to discover it. `PYTHON_BIN` is machine-specific, so `--print`
downgrades it to a warning and a config for another cluster stays inspectable.

## Overrides

```bash
python3 scripts/run.py models/minimax_h3/gb200/baseline.toml --print
python3 scripts/run.py models/minimax_h3/gb200/baseline.toml \
    --set H3_STEPS=2 --set H3_WARMUP=0 --out /tmp/smoke
```

`--print` resolves and shows without running; `--set` overrides any key for one
run without editing the file. Each run writes `config.resolved.json` into its
output dir: the resolved env, the entry that ran, the host, and the exit code.

## Verified

`models/minimax_h3/gb200/baseline.toml` on job 5992033 (nvl72041-T15,
2026-08-09): `request_s` 160.746, `denoise_gpu_s` 148.627, 49 DiT evals, peak
144454 MiB — against the recorded reference of 159.628 / 149.696 / 49 / 144454
from job 5813128. Same peak memory to the byte, same eval count, 0.7% on latency.

## Relationship to `candidates/` + `models/<uid>.toml`

Both paths reach the same arms and neither replaces the other.

| | this path | `candidates/` + `launch_candidate.py` |
|---|---|---|
| files per run | 1 | 2 (candidate + profile) |
| scheduler | none | renders `job.sbatch`, can submit |
| provenance | `config.resolved.json` | full run bundle + `manifest.resolved.toml` |
| best for | running one arm, external users, other sites | experiment matrices, promotion gates |

`minimax_h3` has 23 candidates sharing one profile; that sharing is worth the
extra file. Running one arm once is not.
