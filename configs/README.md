# `configs/` — flat single-file configs for `scripts/run.py`

```bash
python3 scripts/run.py configs/minimax_h3_gb200_baseline.toml
```

That is the whole interface. One command, one file, no scheduler.

## The format is one layer

```toml
name    = "minimax_h3_gb200_baseline"   # lowercase -> the launcher
runtime = "models/minimax_h3/gb200/baseline"
entry   = "scripts/run_minimax_h3_gpu.sh"
gpus    = 1

H3_MODEL_PATH = "/home/yitongl/code/models/MiniMax-H3-diffusers"
H3_STEPS      = "50"                    # UPPERCASE -> exported as env
```

There are no `[tables]` and no references to other files. The split is the whole
schema: **lowercase keys belong to the launcher** (`name`, `runtime`, `entry`,
`out`, `gpus`, `description`), **UPPERCASE keys are exported verbatim as
environment variables**. Environment variables are conventionally uppercase, so
the two namespaces cannot collide, and a lowercase key that is not a launcher
key is a hard error rather than a silently ignored line.

A consequence worth having: the format is parseable by anything that can split
on `=`. `scripts/run.py` carries a 12-line fallback parser and therefore needs
no `tomllib`, so it also runs on the older pythons on draco and cs.

## Slurm is not in the picture

`scripts/run.py` reads no `SLURM_*` variable and calls no `srun`/`sbatch`/
`squeue`. It runs the arm in the current process, on the current machine. That
makes all three of these the same command:

```bash
# no scheduler at all
python3 scripts/run.py configs/minimax_h3_gb200_baseline.toml

# inside an interactive allocation
srun -A nvr_elm_llm -p batch --qos=interactive -N1 --gpus-per-node=4 -t 02:00:00 --pty bash
python3 scripts/run.py configs/minimax_h3_gb200_baseline.toml

# as a batch job
sbatch configs/minimax_h3_gb200.sbatch
```

`configs/minimax_h3_gb200.sbatch` is the only file that mentions Slurm, and its
last line is that same `python3 scripts/run.py ...`. Sites that schedule
differently replace that one file and change nothing else.

## Relative paths resolve from the repo root

`run.py` runs the entry script with the repo root as cwd, so a relative value in
a config (`H3_PROMPT_FILE = "models/minimax_h3/prompts/t2va_example_1.json"`)
means what it reads. The entry scripts locate themselves through `BASH_SOURCE`
and are unaffected by cwd, which is what leaves cwd free to carry this meaning.

## What it checks before it runs

`run.py` fails fast, with the named path, if the runtime dir, the entry script,
or `PYTHON_BIN` does not exist. That check is deliberate: on this stack a
missing entry script otherwise surfaces only after a GPU allocation, which is
the most expensive moment to discover it.

## Overrides

```bash
python3 scripts/run.py configs/minimax_h3_gb200_baseline.toml \
    --set H3_STEPS=10 --set H3_WARMUP=0 \
    --out /tmp/quick-probe
python3 scripts/run.py configs/minimax_h3_gb200_baseline.toml --print   # resolve only
```

Each run writes `config.resolved.json` into its output dir: the resolved env,
the entry that was executed, the host, and the exit code.

## Relationship to `candidates/` + `models/<uid>.toml`

Both paths reach the same arms and neither replaces the other.

| | `configs/` + `run.py` | `candidates/` + `launch_candidate.py` |
|---|---|---|
| files per run | 1 | 2 (candidate + profile) |
| scheduler | none | renders `job.sbatch`, can submit |
| provenance | `config.resolved.json` | full run bundle + `manifest.resolved.toml` |
| best for | running one arm, external users, other sites | experiment matrices, promotion gates |

`minimax_h3` has 23 candidates sharing one profile; that sharing is worth the
extra file. Running one arm once is not.
