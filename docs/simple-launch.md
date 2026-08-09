# Simple launch — one file, one command, no scheduler

```bash
python3 scripts/run.py models/minimax_h3/GB200/dense.toml
```

That is the whole interface.

## Where configs live

Beside the code they launch, one directory per hardware target. There is no
top-level `configs/` directory: a launch config belongs to the implementation it
launches, so it ships and moves with it.

```text
models/minimax_h3/              the model
  gb200/                          the GB200 implementation
    dense.toml                      launch config  ->  the dense control
    fullopt.toml                    launch config  ->  the full stack
    README.md                       what this target is and what it reaches
    <driver + modules>              the code
  h100/  a100/  gb10/  rtx5090/  other targets, same shape
  prompts/                       shared across targets
```

Adding a target means adding a directory with the same shape. Nothing outside it
has to learn about it — no registry, no central list.

## The format is one layer

```toml
name    = "minimax_h3_gb200_diffusers_dense"  # lowercase -> the launcher
runtime = "."                           # the dir this config launches
entry   = "run_minimax_h3_gpu.sh"
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
| `runtime`, `entry` | **this config's directory** | a config beside its arm says `runtime = "."`; move the pair and both still point at each other |
| UPPERCASE values | **the repo root** | they are handed to a process whose cwd is the repo root, so `H3_PROMPT_FILE = "models/minimax_h3/demo_prompt.json"` reads the way it looks |

Entry scripts locate themselves through `BASH_SOURCE` and ignore cwd, which is
what leaves cwd free to carry that second meaning.

## Slurm is not in the picture

`scripts/run.py` reads no `SLURM_*` variable and calls no `srun`/`sbatch`/
`squeue`. It runs the arm in the current process, on the current machine. That
makes all three of these the same command:

```bash
# no scheduler at all
python3 scripts/run.py models/minimax_h3/GB200/dense.toml

# inside an interactive allocation
srun -A nvr_elm_llm -p batch --qos=interactive -N1 --gpus-per-node=4 -t 02:00:00 --pty bash
python3 scripts/run.py models/minimax_h3/GB200/dense.toml

# as a batch job -- your own wrapper, whatever your scheduler is
sbatch my_job.sbatch
```

Job scripts are not tracked in this repo (`*.gitignore`d): the account, partition
and QoS in them are site-specific and of no use to anyone else. Write your own —
the whole thing is a resource header plus the same one-line command:

```bash
#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00

cd "${SLURM_SUBMIT_DIR:-$PWD}"
python3 scripts/run.py models/minimax_h3/GB200/dense.toml
```

That last line is the only thing the scheduler contributes to, which is the
point: `run.py` never learns the scheduler exists, so a site that schedules
differently rewrites this wrapper and changes nothing else.

## What it checks before it runs

`run.py` fails fast, with the named path, if the runtime dir, the entry script,
or `PYTHON_BIN` does not exist. That check is deliberate: on this stack a missing
entry script otherwise surfaces only after a GPU allocation, which is the most
expensive moment to discover it. `PYTHON_BIN` is machine-specific, so `--print`
downgrades it to a warning and a config for another cluster stays inspectable.

## Overrides

```bash
python3 scripts/run.py models/minimax_h3/GB200/dense.toml --print
python3 scripts/run.py models/minimax_h3/GB200/dense.toml \
    --set H3_STEPS=2 --set H3_WARMUP=0 --out /tmp/smoke
```

`--print` resolves and shows without running; `--set` overrides any key for one
run without editing the file. Each run writes `config.resolved.json` into its
output dir: the resolved env, the entry that ran, the host, and the exit code.

## Verified

`models/minimax_h3/GB200/dense.toml` on job 5992033 (nvl72041-T15,
2026-08-09): `request_s` 160.746, `denoise_gpu_s` 148.627, 49 DiT evals, peak
144454 MiB — against the recorded reference of 159.628 / 149.696 / 49 / 144454
from job 5813128. Same peak memory to the byte, same eval count, 0.7% on latency.

## One command, two config dialects

`scripts/run.py` takes either kind of config and produces the same run bundle:

```bash
python3 scripts/run.py models/minimax_h3/GB200/dense.toml        # flat, one file
python3 scripts/run.py transfeat/minimax_h3/h100_dense.toml     # transfeat + profile
```

It tells them apart by shape -- a transfeat declares `model_profile` or a
`[runtime]` table, a flat config carries `runtime` as a plain string -- and hands
both to the same `prepare_run`, so `runs/<stamp>-<id>/` contains `launch.sh`,
`job.sbatch`, `manifest.resolved.toml`, `metadata.json` and `outputs/` either
way. `collect_run.py` reads either.

Use whichever the job calls for. A flat config is one self-contained file, good
for running one arm and for sites outside this cluster. A transfeat shares a
model profile across variants -- `minimax_h3` has 23 of them over one profile --
and carries `kind`, `purpose` and `[requires].capabilities`, which is what drives
the conflict check and the promotion gates.

`scripts/launch_transfeat.py` is still there and is what actually renders the
bundle. Call it directly when you want the scheduler: it is the only one of the
two that submits with `--mode sbatch --confirm-submit`.
