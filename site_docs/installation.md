# Installation

This page covers `NVlabs/Sana` on the `sol-engine` branch. Cosmos3-Super and
LTX-2.3 additionally need
[Sol-Video-Inference-Engine](https://github.com/NVlabs/Sol-Video-Inference-Engine);
see [that section](#the-sol-video-inference-engine-runtime).

## What this repository contains

No package to build and nothing to `pip install`. It holds the acceleration line
for each model: the configs, the per-hardware runtime code, and the Sol-Attn
kernels. The model itself is Diffusers' or SGLang's; the weights are on Hugging
Face.

```bash
git clone -b sol-engine https://github.com/NVlabs/Sana.git
cd Sana
```

Everything is launched by one command, which needs nothing installed beyond a
Python 3.9+ interpreter (3.11+, or the `tomli` backport, to read a config
manifest):

```bash
python3 scripts/run.py config/<model>/<arm>.toml
```

## Launching

### The config

A config is one layer. Lowercase keys belong to the launcher — `name`,
`runtime`, `entry`, `out`, `gpus`, `description`. UPPERCASE keys are exported
verbatim as environment variables. That split is the whole schema: environment
variables are uppercase by convention, so the two namespaces cannot collide, and
a lowercase key that is not a launcher key is an error rather than a silently
ignored line.

```toml
name    = "minimax_h3_gb200_diffusers_dense"
runtime = "."                              # the directory this config launches
entry   = "run_minimax_h3_gpu.sh"
gpus    = 1

H3_MODEL_PATH = "/path/to/MiniMax-H3-diffusers"
H3_STEPS      = "50"
```

Two path bases, one rule each:

| keys | resolved against |
|---|---|
| `runtime`, `entry` | **this config's own directory** — a config beside its arm says `runtime = "."`, and moving the pair keeps them pointing at each other |
| UPPERCASE values | **the repository root** — they are handed to a process whose cwd is the repository root |

`scripts/run.py` also reads the richer manifest dialect under `config/`, which
shares a model profile across variants and carries `kind`, `purpose` and
`[requires].capabilities`. It tells the two apart by shape and renders the same
run bundle either way: `runs/<stamp>-<id>/` with `launch.sh`,
`manifest.resolved.toml`, `metadata.json` and `outputs/`.

### Checking before you spend an allocation

```bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml --print
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml --set H3_STEPS=2 --set H3_WARMUP=0
```

`--print` resolves and shows without running anything; `--set` overrides one key
for one run without editing the file. The run fails immediately, naming the
path, if the runtime directory, the entry script or `PYTHON_BIN` is missing — on
this stack a missing entry script otherwise surfaces only after a GPU
allocation, which is the most expensive moment to find it. `PYTHON_BIN` is
machine-specific, so `--print` downgrades that one to a warning and a config
written for another cluster stays inspectable.

### Slurm

`scripts/run.py` reads no `SLURM_*` variable and calls no `srun`, `sbatch` or
`squeue`. It runs the arm in the current process on the current machine, so
these are the same command:

```bash
# no scheduler
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml

# inside an interactive allocation
srun -A <your-account> -p batch -N1 --gpus-per-node=4 -t 02:00:00 --pty bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml
```

Job scripts are not tracked here — their account, partition and QoS are
site-specific and of no use to anyone else. Write your own; it is a resource
header plus the same one line:

```bash
#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p <your-partition>
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00

cd "${SLURM_SUBMIT_DIR:-$PWD}"
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml
```

That last line is all the scheduler contributes to, which is the point:
`run.py` never learns the scheduler exists, so a site that schedules differently
rewrites this wrapper and changes nothing else. `scripts/launch_config.py`
renders the bundle and is the one that can submit, with
`--mode sbatch --confirm-submit`.

## What each model needs

The interpreter and the weights belong to the model, not to this repository, so
prerequisites differ per model. Every config declares its own, and `--print`
shows the resolved values.

| Model | Runtime it needs |
|---|---|
| Wan2.2 TI2V-5B, Wan2.2-A14B, Wan2.1, HunyuanVideo | A Python env with **PyTorch + Diffusers**; point the config's `PYTHON_BIN` at it. |
| SANA-Video, LingBot-Video | The same, plus a model bundle the runtime fetches on first run. |
| MiniMax-H3 on H100 / A100 / RTX 5090 | An **SGLang container** — `H3_CONTAINER_RUNTIME = "pyxis"`, image pinned in the config. Nothing to install locally. |
| MiniMax-H3 on GB200 / GB10 | A conda env with the **pinned Diffusers PR** that cell's `SOURCE_SNAPSHOT.json` records. |
| Cosmos3-Super, LTX-2.3 | A **Sol-Video-Inference-Engine** checkout, reached by `SOL_LTX_INFER_ROOT`. See below. |

## Sol-Attn kernels

`techniques/sparse_backends` is a standalone package (`sol-attn`) and is the one
piece here that is in no upstream framework:

```bash
pip install ./techniques/sparse_backends
```

Requires Python 3.10+ and an existing PyTorch install — the package does not
pull one in, so that it cannot override a build matched to your CUDA. Install
PyTorch first; `import sol_attn` fails on `No module named 'torch'` otherwise.

It dispatches on compute capability: CuTe kernels for **sm90** (H100),
**sm100** (GB200/B200) and **sm120** (RTX 5090), and a Triton reference
everywhere else. The reference is correct but is not what the published speedups
measure, so on an A100 (sm80) or a DGX Spark GB10 the number you get is not the
number in the tables. `benchmark.json` records the backend that actually ran —
read it rather than assuming.

The CuTe path needs an `nvidia-cutlass-dsl` matching what the kernels were built
against. A newer DSL fails at compile time with
`module 'cutlass.cute.nvgpu' has no attribute 'OperandMajorMode'` instead of
falling back, which is deliberate: a silent fallback reports a dense run as a
sparse one.

## The Sol-Video-Inference-Engine runtime

Cosmos3-Super and LTX-2.3 run inside
[Sol-Video-Inference-Engine](https://github.com/NVlabs/Sol-Video-Inference-Engine),
which holds the SGLang `multimodal_gen` pipelines for them. This repository
vendors the launch body and the official config, and reaches that checkout by
absolute path.

```bash
git clone https://github.com/NVlabs/Sol-Video-Inference-Engine.git
cd Sol-Video-Inference-Engine

PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh
source "$PWD/scripts/use_code_storage_env.sh"
conda activate "$PWD/.conda/ltx23"

uv pip install -e "$PWD/python[diffusion]" --prerelease=allow
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh
```

`create_code_conda_env.sh` creates `.conda/ltx23`, installs `pip` and `uv`, and
adds the activation hooks for `use_code_storage_env.sh`; it defaults to Python
3.11 when `PYTHON_VERSION` is unset. Add `--with-te` to `postinstall_cuda_jit.sh`
for the NVFP4 path — without TransformerEngine those configs fall back to BF16.

Then point this repository at it:

```bash
python3 scripts/run.py config/cosmos3/baseline.toml \
  --set SOL_LTX_INFER_ROOT=/path/to/Sol-Video-Inference-Engine
```

Do not set `PYTHON_BIN` alongside it — the launch body derives the interpreter
as `$SOL_LTX_INFER_ROOT/.conda/ltx23/bin/python`.

## Weights

```bash
export HF_HOME=/somewhere/with/room     # 30–140 GB per model
hf auth login

hf download nvidia/Cosmos3-Super
hf download Lightricks/LTX-2.3
hf download Efficient-Large-Model/SANA-Video_2B_480p_diffusers
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
hf download MiniMaxAI/MiniMax-H3
```

MiniMax-H3's weights live in that repository's `FL2VA` subfolder and the
runtimes resolve it themselves, so pass the repository id rather than a path
into it.

## Verify

Resolve every config without a GPU — this checks each one's runtime root, run
script and prompt path:

```bash
for c in config/*/*.toml; do
  python3 scripts/run.py "$c" --print >/dev/null || echo "FAILED $c"
done
```

Then, inside the environment a given model will use:

```bash
python3 -c "import torch, diffusers; print(torch.__version__, diffusers.__version__, torch.cuda.is_available())"
```
