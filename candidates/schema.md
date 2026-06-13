# Candidate Manifest Schema

Candidate manifests are TOML files.

## Required Top-Level Fields

```toml
id = "baseline"
kind = "baseline"
description = "Official Cosmos3-Super baseline."
submodule = "Sol-LTX-Infer"
base_commit = "3a69b7788a055bed728ec367961c5f25b4ab48dc"
run_script = "scripts/run_cosmos3_sglang.sh"
```

## Supported `kind`

- `baseline`
- `env_only`
- `patch`
- `methodology`

## `official_config`

Use this table for benchmark-defining values. Do not report speedups from a
candidate whose official config differs from baseline.

```toml
[official_config]
model = "nvidia/Cosmos3-Super"
width = 1280
height = 720
frames = 189
fps = 24
steps = 35
guidance_scale = 6.0
flow_shift = 10.0
max_sequence_length = 4096
seed = 42
num_gpus = 4
```

## `env`

Values exported into `launch.sh`. The launcher also injects `OUT_DIR`.

```toml
[env]
MODEL_REPO = "nvidia/Cosmos3-Super"
NUM_GPUS = "4"
SEED = "42"
PROMPT = "..."
NEGATIVE_PROMPT = ""
```

## `artifacts`

Relative to the generated run directory.

```toml
[artifacts]
output_dir = "outputs"
video = "out.mp4"
log = "run.log"
benchmark = "benchmark.json"
frames_dir = "frames"
quality = "quality.json"
risk_notes = "risk_notes.md"
collection = "collection.json"
patch_summary = "patch_summary.md"
```

## `slurm`

The launcher turns this into `job.sbatch`.

```toml
[slurm]
account = "nvr_elm_llm"
partition = "batch"
nodes = 1
gpus_per_node = 4
cpus_per_task = 64
mem = "0"
time = "04:00:00"
job_name = "autovideo-baseline"
exclusive = true
```

## Future Patch Candidates

Patch candidates should add:

```toml
[patch]
summary = "Wire token pruning around the target model's denoise block loop."
touch_points = [
  "python/sglang/multimodal_gen/runtime/efficiency/models/cosmos3_spec.py",
  "python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py",
]
off_identity_required = true
```

## Optional Agent Ownership

Parallel agent goals should add:

```toml
[agent]
goal_id = "token-prune"
owner = "codex"
root_branch = "codex/token-prune"
submodule_branch = "codex/token-prune-sol"
interactive_required = true
write_scope = [
  "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py",
]
```

The orchestration layer should use this block to create isolated worktrees and
avoid two agents editing the same submodule checkout.
