# Config Manifest Schema

Config manifests are TOML files.

## Required Top-Level Fields

```toml
id = "baseline"
kind = "baseline"
description = "Official Cosmos3-Super baseline."
submodule = "Sol-LTX-Infer"
base_commit = "3a69b7788a055bed728ec367961c5f25b4ab48dc"
run_script = "scripts/run_cosmos3_sglang.sh"
# Optional but recommended. Used by launcher metadata, single-flight guards,
# and integration status records.
purpose = "control"
```

## Supported `kind`

- `baseline`
- `env_only`
- `patch`
- `methodology`

## `official_config`

Use this table for benchmark-defining values. Do not report speedups from a
config whose official config differs from baseline.

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

`PYTHON_BIN` is the preferred runtime Python declaration. The launcher records
it in `metadata.json.runtime_python` and validates it before local or Slurm
execution. Do not rely on ambient `python3` for GPU or model-runtime checks.

Configs normally inherit the model profile's environment. Set
`inherit_profile_env = false` at the top level when a config uses an
independent runtime and must not receive legacy conda, cache, or device paths.
Config `[official_config]` values override individual profile values while
retaining unspecified workload fields.

## `purpose`

Supported purposes:

- `control`: baseline, OFF identity, warmup, profile, or other non-scored run.
- `delivery`: config with current public-alignment and quality/speed evidence
  strong enough to become a gated tier delivery profile.
- `frontier`: fan-out config retained for later tier selection.
- `evidence`: measured support that should not become a tier winner.
- `blocker_probe`: bounded probe used to prove a target blocker.
- `unsafe_probe`: intentionally aggressive/unsafe diagnostic probe.

Upper-bound and unsafe probes should be explicit here so single-flight guards,
status records, and release reports do not accidentally promote them as delivery
profiles.

## Algorithm vs Model Glue

Config manifests should preserve the model-agnostic algorithm or policy as
the durable artifact. Cosmos3-specific run scripts, env bridges, dependency
overlays, component names, and fallback plumbing are allowed as reproduction
or validation glue, but they must not be reported as the public algorithm.

Source visibility is not a blocker by itself. A config is blocked for
meaningful GPU evidence only when the pure algorithm is not implemented, the
Cosmos3 runtime does not consume it, the algorithm does not match Cosmos3's
semantics, required quantized weights/online replacement are missing, or the
GPU run lacks a behavior checker against the public boundary.

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

`account` is deliberately absent: the launcher reads `$SLURM_ACCOUNT` and omits
`-A` altogether when that is unset. Put it here only to pin one config to one
account.

```toml
[slurm]
partition = "batch"
nodes = 1
gpus_per_node = 4
cpus_per_task = 64
mem = "0"
time = "04:00:00"
job_name = "autovideo-baseline"
exclusive = true
```

## Future Patch Configs

Patch config should add:

```toml
[patch]
summary = "Wire token pruning around the target model's denoise block loop."
touch_points = [
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
  "techniques/transforms/sparse_attention.py",
]
```

The orchestration layer should use this block to create isolated worktrees and
avoid two agents editing the same submodule checkout.

## Model-Agnostic Efficiency Configs

Efficiency config may use the layered schema below. The launcher accepts
these manifests directly; `model_profile = "cosmos3"` fills the standard
runtime, env, and Slurm defaults from `models/cosmos3.toml`.

```toml
kind = "methodology"
purpose = "frontier"
description = "Short runnable config summary."
model_profile = "cosmos3"

[id]
name = "semantic_permutation"
dimension = "sparse_attention"
family = "svg2_semantic_permutation"

[references.external]
paper = "..."
code = "..."
notes = "Canonical paper/repo or closest open-source implementation."

[references.local]
generic_impl = "techniques/transforms/sparse_attention.py"

[requirements]
capabilities = [
  "has_attention_layers",
  "has_spatiotemporal_token_layout",
  "has_attention_backend_switch",
]

[adapter]
# Descriptive target label only. Dry-run validation now synthesizes a minimal
# ModelSpec from [requirements].capabilities instead of importing built-in
# per-model specs.
model_spec = "Cosmos3"

[efficiency]
kind = "build_transform"
name = "sparse_attention"

[efficiency.params]
component = "transformer"
route_mode = "semantic_permutation"
backend = "sparse_video_gen_2_attn"
svg2_num_q_centroids = 400
svg2_num_k_centroids = 1000
svg2_top_p_kmeans = 0.9
svg2_min_kc_ratio = 0.1
svg2_kmeans_iter_init = 50
svg2_kmeans_iter_step = 2
svg2_first_layers_fp = 0.03
svg2_first_times_fp = 0.3

[verification]
mode = "gpu"
allow_non_bit_exact = true
quality_gate = "baseline-comparable"
```

Layer-1 dry-run validation checks schema, required capabilities, adapter
discovery, compose compatibility, and transform env preview. For build/load
transforms the preview is also merged into `launch.sh`, so the generated run
bundle is inspectable before GPU submission.

## Cosmos3 GPU Readiness Gate

`verification.mode = "gpu"` means the config ultimately needs GPU evidence.
It does not mean the current Cosmos3 runtime already consumes every advertised
optimization. `scripts/launch_config.py` therefore allows `--mode dry-run`
for all valid config, but refuses `--mode local` and `--mode sbatch` for
Cosmos3 config that are currently only pure policies without runtime
consumers, wiring probes, unconsumed env/config adapters, or wired paths with
missing runtime dependencies.

Use `--allow-unsupported-gpu` only for an explicit diagnostic env/export run.
Those runs must not be reported as proof that the public-reference technique is
implemented or active.

## Public Reference Alignment Gate

`scripts/audit_public_reference_alignment.py` records each config's actual
scope relative to its public/canonical references. This is separate from URL
validation:

- A public URL proves provenance, not full implementation equivalence.
- Public source access means an implementation can be attempted; it does not
  prove the local config already implements or consumes that algorithm.
- Baselines, pure policy layers, env adapters, and blocker probes must not be
  promoted as line-for-line public-reference ports.
- A config with any current public-alignment `true_blocker` must not use
  `purpose = "delivery"`. At minimum the Cosmos3 runtime must consume the
  advertised env/config, a GPU run must prove that path is active, and current
  quality/speed evidence must not be blocked.

The review matrix is generated on demand with
`scripts/audit_public_reference_alignment.py --markdown-out <path>`; it is not
required to live in the repository.
