# Run

```bash
python3 scripts/run.py config/<model>/<arm>.toml
```

That is the whole interface. `<model>` is a directory under `config/`, `<arm>`
is one configuration of it — usually `baseline` plus one or more optimized
arms.

```bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml
python3 scripts/run.py config/wan22_ti2v_5b/wan5b_kernel_easycache_pisa.toml
```

Each run writes a bundle under `runs/<timestamp>-<id>/`: `launch.sh`,
`manifest.resolved.toml`, `metadata.json`, and `outputs/` with the video and
`benchmark.json`.

## Before you run

```bash
python3 scripts/run.py <config> --print
```

Resolves the config and checks its paths without running anything — the runtime
directory, the entry script, the interpreter — and fails naming whichever is
missing.

## Overriding a value

```bash
python3 scripts/run.py <config> --set KEY=VALUE [--set KEY=VALUE ...]
```

Changes a key for one run without editing the file. For example, a short smoke
run instead of the full 50 steps:

```bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml \
    --set WAN22_STEPS=4 --set WAN22_WARMUP_PASSES=0
```

## The config format

```toml
id            = "<model>_<arm>"
kind          = "baseline"           # or "optimized"
model_profile = "<model>"            # inherits models/<model>.toml

[runtime]
root = "models/<model>/<variant>"    # the code this config launches

[env]
KEY = "value"                        # exported as an environment variable
```

The model profile carries the workload and whatever every arm of that model
shares; the config carries what makes this arm different. `--print` shows the
merged result.

```toml
id            = "wan22_ti2v_5b_baseline"
kind          = "baseline"
model_profile = "wan22_ti2v_5b"

[runtime]
root = "models/wan22_ti2v_5b/baseline"

[env]
WAN22_STEPS = "50"
```
