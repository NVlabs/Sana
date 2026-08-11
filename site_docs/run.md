# Run

```bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml
```

That is the whole interface. One config per arm, under `config/<model>/`.

Each run writes a bundle under `runs/<timestamp>-<id>/`: `launch.sh`,
`manifest.resolved.toml`, `metadata.json`, and `outputs/` with the video and
`benchmark.json`.

## Before you run

```bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml --print
```

Resolves the config and checks its paths without running anything. It reports
the runtime directory, the entry script and the interpreter, and fails naming
the one that is missing.

## Overriding a value

```bash
python3 scripts/run.py config/wan22_ti2v_5b/baseline.toml \
    --set WAN22_STEPS=4 --set WAN22_WARMUP_PASSES=0
```

`--set` changes one key for one run without editing the file. Repeatable.

## The config format

```toml
id            = "wan22_ti2v_5b_baseline"
kind          = "baseline"
model_profile = "wan22_ti2v_5b"          # inherits models/wan22_ti2v_5b.toml

[runtime]
root = "models/wan22_ti2v_5b/baseline"   # the code this config launches

[env]
WAN22_STEPS = "50"                        # exported as an environment variable
```

The model profile supplies the workload and the defaults every arm of that
model shares; the config carries what makes this arm different. `--print` shows
the merged result.
