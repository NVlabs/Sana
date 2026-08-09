#!/usr/bin/env python3
"""Run one model arm from a single flat config file.

    python3 scripts/run.py models/minimax_h3/gb200/dense.toml

That is the whole interface. There is no scheduler in this file: it runs the arm
here, in this process, on whatever machine you are on. To run it under Slurm,
put this exact command inside your own sbatch script -- job scripts are not
tracked here because their account/partition/QoS are site-specific; see
docs/simple-launch.md for the four-line wrapper. Nothing here reads SLURM_*, and
nothing here calls srun/sbatch/squeue, so the same command is correct on a login
node inside an salloc, on a bare workstation, and inside a batch job.

WHERE CONFIGS LIVE

Beside the code they launch, one directory per hardware target:

    models/minimax_h3/            the model
      gb200/                        the GB200 implementation
        dense.toml                    launch config  ->  the dense control
        fullopt.toml                  launch config  ->  the full stack
        <driver + modules>            the code
      h100/  a100/  gb10/  rtx5090/  other targets, same shape

There is no top-level configs/ directory. A config is part of the hardware
implementation it launches, so it ships and moves with it.

CONFIG FORMAT -- one layer, no nesting:

    lowercase keys  are for this launcher   (name, runtime, entry, out, gpus)
    UPPERCASE keys  are exported verbatim as environment variables

That split is the entire schema. It needs no [sections] because environment
variables are conventionally uppercase already, so the two namespaces cannot
collide. A config is therefore also readable by anything that can split on '=';
this file carries a 12-line fallback parser so it does not even need tomllib.

PATHS -- two bases, one rule each:

    runtime, entry   resolve against THIS CONFIG'S directory. A config sitting
                     beside its code says runtime = ".", not the long repo path.
    UPPERCASE values are handed to a process whose cwd is the repo root, so a
                     path in them reads from there (H3_PROMPT_FILE =
                     "models/minimax_h3/prompts/t2va_example_1.json").
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Keys this launcher consumes. Everything else that is uppercase becomes env.
RESERVED = {"name", "runtime", "entry", "out", "description", "gpus"}


def load_config(path: Path) -> dict[str, object]:
    text = path.read_text()
    try:
        import tomllib

        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    # Fallback for pythons without tomllib. The flat format makes this honest:
    # there are no tables to track, so a line-splitter is a complete parser.
    out: dict[str, object] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if not line or "=" not in line:
            continue
        if line.startswith("["):
            raise SystemExit(
                f"{path}: [{line}] -- this launcher takes flat configs only, no tables"
            )
        key, value = (part.strip() for part in line.split("=", 1))
        out[key] = value.strip('"').strip("'")
    return out


def split_config(cfg: dict[str, object], path: Path) -> tuple[dict, dict]:
    """-> (launcher keys, environment keys). Anything neither is an error."""
    launcher, env, unknown = {}, {}, []
    for key, value in cfg.items():
        if key in RESERVED:
            launcher[key] = value
        elif key == key.upper():
            # Covers names that str.isupper() rejects for having no cased
            # characters of its own, e.g. a key that is all digits/underscores.
            env[key] = str(value)
        else:
            unknown.append(key)
    if unknown:
        raise SystemExit(
            f"{path}: unknown lowercase key(s) {unknown}. Launcher keys are "
            f"{sorted(RESERVED)}; everything else must be UPPERCASE (exported as env)."
        )
    return launcher, env


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", help="flat TOML config, e.g. configs/foo.toml")
    ap.add_argument("--out", help="output dir (default: runs/<name>-<utc stamp>)")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key; repeatable",
    )
    ap.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="resolve and print, run nothing",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    cfg = load_config(config_path)

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        cfg[key] = value

    launcher, env = split_config(cfg, config_path)

    name = str(launcher.get("name") or config_path.stem)
    runtime_rel = launcher.get("runtime")
    entry_rel = launcher.get("entry")
    if not runtime_rel or not entry_rel:
        raise SystemExit(f"{config_path}: 'runtime' and 'entry' are required")

    # Resolved against the config's own directory, with no repo-root fallback.
    # A single base keeps it predictable: a config that sits with its code
    # says runtime = ".", and moving the pair moves both.
    runtime = (config_path.parent / str(runtime_rel)).resolve()
    entry = (runtime / str(entry_rel)).resolve()

    # The checks that matter, done before anything is allocated or launched.
    # A missing entry script is the failure that costs the most when it is found
    # late: on this stack it would otherwise surface after a GPU allocation.
    # runtime/entry live in the repo, so they are wrong everywhere and are always
    # fatal. PYTHON_BIN is machine-specific -- a config for another cluster is
    # still worth inspecting here, so --print downgrades it to a warning.
    if not runtime.is_dir():
        raise SystemExit(f"runtime dir does not exist: {runtime}")
    if not entry.is_file():
        raise SystemExit(f"entry script does not exist: {entry}")
    python_bin = env.get("PYTHON_BIN")
    if python_bin and not Path(python_bin).exists():
        message = f"PYTHON_BIN does not exist: {python_bin}"
        if not args.print_only:
            raise SystemExit(message)
        print(f"WARN: {message}", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(
        args.out or launcher.get("out") or REPO / "runs" / f"{name}-{stamp}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "name": name,
        "config": str(config_path),
        "runtime": str(runtime),
        "entry": str(entry),
        "out_dir": str(out_dir),
        "started_at_utc": stamp,
        "hostname": os.uname().nodename,
        "env": dict(sorted(env.items())),
    }
    (out_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2))

    print(f"name    : {name}")
    print(f"runtime : {runtime}")
    print(f"entry   : {entry}")
    print(f"out_dir : {out_dir}")
    print(f"env     : {len(env)} vars")
    if args.print_only:
        for key in sorted(env):
            print(f"    {key}={env[key]}")
        print("status: printed only, nothing run")
        return 0

    # cwd is the repo root, not the runtime dir. Entry scripts locate themselves
    # through BASH_SOURCE and so do not care, which leaves cwd free to mean the
    # one useful thing: relative paths inside a config (H3_PROMPT_FILE and the
    # like) resolve the way they read, i.e. from the repo root, matching how
    # every other config in this tree writes a path.
    run_env = {**os.environ, **env, "OUT_DIR": str(out_dir)}
    proc = subprocess.run(["bash", str(entry)], env=run_env, cwd=str(REPO))
    (out_dir / "config.resolved.json").write_text(
        json.dumps({**resolved, "returncode": proc.returncode}, indent=2)
    )
    print(f"status: exit {proc.returncode}  ({out_dir})")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
