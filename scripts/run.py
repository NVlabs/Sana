#!/usr/bin/env python3
"""Run one model arm from a single flat config file.

    python3 scripts/run.py models/minimax_h3/GB200/dense.toml

That is the whole interface. There is no scheduler in this file: it runs the arm
here, in this process, on whatever machine you are on. To run it under Slurm,
put this exact command inside your own sbatch script -- job scripts are not
tracked here because their account/partition/QoS are site-specific; see
site_docs/installation.md for the four-line wrapper. Nothing here reads SLURM_*, and
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
                     "models/minimax_h3/demo_prompt.json").
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import launch_config  # noqa: E402

# Keys this launcher consumes. Everything else that is uppercase becomes env.
RESERVED = {"name", "runtime", "entry", "out", "description", "gpus"}

# Artifact names prepare_run resolves under <run>/outputs/. Spelled out so a flat
# config produces the same layout as a config manifest and collect_run.py can
# read either.
DEFAULT_ARTIFACTS = {
    "output_dir": "outputs",
    "video": "out.mp4",
    "log": "run.log",
    "benchmark": "benchmark.json",
}


def is_config_manifest(cfg: dict[str, object]) -> bool:
    """Tell the two dialects apart.

    Tested on the fields prepare_run actually requires, not on the two that are
    merely common. config/cosmos3/sglang_baseline.toml is a manifest with
    neither model_profile nor a [runtime] table -- it names its submodule and
    run_script directly -- and a model_profile/runtime test called it a flat
    config and then rejected every lowercase key it has.
    """
    if "model_profile" in cfg or isinstance(cfg.get("runtime"), dict):
        return True
    return "submodule" in cfg and "run_script" in cfg


def to_manifest(
    name: str,
    launcher: dict,
    env: dict,
    runtime: Path,
    entry: Path,
    config_path: Path,
) -> dict:
    """Flat config -> the manifest shape prepare_run expects.

    submodule and run_script are written as repo-relative paths because that is
    what prepare_run joins them from; the flat dialect's own bases (config-dir
    for runtime/entry) have already been resolved by the time we get here.
    """
    return {
        "id": name,
        "kind": str(launcher.get("kind") or "baseline"),
        "purpose": str(launcher.get("description") or "single-file config run"),
        "submodule": str(runtime.relative_to(REPO)),
        "run_script": str(entry.relative_to(runtime)),
        "runtime": {"root": str(runtime.relative_to(REPO))},
        "env": dict(env),
        "artifacts": dict(DEFAULT_ARTIFACTS),
        "source_config": str(config_path.relative_to(REPO)),
    }


def load_config(path: Path) -> dict[str, object]:
    text = path.read_text()
    try:
        import tomllib

        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    try:
        # draco and cs ship python 3.9. A config manifest has nested tables the
        # fallback below cannot represent, so the backport comes first --
        # launch_config and collect_run already prefer it for the same reason.
        import tomli

        return tomli.loads(text)
    except ModuleNotFoundError:
        pass
    # Last resort. The flat format makes this honest: there are no tables to
    # track, so a line-splitter is a complete parser -- of a flat config. It is
    # not one for a manifest, which is what the guard below is for.
    out: dict[str, object] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if not line:
            continue
        # Checked before the '=' filter, not after it. After was unreachable: a
        # header like [env] contains no '=', so the `continue` fired first and
        # the header was skipped in silence. Every key under it was then
        # flattened into the top level, prepare_run merged an [env] it read as
        # empty, and the run proceeded with the model profile's defaults alone.
        # On draco that dropped H3_CONTAINER_RUNTIME = "pyxis" from an A100
        # config, so the job never entered its container and died on `No module
        # named sglang` -- while manifest.resolved.toml, written from a copy
        # parsed properly, showed the value present the whole time.
        if line.startswith("["):
            raise SystemExit(
                f"{path}: {line} -- this parser reads flat configs only. To read "
                "a config manifest on python < 3.11, install the tomli backport."
            )
        if "=" not in line:
            continue
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
    ap.add_argument(
        "config",
        help="a flat single-file config, or a config manifest from config/",
    )
    ap.add_argument("--run-root", default="runs", help="run bundle root")
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

    # --set has to land where the dialect keeps that key. A flat config holds
    # its environment at the top level, a manifest nests it under [env], so
    # writing every override to the top level made --set PYTHON_BIN=... silently
    # do nothing for manifests -- it added an ignored top-level key while the
    # real value stayed in [env]. Uppercase goes to the env table for manifests;
    # lowercase is a launcher key either way.
    manifest = is_config_manifest(cfg)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if manifest and key == key.upper():
            cfg.setdefault("env", {})[key] = value
        else:
            cfg[key] = value

    # A config manifest needs none of the flat dialect's translation: it is
    # already the shape prepare_run reads, including the [requires] block that
    # drives the capability/conflict check. Run it directly.
    if manifest:
        launch_args = argparse.Namespace(
            config=str(config_path), mode="local", run_root=args.run_root,
            name_suffix="", runtime_root=None, env=None, strict_commit=False,
            confirm_submit=False, allow_unsupported_gpu=True,
        )
        if args.print_only:
            # dry-run mode so machine-specific paths -- PYTHON_BIN, weights,
            # anything outside the repo -- do not fail a config written for a
            # different cluster. It also sets allow_missing_runtime, which is
            # how five GB200 configs kept pointing at baseline/ and optimized/
            # after those were flattened away, so the repo-side paths are
            # checked here instead. Fatal for what is wrong everywhere, silent
            # for what is only wrong here.
            launch_args.mode = "dry-run"
            # Only [runtime].root, never submodule: root is repo-relative by
            # contract, while submodule may name an external checkout
            # (config/cosmos3/sglang_baseline.toml points at the SGLang runtime),
            # which is machine-specific and belongs in the warn bucket.
            root = cfg.get("runtime", {}).get("root") if isinstance(cfg.get("runtime"), dict) else None
            if root and not (REPO / str(root)).is_dir():
                raise SystemExit(f"runtime root does not exist: {REPO / str(root)}")
            script = cfg.get("run_script")
            if root and script and not (REPO / str(root) / str(script)).is_file():
                raise SystemExit(
                    f"run script does not exist: {REPO / str(root) / str(script)}"
                )
        run_dir, launch_script, _job, _data = launch_config.prepare_run(
            launch_args, launch_config.merge_model_profile(cfg)
        )
        print(f"config: {cfg.get('id', config_path.stem)}")
        print(f"run_dir  : {run_dir}")
        if args.print_only:
            print("status: printed only, nothing run")
            return 0
        proc = subprocess.run([str(launch_script)])
        launch_config.update_metadata(
            run_dir, {"status": "completed", "returncode": proc.returncode}
        )
        print(f"status: exit {proc.returncode}  ({run_dir})")
        return proc.returncode

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

    print(f"name    : {name}")
    print(f"runtime : {runtime}")
    print(f"entry   : {entry}")
    print(f"env     : {len(env)} vars")
    if args.print_only:
        for key in sorted(env):
            print(f"    {key}={env[key]}")
        print("status: printed only, nothing run")
        return 0

    # Hand the assembled manifest to launch_config's prepare_run rather than
    # executing here. That is what makes this a front-end and not a second
    # implementation: one bundle format, one set of provenance files, and
    # collect_run.py finds artifacts under outputs/ no matter which config
    # dialect was used to start the run.
    data = to_manifest(name, launcher, env, runtime, entry, config_path)
    launch_args = argparse.Namespace(
        config=str(config_path),
        mode="local",
        run_root=args.run_root,
        name_suffix="",
        runtime_root=str(runtime),
        env=None,
        strict_commit=False,
        confirm_submit=False,
        allow_unsupported_gpu=True,
    )
    run_dir, launch_script, _job, _data = launch_config.prepare_run(launch_args, data)
    print(f"run_dir : {run_dir}")
    proc = subprocess.run([str(launch_script)])
    launch_config.update_metadata(run_dir, {"status": "completed", "returncode": proc.returncode})
    print(f"status: exit {proc.returncode}  ({run_dir})")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
