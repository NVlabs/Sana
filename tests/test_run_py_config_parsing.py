"""`scripts/run.py` must read a config the same way on every interpreter.

run.py parses the config itself and hands the dict to `launch_config.prepare_run`,
so a parser that loses part of the file does not fail -- it launches a different
run. That happened: on python 3.9 the tomllib import fails, and the line-splitter
fallback dropped every `[table]` header, flattening the keys beneath into the top
level. `prepare_run` then read `data["env"]` as empty, merged only the model
profile's defaults, and an A100 config lost `H3_CONTAINER_RUNTIME = "pyxis"`. The
job ran outside its container and died on `No module named sglang`, six seconds
in, with `manifest.resolved.toml` -- written from a properly parsed copy -- showing
the setting present all along.

The tests below are the two halves of that: the fallback must refuse a file it
cannot represent instead of silently returning a partial dict, and the loader must
produce the same nesting whichever backend it used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("_run_py", ROOT / "scripts/run.py")
run_py = importlib.util.module_from_spec(_spec)
sys.modules["_run_py"] = run_py
_spec.loader.exec_module(run_py)

MANIFEST = """\
id = "example"
model_profile = "minimax_h3"

[env]
H3_CONTAINER_RUNTIME = "pyxis"
"""

FLAT = """\
name = "example"
runtime = "."
entry = "run.sh"
H3_STEPS = "50"
"""


def _fallback_only(monkeypatch) -> None:
    """Make both TOML backends unimportable, as python 3.9 without tomli is."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name in {"tomllib", "tomli"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)


def test_fallback_refuses_a_manifest_instead_of_dropping_its_tables(
    tmp_path, monkeypatch
) -> None:
    """The guard has to fire on the header itself.

    It used to sit after `if "=" not in line: continue`, which made it dead code
    -- `[env]` has no `=`. The parser returned a dict that looked fine and was
    missing a table.
    """
    path = tmp_path / "manifest.toml"
    path.write_text(MANIFEST)
    _fallback_only(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        run_py.load_config(path)
    assert "[env]" in str(excinfo.value)


def test_fallback_still_reads_a_flat_config(tmp_path, monkeypatch) -> None:
    """The flat dialect has no tables, so the line-splitter remains complete."""
    path = tmp_path / "flat.toml"
    path.write_text(FLAT)
    _fallback_only(monkeypatch)
    cfg = run_py.load_config(path)
    assert cfg == {
        "name": "example",
        "runtime": ".",
        "entry": "run.sh",
        "H3_STEPS": "50",
    }


@pytest.mark.parametrize(
    "config",
    sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").rglob("*.toml")),
)
def test_env_table_survives_parsing(config: str) -> None:
    """Every manifest that declares [env] must still have it after loading.

    The regression was invisible per-file: the dict was well-formed, just
    shallower. Comparing against the file's own text is what catches it.
    """
    path = ROOT / config
    if "[env]" not in path.read_text():
        pytest.skip("no [env] table")
    cfg = run_py.load_config(path)
    assert isinstance(cfg.get("env"), dict) and cfg["env"], (
        f"{config} declares [env] but it did not survive load_config; its keys "
        "were flattened into the top level, where prepare_run does not look."
    )
