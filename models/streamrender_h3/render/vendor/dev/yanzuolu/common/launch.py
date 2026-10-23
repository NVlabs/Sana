"""Project-agnostic dynamic launcher.

The config owns the experiment entry point::

    entry:
      module: projects.my_project.engines.train
      class_name: TrainEngine

Command-line overrides use KEY VALUE pairs::

    python -m common.launch --config exp.yaml engine.lr 1e-5 runtime.seed 42

The engine is instantiated as ``cls(config, meta_model)`` and must expose
``run()``; ``meta_model`` is resolved the same way from ``config.meta_model``
(``module`` / ``class_name``) and receives the whole config. This file
intentionally has no project registry and no imports from ``projects`` /
``diffusion`` / ``tools``.
"""

from __future__ import annotations

import argparse
import importlib
import traceback
from pathlib import Path
from typing import Any

from .config import CfgNode
from .distributed.init import configure_distributed, destroy_distributed
from .logging import configure_logging
from .persistence import configure_persistence, resolve_run_dirs


def main(argv: list[str] | None = None) -> Any:
    parser = argparse.ArgumentParser(description="Run an experiment config via dynamic target import.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = CfgNode.from_file(config_path)
    config.merge_from_list(args.opts)
    config = resolve_run_dirs(config, config_path)   # pure path derivation
    config = configure_logging(config)               # text logging up first ...
    config = configure_distributed(config)           # ... so these two can log
    config = configure_persistence(config, config_path, args.opts)

    module = importlib.import_module(config.entry.module)
    engine_cls = getattr(module, config.entry.class_name)
    meta_cls = getattr(importlib.import_module(config.meta_model.module), config.meta_model.class_name)
    exit_code = 0
    try:
        return engine_cls(config, meta_cls(config)).run()
    except BaseException:
        exit_code = 1
        # Print the traceback NOW: the finally below may block forever (peer ranks
        # stuck in a collective make destroy_distributed / finalize hang), which
        # would otherwise swallow the crash diagnostics entirely.
        traceback.print_exc()
        raise
    finally:
        # Flush any in-flight async checkpoint before tearing down the process
        # group it writes through; then close trackers with the real exit code
        # so a crashed run is not marked as finished.
        config.checkpointer.finalize()
        config.logger.close(exit_code=exit_code)
        # Only tear the process group down on a clean run. After a crash the
        # surviving ranks are stuck in collectives; destroy_distributed() would
        # block on them forever and keep the whole job alive in a zombie state.
        if exit_code == 0:
            destroy_distributed()


if __name__ == "__main__":
    main()
