"""Local JSON-line adapter for one persistent model session.

Framework output goes to the worker log; atomic JSON receipts carry results.
Only the owning pipeline writes commands to this process's standard input.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=("stage1", "stage2", "qwen"), required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    session = None
    started = time.monotonic_ns()
    try:
        module = importlib.import_module("runtime." + args.module)
        session = module.Session(**json.loads(args.options.read_text()))
        write_json(args.ready, {"status": "READY", "started_monotonic_ns": started,
                               "ready_monotonic_ns": time.monotonic_ns()})
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "close":
                break
            if command["op"] not in ("run", "prepare", "release_idle_cache"):
                raise ValueError("unsupported session operation")
            try:
                result = getattr(session, command["op"])(**command["kwargs"])
                write_json(command["response"], {"status": "PASS", "result": result})
            except Exception as error:
                write_json(command["response"], {
                    "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
                raise
    except BaseException as error:
        if not args.ready.exists():
            write_json(args.ready, {"status": "FAIL", "error": f"{type(error).__name__}: {error}"})
        traceback.print_exc()
        raise
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
