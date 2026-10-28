#!/usr/bin/env python3
"""Sol-Engine GB200 server entry point: picks the model's server from the checkpoint and execs it.

    serve.py --model /models/<checkpoint> [server options]

    MiniMax-H3   (modular_model_index.json)           -> models/minimax_h3/GB200/serve.py
    Wan2.2 5B/A14B (model_index.json: WanPipeline)    -> models/wan22/GB200/serve.py

Every option after the model is passed through unchanged; see each server's --help.
"""
import json
import os
import sys

ROOT = os.environ.get("SOL_ENGINE_ROOT", "/opt/sol-engine")
SERVERS = {"minimax_h3": "models/minimax_h3/GB200/serve.py", "wan22": "models/wan22/GB200/serve.py"}


def model_family(path: str) -> str:
    if os.path.exists(os.path.join(path, "modular_model_index.json")):
        return "minimax_h3"
    index = os.path.join(path, "model_index.json")
    if os.path.exists(index) and json.load(open(index)).get("_class_name", "").startswith("Wan"):
        return "wan22"
    raise SystemExit(f"{path}: not a MiniMax-H3 or Wan2.2 diffusers checkpoint")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    model = os.environ.get("MODEL_PATH")
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
        elif a.startswith("--model="):
            model = a.split("=", 1)[1]
    if not model:
        raise SystemExit("--model <checkpoint dir> is required")
    family = model_family(model)
    if "--model" not in argv and not any(a.startswith("--model=") for a in argv):
        argv = ["--model", model] + argv
    server = os.path.join(ROOT, SERVERS[family])
    print(f"[sol-engine] {family}: {server} {' '.join(argv)}", flush=True)
    os.execv(sys.executable, [sys.executable, server] + argv)


if __name__ == "__main__":
    main()
