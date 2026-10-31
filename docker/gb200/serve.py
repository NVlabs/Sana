#!/usr/bin/env python3
"""Sol-Engine GB200 server entry point: picks the model's server from the checkpoint and execs it.

    serve.py --model /models/<checkpoint> [server options]

    MiniMax-H3   (modular_model_index.json)           -> models/minimax_h3/GB200/serve.py      (GPUs >= 120 GB)
                                                      -> models/minimax_h3/H100/serve_h100.py  (80 GB GPUs; 1-8)
    Wan2.2 5B/A14B (model_index.json: WanPipeline)    -> models/wan22/GB200/serve.py

Every option after the model is passed through unchanged; see each server's --help.
H3_SERVER=gb200|h100 overrides the choice by GPU memory.
"""
import json
import os
import sys

ROOT = os.environ.get("SOL_ENGINE_ROOT", "/opt/sol-engine")
SERVERS = {"minimax_h3": "models/minimax_h3/GB200/serve.py", "wan22": "models/wan22/GB200/serve.py",
           "minimax_h3_h100": "models/minimax_h3/H100/serve_h100.py"}


def h3_server() -> str:
    """MiniMax-H3 needs ~121 GB on one GPU: the single-GPU server on GB200-class memory, the sharded /
    offloading one on 80 GB GPUs."""
    choice = os.environ.get("H3_SERVER")
    if choice:
        return {"gb200": "minimax_h3", "h100": "minimax_h3_h100"}[choice]
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()
    return "minimax_h3" if min(int(x) for x in out) >= 120 * 1024 else "minimax_h3_h100"


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
    if family == "minimax_h3":
        family = h3_server()
    server = os.path.join(ROOT, SERVERS[family])
    print(f"[sol-engine] {family}: {server} {' '.join(argv)}", flush=True)
    os.execv(sys.executable, [sys.executable, server] + argv)


if __name__ == "__main__":
    main()
