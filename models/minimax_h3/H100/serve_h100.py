#!/usr/bin/env python3
"""MiniMax-H3 HTTP server for 80 GB GPUs (H100), SGLang-compatible video API.

    python serve_h100.py --model /models/MiniMax-H3-diffusers --num-gpus 4 --attention sol --port 30000

Same endpoints, request fields and response schema as the GB200 server (`../GB200/serve.py`, whose HTTP
layer and request parsing this reuses). The engine is `h3_h100.build_engine`: on 1 GPU every DiT block and
text-encoder layer streams in from pinned host memory per layer; on N GPUs the DiT blocks are sharded 1/N per
rank and all-gathered per layer, the block stack runs Ulysses over N ranks (below 4 GPUs the text encoder
still streams from host memory). The server always runs under torchrun (it re-launches itself); rank 0
serves HTTP and hands each request to the other ranks over a gloo control group. Host memory: about 150 GB
per GPU while loading.
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
GB200 = os.path.join(HERE, "..", "GB200")
sys.path[:0] = [HERE, GB200]

import serve as gb200_serve                              # ../GB200/serve.py: HTTP app, request parsing  # noqa: E402

log = gb200_serve.log
WARMUP_PROMPT = "A calm lake at sunrise, gentle waves, birds singing."


class H3H100Engine:
    """The interface `gb200_serve.build_app` drives: generate(), progress, precision, attention, applied."""

    def __init__(self, model_path: str, attention: str) -> None:
        import torch
        import torch.distributed as dist
        import h3_h100

        self.torch, self.dist, self.h3 = torch, dist, h3_h100
        self.precision, self.attention = "bf16", attention
        self.engine = h3_h100.build_engine(model_path, attention)
        self.rank, self.world = self.engine.rank, self.engine.world
        # requests arrive at arbitrary times: wait for them on CPU (gloo), never inside an NCCL call
        self.ctrl = dist.new_group(backend="gloo", timeout=timedelta(days=365))
        self.applied = {"num_gpus": self.world, "attention": attention, "precision": "bf16",
                        "placement": "offload" if self.world == 1 else f"sharded x{self.world} + ulysses"}
        tr = self.engine.pipe.transformer
        self._calls, self._steps = 0, 1
        inner = tr.forward

        def counted(*a, **kw):
            self._calls += 1
            return inner(*a, **kw)
        tr.forward = counted

    @property
    def progress(self) -> int:
        return min(99, int(100 * self._calls / max(1, self._steps)))

    def run(self, req: dict, out_path: str | None) -> dict:
        """One request on every rank; rank 0 encodes the mp4."""
        torch = self.torch
        self._calls, self._steps = 0, max(1, req["steps"] - 1)
        torch.cuda.reset_peak_memory_stats()
        state, seconds = self.h3.run_request(self.engine, req["prompt"], req["height"], req["width"],
                                             req["num_frames"], req["steps"], req["seed"])
        stats = {"inference_time_s": round(seconds, 3),
                 "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        if self.rank == 0 and out_path:
            t = time.perf_counter()
            self.h3.write_state_mp4(state, out_path, req.get("preset", gb200_serve.DEFAULT_PRESET),
                                    req.get("crf", gb200_serve.DEFAULT_CRF))
            stats["encode_s"] = round(time.perf_counter() - t, 3)
        return stats

    def broadcast(self, msg):
        box = [msg]
        self.dist.broadcast_object_list(box, src=0, group=self.ctrl)
        return box[0]

    # rank 0, from the HTTP worker thread
    def generate(self, out_path, prompt, width, height, num_frames, steps, seed,
                 preset=gb200_serve.DEFAULT_PRESET, crf=gb200_serve.DEFAULT_CRF) -> dict:
        """`steps` = sigma grid points (diffusers), i.e. denoising updates + 1."""
        req = {"prompt": prompt, "width": width, "height": height, "num_frames": num_frames, "steps": steps,
               "seed": seed, "preset": preset, "crf": crf}
        self.broadcast(req)
        return self.run(req, out_path)

    # ranks > 0
    def worker_loop(self) -> None:
        while True:
            req = self.broadcast(None)
            if req is None:
                return
            self.run(req, None)


def relaunch_under_torchrun(num_gpus: int) -> None:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={num_gpus}"]
    os.execv(sys.executable, cmd + [os.path.abspath(__file__)] + sys.argv[1:])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("H3_MODEL_PATH", "/models/MiniMax-H3-diffusers"),
                   help="converted diffusers checkpoint directory")
    p.add_argument("--num-gpus", type=int, choices=(1, 2, 4, 8), default=int(os.environ.get("NUM_GPUS", 1)))
    p.add_argument("--attention", choices=("sol", "dense"), default=os.environ.get("H3_ATTENTION", "sol"),
                   help="sol = Sol-Attn sparse attention (SM90 kernel); dense = FlashAttention-4 sm90")
    p.add_argument("--precision", choices=("bf16",), default="bf16")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 30000)))
    p.add_argument("--output-dir", default=os.environ.get("H3_OUTPUT_DIR", "/outputs"))
    p.add_argument("--served-model-name", default=gb200_serve.MODEL_ID)
    p.add_argument("--warmup", type=int, default=1, help="warmup requests before /health answers")
    p.add_argument("--default-seconds", type=float, default=5.0)
    p.add_argument("--default-steps", type=int, default=49, help="denoising updates when a request names none")
    p.add_argument("--disable-conditioning-cache", action="store_true",
                   help="accepted for SGLang parity; this server never caches prompt conditioning")
    a = p.parse_args()
    if "LOCAL_RANK" not in os.environ:
        relaunch_under_torchrun(a.num_gpus)

    rank = int(os.environ.get("RANK", "0"))
    os.makedirs(a.output_dir, exist_ok=True)
    defaults = SimpleNamespace(seconds=a.default_seconds, steps=a.default_steps, seed=0)
    scratch = os.environ.get("H3_SCRATCH", os.path.join(a.output_dir, ".scratch"))
    model = gb200_serve.relocate_checkpoint(a.model, os.path.join(scratch, f"rank{rank}"))
    engine = H3H100Engine(model, a.attention)

    frames = gb200_serve.align_frame_count(round(a.default_seconds * gb200_serve.FPS))
    for i in range(a.warmup):                            # every rank runs the warmups in lockstep
        t = time.perf_counter()
        r = engine.run({"prompt": WARMUP_PROMPT, "width": 1344, "height": 768, "num_frames": frames,
                        "steps": a.default_steps + 1, "seed": 0}, os.path.join(a.output_dir, f"_warmup_{i}.mp4"))
        log(f"warmup {i}: {time.perf_counter() - t:.1f}s {json.dumps(r)}")
    if rank != 0:
        engine.worker_loop()
        return 0

    import uvicorn
    app = gb200_serve.build_app(engine, a.output_dir, defaults, a.served_model_name)
    log(f"serving on {a.host}:{a.port} ({a.num_gpus} GPU, {a.attention})")
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        engine.broadcast(None)                           # release the worker ranks
    return 0


if __name__ == "__main__":
    sys.exit(main())
