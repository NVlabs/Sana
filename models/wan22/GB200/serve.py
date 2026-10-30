#!/usr/bin/env python3
"""Wan2.2 (TI2V-5B / T2V-A14B) on 1-8 GB200s: SGLang-compatible HTTP video server (Sol-Engine stack).

    python serve.py --model /models/Wan2.2-T2V-A14B-Diffusers --num-gpus 4 --attention fa4sol --port 30000

Endpoints (same paths and response schema as `sglang serve --model-type diffusion`):

    POST   /v1/videos                 create a job; body = SGLang VideoGenerationsRequest subset
    GET    /v1/videos                 list jobs
    GET    /v1/videos/{id}            job status
    GET    /v1/videos/{id}/content    the mp4 (H.264)
    DELETE /v1/videos/{id}
    GET    /health  /v1/models  /model_info

The engine is the measured stack: `wan_fast` (own block forward, CFG batching, fused Triton kernels,
cuDNN attention), sparse attention `fa4sol` (Sol routing on FA4 block-sparse tiles, default), `sol`
(the Sol-Attn kernel) or `dense`.  With --num-gpus N > 1 the server re-launches itself under torchrun:
CFG parallel 2 x Ulysses N/2 with symmetric-memory all-to-all, and a row-sharded VAE decode.  Rank 0
serves HTTP and broadcasts each request to the other ranks over a gloo control group; across two
nodes, start one server per node with --nnodes 2 --node-rank {0,1} --dist-init-addr <node0>:<port>.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
       "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
       "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
# the official standard workloads; request fields override them
DEFAULTS = {
    "a14b": dict(name="Wan-AI/Wan2.2-T2V-A14B-Diffusers", align=16, width=1280, height=720, num_frames=81, fps=16,
                 num_inference_steps=40, guidance_scale=4.0, guidance_scale_2=3.0, flow_shift=12.0),
    "5b": dict(name="Wan-AI/Wan2.2-TI2V-5B-Diffusers", align=32, width=1280, height=704, num_frames=121, fps=24,
               num_inference_steps=50, guidance_scale=5.0, guidance_scale_2=None, flow_shift=5.0),
}


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[wan-serve] {msg}", flush=True)


def model_kind(path: str) -> str:
    return "a14b" if os.path.isdir(os.path.join(path, "transformer_2")) else "5b"


# -- engine -------------------------------------------------------------------------------------

class WanEngine:
    def __init__(self, model_path: str, precision: str, attention: str, rho: float, vae_compile: bool,
                 expert_swap: str = "auto"):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        self.kind = model_kind(model_path)
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.ctrl = None
        if self.world > 1:
            lr = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(lr)
            dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
            # requests arrive at arbitrary times: wait for them on CPU (gloo), never inside an NCCL call
            self.ctrl = dist.new_group(backend="gloo", timeout=timedelta(days=365))
        self.precision, self.attention = precision, attention
        os.environ["WAN_SPARSE_IMPL"] = "fa4sol" if attention == "fa4sol" else "sol"
        os.environ["WAN_FA4SOL_RHO"] = str(rho)

        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
        import wan_fast

        t0 = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.bfloat16)
        self.pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
        self._sched_cfg = self.pipe.scheduler.config
        self._shift = None
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        if vae_compile and self.world == 1:          # the row-sharded VAE is eager
            self.pipe.vae.decoder = torch.compile(self.pipe.vae.decoder, mode="default", dynamic=False)
        info = wan_fast.install_pipeline(self.pipe, precision, "dense" if attention == "dense" else "sol")
        # A14B holds two 14B experts; on GPUs under 120 GB keep only the expert in use resident
        gpu_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
        self.expert_swap = self.kind == "a14b" and (expert_swap == "on" or (expert_swap == "auto" and gpu_gb < 120))
        if self.expert_swap:
            info.update(wan_fast.enable_expert_swap(self.pipe))
        self.sol = info.get("sol")
        self.parallel = {"world": self.world}
        if self.world > 1:
            self.parallel.update(wan_fast.init_parallel(True))
            self.parallel.update(wan_fast.install_parallel_vae(self.pipe.vae))
            # graph replay needs fixed input addresses, which only the symmetric-memory Ulysses path has
            if attention == "fa4sol" and self.parallel.get("P", 1) > 1:
                os.environ["WAN_FA4SOL_GRAPH"] = "1"
        self._calls, self._steps = 0, 1
        log(f"{self.kind} loaded in {time.perf_counter() - t0:.1f}s: expert_swap={self.expert_swap} precision={precision} attention={attention} "
            f"rho={rho} parallel={self.parallel}")

    @property
    def progress(self) -> int:
        return min(99, int(100 * self._calls / max(1, self._steps)))

    def _set_shift(self, shift: float) -> None:
        if shift != self._shift:
            from diffusers import UniPCMultistepScheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self._sched_cfg, flow_shift=shift)
            self._shift = shift

    def generate(self, req: dict) -> tuple:
        """Every rank calls this with the same request; rank 0 gets the frames."""
        torch = self.torch
        self._set_shift(req["flow_shift"])
        self._calls, self._steps = 0, req["num_inference_steps"]

        def cb(p, i, t, kw):
            self._calls = i + 1
            return kw
        kw = dict(prompt=req["prompt"], negative_prompt=req["negative_prompt"], height=req["height"],
                  width=req["width"], num_frames=req["num_frames"], num_inference_steps=req["num_inference_steps"],
                  guidance_scale=req["guidance_scale"], output_type="np", callback_on_step_end=cb,
                  generator=torch.Generator(device="cuda").manual_seed(req["seed"]))
        if self.kind == "a14b" and req.get("guidance_scale_2") is not None:
            kw["guidance_scale_2"] = req["guidance_scale_2"]
        if self.sol is not None:
            self.sol.sparse_calls = self.sol.dense_calls = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        if self.world > 1:
            self.dist.barrier(group=self.ctrl)
        t0 = time.perf_counter()
        frames = self.pipe(**kw).frames[0]
        torch.cuda.synchronize()
        stats = {"inference_time_s": round(time.perf_counter() - t0, 3),
                 "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        if self.sol is not None:
            stats.update(sparse_calls=self.sol.sparse_calls, dense_calls=self.sol.dense_calls)
        return frames, stats

    # rank 0: hand a request to every rank; other ranks: wait for one
    def broadcast(self, req):
        if self.world > 1:
            box = [req]
            self.dist.broadcast_object_list(box, src=0, group=self.ctrl)
            return box[0]
        return req

    def worker_loop(self) -> None:
        while True:
            req = self.broadcast(None)
            if req is None:
                return
            self.generate(req)


def write_mp4(frames, fps: int, path: str) -> float:
    from diffusers.utils import export_to_video
    t = time.perf_counter()
    export_to_video(list(frames), path, fps=fps)
    return round(time.perf_counter() - t, 3)


# -- request parsing ----------------------------------------------------------------------------

def resolve_request(body: dict, d: dict) -> dict:
    if body.get("input_reference") or body.get("image") or body.get("image_path") or body.get("reference_url"):
        raise ValueError("image-conditioned requests are not served (text-to-video only)")
    if int(body.get("n") or body.get("num_outputs_per_prompt") or 1) != 1:
        raise ValueError("only one output per request is served")
    width, height = body.get("width"), body.get("height")
    if not (width and height) and body.get("size"):
        width, height = (int(x) for x in str(body["size"]).lower().split("x"))
    width, height = int(width or d["width"]), int(height or d["height"])
    # VAE spatial stride x patch size: 8 x 2 for A14B, 16 x 2 for TI2V-5B
    if width % d["align"] or height % d["align"]:
        raise ValueError(f"width/height must be multiples of {d['align']}, got {width}x{height}")
    frames = int(body.get("num_frames") or d["num_frames"])
    if (frames - 1) % 4:
        raise ValueError(f"num_frames must be 4n+1, got {frames}")
    steps = int(body.get("num_inference_steps") or d["num_inference_steps"])
    if not 2 <= steps <= 100:
        raise ValueError(f"num_inference_steps must be within [2, 100], got {steps}")
    seed = body.get("seed")
    seed = int(seed[0] if isinstance(seed, list) else (seed if seed is not None else 1024))
    g2 = body.get("guidance_scale_2", d["guidance_scale_2"])
    return {"prompt": str(body["prompt"]), "negative_prompt": body.get("negative_prompt") or NEG,
            "width": width, "height": height, "num_frames": frames, "num_inference_steps": steps, "seed": seed,
            "guidance_scale": float(body.get("guidance_scale") or d["guidance_scale"]),
            "guidance_scale_2": float(g2) if g2 is not None else None,
            "flow_shift": float(body.get("flow_shift") or d["flow_shift"]),
            "fps": int(body.get("fps") or d["fps"])}


# -- HTTP (rank 0) ------------------------------------------------------------------------------

def build_app(engine: WanEngine, out_dir: str, d: dict, served_name: str):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="Wan2.2 GB200 (Sol-Engine)")
    jobs: dict = {}
    order: list = []
    executor = ThreadPoolExecutor(max_workers=1)        # one request at a time on the GPUs
    lock = threading.Lock()

    def view(job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "object", "model", "status", "created_at", "size", "seconds",
                                       "quality", "url", "urls", "completed_at", "error", "file_path",
                                       "file_paths", "num_outputs", "peak_memory_mb", "inference_time_s")}
        out["progress"] = 100 if job["status"] == "completed" else (
            engine.progress if job["status"] == "in_progress" else 0)
        return out

    def run(job_id: str) -> None:
        job = jobs[job_id]
        job["status"] = "in_progress"
        path = os.path.join(out_dir, f"{job_id}.mp4")
        try:
            with lock:
                req = engine.broadcast(job["_req"])
                frames, r = engine.generate(req)
            r["encode_s"] = write_mp4(frames, req["fps"], path)
            job.update(status="completed", completed_at=int(time.time()), file_path=path, file_paths=[path],
                       url=f"/v1/videos/{job_id}/content", urls=[f"/v1/videos/{job_id}/content"], num_outputs=1,
                       inference_time_s=r["inference_time_s"], peak_memory_mb=r["peak_memory_mb"])
            log(f"{job_id} done: {json.dumps(r)}")
        except Exception as e:  # noqa: BLE001
            job.update(status="failed", completed_at=int(time.time()),
                       error={"code": "generation_failed", "message": repr(e)[:2000]})
            log(f"{job_id} failed: {e!r}")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "sol-engine",
                                            "created": int(time.time())}]}

    @app.get("/model_info")
    async def model_info():
        return {"model_path": served_name, "is_generation": True, "task_types": ["t2v"], "model": engine.kind,
                "precision": engine.precision, "attention": engine.attention, "parallel": engine.parallel,
                "version": os.environ.get("SOL_ENGINE_VERSION"), "git_sha": os.environ.get("SOL_ENGINE_GIT_SHA")}

    @app.post("/v1/videos")
    async def create(request: Request):
        ctype = request.headers.get("content-type", "")
        if "multipart" in ctype or "form" in ctype:
            form = await request.form()
            body = {k: v for k, v in form.items()}
            if isinstance(body.get("extra_params"), str):
                body.update(json.loads(body.pop("extra_params")))
        else:
            body = await request.json()
        try:
            req = resolve_request(body, d)
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request"}})
        job_id = f"video_{uuid.uuid4().hex}"
        jobs[job_id] = {"id": job_id, "object": "video", "model": body.get("model") or served_name,
                        "status": "queued", "created_at": int(time.time()),
                        "size": f"{req['width']}x{req['height']}", "seconds": f"{req['num_frames'] / req['fps']:.3f}",
                        "quality": "standard", "_req": req}
        order.append(job_id)
        asyncio.get_running_loop().run_in_executor(executor, run, job_id)
        return view(jobs[job_id])

    @app.get("/v1/videos")
    async def list_jobs():
        return {"object": "list", "data": [view(jobs[j]) for j in order if j in jobs]}

    @app.get("/v1/videos/{video_id}")
    async def retrieve(video_id: str):
        if video_id not in jobs:
            raise HTTPException(status_code=404, detail="video not found")
        return view(jobs[video_id])

    @app.get("/v1/videos/{video_id}/content")
    async def content(video_id: str):
        job = jobs.get(video_id)
        if job is None:
            raise HTTPException(status_code=404, detail="video not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"video is {job['status']}")
        return FileResponse(job["file_path"], media_type="video/mp4", filename=os.path.basename(job["file_path"]))

    @app.delete("/v1/videos/{video_id}")
    async def delete(video_id: str):
        job = jobs.get(video_id)
        if job is None:
            raise HTTPException(status_code=404, detail="video not found")
        if job["status"] in ("queued", "in_progress"):
            raise HTTPException(status_code=409, detail="cannot delete a running or queued job")
        jobs.pop(video_id)
        if job.get("file_path") and os.path.exists(job["file_path"]):
            os.remove(job["file_path"])
        out = view(job)
        out["status"] = "deleted"
        return out

    return app


def relaunch_under_torchrun(a) -> None:
    """--num-gpus N > 1 outside torchrun: exec torchrun with N/nnodes local ranks running this file."""
    if a.num_gpus % a.nnodes:
        raise SystemExit(f"--num-gpus {a.num_gpus} must be divisible by --nnodes {a.nnodes}")
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={a.num_gpus // a.nnodes}"]
    if a.nnodes == 1:
        cmd += ["--standalone"]
    else:
        if not a.dist_init_addr:
            raise SystemExit("--dist-init-addr host:port is required with --nnodes > 1")
        host, port = a.dist_init_addr.rsplit(":", 1)
        cmd += [f"--nnodes={a.nnodes}", f"--node_rank={a.node_rank}", f"--master_addr={host}",
                f"--master_port={port}"]
    os.execv(sys.executable, cmd + [os.path.abspath(__file__)] + sys.argv[1:])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("WAN_MODEL_PATH", "/models/Wan2.2-T2V-A14B-Diffusers"),
                   help="diffusers checkpoint directory (A14B if it has transformer_2/, else TI2V-5B)")
    p.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", 1)))
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--dist-init-addr", default=None, help="host:port of node 0 (multi-node)")
    p.add_argument("--precision", choices=("bf16", "fp8"), default=os.environ.get("WAN_PRECISION", "bf16"))
    p.add_argument("--attention", choices=("fa4sol", "sol", "dense"),
                   default=os.environ.get("WAN_ATTENTION", "fa4sol"))
    p.add_argument("--rho", type=float, default=float(os.environ.get("WAN_FA4SOL_RHO", 0.35)),
                   help="fa4sol: fraction of (q256, kv128) blocks computed exactly")
    p.add_argument("--no-vae-compile", action="store_true")
    p.add_argument("--expert-swap", choices=("auto", "on", "off"), default=os.environ.get("WAN_EXPERT_SWAP", "auto"),
                   help="A14B: keep only the expert in use on the GPU (auto: when the GPU has < 120 GB)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 30000)))
    p.add_argument("--output-dir", default=os.environ.get("WAN_OUTPUT_DIR", "/outputs"))
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--warmup", type=int, default=2, help="standard-workload requests before /health answers "
                   "(2: the compiled VAE re-specializes once)")
    a = p.parse_args()
    if a.num_gpus > 1 and "LOCAL_RANK" not in os.environ:
        relaunch_under_torchrun(a)

    engine = WanEngine(a.model, a.precision, a.attention, a.rho, vae_compile=not a.no_vae_compile,
                       expert_swap=a.expert_swap)
    d = DEFAULTS[engine.kind]
    for i in range(a.warmup):                         # every rank runs the warmups in lockstep
        t = time.perf_counter()
        _, r = engine.generate(resolve_request({"prompt": "A calm lake at sunrise, gentle waves, birds flying."}, d))
        log(f"warmup {i}: {time.perf_counter() - t:.1f}s {json.dumps(r)}")
    if engine.rank != 0:
        engine.worker_loop()
        return 0

    os.makedirs(a.output_dir, exist_ok=True)
    import uvicorn
    app = build_app(engine, a.output_dir, d, a.served_model_name or d["name"])
    log(f"serving on {a.host}:{a.port}")
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        engine.broadcast(None)                       # release the worker ranks
    return 0


if __name__ == "__main__":
    sys.exit(main())
