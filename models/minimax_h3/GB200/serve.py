#!/usr/bin/env python3
"""MiniMax-H3 single-GB200 HTTP server, SGLang-compatible video API.

    python serve.py --model /models/MiniMax-H3-diffusers --precision fp8 --attention sol --port 30000

Endpoints (same paths and response schema as `sglang serve --model-type diffusion`):

    POST   /v1/videos                 create a job; body = SGLang VideoGenerationsRequest subset
    GET    /v1/videos                 list jobs
    GET    /v1/videos/{id}            job status (status/progress/inference_time_s/peak_memory_mb/...)
    GET    /v1/videos/{id}/content    the mp4 (H.264 + AAC stereo 32 kHz)
    DELETE /v1/videos/{id}
    GET    /health  /v1/models  /model_info

Request size/length, in the order they are honoured:
    width/height | size "WxH" | target.short_edge + target.aspect_ratio (H3 extension)   -> 1344x768
    num_frames   | target.duration_seconds | seconds -> frames = align_17n+5(round(s * 24))  -> 124

The engine is the measured single-GPU stack: AdaLN table, Triton fusions, Sol-Attn (or dense
cuDNN), the owned block loop of `fast_forward` (v2 kernels), optional MXFP8 linears, compiled VAE
decode. One request runs at a time on the GPU; the rest queue in arrival order.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FPS = 24
MODEL_ID = "MiniMaxAI/MiniMax-H3"


def log(msg: str) -> None:
    print(f"[h3-serve] {msg}", flush=True)


def relocate_checkpoint(model: str, scratch: str) -> str:
    """The converter writes absolute paths into `modular_model_index.json`. When the checkpoint is
    mounted somewhere else (a container), point every component at the mounted directory through a
    shadow directory of symlinks plus a rewritten index; the weights are never copied."""
    import hashlib

    model = os.path.abspath(model)
    with open(os.path.join(model, "modular_model_index.json")) as f:
        index = json.load(f)
    # diffusers only warns when a component fails to load and the request dies much later, so check
    # here. The converter symlinks text_encoder/processor/tokenizer into the original FL2VA folder;
    # in a container that folder must be mounted at the path the links name.
    broken = [f"{name} -> {os.readlink(os.path.join(model, name))}" for name in sorted(os.listdir(model))
              if os.path.islink(os.path.join(model, name)) and not os.path.exists(os.path.join(model, name))]
    if broken:
        raise SystemExit("checkpoint components point at paths that are not mounted: " + "; ".join(broken))
    changed = False
    for value in index.values():
        if isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict):
            path = value[2].get("pretrained_model_name_or_path")
            if path and os.path.isabs(path) and os.path.realpath(path) != os.path.realpath(model):
                value[2]["pretrained_model_name_or_path"] = model
                changed = True
    if not changed:
        return model
    shadow = os.path.join(scratch, "h3_ckpt_" + hashlib.sha1(model.encode()).hexdigest()[:10])
    os.makedirs(shadow, exist_ok=True)
    for name in os.listdir(model):
        dst = os.path.join(shadow, name)
        if name != "modular_model_index.json" and not os.path.lexists(dst):
            os.symlink(os.path.join(model, name), dst)
    with open(os.path.join(shadow, "modular_model_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    log(f"checkpoint index pointed at {model} via {shadow}")
    return shadow


def align_frame_count(frames: int) -> int:
    """MiniMax-H3 frame counts are 17n+5 (SGLang `minimax_h3_align_frame_count`)."""
    if frames <= 0:
        return 1
    return frames + (5 - frames) % 17


# -- engine -------------------------------------------------------------------------------------

class H3Engine:
    def __init__(self, model_path: str, precision: str, attention: str, fp8_blocks: str,
                 vae_compile: bool) -> None:
        import torch
        import fast_forward
        import gpu_infer
        from gpu_infer_cp import build_pipeline

        self.torch = torch
        self.precision, self.attention = precision, attention
        t0 = time.perf_counter()
        args = SimpleNamespace(model_path=model_path, attention_backend=None)
        self.pipe, _ = build_pipeline(args, "cuda:0")
        log(f"pipeline loaded in {time.perf_counter() - t0:.1f}s")
        config = dict(kernel=True, sol_attn=attention == "sol", sol_tau=1.0, sol_thresh_type="diag",
                      sol_dense_steps=10, sol_dense_layers=2, sol_sink_mode="prefix", cache=None,
                      shard_vae=vae_compile, vae_batched=True, vae_compile="default" if vae_compile else None)
        _, self.sparse = gpu_infer.install_stack(self.pipe, config)
        tr = self.pipe.transformer
        if self.sparse is not None:
            tr._h3_attn_core = self.sparse
        else:
            from sol_attn_h3 import _dense
            tr._h3_attn_core = _dense
        n = len(tr.transformer_blocks)
        blocks = () if precision == "bf16" else _parse_blocks(fp8_blocks, n)
        self.applied = fast_forward.install(tr, blocks, kernels="v2")
        self._calls = 0
        self._steps = 1
        original = tr.forward

        def counted(*a, **kw):
            self._calls += 1
            return original(*a, **kw)
        tr.forward = counted
        log(f"stack installed: precision={precision} attention={attention} fp8_blocks={len(blocks)}")

    @property
    def progress(self) -> int:
        return min(99, int(100 * self._calls / max(1, self._steps)))

    def generate(self, prompt: str, width: int, height: int, num_frames: int, steps: int, seed: int,
                 out_path: str) -> dict:
        torch = self.torch
        # Sol-Attn closes the previous request itself when the timestep jumps back; closing it here
        # as well would hand the strict guard an empty request.
        self._calls, self._steps = 0, max(1, steps - 1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = self.pipe(prompt=prompt, height=height, width=width, num_frames=num_frames,
                          num_inference_steps=steps, generator=torch.Generator().manual_seed(seed))
        torch.cuda.synchronize()
        inference_s = time.perf_counter() - t0
        videos = getattr(state, "videos", None)
        if videos is None:
            videos = getattr(state, "frames")
        audio = getattr(state, "audio", None)
        rate = int(getattr(state, "sampling_rate", 32000) or 32000)
        t1 = time.perf_counter()
        _write_mp4(videos[0], audio, rate, out_path)
        return {"inference_time_s": round(inference_s, 3), "encode_s": round(time.perf_counter() - t1, 3),
                "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def _parse_blocks(spec: str, n: int):
    if spec == "all":
        return range(n)
    if "-" in spec:
        a, b = spec.split("-")
        return range(int(a), int(b) + 1)
    return [int(x) for x in spec.split(",")]


def _write_mp4(frames, audio, rate: int, path: str) -> None:
    """H.264 video + AAC stereo in one ffmpeg call, frames piped as raw RGB."""
    import numpy as np
    import imageio_ffmpeg
    from scipy.io import wavfile

    arr = np.stack([np.asarray(f) for f in frames]) if isinstance(frames, (list, tuple)) else np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-"]
        if audio is not None:
            wav = os.path.join(tmp, "a.wav")
            a = audio.detach().float().cpu().numpy()
            a = a.reshape(-1, a.shape[-1]).T            # (samples, channels)
            wavfile.write(wav, rate, np.clip(a, -1, 1).astype(np.float32))
            cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", path]
        subprocess.run(cmd, input=arr.tobytes(), check=True)


# -- request parsing ----------------------------------------------------------------------------

def resolve_shape(ratio: float, short_edge: int = 768) -> tuple[int, int]:
    """SGLang's `minimax_h3_resolve_spatial_shape`: short edge x ratio, capped at 768*1344 pixels,
    each side rounded to the nearest multiple of 32. 16:9 at 768 -> 1344x768."""
    import math

    if not 0.25 <= ratio <= 4.0:
        raise ValueError(f"aspect ratio must be within 1:4 to 4:1, got {ratio:g}")
    w, h = (short_edge * ratio, float(short_edge)) if ratio >= 1.0 else (float(short_edge), short_edge / ratio)
    cap = 768 * 1344
    if w * h > cap:
        s = math.sqrt(cap / (w * h))
        w, h = w * s, h * s
    return max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)


def resolve_request(body: dict, defaults: SimpleNamespace) -> dict:
    task = body.get("task") or body.get("task_type") or "t2va"
    if task != "t2va":
        raise ValueError(f"only task 't2va' is served, got {task!r}")
    if body.get("conditions") or body.get("input_reference") or body.get("reference_url") \
            or body.get("video_path") or body.get("video_url"):
        raise ValueError("conditioned tasks (fl2va/ref2va inputs) are not served")
    if int(body.get("n") or body.get("num_outputs_per_prompt") or 1) != 1:
        raise ValueError("only one output per request is served")
    target = body.get("target") or {}
    width, height = body.get("width"), body.get("height")
    if not (width and height) and body.get("size"):
        width, height = (int(x) for x in str(body["size"]).lower().split("x"))
    if not (width and height):
        rw, rh = (float(x) for x in str(target.get("aspect_ratio", "16:9")).split(":"))
        width, height = resolve_shape(rw / rh, int(target.get("short_edge", 768)))
    width, height = int(width), int(height)
    if width % 32 or height % 32:
        raise ValueError(f"width/height must be multiples of 32, got {width}x{height}")
    if body.get("num_frames"):
        frames = align_frame_count(int(body["num_frames"]))
    else:
        seconds = float(target.get("duration_seconds") or body.get("seconds") or defaults.seconds)
        if not 4.0 <= seconds <= 15.0:
            raise ValueError(f"duration must be within [4, 15] s, got {seconds}")
        frames = align_frame_count(int(round(seconds * FPS)))
    seed = body.get("seed")
    seed = int(seed[0] if isinstance(seed, list) else (seed if seed is not None else defaults.seed))
    steps = int(body.get("num_inference_steps") or defaults.steps)
    if not 2 <= steps <= 100:
        raise ValueError(f"num_inference_steps must be within [2, 100], got {steps}")
    # The released checkpoint fixes these (guidance-distilled, flow shifts 12 / 3); report any request
    # that asks for something else rather than silently pretending to honour it.
    fixed = {"flow_shift": 12.0, "audio_flow_shift": 3.0, "guidance_scale": None}
    ignored = {k: body[k] for k, v in fixed.items() if body.get(k) is not None and float(body[k]) != v}
    return {"prompt": str(body["prompt"]), "width": width, "height": height, "num_frames": frames,
            "steps": steps, "seed": seed, "seconds": frames / FPS, "ignored": ignored}


# -- HTTP ---------------------------------------------------------------------------------------

def build_app(engine: H3Engine, out_dir: str, defaults: SimpleNamespace, served_name: str):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="MiniMax-H3 GB200 (Sol-Engine)")
    jobs: dict[str, dict] = {}
    order: list[str] = []
    executor = ThreadPoolExecutor(max_workers=1)     # one request at a time on the GPU
    lock = threading.Lock()

    def view(job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "object", "model", "status", "created_at", "size", "seconds",
                                       "quality", "url", "urls", "remixed_from_video_id", "completed_at",
                                       "expires_at", "error", "file_path", "file_paths", "num_outputs",
                                       "peak_memory_mb", "inference_time_s", "revised_prompt", "action")}
        out["progress"] = 100 if job["status"] == "completed" else (
            engine.progress if job["status"] == "in_progress" else 0)
        return out

    def run(job_id: str) -> None:
        job = jobs[job_id]
        job["status"] = "in_progress"
        path = os.path.join(out_dir, f"{job_id}.mp4")
        try:
            with lock:
                r = engine.generate(out_path=path, **{k: job["_req"][k] for k in
                                                     ("prompt", "width", "height", "num_frames", "steps", "seed")})
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
        return {"model_path": served_name, "is_generation": True, "task_types": ["t2va"],
                "precision": engine.precision, "attention": engine.attention, "applied": engine.applied}

    @app.post("/v1/videos")
    async def create(request: Request):
        ctype = request.headers.get("content-type", "")
        if "multipart" in ctype or "form" in ctype:
            form = await request.form()
            body = {k: v for k, v in form.items()}
            for k in ("extra_params", "target"):
                if isinstance(body.get(k), str):
                    body.update(json.loads(body.pop(k))) if k == "extra_params" else body.update(
                        {k: json.loads(body[k])})
        else:
            body = await request.json()
        try:
            req = resolve_request(body, defaults)
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request"}})
        job_id = f"video_{uuid.uuid4().hex}"
        jobs[job_id] = {"id": job_id, "object": "video", "model": body.get("model") or served_name,
                        "status": "queued", "created_at": int(time.time()),
                        "size": f"{req['width']}x{req['height']}", "seconds": f"{req['seconds']:.3f}",
                        "quality": "standard", "_req": req}
        order.append(job_id)
        if req["ignored"]:
            log(f"{job_id}: fields served at checkpoint defaults, not as requested: {req['ignored']}")
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("H3_MODEL_PATH", "/models/MiniMax-H3-diffusers"),
                   help="converted diffusers checkpoint directory")
    p.add_argument("--precision", choices=("bf16", "fp8"), default=os.environ.get("H3_PRECISION", "fp8"))
    p.add_argument("--attention", choices=("sol", "dense"), default=os.environ.get("H3_ATTENTION", "sol"))
    p.add_argument("--fp8-blocks", default=os.environ.get("H3_FP8_BLOCKS", "all"))
    p.add_argument("--no-vae-compile", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 30000)))
    p.add_argument("--output-dir", default=os.environ.get("H3_OUTPUT_DIR", "/outputs"))
    p.add_argument("--served-model-name", default=MODEL_ID)
    p.add_argument("--warmup", type=int, default=1, help="warmup requests before /health answers")
    p.add_argument("--default-seconds", type=float, default=5.0)
    p.add_argument("--default-steps", type=int, default=50)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    defaults = SimpleNamespace(seconds=a.default_seconds, steps=a.default_steps, seed=0)

    model = relocate_checkpoint(a.model, os.environ.get("H3_SCRATCH", os.path.join(a.output_dir, ".scratch")))
    engine = H3Engine(model, a.precision, a.attention, a.fp8_blocks, vae_compile=not a.no_vae_compile)
    for i in range(a.warmup):
        t = time.perf_counter()
        r = engine.generate("A calm lake at sunrise, gentle waves, birds singing.", 1344, 768,
                            align_frame_count(round(a.default_seconds * FPS)), a.default_steps, 0,
                            os.path.join(a.output_dir, f"_warmup_{i}.mp4"))
        log(f"warmup {i}: {time.perf_counter() - t:.1f}s {json.dumps(r)}")

    import uvicorn
    app = build_app(engine, a.output_dir, defaults, a.served_model_name)
    log(f"serving on {a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
