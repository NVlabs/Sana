"""torchrun entry point: rank-0 HTTP bridge, synchronized multi-GPU worker."""
import argparse
import io
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .config import ROOT, load_config, mount_vendor
from .server import Bridge, serve


def run_h3(settings, assets, bridge):
    mount_vendor()
    import numpy as np
    from PIL import Image
    import torch
    import torch.distributed as dist
    from dev.yanzuolu.common.config import CfgNode
    from dev.yanzuolu.common.distributed.init import configure_distributed
    from dev.yanzuolu.common.distributed.unified_parallel import init_unified_parallel, use_unified_parallel
    from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
    from dev.yanzuolu.common.model import build_models

    cfg = CfgNode(settings["inference"])
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (4, 8):
        raise ValueError("Release runtime requires 4 or 8 GPUs; use 8 for the production profile")
    cfg.distributed.up_size = [1, world]
    # Qwen checkpoint shards are a 4-rank layout; the wrapper handles 8 via replication.
    configure_distributed(cfg)
    from render.pipeline import StreamingPipeline
    init_unified_parallel((1, world), 1)
    rank = dist.get_rank()
    models = build_models(cfg.models)
    if len(models["backbone"].dit.blocks) != 50:
        raise RuntimeError("The H3 backbone must execute all 50 layers")
    pipe = StreamingPipeline(cfg, models, settings)
    outputs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgb-publication")

    def publish(session, tensor, metrics):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        pixels = tensor[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        images = []
        for frame in pixels:
            out = io.BytesIO()
            Image.fromarray(frame).save(out, format="JPEG", quality=85)
            images.append(out.getvalue())
        while not session.closed:
            try:
                session.outputs.put((images, metrics), timeout=.5)
                break
            except queue.Full:
                continue

    def broadcast(value):
        data = [value]
        dist.broadcast_object_list(data, src=0)
        return data[0]

    # Warm identical bootstrap/continuation shapes, then discard all stream state.
    with execution_phase(ExecutionPhase.VALIDATION), use_unified_parallel(world):
        pipe.start(prompt=assets["prompt"], picture=assets["reference_image"], seed=cfg.seed)
        for _ in range(settings["warmup_rounds"]):
            _, frames = pipe.next_counts()
            pixels = torch.zeros((3, frames, settings["reference"]["height"],
                                  settings["reference"]["width"]), device=pipe.device)
            pipe.render(pixels)
        dist.barrier()
        if rank == 0:
            bridge.worker_ready = True
            print("STREAMRENDER_READY", json.dumps({"gpus": world, "layers": 50, "nfe": 2}), flush=True)
        while True:
            session = bridge.starts.get() if rank == 0 else None
            header = broadcast(({"stop": True} if session is None else
                                {"id": session.id, "seed": cfg.seed}) if rank == 0 else None)
            if header.get("stop"):
                outputs.shutdown(wait=True)
                return
            pipe.start(prompt=assets["prompt"], picture=assets["reference_image"], seed=header["seed"])
            if rank == 0:
                session.ready = True
            pending_output = None
            while True:
                count, frames = pipe.next_counts()
                records = session.collect(frames, settings["session_idle_seconds"]) if rank == 0 else None
                packet = broadcast(None if rank == 0 and records is None else
                                   [(item[0], item[1]) for item in records] if rank == 0 else None)
                if packet is None:
                    break
                arrays = [np.array(Image.open(io.BytesIO(png)).convert("RGB")) for _, png in packet]
                pixels = torch.from_numpy(np.stack(arrays)).to(pipe.device).permute(3, 0, 1, 2).float().div_(255)
                # Semantic palette labels must not be bilinearly blended.
                pixels = torch.nn.functional.interpolate(
                    pixels.permute(1, 0, 2, 3),
                    size=(settings["reference"]["height"], settings["reference"]["width"]),
                    mode="nearest").permute(1, 0, 2, 3).contiguous()
                rgb, metrics = pipe.render(pixels)
                if rank == 0:
                    metrics["session"] = session.id
                    metrics["reference_first_frame"] = records[0][0]
                    metrics["reference_last_frame"] = records[-1][0]
                    metrics["oldest_input_to_rgb_ready_ms"] = (time.monotonic() - records[0][3]) * 1000
                    session.metrics = metrics
                    output_dir = Path(settings.get("output_dir", str(ROOT / "outputs"))) / session.id
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with (output_dir / "controls.jsonl").open("a") as transcript:
                        for frame_index, _, controls, _ in records:
                            transcript.write(json.dumps({"session": session.id,
                                "frame": frame_index, "timestamp": frame_index / 24,
                                "chunk": metrics["round"], "controls": controls}) + "\n")
                    with (output_dir / "timings.jsonl").open("a") as timings:
                        timings.write(json.dumps(metrics) + "\n")
                    print("STREAMRENDER_CHUNK", json.dumps(metrics), flush=True)
                    if pending_output is not None:
                        pending_output.result()
                    if rgb is not None:
                        pending_output = outputs.submit(publish, session, rgb, metrics)
            if pending_output is not None:
                pending_output.result()
            # Reset all encoder/decoder/KV state on the next start().


def run_passthrough(settings, bridge):
    """Explicit CPU transport diagnostic; never reported as H3 inference."""
    from PIL import Image
    bridge.worker_ready = True
    while True:
        session = bridge.starts.get()
        if session is None:
            return
        session.ready = True
        index = 0
        while not session.closed:
            values = session.collect(8, settings["session_idle_seconds"])
            if values is None:
                break
            images = []
            for _, png, _, _ in values:
                out = io.BytesIO()
                Image.open(io.BytesIO(png)).convert("RGB").save(out, "JPEG")
                images.append(out.getvalue())
            session.metrics = {"round": index, "backend": "passthrough", "output_frames": len(images)}
            while not session.closed:
                try:
                    session.outputs.put((images, session.metrics), timeout=.5)
                    break
                except queue.Full:
                    pass
            index += 1


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/runtime.json"))
    parser.add_argument("--assets", default=str(ROOT / "configs/assets.local.json"))
    parser.add_argument("--backend", choices=["h3", "passthrough"])
    parser.add_argument("--port", type=int, help="Override loopback HTTP port")
    parser.add_argument("--validate-browser", action="store_true",
                        help="Run the actual engine loop, then stop this validation worker")
    args = parser.parse_args()
    settings, assets = load_config(args.config, args.assets)
    if args.port:
        settings["port"] = args.port
    os.environ["STREAMRENDER_SERVER"] = f"http://{settings['host']}:{settings['port']}"
    if args.backend:
        settings["backend"] = args.backend
    rank = int(os.environ.get("RANK", "0"))
    bridge = Bridge(settings) if rank == 0 else None
    httpd = serve(bridge) if rank == 0 else None
    validation_errors = []
    validation_thread = None
    if args.validate_browser and rank == 0:
        def validate():
            import subprocess
            try:
                subprocess.run([os.environ.get("NODE_BIN", "node"),
                                str(ROOT / "benchmarks/browser_loop.mjs")], check=True)
            except BaseException as error:
                validation_errors.append(error)
            finally:
                for session in bridge.sessions.values():
                    session.closed = True
                # A terminated validation producer releases the collective loop.
                bridge.starts.put(None)
        validation_thread = threading.Thread(target=validate, daemon=True)
        validation_thread.start()
    try:
        if settings["backend"] == "h3":
            run_h3(settings, assets, bridge)
            # Unified-parallel context exit performs a barrier, so tear down
            # the default group only after run_h3() has left that context.
            from dev.yanzuolu.common.distributed.init import destroy_distributed
            destroy_distributed()
        else:
            if rank != 0:
                raise ValueError("passthrough diagnostic is a single CPU process")
            run_passthrough(settings, bridge)
        if validation_thread:
            validation_thread.join()
        if validation_errors:
            raise RuntimeError(f"Browser validation failed: {validation_errors[0]}")
    except BaseException as error:
        if bridge:
            bridge.fail(error)
        raise
    finally:
        if httpd:
            httpd.shutdown()


if __name__ == "__main__":
    main()
