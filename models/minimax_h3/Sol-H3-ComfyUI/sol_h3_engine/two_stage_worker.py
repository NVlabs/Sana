"""Sol-H3 two stage (H3 draft → learned upscale → LTX refine) behind the ComfyUI worker protocol.

The pipeline is NVlabs/Sana sol-engine `models/minimax_h3/Sol-H3-RTX5090` (DGX Spark: `Sol-H3-Spark`, the same
runtime without CPU offload), used unchanged: its own Qwen / Stage1 / Stage2 processes, recipe and request → MP4 clock.
This module only maps ComfyUI requests onto `runtime.pipeline.Pipeline` and reports its stages as progress.
Profile options: {"two_stage": {"package": <Sol-H3-RTX5090 or Sol-H3-Spark directory>, "offload": "cpu"|"none",
                  "resident_qwen": bool}}.
resident_qwen (RTX 5090): the package loads Qwen for every request (one_request lifetime), rereading its 15.7 GB
checkpoint from disk — on a host whose RAM cap leaves no page cache for it that costs ~56 s per request. With this
option one Qwen session stays resident on the card (the package's own resident Session, as on DGX Spark) while the
video stages keep their CPU offload; the request sequence and recipe are otherwise the package's own.
"""
import inspect
import json
import signal
import sys
import time
from pathlib import Path

# Worker name → the Studio stage it covers.
STAGES = {"qwen": "conditioning", "stage1": "denoising", "stage2": "refining"}


def serve(options, write_json):
    settings = options["two_stage"]
    package = Path(settings["package"]).resolve(strict=True)
    sys.path.insert(0, str(package))
    from runtime.config import load_paths, normalize_case
    from runtime import pipeline as runtime_pipeline
    paths = load_paths(options["paths"])
    root = Path(options["work_dir"])
    progress_path = [None]

    def progress(stage, step=None, steps=None):
        if progress_path[0]:
            write_json(progress_path[0], {"stage": stage, "step": step, "steps": steps, "time": time.time()})

    class ReportingWorker(runtime_pipeline.Worker):
        """The package's own worker; announces each stage before its call."""

        def __init__(self, name, *args, **kwargs):
            self.stage = STAGES.get(name)
            super().__init__(name, *args, **kwargs)

        def submit(self, op="run", **kwargs):
            if op == "run" and self.stage:
                progress(self.stage)
            return super().submit(op, **kwargs)

    class ResidentQwenPipeline(runtime_pipeline.Pipeline):
        """CPU-offloaded video stages with one resident Qwen session (loaded once, after the warmup)."""

        def start(self, case):
            super().start(case)
            self.resident = self.worker_factory("qwen", self.root / "qwen-resident", self.paths, self.config)

        def generate(self, case):
            # The package's cpu-offload request path with the per-request Qwen worker replaced by the resident one.
            original = self.worker_factory
            resident = self.resident

            class Borrowed:
                def __init__(self, *_args, **_kwargs):
                    pass

                def call(self, *args, **kwargs):
                    return resident.call(*args, **kwargs)

                def close(self):
                    pass  # the session stays loaded

            self.worker_factory = lambda name, *args, **kwargs: Borrowed() if name == "qwen" else original(name, *args, **kwargs)
            try:
                return super().generate(case)
            finally:
                self.worker_factory = original

        def close(self):
            try:
                if getattr(self, "resident", None) is not None:
                    self.resident.close()
            finally:
                super().close()

    pipeline = None
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        if not stopped:
            stopped = True
            raise KeyboardInterrupt("Closing this ComfyUI worker")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "close":
                break
            if request["op"] != "generate":
                raise ValueError("Unsupported operation")
            progress_path[0] = request.get("progress")
            try:
                if request.get("mode", "text") != "text":
                    raise ValueError("Two stage runs text-to-video on this device")
                if request.get("resolution", "768p") != "768p":
                    raise ValueError("Two stage renders 768p only")
                case = normalize_case({"case_id": request["case_id"], "prompt": request["prompt"],
                                       "seed": request["seed"]})
                cold = pipeline is None
                if cold:
                    progress("loading")
                    factory = ResidentQwenPipeline if settings.get("resident_qwen") else runtime_pipeline.Pipeline
                    kwargs = {"worker_factory": ReportingWorker}
                    if "offload_mode" in inspect.signature(runtime_pipeline.Pipeline).parameters:  # not on Spark
                        kwargs["offload_mode"] = settings.get("offload", "cpu")
                    pipeline = factory(paths, root / "pipeline", **kwargs)
                    pipeline.start(case)  # loads every stage and runs one full warmup request (seed 999)
                row = pipeline.generate(case)
                progress("saving")
                write_json(request["response"], {
                    **row, "profile": options["name"], "precision": "two_stage", "decoder": "ltx",
                    "single_stage": False, "cold_start": cold, "prompt": case["prompt"],
                    "startup_and_warmup_s": pipeline.report["startup_and_warmup_s"] if cold else 0,
                    "h3_phases_s": {"qwen_s": row.get("qwen_s"), "stage1_s": row.get("stage1_s"),
                                    "stage2_s": row.get("stage2_s"), **(row.get("stage2_phases_s") or {})},
                    "timing": "request through completed MP4 (the package's own clock); startup/warmup separate",
                    "report": str(pipeline.root / "results.json")})
            except Exception as error:
                write_json(request["response"], {"status": "FAIL", "error": "%s: %s" % (type(error).__name__, error)})
                raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if pipeline is not None:
            pipeline.close()
