"""Run only the single-stage H3 pipeline in its existing Python environment."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=Path, required=True)
    options = json.loads(parser.parse_args().options.read_text())
    if options["precision"] == "hyperflow":
        from hyperflow_worker import serve
        return serve(options, write_json)
    if options["precision"] == "two_stage":
        from two_stage_worker import serve
        return serve(options, write_json)
    runtime = Path(__file__).parent / "runtimes" / ("h3_" + options["precision"])
    sys.path.insert(0, str(runtime))
    from runtime.config import load_paths, normalize_case, load_recipe
    from runtime.pipeline import Pipeline
    recipe = load_recipe()
    if "stage2" in recipe or recipe["stage1"]["width"] != 1344:
        raise RuntimeError("This integration requires single-stage direct 768p H3")
    paths = load_paths(options["paths"])
    decoder = options.get("decoder", "original")
    os.environ.pop("SOL_H3_FAST_VAE", None)
    if decoder == "fast_noncausal":
        fast = dict(options["fast_vae"], vae_config=str(Path(paths["h3_model"]) / "vae/config.json"))
        os.environ["SOL_H3_FAST_VAE"] = json.dumps(fast)
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
            try:
                if request.get("decoder", "original") != decoder:
                    raise ValueError("Decoder differs from this worker's loaded profile")
                case = normalize_case({"case_id": request["case_id"],
                                       "prompt": request["prompt"], "seed": request["seed"]})
                cold = pipeline is None
                if cold:
                    pipeline = Pipeline(paths, Path(options["work_dir"]) / "pipeline",
                                        offload_mode="cpu")
                    if decoder == "fast_noncausal":
                        pipeline.config["decoder"] = {"name": decoder, "weights": fast["weights"],
                            "input": "normalized BTCHW", "rgb": "ImageNet revert_tensor",
                            "cpu_offload": True, "cuda_graph": False,
                            "experimental": True}
                    pipeline.start(case)
                row = pipeline.generate(case)
                pipeline.finish()
                write_json(request["response"], {
                    **row, "profile": options["name"], "precision": options["precision"],
                    "decoder": decoder, "single_stage": True,
                    "startup_and_warmup_s": pipeline.report["startup_and_warmup_s"] if cold else 0,
                    "cold_start": cold, "prompt": case["prompt"],
                    "timing": "request through completed MP4; startup/warmup reported separately",
                    "report": str(pipeline.root / "results.json"),
                })
            except Exception as error:
                write_json(request["response"], {"status": "FAIL", "error": "%s: %s" %
                                                 (type(error).__name__, error)})
                raise
    finally:
        # A failed response may make the caller send SIGTERM while cleanup is
        # already running. Do not interrupt nested workers' graceful close.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    main()
