"""Save one request's NVFP4 Qwen conditioning payload, the input build_adaln_table.py needs.

The AdaLN table depends only on each task's (timestep, endpoint) schedule, so any prompt works:
one text payload (t2va) and one first-frame payload (fl2va). Runs the worker in its
conditioning-only mode (Qwen only, no DiT load) with a profile from SOL_ENGINE_CONFIG.
Usage: SOL_ENGINE_CONFIG=config.json save_payload.py PROFILE OUT.pt "PROMPT" [FIRST_FRAME.png]
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sol_h3_engine import bridge  # noqa: E402
from sol_h3_engine.config import load_profile  # noqa: E402


def main():
    profile, out, prompt = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    image = sys.argv[4] if len(sys.argv) > 4 else None
    settings = load_profile(profile)
    root = out.resolve().parent / "payload-worker"
    engine = bridge.EngineProcess(settings, root)
    try:
        engine.sequence += 1
        response = engine.root / "response.json"
        command = {"op": "generate", "op_detail": "conditioning_only", "prompt": prompt, "seed": 42,
                   "case_id": "payload", "response": str(response),
                   "mode": "first_frame" if image else "text", "images": [str(Path(image).resolve())] if image else []}
        engine.process.stdin.write(json.dumps(command) + "\n")
        engine.process.stdin.flush()
        import time
        while not response.is_file():
            if engine.process.poll() is not None:
                raise RuntimeError("worker exited; see %s" % (engine.root / "worker.log"))
            time.sleep(0.2)
        result = json.loads(response.read_text())
        if result["status"] != "PASS":
            raise RuntimeError(result.get("error"))
        shutil.copyfile(result["payload"], out)
        print("saved", out, result["task"])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
