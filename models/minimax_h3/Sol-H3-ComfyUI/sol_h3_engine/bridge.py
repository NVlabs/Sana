"""Persistent, serialized local inference with cancellable owned workers."""
import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent


class EngineProcess:
    def __init__(self, profile, output_root):
        self.profile = profile
        self.root = Path(output_root).resolve() / "sol_h3" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.sequence = 0
        self.log = (self.root / "worker.log").open("x")
        options = dict(profile, work_dir=str(self.root))
        (self.root / "options.json").write_text(json.dumps(options, indent=2))
        env = os.environ.copy()
        # ComfyUI enables cudaMallocAsync at startup. H3's compiled VAE uses
        # CUDA graph pool accounting, which requires the native allocator.
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env["PYTORCH_ALLOC_CONF"] = "backend:native"
        env.update(CUDA_VISIBLE_DEVICES=profile["gpu"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        self.command = [profile["python"], str(ROOT / "worker.py"), "--options",
                        str(self.root / "options.json")]
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                stdout=self.log, stderr=subprocess.STDOUT, text=True,
                cwd=ROOT, env=env, start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def generate(self, prompt, seed, decoder="original", cancel=None, mode="text", images=(),
                 progress=None, resolution="768p"):
        self.sequence += 1
        response = self.root / ("response-%05d.json" % self.sequence)
        # A worker MAY write stage/step updates here; absence is normal.
        progress_file = self.root / ("progress-%05d.json" % self.sequence)
        command = {"op": "generate", "prompt": prompt, "seed": seed,
                   "decoder": decoder, "response": str(response),
                   "case_id": "request-%05d" % self.sequence,
                   "mode": mode, "images": [str(path) for path in images], "resolution": resolution,
                   "progress": str(progress_file)}
        started = time.monotonic()
        reported = 0.0
        try:
            self.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            while not response.is_file():
                if cancel is not None:
                    cancel()
                now = time.monotonic()
                if progress is not None and now - reported >= 0.5:
                    reported = now
                    progress(self._progress(progress_file, now - started))
                if self.process.poll() is not None:
                    raise RuntimeError("Sol-H3 exited %s; see %s" %
                                       (self.process.returncode, self.root / "worker.log"))
                if time.monotonic() - started > self.profile.get("timeout_s", 3600):
                    raise TimeoutError("Sol-H3 request timed out; see %s" % (self.root / "worker.log"))
                time.sleep(0.1)
            result = json.loads(response.read_text())
            if result["status"] != "PASS":
                raise RuntimeError(result["error"] + "; see " + str(self.root / "worker.log"))
            output = Path(result["output"]).resolve(strict=True)
            output.relative_to(self.root)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("Sol-H3 returned no completed video")
            result["node_wall_s"] = time.monotonic() - started
            result["worker_log"] = str(self.root / "worker.log")
            (response.with_suffix(".node.json")).write_text(json.dumps(result, indent=2))
            return result
        except BaseException:
            self.close()
            raise

    def _progress(self, path, elapsed):
        update = {}
        try:
            update = json.loads(path.read_text())
        except (OSError, ValueError):
            pass  # Not written (current runtimes) or mid-write; elapsed still advances.
        if not isinstance(update, dict):
            update = {}
        update.setdefault("stage", "generating")
        update["elapsed_s"] = elapsed
        return update

    def close(self):
        child = self.process
        if child.poll() is None:
            # SIGTERM is handled by worker.py, which closes its own H3/Qwen workers.
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=50)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child.stdin is not None:
            child.stdin.close()
        self.log.close()


_lock = threading.RLock()
_engine = None
_key = None


def generate(profile, output_root, prompt, seed, decoder=None, cancel=None, mode="text",
             images=(), progress=None, resolution="768p"):
    global _engine, _key
    # One resident model session. Switching precision/decoder releases the old one.
    decoder = decoder or profile.get("decoder", "original")
    key = json.dumps([profile, str(output_root), decoder], sort_keys=True)
    with _lock:
        if _engine is not None and (_key != key or _engine.process.poll() is not None):
            _engine.close()
            _engine = None
        if _engine is None:
            _engine = EngineProcess(profile, output_root)
            _key = key
        try:
            return _engine.generate(prompt, seed, decoder, cancel, mode=mode, images=images,
                                    progress=progress, resolution=resolution)
        except BaseException:
            _engine = None
            _key = None
            raise


def is_warm(profile, output_root, decoder=None):
    """Whether the next request for this profile reuses the resident worker."""
    decoder = decoder or profile.get("decoder", "original")
    key = json.dumps([profile, str(output_root), decoder], sort_keys=True)
    with _lock:
        return _engine is not None and _key == key and _engine.process.poll() is None


def close():
    global _engine, _key
    with _lock:
        if _engine is not None:
            _engine.close()
        _engine = None
        _key = None


atexit.register(close)
