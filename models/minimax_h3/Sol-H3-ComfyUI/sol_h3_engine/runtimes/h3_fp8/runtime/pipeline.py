"""Persistent single-Spark pipeline with a continuous request-to-MP4 clock."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .config import PACKAGE, load_recipe
from .worker import write_json


def await_json(path, process, timeout=1800):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"Worker exited with code {process.returncode}; see {path.parent / 'worker.log'}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"No worker response: {path}")
        time.sleep(0.01)
    receipt = json.loads(path.read_text())
    if receipt.get("status") not in ("PASS", "READY"):
        raise RuntimeError(receipt.get("error", "Worker failed"))
    return receipt


def worker_environment(name, paths):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    # Downloads are a separate preparation step. No token is needed by inference.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(key, None)
    roots = [str(PACKAGE), str(PACKAGE.parents[1])]
    if name == "stage1" and os.environ.get("SOL_H3_FAST_VAE"):
        fast = json.loads(os.environ["SOL_H3_FAST_VAE"])
        roots.insert(0, fast["diffusers_source"])
    if paths.get(name + "_dependencies"):
        roots.append(paths[name + "_dependencies"])
    if name == "stage1":
        if paths.get("fa4_dependencies"):
            roots.extend([paths["fa4_dependencies"],
                          str(Path(paths["fa4_dependencies"]) / "nvidia_cutlass_dsl/python_packages")])
        roots.extend([paths["fa4_root"], str(Path(paths["fa4_root"]) / "nvidia_cutlass_dsl/python_packages"),
                      paths["fastvideo_root"], str(Path(paths["fastvideo_root"]) / "fastvideo-kernel/python")])
        roots.append(str(Path(paths["ltx_root"]) / "packages/ltx-core/src"))
        env["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
        env.pop("LD_LIBRARY_PATH", None)
    elif name == "stage2":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


class Worker:
    """One owned process group, one model session and a small command channel."""

    def __init__(self, name, root, paths, config):
        self.root = Path(root)
        self.root.mkdir()
        self.sequence = 0
        self.process = None
        self.log = (self.root / "worker.log").open("x")
        options = {"paths": paths, "work_dir": str(self.root / "model"), "config": config}
        write_json(self.root / "options.json", options)
        command = [paths[name + "_python"]]
        if name == "stage2":
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "--module"]
        else:
            command += ["-m"]
        command += ["runtime.worker", "--module", name, "--options", str(self.root / "options.json"),
                    "--ready", str(self.root / "ready.json")]
        try:
            self.process = subprocess.Popen(command, cwd=PACKAGE, env=worker_environment(name, paths),
                                            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                                            text=True, start_new_session=True)
            self.ready = await_json(self.root / "ready.json", self.process)
        except BaseException:
            self.close()
            raise

    def submit(self, op="run", **kwargs):
        if op not in ("run", "prepare", "release_idle_cache", "offload", "onload"):
            raise ValueError("Unsupported session operation")
        self.sequence += 1
        response = self.root / f"response-{self.sequence:05d}.json"
        self.process.stdin.write(json.dumps({"op": op, "response": str(response), "kwargs": kwargs}) + "\n")
        self.process.stdin.flush()
        return response

    def wait(self, response):
        return await_json(response, self.process)["result"]

    def call(self, op="run", **kwargs):
        return self.wait(self.submit(op, **kwargs))

    def close(self):
        child = self.process
        if child is not None and child.poll() is None:
            try:
                child.stdin.write('{"op":"close"}\n')
                child.stdin.flush()
                child.wait(timeout=30)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                # This process group was created by this Worker, never a shared service.
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        if child is not None and child.stdin is not None:
            child.stdin.close()
        self.log.close()


class Pipeline:
    """Direct H3 four-update768p trial with the existing worker/Qwen lifecycle."""

    def __init__(self, paths, output_root, *, task="t2va", ref_image_match=None,
                 ref_stage1_attn=None, offload_mode="none", worker_factory=Worker):
        if task != "t2va" or offload_mode != "cpu" or ref_image_match or ref_stage1_attn:
            raise ValueError("This experiment is fixed T2VA with CPU offload")
        self.paths, self.task = paths, task
        self.config = load_recipe(task)
        self.config["execution"] = {"offload": "cpu", "gpu_count": 1, "qwen_lifetime": "one_request", "overlap_stages": False}
        self.root = Path(output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.worker_factory = worker_factory
        self.stage1 = self.qwen = None
        self.report = {"status": "STARTING", "task": task, "recipe": self.config, "requests": [],
                       "timing": "same-host monotonic request entry to completed muxed MP4",
                       "warmup_excluded": True, "first_formal_request_included": True, "phase_sum_used": False}
        self.save()

    def save(self):
        write_json(self.root / "results.json", self.report)

    def condition(self, case, request, worker_root):
        self.qwen = self.worker_factory("qwen", worker_root, self.paths, self.config)
        try:
            return self.qwen.call(case=case, output_root=str(request / "qwen"))
        finally:
            self.qwen.close()
            self.qwen = None

    def start(self, case):
        started = time.monotonic_ns()
        self.stage1 = self.worker_factory("stage1", self.root / "stage1-worker", self.paths, self.config)
        request = self.root / "warmup"
        request.mkdir()
        warmup = dict(case, seed=999, request_id="warmup")
        conditioning = self.condition(warmup, request, self.root / "warmup-qwen-worker")
        s1 = self.stage1.call(case=warmup, conditioning_path=conditioning["conditioning_path"], outputdir=str(request / "stage1"))
        self.report.update(status="READY", startup_and_warmup_s=(time.monotonic_ns() - started) / 1e9,
                           warmup={"case_id": case["case_id"], "seed": 999, "output": s1["output"]})
        self.save()

    def generate(self, case):
        if self.stage1 is None or case.get("task", "t2va") != self.task:
            raise RuntimeError("Start this T2VA session before generating")
        request = self.root / case["case_id"]
        request.mkdir()
        row = {"case_id": case["case_id"], "seed": case["seed"], "task": self.task, "status": "RUNNING"}
        self.report["requests"].append(row)
        started = time.monotonic_ns()
        row["request_start_monotonic_ns"] = started
        try:
            conditioning = self.condition(case, request, request / "qwen-worker")
            s1 = self.stage1.call(case=case, conditioning_path=conditioning["conditioning_path"], outputdir=str(request / "stage1"))
            ended = s1["final_mp4_complete_monotonic_ns"]
            if not started < ended <= time.monotonic_ns() or not Path(s1["output"]).is_file():
                raise RuntimeError("Inconsistent request-to-completed-MP4 boundary")
            row.update(status="PASS", output=s1["output"], e2e_s=(ended-started)/1e9,
                       final_mp4_complete_monotonic_ns=ended, qwen_s=conditioning["request_wall_s"],
                       stage1_s=s1["request_wall_s"], h3_phases_s=s1["capture"]["phases_s"])
        except BaseException as error:
            row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            self.report["status"] = "FAIL"
            raise
        finally:
            self.save()
        return row

    def finish(self):
        rows = self.report["requests"]
        if not rows or any(row["status"] != "PASS" for row in rows):
            raise RuntimeError("No complete successful batch")
        self.report.update(status="PASS", mean_e2e_s=sum(row["e2e_s"] for row in rows)/len(rows))
        self.save()

    def close(self):
        try:
            if self.qwen is not None:
                self.qwen.close()
        finally:
            if self.stage1 is not None:
                self.stage1.close()
