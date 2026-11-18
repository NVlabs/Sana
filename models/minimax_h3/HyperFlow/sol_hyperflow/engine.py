"""Resident eight-GPU HyperFlow engine, retaining both upstream DiT partitions."""
from __future__ import annotations

import os
from pathlib import Path

from .config import Request, validate_world_size


class HyperFlowInference:
    """Call every operation on all ranks; only rank zero returns media.

    Requests must run serially and in the same order on every rank. The two DiTs
    share the conditioner and VAEs, but their weights remain distinct.
    """

    def __init__(self, model_path, adapter_path, *, lora_mode="fused", trim_conditioner=True):
        validate_world_size(int(os.environ.get("WORLD_SIZE", "1")))
        if lora_mode not in {"fused", "separate"}:
            raise ValueError("lora_mode must be fused or separate; merged/MXFP8 is not this profile.")
        model_path = Path(model_path).expanduser().resolve()
        adapter_path = Path(adapter_path).expanduser().resolve()
        if not model_path.is_dir() or not adapter_path.is_file():
            raise FileNotFoundError("Pass a local MiniMax-H3 snapshot and a local HyperFlow adapter file.")
        for partition in ("transformer", "transformer_ref"):
            if not (model_path / partition / "config.json").is_file():
                raise FileNotFoundError(f"The model snapshot is missing {partition}/config.json.")
        if (model_path / "transformer").resolve() == (model_path / "transformer_ref").resolve():
            raise ValueError("transformer and transformer_ref must be distinct upstream partitions.")

        import torch
        import torch.distributed as dist
        from .runtime import build_all

        self._owns_process_group = False
        self._closed = False
        self.lora_mode = lora_mode
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=self.device)
            self._owns_process_group = True
        self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
        validate_world_size(self.world_size)
        try:
            self.pipes, self.attention, self.metadata = build_all(
                str(model_path), str(adapter_path), self.device, self.world_size,
                lora_mode=lora_mode, trim_conditioner=trim_conditioner,
            )
        except Exception:
            if self._owns_process_group:
                dist.destroy_process_group()
                self._owns_process_group = False
            raise

    def generate(self, prompt, *, task="t2v", duration=5, seed=0, image=None,
                 references=None, attention_backend="sol_bsa"):
        if self._closed:
            raise RuntimeError("This HyperFlow engine is closed.")
        request = Request(prompt, task, duration, seed, attention_backend,
                          image, tuple(references or ())).payload()
        from h3_runtime.engine import GeneratedMedia
        from .runtime import run

        sparse = self.attention["ref2v" if task == "ref2va" else "t2v"]
        state, elapsed, counts = run(self.pipes[request["task"]], request, sparse,
                                    expect_fused=self.lora_mode == "fused")
        self.last_request = dict(task=task, duration=duration, seed=seed, steps=8,
                                 attention_backend=attention_backend,
                                 generation_seconds=elapsed, **counts)
        if self.rank != 0:
            return None
        audio = state.get("audio")
        return GeneratedMedia(video=state["videos"][0], audio=None if audio is None else audio[0],
                              audio_sample_rate=state["sampling_rate"],
                              elapsed_s=elapsed, duration=duration, seed=seed)

    def warmup(self, *, prompt=None, task="t2v", duration=5, image=None, references=None,
               attention_backend="sol_bsa"):
        self.generate(prompt or "A continuous cinematic scene in natural light.", task=task,
                      duration=duration, image=image, references=references,
                      attention_backend=attention_backend)

    def close(self):
        import torch.distributed as dist

        if self._owns_process_group and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            self._owns_process_group = False
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
