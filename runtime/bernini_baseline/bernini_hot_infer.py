#!/usr/bin/env python3
"""Clean hot-timing driver for the Bernini t2v PRISTINE baseline.

Runs under torchrun *inside* the vendored pristine bernini_src. It only MEASURES
per-stage latency of each full-Bernini pipeline call — semantic planner
(vit_mllm), text encode (t5), the DiT diffusion loop, and VAE decode — then
prints a `[HOT_TIMING]` line and appends the record to $BERNINI_TIMING_JSON.

It only measures and changes no runtime configuration. Warmup-vs-hot is decided by the
caller through the `--inputs` ordering (a warmup pass followed by a measured
pass); this driver simply times every call in order.

Primary metric is `text_to_vae_decode` (from the start of text/condition
encoding to the end of VAE decode), matching the baseline measurement contract.
"""
from __future__ import annotations

import json
import os
import time

import torch

import bernini.pipeline as pipeline_mod
import infer_multi_gpu as infer_mod
from bernini.pipeline import BerniniPipeline


CUR = None        # current call's timing state (for the module-level _vae_decode)
RECORDS = []      # per-call timing records (rank 0 only)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def is_rank0():
    return os.environ.get("RANK", "0") == "0"


_orig_call = BerniniPipeline.__call__
_orig_vit = BerniniPipeline.sample_vit_embed
_orig_vae = pipeline_mod._vae_decode


def timed_call(self, *args, **kwargs):
    global CUR
    call_id = getattr(self, "_call_id", 0) + 1
    self._call_id = call_id
    state = {
        "call_id": call_id,
        "output_path": kwargs.get("output_path", ""),
        "vit_mllm": 0.0,
        "t5": 0.0,
        "diffusion": 0.0,
        "vae_decode": 0.0,
        "text_start": None,
        "decode_end": None,
    }
    self._timing_state = state
    CUR = state
    sync()
    state["call_start"] = time.perf_counter()
    result = _orig_call(self, *args, **kwargs)
    sync()
    total = time.perf_counter() - state["call_start"]
    t2v = None
    if state["text_start"] is not None and state["decode_end"] is not None:
        t2v = state["decode_end"] - state["text_start"]
    if is_rank0():
        print(
            "[HOT_TIMING] call=%d vit_mllm=%.3fs t5=%.3fs diffusion=%.3fs "
            "vae_decode=%.3fs text_to_vae_decode=%s pipeline_total_with_save=%.3fs"
            % (
                call_id,
                state["vit_mllm"],
                state["t5"],
                state["diffusion"],
                state["vae_decode"],
                ("%.3f" % t2v) if t2v is not None else "n/a",
                total,
            ),
            flush=True,
        )
        RECORDS.append(
            {
                "call_id": call_id,
                "output": os.path.basename(str(state["output_path"])),
                "vit_mllm": state["vit_mllm"],
                "t5": state["t5"],
                "diffusion": state["diffusion"],
                "vae_decode": state["vae_decode"],
                "text_to_vae_decode": t2v,
                "pipeline_total_with_save": total,
            }
        )
        json_path = os.environ.get("BERNINI_TIMING_JSON", "")
        if json_path:
            try:
                with open(json_path, "w") as f:
                    json.dump(RECORDS, f, indent=2)
            except OSError:
                pass
    CUR = None
    self._timing_state = None
    return result


def timed_vit(self, *args, **kwargs):
    state = getattr(self, "_timing_state", None)
    sync()
    start = time.perf_counter()
    if state is not None and state["text_start"] is None:
        state["text_start"] = start
    result = _orig_vit(self, *args, **kwargs)
    sync()
    if state is not None:
        state["vit_mllm"] += time.perf_counter() - start
    return result


def timed_vae(vae, latents):
    sync()
    start = time.perf_counter()
    result = _orig_vae(vae, latents)
    sync()
    end = time.perf_counter()
    if CUR is not None:
        CUR["vae_decode"] += end - start
        CUR["decode_end"] = end
    return result


BerniniPipeline.__call__ = timed_call
BerniniPipeline.sample_vit_embed = timed_vit
pipeline_mod._vae_decode = timed_vae


_orig_build = infer_mod.build_pipeline


def timed_build_pipeline(args, device):
    pipeline = _orig_build(args, device)
    if not isinstance(pipeline, BerniniPipeline):
        return pipeline
    model = pipeline.model

    _orig_t5 = model.get_t5_text_embeddings_sample

    def timed_t5(*a, **k):
        state = getattr(pipeline, "_timing_state", None)
        sync()
        start = time.perf_counter()
        if state is not None and state["text_start"] is None:
            state["text_start"] = start
        r = _orig_t5(*a, **k)
        sync()
        if state is not None:
            state["t5"] += time.perf_counter() - start
        return r

    model.get_t5_text_embeddings_sample = timed_t5

    _orig_diff = model.diff_dec.sample_bernini_wvitcfg

    def timed_diff(*a, **k):
        state = getattr(pipeline, "_timing_state", None)
        sync()
        start = time.perf_counter()
        r = _orig_diff(*a, **k)
        sync()
        if state is not None:
            state["diffusion"] += time.perf_counter() - start
        return r

    model.diff_dec.sample_bernini_wvitcfg = timed_diff
    return pipeline


infer_mod.build_pipeline = timed_build_pipeline


if __name__ == "__main__":
    infer_mod.main()
