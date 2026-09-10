"""W8A8 FastH3 VSA Stage1: four updates, normalized H3 latent and native PCM."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import time

from .stage1_ops.lookup import install_exact_t2va_lookup, quantized_linear_name
from .stage1_ops import native, vsa

def validate_native_attention(before, after, *, require_calls=True):
    if (len(before) != 1 or len(after) != 1 or after[0]["calls"] < before[0]["calls"]
            or (require_calls and after[0]["calls"] == before[0]["calls"])
            or after[0]["fa_version"] != "4" or after[0]["fallback_calls"] != 0
            or not set(after[0]["q_dtypes"]) <= {"torch.bfloat16", "torch.float16"}
            or any(len(after[0].get(name, [])) != 4 for name in ("scheduler_timesteps", "audio_scheduler_timesteps"))):
        raise RuntimeError(f"native FA4 or four-update schedule drift: {after}")
    return [{**after[0], "request_fa4_calls": after[0]["calls"] - before[0]["calls"],
             "resolved_precision_scope": "original constructor config; actual DiT converted to FP8 afterwards; Qwen external NVFP4"}]


def install_worker(worker):
    import torch
    from fastvideo.pipelines.lazy_module import is_lazy_module
    from fastvideo.layers.lora.linear import BaseLayerWithLoRA
    from fastvideo.layers.linear import LinearBase
    from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod, convert_model_to_fp8
    from .stage1_ops import native as control, capture, vsa, regional
    pipe = worker.pipeline
    state = {"transformer_materializations": 0, "video_vae_materializations": 0,
             "conditioning_calls": 0, "attention_calls": 0, "compression_gates_unused": 0,
             "transformer_forwards": 0,
             "fp8_route": "native_FastVideo_tensorwise_W8A8_after_original_BF16_LoRA_merge",
             "qwen_builtin_materializations": 0, "case": None}
    worker._sol_h3_stage1 = state
    if not is_lazy_module(pipe.modules["transformer"]) or not is_lazy_module(pipe.modules["text_encoder"]):
        raise RuntimeError("existing native lazy construction required")
    # Refuse accidental native Qwen loading, including eager parameter probes.
    def forbidden_qwen():
        state["qwen_builtin_materializations"] += 1
        raise RuntimeError("native BF16 Qwen must not load in external NVFP4 mode")
    object.__setattr__(pipe.modules["text_encoder"], "_lazy_loader", forbidden_qwen)
    vae = pipe.modules["vae"]
    if not is_lazy_module(vae) or object.__getattribute__(vae, "_lazy_module") is not None:
        raise RuntimeError("latent capture requires an unmaterialized native video VAE")
    def forbidden_video_vae():
        state["video_vae_materializations"] += 1
        raise RuntimeError("latent capture must never materialize the native video VAE")
    object.__setattr__(vae, "_lazy_loader", forbidden_video_vae)
    def quantize_after_merge(model):
        state["transformer_materializations"] += 1
        if state["transformer_materializations"] != 1:
            raise RuntimeError("resident Stage1 transformer was unexpectedly reloaded")
        merged = 0
        # The original constructor already merged BF16 low-rank deltas. Keep
        # each identical base layer, remove wrappers and their dead A/B storage.
        for name, module in list(model.named_modules()):
            if isinstance(module, BaseLayerWithLoRA):
                if not module.merged and not module.disable_lora:
                    raise RuntimeError(f"LoRA not merged before FP8: {name}")
                parent, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent) if parent else model, leaf, module.base_layer)
                merged += int(module.merged)
        pipe.lora_layers.clear()
        pipe.lora_adapters.clear()
        state["fixed_lookup"] = install_exact_t2va_lookup(model, pipe)
        tagged = []
        for name, module in model.named_modules():
            if isinstance(module, LinearBase) and quantized_linear_name(name):
                module.quant_config = FP8Config()
                module.quant_method = FP8QuantizeMethod("tensor")
                tagged.append(name)
        if not tagged or merged == 0:
            raise RuntimeError("missing real LoRA merge or native FP8 targets")
        convert_model_to_fp8(model)
        state.update(merged_lora_layers=merged, fp8_linears=len(tagged), fp8_linear_names=tagged,
                     fp8_weight_bytes=sum(m._fp8_weight.numel() for m in model.modules() if hasattr(m, "_fp8_weight")))
        original_model_forward = model.forward
        def observed_model_forward(*args, **kwargs):
            state["transformer_forwards"] += 1
            return original_model_forward(*args, **kwargs)
        model.forward = observed_model_forward
        if len(tagged) != 312:
            raise RuntimeError(f"expected 312 native W8A8 linears, got {len(tagged)}")
        vsa.install_model(model, state)
        regional.install(model, state)
        return model
    pipe.modules["transformer"].set_materialize_transform(quantize_after_merge)
    control.install_fa4_audit(worker)  # Native FA4 handles the two text-refiner blocks.
    original_post_init = pipe.post_init
    def post_init():
        original_post_init()
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
        stage = pipe._stage_name_mapping["conditioning_stage"]
        def condition(batch, args):
            record = state["case"]
            if batch.prompt != record["case"]["prompt"] or batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY):
                raise RuntimeError("native T2VA prompt/keyframe mismatch")
            payload = record["payload"]
            cond, tags = payload["prompt_embeds"], payload["text_token_tags"]
            if (cond.dtype != torch.bfloat16 or cond.ndim != 3 or cond.shape[0] != 1 or cond.shape[2] != 5120
                    or not bool(torch.isfinite(cond).all()) or tags.dtype != torch.long
                    or tuple(tags.shape) != (cond.shape[1],) or not bool((tags == MINIMAX_H3_TEXT_TAG).all())):
                raise RuntimeError("NVFP4 prompt-only feature ABI mismatch")
            batch.prompt_embeds = [cond.to(device="cuda:0")]
            batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = tags
            state["conditioning_calls"] += 1
            return batch
        stage.forward = condition
        # Preserve native per-stage machinery, retaining only video models;
        # normal process exit still frees them and abort cleanup is unchanged.
        keep = {id(pipe.modules[name]) for name in ("transformer", "vae", "audio_vae")}
        for stage in pipe._stages:
            stage._lazy_modules_to_release = tuple(m for m in stage._lazy_modules_to_release if id(m) not in keep)
        capture.install_capture(worker)
        capture.arm_capture(worker, state["case"]["target"])
        pipe.post_init = original_post_init
    pipe.post_init = post_init
    return {"status": "READY_FOR_FIRST_REQUEST"}


def arm_worker(worker, case, conditioning_path, outputdir):
    import torch
    from .stage1_ops import capture, vsa
    root = Path(conditioning_path).resolve()
    receipt = json.loads(root.read_text())
    payload_path = root.parent / "conditioning.pt"
    if (receipt.get("status") != "PASS" or receipt.get("task") != "t2va" or receipt.get("external_anchor_used") is not False
            or receipt.get("prompt") != case["prompt"]
            or receipt.get("case_id") != case["case_id"] or receipt.get("seed") != case["seed"]
            or hashlib.sha256(payload_path.read_bytes()).hexdigest() != receipt["payload_sha256"]):
        raise RuntimeError("fresh NVFP4 conditioning receipt mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if payload["prompt"] != case["prompt"] or payload["task"] != "t2va" or payload["external_anchor_used"]:
        raise RuntimeError("NVFP4 payload presentation mismatch")
    target = {**case, "directory": str(outputdir)}
    worker._sol_h3_stage1["case"] = {"case": case, "target": target, "payload": payload}
    if worker.pipeline.post_init_called:
        capture.arm_capture(worker, target)
    result = {key: worker._sol_h3_stage1[key] for key in ("attention_calls", "conditioning_calls", "transformer_forwards")}
    result["vsa_snapshot"] = vsa.snapshot(worker._sol_h3_stage1)
    return result


def finish_worker_request(worker):
    """Trim only inactive CUDA cache after output; keep all loaded models."""
    import gc
    import torch
    started = time.monotonic_ns()
    video_vae_materialized = object.__getattribute__(worker.pipeline.modules["vae"], "_lazy_module") is not None
    if video_vae_materialized or worker._sol_h3_stage1["video_vae_materializations"] != 0:
        raise RuntimeError("latent-only transfer unexpectedly loaded the video VAE")
    def models():
        return {name: id(object.__getattribute__(worker.pipeline.modules[name], "_lazy_module"))
                for name in ("transformer", "vae", "audio_vae")}
    identities = models()
    torch.cuda.synchronize()
    gc.collect()
    before = {"allocated_bytes": int(torch.cuda.memory_allocated()),
              "reserved_bytes": int(torch.cuda.memory_reserved())}
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after = {"allocated_bytes": int(torch.cuda.memory_allocated()),
             "reserved_bytes": int(torch.cuda.memory_reserved())}
    cleanup = {"before": before, "after": after,
               "freed_reserved_bytes": before["reserved_bytes"] - after["reserved_bytes"],
               "model_identities_unchanged": identities == models(),
               "started_monotonic_ns": started, "completed_monotonic_ns": time.monotonic_ns(),
               "video_vae_materialized": video_vae_materialized,
               "placement": "DiT/audio VAE resident; video VAE never materialized"}
    if after["allocated_bytes"] != before["allocated_bytes"] or not cleanup["model_identities_unchanged"]:
        raise RuntimeError(f"inactive-cache trim unexpectedly changed active storage/models: {cleanup}")
    return {**{key: value for key, value in worker._sol_h3_stage1.items() if key != "case"},
            "inactive_cuda_cache_cleanup": cleanup}


class Session:
    """One dedicated FastVideo executor on a single SM121 GPU."""

    def __init__(self, paths: dict, work_dir: str, config: dict):
        self.closed = False
        self.failed = False
        self.root = Path(work_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.basic = native.load_official_entry(
            Path(paths["fastvideo_root"]).expanduser().resolve(strict=True))
        profile = self.basic.parse_args([
            "--model-path", str(Path(paths["h3_model"]).expanduser().resolve(strict=True)),
            "--prompt", "session", "--output", str(self.root), "--num-gpus", "1",
            "--execution-backend", "mp", "--height", "384", "--width", "672",
            "--num-frames", "124", "--steps", "5", "--lazy-module-load",
            "--no-warmup", "--repeats", "1"])
        policy = vsa.configure_profile(profile)
        profile.lora_path = str(Path(paths["vsa_lora"]).expanduser().resolve(strict=True))
        profile.lora_strength = 1.0
        self.profile = profile
        # Make the fixed release route independent of inherited VSA selector flags.
        os.environ.update(FASTVIDEO_VSA_SM100A="0", FASTVIDEO_VSA_CUTEDSL="0",
                          FASTVIDEO_NVFP4_FA4="0")
        environment = self.basic.configure_environment(profile)
        self.basic.validate_profile_dependencies(profile)
        generator_config = self.basic.build_generator_config(profile)
        native.atomic_json(self.root / "resolved_config.json", {
            "profile": vars(profile), "generator": dataclasses.asdict(generator_config),
            "environment": environment, "attention": policy,
            "quantization": "native W8A8 after BF16 LoRA merge; exact BF16 AdaLN lookup",
            "video_decode": False, "audio_decode": "original H3 PCM"})
        from fastvideo import VideoGenerator
        import cloudpickle
        self.generator = VideoGenerator.from_config(generator_config)
        self.rpc = lambda fn: self.generator.executor.collective_rpc(cloudpickle.dumps(fn))
        try:
            self.rpc(install_worker)
        except BaseException:
            self.generator.shutdown()
            self.closed = True
            raise

    def run(self, case, conditioning_path, outputdir):
        if self.closed or self.failed:
            raise RuntimeError("Stage1 session is unavailable")
        outputdir = Path(outputdir).resolve()
        outputdir.mkdir(parents=True, exist_ok=False)
        case = {**case, "source_index": case.get("source_index", 0)}
        try:
            before = self.rpc(functools.partial(
                arm_worker, case=case, conditioning_path=str(conditioning_path),
                outputdir=str(outputdir)))[0]
            attention_before = self.rpc(native.read_fa4_audit)
            self.profile.prompt = case["prompt"]
            output = outputdir / "native_384p.mp4"
            request = self.basic.build_request(self.profile, output, case["seed"])
            request = dataclasses.replace(request, output=dataclasses.replace(
                request.output, save_video=False, return_frames=False))
            started = time.monotonic_ns()
            self.generator.generate(request)
            ended = time.monotonic_ns()
            actual = self.rpc(finish_worker_request)[0]
            attention_calls = actual["attention_calls"] - before["attention_calls"]
            forwards = actual["transformer_forwards"] - before["transformer_forwards"]
            if (forwards != 4 or attention_calls != 202
                    or actual["conditioning_calls"] - before["conditioning_calls"] != 1):
                raise RuntimeError(f"four-update native H3 call count mismatch: {actual}")
            attention_actual = validate_native_attention(
                attention_before, self.rpc(native.read_fa4_audit))
            sparse_actual = vsa.validate_request(before["vsa_snapshot"], actual["official_vsa"])
            regional = actual["vsa_regional_compile"]
            if (regional["body_runtime_calls"] != actual["official_vsa"]["vsa_forward_calls"]
                    or regional["text_runtime_calls"] != actual["official_vsa"]["dense_text_calls"]
                    or regional["regions_wrapped"] != 52):
                raise RuntimeError("regional call counts differ from native VSA")
            regional["status"] = "PASS_RUNTIME_FULLGRAPH_AND_CALL_COUNTS"
            capture_path = outputdir / "capture.json"
            capture = json.loads(capture_path.read_text())
            if (capture["status"] != "PASS" or not capture["same_request"]
                    or capture["external_anchor_used"]
                    or capture.get("latent_only_transfer") is not True
                    or capture.get("video_decode_calls") != 0
                    or capture.get("h3_decoder_calls") != 0
                    or capture.get("generated_first_frame_path") is not None
                    or output.exists() or (outputdir / "generated_first_frame.png").exists()):
                raise RuntimeError("same-request latent-only capture failed")
            receipt = {
                **case, "status": "PASS", "capture": capture, "actual": actual,
                "capture_dir": str(outputdir), "capture_path": str(capture_path),
                "native_fa4_audit": attention_actual, "official_vsa_actual": sparse_actual,
                "request_transformer_forwards": forwards,
                "request_attention_calls": attention_calls,
                "first_request_compile_included": not bool(before["transformer_forwards"]),
                "conditioning_path": str(conditioning_path), "native_video_saved": False,
                "latent_only_transfer": True,
                "started_monotonic_ns": started, "completed_monotonic_ns": ended,
                "request_wall_s": (ended - started) / 1e9,
                "timing_scope": "Stage1 generation and capture; Qwen and full E2E owned by caller"}
            native.atomic_json(outputdir / "manifest.json", receipt)
            return receipt
        except BaseException:
            self.failed = True
            raise

    def release_idle_cache(self):
        if self.closed or self.failed:
            raise RuntimeError("Stage1 session is unavailable")
        return self.rpc(finish_worker_request)[0]["inactive_cuda_cache_cleanup"]

    def close(self):
        if not self.closed:
            self.closed = True
            self.generator.shutdown()
