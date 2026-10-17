"""Original H3 baseline: no LoRA, official 50-point schedule, same Qwen/VAE/offload as Sol-H3."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import time

from .stage1_ops.lookup import quantized_linear_name
from .stage1_ops import native, vsa, sol
from . import offload
from .stage1_ops.tasks import input_spec, native_inputs, ref_lora_alpha, task_of, validate_prepared_media
from .qwen_ops.media import reference_target_area, resolve_reference_image_size

# Official basic_minimax_h3_t2v default: 50 sigma-grid points -> 49 DiT forwards.
GRID_POINTS = 50
FORWARDS = GRID_POINTS - 1


def install_reference_image_size(reference_module, target_area):
    """One dedicated worker seam; do not modify the external FastVideo files."""
    reference_module.resolve_reference_image_size = functools.partial(
        resolve_reference_image_size, target_area=target_area)


def validate_native_prompt(actual, original):
    # Pinned FastVideo video_generator.py normalizes only boundary whitespace.
    if actual != original.strip():
        raise RuntimeError("native H3 prompt mismatch")

def validate_native_attention(before, after, *, require_calls=True):
    if (len(before) != 1 or len(after) != 1 or after[0]["calls"] < before[0]["calls"]
            or (require_calls and after[0]["calls"] == before[0]["calls"])
            or after[0]["fa_version"] != "4" or after[0]["fallback_calls"] != 0
            or not set(after[0]["q_dtypes"]) <= {"torch.bfloat16", "torch.float16"}
            or any(len(after[0].get(name, [])) != FORWARDS for name in ("scheduler_timesteps", "audio_scheduler_timesteps"))):
        raise RuntimeError(f"native FA4 or 50-point schedule drift: {after}")
    return [{**after[0], "request_fa4_calls": after[0]["calls"] - before[0]["calls"],
             "resolved_precision_scope": "original constructor config; DiT remains BF16; Qwen external NVFP4"}]


def install_worker(worker, task="t2va", attention=None, reference_area=None,
                   reference_budget_source=None, cpu_offload=False):
    import torch
    from fastvideo.pipelines.lazy_module import is_lazy_module
    from fastvideo.layers.lora.linear import BaseLayerWithLoRA
    from fastvideo.layers.linear import LinearBase
    from .stage1_ops import native as control, capture, vsa, regional
    pipe = worker.pipeline
    state = {"task": task, "transformer_materializations": 0, "video_vae_materializations": 0,
             "conditioning_calls": 0, "attention_calls": 0, "compression_gates_unused": 0,
             "transformer_forwards": 0,
             "fp8_route": "none; original BF16 MiniMax-H3, no LoRA",
             "qwen_builtin_materializations": 0, "case": None}
    worker._sol_h3_stage1 = state
    state["cpu_offload"] = cpu_offload
    state["attention_backend"] = attention
    if cpu_offload:
        offload.install_stage1_loader()
    if task == "ref2va" and reference_area is not None:
        from fastvideo.pipelines.basic.minimax_h3 import reference
        install_reference_image_size(reference, reference_area)
        state["reference_image_resize"] = {"mode": "match", "pixel_budget": reference_area,
                                           "budget_source": reference_budget_source}
    sol_route = sol.RequestRoute(state) if attention == sol.POLICY else None
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
    if task != "t2va":
        raise RuntimeError("direct768 experiment supports only T2VA")
    def record_decoder(model):
        state["video_vae_materializations"] += 1
        if state["video_vae_materializations"] != 1:
            raise RuntimeError("Direct H3 VAE unexpectedly reloaded")
        return model
    vae.set_materialize_transform(record_decoder)
    if task == "ref2va":
        # The pinned compatibility patch exposes this lifecycle option. The
        # dedicated session never switches/unmerges its constructor adapter.
        import fastvideo.pipelines.lora_pipeline as lora_pipeline
        original_lora_layer = lora_pipeline.get_lora_layer
        def inference_lora_layer(*args, **kwargs):
            if not kwargs.get("training_mode", False):
                kwargs["retain_cpu_weight"] = False
            layer = original_lora_layer(*args, **kwargs)
            if layer is not None and task == "ref2va":
                original_set = layer.set_lora_weights
                def set_ref_weights(A, B, **options):
                    # The released adapter stores alpha only in safetensors
                    # metadata. Native FastVideo reads per-tensor alpha keys.
                    options["lora_alpha"] = ref_lora_alpha(A.shape[0], options.get("lora_alpha"))
                    return original_set(A, B, **options)
                layer.set_lora_weights = set_ref_weights
            return layer
        lora_pipeline.get_lora_layer = inference_lora_layer
    def prepare_bf16_after_merge(model):
        state["transformer_materializations"] += 1
        if state["transformer_materializations"] != 1:
            raise RuntimeError("resident Stage1 transformer was unexpectedly reloaded")
        merged = 0
        # The original constructor already merged BF16 low-rank deltas. Keep
        # each identical base layer, remove wrappers and their dead A/B storage.
        for name, module in list(model.named_modules()):
            if isinstance(module, BaseLayerWithLoRA):
                raise RuntimeError(f"baseline must not load a LoRA: {name}")
        pipe.lora_layers.clear()
        pipe.lora_adapters.clear()
        # The exact AdaLN lookup is specific to the distilled four-update grid.
        tagged = []
        bf16_weight_bytes = 0
        for name, module in model.named_modules():
            if isinstance(module, LinearBase) and quantized_linear_name(name):
                if module.weight.dtype != torch.bfloat16 or module.quant_config is not None:
                    raise RuntimeError(f"BF16 control has a quantized/non-BF16 target: {name}")
                tagged.append(name)
                bf16_weight_bytes += module.weight.numel() * module.weight.element_size()
        if not tagged or merged != 0 or any(hasattr(m, "_fp8_weight") for m in model.modules()):
            raise RuntimeError("baseline requires original BF16 weights, no LoRA and no FP8")
        state.update(merged_lora_layers=merged, bf16_linears=len(tagged),
                     bf16_linear_names=tagged, bf16_weight_bytes=bf16_weight_bytes,
                     fp8_linears=0, fp8_weight_bytes=0)
        original_model_forward = model.forward
        def observed_model_forward(*args, **kwargs):
            state["transformer_forwards"] += 1
            if sol_route is not None:
                sol_route.start_forward()
            state["dit_active"] = True
            try:
                return original_model_forward(*args, **kwargs)
            finally:
                state["dit_active"] = False
        model.forward = observed_model_forward
        if len(tagged) != 312:
            raise RuntimeError(f"expected 312 native BF16 linears, got {len(tagged)}")
        manager = offload.prepare_stage1_compile(model) if cpu_offload else None
        if task == "ref2va" or attention == "FA4_dense":
            if sol_route is None:
                regional.install_dense(model, state)
            else:
                sol.install(model, state, sol_route)
        else:
            vsa.install_model(model, state)
            regional.install(model, state)
        if manager is not None:
            manager.install_inference_hooks()
            state["offload"] = manager.stats
        return model
    pipe.modules["transformer"].set_materialize_transform(prepare_bf16_after_merge)
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
            validate_native_prompt(batch.prompt, record["case"]["prompt"])
            payload = record["payload"]
            validate_prepared_media(record["case"], payload, batch)
            cond, tags = payload["prompt_embeds"], payload["text_token_tags"]
            if (cond.dtype != torch.bfloat16 or cond.ndim != 3 or cond.shape[0] != 1 or cond.shape[2] != 5120
                    or not bool(torch.isfinite(cond).all()) or tags.dtype != torch.long
                    or tuple(tags.shape) != (cond.shape[1],)
                    or not bool(((tags == 1) | ((tags == 0) & (task != "t2va"))).all())):
                raise RuntimeError("NVFP4 task conditioning feature ABI mismatch")
            batch.prompt_embeds = [cond.to(device="cuda:0")]
            batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = tags
            state["conditioning_calls"] += 1
            return batch
        stage.forward = condition
        if sol_route is not None:
            from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
            denoising = pipe._stage_name_mapping["denoising_stage"]
            original_denoising = denoising.forward
            def denoise(batch, args):
                sol_route.prepare(batch.extra[MINIMAX_H3_LAYOUT_KEY])
                return original_denoising(batch, args)
            denoising.forward = denoise
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
    task = task_of(case)
    if task != worker._sol_h3_stage1["task"]:
        raise RuntimeError("one task/model partition is required per Stage1 session")
    if (receipt.get("status") != "PASS" or receipt.get("task") != task or receipt.get("external_anchor_used") is not False
            or receipt.get("prompt") != case["prompt"]
            or receipt.get("case_id") != case["case_id"] or receipt.get("seed") != case["seed"]
            or hashlib.sha256(payload_path.read_bytes()).hexdigest() != receipt["payload_sha256"]):
        raise RuntimeError("fresh NVFP4 conditioning receipt mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if (payload["prompt"] != case["prompt"] or payload["task"] != task or payload["external_anchor_used"]
            or (task != "t2va" and payload.get("input_spec") != input_spec(case))):
        raise RuntimeError("NVFP4 payload presentation mismatch")
    target = {**case, "directory": str(outputdir)}
    worker._sol_h3_stage1["case"] = {"case": case, "target": target, "payload": payload}
    if worker.pipeline.post_init_called:
        capture.arm_capture(worker, target)
    result = {key: worker._sol_h3_stage1[key] for key in ("attention_calls", "conditioning_calls", "transformer_forwards")}
    result["vsa_snapshot"] = vsa.snapshot(worker._sol_h3_stage1) if task != "ref2va" and worker._sol_h3_stage1["attention_backend"] != "FA4_dense" else None
    return result


def finish_worker_request(worker):
    """Trim only inactive CUDA cache after output; keep all loaded models."""
    import gc
    import torch
    started = time.monotonic_ns()
    video_vae_materialized = object.__getattribute__(worker.pipeline.modules["vae"], "_lazy_module") is not None
    fast_decoder = bool(os.environ.get("SOL_H3_FAST_VAE"))
    if video_vae_materialized != (not fast_decoder) or worker._sol_h3_stage1["video_vae_materializations"] != (0 if fast_decoder else 1):
        raise RuntimeError("Direct output did not retain one original H3 VAE")
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
               "placement": "DiT body/video VAE/audio VAE CPU offload; shared DiT GPU" if video_vae_materialized
                            else "DiT body/audio VAE CPU offload; noncausal VAE CPU between requests; original video VAE never materialized"}
    if after["allocated_bytes"] != before["allocated_bytes"] or not cleanup["model_identities_unchanged"]:
        raise RuntimeError(f"inactive-cache trim unexpectedly changed active storage/models: {cleanup}")
    return {**{key: value for key, value in worker._sol_h3_stage1.items() if key != "case"},
            "inactive_cuda_cache_cleanup": cleanup}


class Session:
    """One dedicated FastVideo executor on a single SM121 GPU."""

    def __init__(self, paths: dict, work_dir: str, config: dict):
        self.closed = False
        self.failed = False
        self.cpu_offload = offload.enabled(config)
        self.root = Path(work_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.task = task_of(config.get("stage1", config))
        self.attention = config.get("stage1", config).get("attention")
        self.reference_area = reference_target_area(config)
        self.reference_budget_source = config.get("stage1", config).get(
            "reference_image_resize", {}).get("budget_source")
        if self.attention == sol.POLICY and self.task != "ref2va":
            raise ValueError("this fixed Sol policy is Ref2VA only")
        self.basic = native.load_official_entry(
            Path(paths["fastvideo_root"]).expanduser().resolve(strict=True))
        profile = self.basic.parse_args([
            "--model-path", str(Path(paths["h3_model"]).expanduser().resolve(strict=True)),
            "--prompt", "session", "--output", str(self.root), "--num-gpus", "1",
            "--execution-backend", "mp", "--height", "768", "--width", "1344",
            "--num-frames", "124", "--steps", str(GRID_POINTS), "--lazy-module-load",
            "--no-warmup", "--repeats", "1"])
        if self.task == "ref2va" or self.attention == "FA4_dense":
            profile.vsa = False
            profile.inference_torch_compile = False
            policy = {"attention_backend": "FLASH_ATTN", "selected_attention_backend": self.attention or "FA4_dense",
                      "compression_gates": "none; FastH3 dense-datafree adapter"}
        else:
            policy = vsa.configure_profile(profile)
        if self.task != "t2va" or self.attention != "FA4_dense":
            raise ValueError("baseline is T2VA dense FA4 only")
        profile.lora_path = None
        self.profile = profile
        # Make the fixed release route independent of inherited VSA selector flags.
        os.environ.update(FASTVIDEO_VSA_SM100A="0", FASTVIDEO_VSA_CUTEDSL="0",
                          FASTVIDEO_NVFP4_FA4="0")
        environment = self.basic.configure_environment(profile)
        self.basic.validate_profile_dependencies(profile)
        generator_config = self.basic.build_generator_config(profile)
        if self.cpu_offload:
            engine = generator_config.engine
            generator_config = dataclasses.replace(generator_config, engine=dataclasses.replace(
                engine, offload=dataclasses.replace(engine.offload, dit=True, dit_layerwise=True)))
        if self.task != "t2va":
            components = generator_config.pipeline.components
            if self.task == "ref2va":
                components = dataclasses.replace(components,
                    override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline")
            generator_config = dataclasses.replace(generator_config,
                pipeline=dataclasses.replace(generator_config.pipeline, workload_type="i2v", components=components))
        native.atomic_json(self.root / "resolved_config.json", {
            "profile": vars(profile), "generator": dataclasses.asdict(generator_config),
            "environment": environment, "attention": policy,
            "quantization": "none; original BF16 MiniMax-H3, no LoRA, native AdaLN",
            "video_decode": True, "audio_decode": "original H3 PCM"})
        from fastvideo import VideoGenerator
        import cloudpickle
        self.generator = VideoGenerator.from_config(generator_config)
        self.rpc = lambda fn: self.generator.executor.collective_rpc(cloudpickle.dumps(fn))
        try:
            self.rpc(functools.partial(install_worker, task=self.task, attention=self.attention,
                                      reference_area=self.reference_area,
                                      reference_budget_source=self.reference_budget_source,
                                      cpu_offload=self.cpu_offload))
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
        if task_of(case) != self.task:
            raise ValueError("case task differs from the loaded Stage1 session")
        try:
            before = self.rpc(functools.partial(
                arm_worker, case=case, conditioning_path=str(conditioning_path),
                outputdir=str(outputdir)))[0]
            attention_before = self.rpc(native.read_fa4_audit)
            self.profile.prompt = case["prompt"]
            output = outputdir / "native_768p.mp4"
            request = self.basic.build_request(self.profile, output, case["seed"])
            request = dataclasses.replace(request, inputs=native_inputs(case), output=dataclasses.replace(
                request.output, save_video=False, return_frames=False))
            started = time.monotonic_ns()
            self.generator.generate(request)
            ended = time.monotonic_ns()
            actual = self.rpc(finish_worker_request)[0]
            attention_calls = actual["attention_calls"] - before["attention_calls"]
            forwards = actual["transformer_forwards"] - before["transformer_forwards"]
            if (forwards != FORWARDS or attention_calls != 50 * FORWARDS + 2
                    or actual["conditioning_calls"] - before["conditioning_calls"] != 1):
                raise RuntimeError(f"50-point native H3 call count mismatch: {actual}")
            attention_actual = validate_native_attention(
                attention_before, self.rpc(native.read_fa4_audit))
            if self.task == "ref2va" or self.attention == "FA4_dense":
                if attention_actual[0]["request_fa4_calls"] != 50 * FORWARDS + 2:
                    raise RuntimeError("physical FA4 call count drift")
                if self.attention == sol.POLICY:
                    actual["sol_attention"] = sol.validate_request(actual["sol_attention"])
                sparse_actual = None
            else:
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
                    or capture.get("latent_only_transfer") is not False
                    or capture.get("h3_decoder_calls") != 1 or not output.is_file()):
                raise RuntimeError("Same-request original H3 direct decode failed")
            receipt = {
                **case, "status": "PASS", "capture": capture, "actual": actual,
                "capture_dir": str(outputdir), "capture_path": str(capture_path),
                "native_fa4_audit": attention_actual, "official_vsa_actual": sparse_actual,
                "request_transformer_forwards": forwards,
                "request_attention_calls": attention_calls,
                "first_request_compile_included": not bool(before["transformer_forwards"]),
                "conditioning_path": str(conditioning_path), "native_video_saved": True,
                "latent_only_transfer": False,
                "output": str(output),
                "final_mp4_complete_monotonic_ns": capture["final_mp4_complete_monotonic_ns"],
                "started_monotonic_ns": started, "completed_monotonic_ns": ended,
                "request_wall_s": (ended - started) / 1e9,
                "timing_scope": "H3 generation, original VAE decode and MP4; Qwen/full E2E owned by caller"}
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
