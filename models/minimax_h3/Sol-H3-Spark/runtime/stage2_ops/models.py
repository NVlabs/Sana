"""Construct only the model fleet consumed by the Spark latent route.

Builders and joint AV sampling are the public Super Acceleration/LTX classes.
This factory avoids the older diagnostic constructor's unused image, text,
pixel-input VAE, LTX upsampler and preview-decoder models.
"""
from __future__ import annotations

from collections import Counter
import gc
import importlib
import os
from pathlib import Path
import sys

from .h3_upscale import load_upscaler
from .single_gpu_all2all import single_gpu_all2all_factory

SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
TILES = {"frames": {"tile_size": 128, "overlap": 24},
         "height": {"tile_size": 448, "overlap": 64},
         "width": {"tile_size": 768, "overlap": 64}}


def public_refiner_modules(ltx_root):
    ltx_root = Path(ltx_root).resolve(strict=True)
    for package in ("ltx-core", "ltx-pipelines", "ltx-kernels"):
        source = ltx_root / "packages" / package / "src"
        if not source.is_dir():
            raise FileNotFoundError(source)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    model_root = Path(__file__).resolve().parents[3]
    for path in (model_root / "super_acceleration/stage2", model_root.parents[1]):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["H3_LTX_SOURCE_FRAMES"] = "124"
    os.environ["H3_LTX_OUTPUT_FRAMES"] = "121"
    os.environ["SOL_ATTN_STRICT"] = "1"
    base = importlib.import_module("refiner_encoder_ablation_single_gpu")
    compat = importlib.import_module("official_compat_h3_refiner_diagnostic")
    if (base.FRAME_COUNT, base.WIDTH, base.HEIGHT, tuple(base.STAGE2_SIGMAS)) != (121, 1344, 768, SIGMAS):
        raise RuntimeError("public Refiner geometry or sigma contract changed")
    return base, compat


def construct(models, paths, *, torch, dist, compile_enabled, base, compat):
    from .h3_ltx_adapter import H3ToLTXAdapter
    from ltx_core.model.transformer.model import LTXModelType

    models.rank, models.world_size = 0, 1
    models.device, models.dtype = torch.device("cuda", 0), torch.bfloat16
    models.compile_enabled = compile_enabled
    models.attention_backend = "sol"
    models.variant_config = {"encoder": "h3_adapter", "latent_upsampler": False,
                             "pixel_width": 1344, "pixel_height": 768}
    models.stage2_audio_enabled, models.retain_allocator_cache = True, True
    models.output_decoder_backend = "official_vae"
    models.stage2_sigmas = SIGMAS
    models.sigmas = torch.tensor(SIGMAS, device=models.device, dtype=torch.float32)
    models.last_decoder_stream_info = None
    models.h3_upscaler_calls = models.adapter_convert_calls = 0
    models.h3_upscaler, models.h3_spatial_route = load_upscaler(
        paths["h3_upscaler_source"], paths["h3_upscaler_checkpoint"],
        device=models.device, torch_module=torch)
    models.h3_ltx_adapter = H3ToLTXAdapter.from_pretrained(
        paths["adapter_dir"], device=models.device, dtype=models.dtype)

    pixel_shape = base.VideoPixelShape(batch=1, frames=121, height=768, width=1344, fps=24)
    latent_shape = base.VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=base.VIDEO_SCALE_FACTORS)
    models.video_tools = base.VideoLatentTools(base.VideoLatentPatchifier(patch_size=1),
        latent_shape, 24.0, scale_factors=base.VIDEO_SCALE_FACTORS)
    registry = base.ModelRegistry()
    compilation = (base.CompilationConfig(mode=base.COMPILE_MODE, fullgraph=False,
        dynamic=None, seq_dim_dynamic=False, capture=False) if compile_enabled else None)
    lora = base.LoraPathStrengthAndSDOps(str(paths["refiner_lora"]), 0.8, base.LTXV_LORA_COMFY_RENAMING_MAP)
    stage = base.DiffusionStage.from_checkpoint(str(paths["transformer"]), models.dtype,
        models.device, loras=(lora,), quantization=None, registry=registry,
        compilation_config=compilation, alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER,
        offload_mode=base.OffloadMode.NONE)
    model_config = stage._transformer_builder.model_config()["transformer"]
    heads, head_dim = int(model_config["num_attention_heads"]), int(model_config["attention_head_dim"])
    if (int(model_config["num_layers"]), heads, head_dim) != (48, 32, 128):
        raise RuntimeError("expected the original 48-layer, 32-head LTX joint refiner")
    with single_gpu_all2all_factory(torch_module=torch) as all2all:
        manager = base.AttentionManager(max_tokens=65536, num_heads=heads,
            head_dim=head_dim, tensor_dtype=models.dtype, group=dist.group.WORLD)
    models.all2all_receipt = all2all
    if all2all["single_gpu_instances"] != 4 or all2all["delegated_instances"] != 0 or manager.copy_out:
        raise RuntimeError("expected four singleton copy-eliding All2All buffers")
    models.attention_manager = manager
    models.steady_all2all_timeout_s = manager.all2all_timeout_seconds
    if compile_enabled:
        manager.all2all_timeout_seconds = base.COMPILE_WARMUP_ALL2ALL_TIMEOUT_S
    stage._transformer_builder = base.SequenceParallelBuilder(inner=stage._transformer_builder,
        attn_mgr=manager, registry=registry,
        tracker=base.TransformerWeightTracker(group=dist.group.WORLD, no_lora_swap=True))
    models.transformer = stage._build_transformer(video_tools=models.video_tools).requires_grad_(False)
    models.sol_attention = base.Stage2SolAttention(models.transformer,
        isolate_sol_from_compile=compile_enabled)
    models.sol_backend = base._actual_sol_backend()
    if models.sol_backend != "triton":
        raise RuntimeError("Spark release requires the original Triton Sol backend")
    core = models.transformer.velocity_model
    while hasattr(core, "model"):
        core = core.model
    if core.model_type != LTXModelType.AudioVideo:
        raise RuntimeError("Spark requires the full joint AudioVideo model")
    models.wrapped_transformer = base.BatchSplitAdapter(models.transformer, max_batch_size=1)
    audio_block = compat.AudioConditioner(str(paths["audio_vae"]), models.dtype, models.device,
        registry=registry, alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER)
    models.audio_encoder = audio_block._encoder_builder.build(
        device=models.device, dtype=models.dtype).eval().requires_grad_(False)
    models.audio_tools = compat.AudioLatentTools(compat.AudioPatchifier(patch_size=1),
        compat.AudioLatentShape.from_video_pixel_shape(pixel_shape))
    decoder_block = compat.VideoDecoder(str(paths["output_video_vae"]), models.dtype,
        models.device, registry=registry, alloc_trim_strategy=base.AllocatorTrimStrategy.DEFER)
    # Original explicit Conv tiles. Resolve before constructing decoder weights.
    models.tiling_config = compat.TileSizeConfig(**{
        name: compat.DimensionSizeConfig(**value) for name, value in TILES.items()})
    models.tiling_config.validate(
        compat.tiling_scale_factors_for_vae(str(paths["output_video_vae"])), pixel_shape)
    models.tiling_source = "original_Conv128_448x768"
    gc.collect()
    torch.cuda.empty_cache()
    models.video_decoder = decoder_block._decoder_builder.build(
        device=models.device, dtype=models.dtype).eval().requires_grad_(False)
    if type(models.video_decoder).__name__ != "ConvVideoDecoder":
        raise RuntimeError("the output checkpoint must build the original ConvVideoDecoder")
    models.parameter_dtypes = dict(Counter(str(p.dtype) for p in models.transformer.parameters()))
    if (not models.parameter_dtypes.get("torch.bfloat16")
            or not set(models.parameter_dtypes) <= {"torch.bfloat16", "torch.float32"}):
        raise RuntimeError("refiner parameters must retain BF16 and original FP32 parameters")
    models.residency = {name: base._module_residency(name, module) for name, module in (
        ("transformer", models.transformer), ("audio_encoder", models.audio_encoder),
        ("video_decoder", models.video_decoder), ("h3_upscaler", models.h3_upscaler),
        ("h3_ltx_adapter", models.h3_ltx_adapter.model))}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
