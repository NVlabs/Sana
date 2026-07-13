from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class LightSglangMoeRunnerConfig:
    num_experts: Optional[int] = None
    num_local_experts: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size_per_partition: Optional[int] = None
    layer_id: Optional[int] = None
    top_k: Optional[int] = None
    num_fused_shared_experts: Optional[int] = None
    params_dtype: Optional[torch.dtype] = None
    routing_method_type: Optional[object] = None
    activation: str = "silu"
    is_gated: bool = True
    apply_router_weight_on_input: bool = False
    inplace: bool = False
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None
    swiglu_limit: Optional[float] = None
    gate_up_interleaved: bool = False


class LightSglangStandardTopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


SGLANG_MOE_SERVER_ARGS = SimpleNamespace(
    enable_deterministic_inference=False,
    enable_fused_moe_sum_all_reduce=False,
)
FP8_E4M3_MAX = 448.0
_SERVER_ARGS_READY = False


def sglang_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class SglangEnvFlag:
    def __init__(self, name: str, default: bool = False):
        self.name = name
        self.default = default

    def get(self) -> bool:
        return sglang_env_flag(self.name, self.default)


def ensure_module(name: str, package_path: str | None = None) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    if package_path is not None:
        module.__path__ = [package_path]
    return module


def fp8_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    return torch.clamp(amax.float() / FP8_E4M3_MAX, min=1e-12)


def quantize_to_fp8_e4m3fn(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(input.float() / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )


def raise_unsupported_sglang_quantization(*args, **kwargs):
    raise RuntimeError("This LingBotVideo SGLang MoE shim only enables FP8 W8A8 quantization")


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    use_per_token_if_dynamic: bool = False,
):
    if input.ndim != 2:
        raise ValueError(f"Expected 2D input tensor, got {input.ndim}D")
    rows = input.shape[0]
    output_rows = max(int(num_token_padding or 0), rows)
    output = torch.empty((output_rows, input.shape[1]), device=input.device, dtype=torch.float8_e4m3fn)
    input_contiguous = input.contiguous()
    if scale is None:
        if use_per_token_if_dynamic:
            scale = fp8_scale_from_amax(input_contiguous.float().abs().amax(dim=1, keepdim=True))
            if output_rows > rows:
                padded_scale = torch.ones((output_rows, 1), device=input.device, dtype=torch.float32)
                padded_scale[:rows] = scale
                scale = padded_scale
        else:
            scale = fp8_scale_from_amax(input_contiguous.float().abs().amax()).reshape(1)
    quant_scale = scale[:rows] if scale.ndim == 2 else scale
    output[:rows] = quantize_to_fp8_e4m3fn(input_contiguous, quant_scale)
    if output_rows > rows:
        output[rows:] = torch.zeros(
            (output_rows - rows, input.shape[1]), device=input.device, dtype=torch.float8_e4m3fn
        )
    return output, scale.contiguous()


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    if fuse_silu_and_mul:
        half = x.shape[-1] // 2
        x = (F.silu(x[..., :half]) * x[..., half:]).contiguous()
    if x.shape[-1] % group_size != 0:
        raise ValueError("The last dimension must be divisible by group_size")
    x_view = x.contiguous().view(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    scales = fp8_scale_from_amax(x_view.float().abs().amax(dim=-1)).clamp_min(eps)
    x_q = quantize_to_fp8_e4m3fn(x_view, scales.unsqueeze(-1)).view(x.shape)
    if column_major_scales:
        scales = scales.transpose(0, 1).contiguous()
    return x_q.contiguous(), scales.contiguous()


def sglang_silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None, *args, **kwargs):
    half = x.shape[-1] // 2
    result = F.silu(x[..., :half]) * x[..., half:]
    if out is not None:
        out.copy_(result)
        return out
    return result


def sglang_gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None, *args, **kwargs):
    half = x.shape[-1] // 2
    result = F.gelu(x[..., :half]) * x[..., half:]
    if out is not None:
        out.copy_(result)
        return out
    return result


@contextmanager
def null_sglang_config_override(config):
    yield


def install_sglang_moe_import_shims() -> None:
    try:
        spec = importlib.util.find_spec("sglang")
    except Exception:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return

    package_dir = next(iter(spec.submodule_search_locations))
    srt_dir = os.path.join(package_dir, "srt")
    layers_dir = os.path.join(srt_dir, "layers")
    moe_dir = os.path.join(layers_dir, "moe")
    moe_runner_dir = os.path.join(moe_dir, "moe_runner")
    triton_utils_dir = os.path.join(moe_runner_dir, "triton_utils")
    jit_kernel_dir = os.path.join(package_dir, "jit_kernel")

    ensure_module("sglang", package_dir)
    ensure_module("sglang.srt", srt_dir)
    ensure_module("sglang.srt.layers", layers_dir)
    ensure_module("sglang.srt.layers.moe", moe_dir)
    moe_runner_module = ensure_module("sglang.srt.layers.moe.moe_runner", moe_runner_dir)
    moe_runner_module.MoeRunnerConfig = LightSglangMoeRunnerConfig
    triton_utils_module = ensure_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils", triton_utils_dir
    )
    triton_utils_module.get_config = lambda: None
    triton_utils_module.override_config = null_sglang_config_override
    ensure_module("sglang.jit_kernel", jit_kernel_dir)
    activation_module = ensure_module("sglang.jit_kernel.activation")
    activation_module.silu_and_mul = sglang_silu_and_mul
    activation_module.gelu_and_mul = sglang_gelu_and_mul

    server_args_module = ensure_module("sglang.srt.server_args")
    server_args_module.get_global_server_args = lambda: SGLANG_MOE_SERVER_ARGS
    server_args_module.set_global_server_args_for_scheduler = lambda args: None

    batch_module = ensure_module("sglang.srt.batch_invariant_ops")
    batch_module.is_batch_invariant_mode_enabled = lambda: False

    environ_module = ensure_module("sglang.srt.environ")
    environ_module.envs = SimpleNamespace(
        SGLANG_OPT_SWIGLU_CLAMP_FUSION=SglangEnvFlag("SGLANG_OPT_SWIGLU_CLAMP_FUSION"),
        SGLANG_EXPERIMENTAL_LORA_OPTI=SglangEnvFlag("SGLANG_EXPERIMENTAL_LORA_OPTI"),
    )

    utils_module = ensure_module("sglang.srt.utils")
    utils_module.cpu_has_amx_support = lambda: False
    utils_module.get_bool_env_var = sglang_env_flag
    utils_module.is_cpu = lambda: not torch.cuda.is_available()
    utils_module.is_cuda = torch.cuda.is_available
    utils_module.is_hip = lambda: False
    utils_module.is_musa = lambda: False
    utils_module.is_xpu = lambda: False
    utils_module.use_intel_xpu_backend = lambda: False
    utils_module.get_device_name = lambda: torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    utils_module.is_sm90_supported = (
        lambda: torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
    )

    custom_op_module = ensure_module("sglang.srt.utils.custom_op")
    custom_op_module.register_custom_op = lambda *args, **kwargs: (lambda fn: fn)

    moe_utils_module = ensure_module("sglang.srt.layers.moe.utils")
    moe_utils_module.get_moe_padding_size = lambda is_aiter_moe: 0

    fp8_module = ensure_module("sglang.srt.layers.quantization.fp8_kernel")
    fp8_module.per_token_group_quant_fp8 = sglang_per_token_group_quant_fp8
    fp8_module.scaled_fp8_quant = sglang_scaled_fp8_quant
    fp8_module.sglang_per_token_group_quant_fp8 = sglang_per_token_group_quant_fp8

    int8_module = ensure_module("sglang.srt.layers.quantization.int8_kernel")
    int8_module.per_token_group_quant_int8 = raise_unsupported_sglang_quantization
    int8_module.per_token_quant_int8 = raise_unsupported_sglang_quantization
    int8_module.sglang_per_token_group_quant_int8 = raise_unsupported_sglang_quantization


try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts as sglang_fused_experts,
    )
except Exception as direct_exc:  # pragma: no cover - optional deployment dependency
    install_sglang_moe_import_shims()
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts as sglang_fused_experts,
        )
    except Exception as shim_exc:  # pragma: no cover - optional deployment dependency
        sglang_fused_experts = None
        SGLANG_MOE_IMPORT_ERROR = shim_exc
    else:
        SGLANG_MOE_IMPORT_ERROR = None
else:
    SGLANG_MOE_IMPORT_ERROR = None

try:
    from sglang.srt.server_args import (
        get_global_server_args as sglang_get_global_server_args,
        set_global_server_args_for_scheduler as sglang_set_global_server_args_for_scheduler,
    )
except Exception:  # pragma: no cover - optional deployment dependency
    sglang_get_global_server_args = None
    sglang_set_global_server_args_for_scheduler = None


def _ensure_moe_server_args_attrs(args) -> None:
    if not hasattr(args, "enable_deterministic_inference"):
        args.enable_deterministic_inference = False
    if not hasattr(args, "enable_fused_moe_sum_all_reduce"):
        args.enable_fused_moe_sum_all_reduce = False


def ensure_sglang_moe_ready() -> None:
    global _SERVER_ARGS_READY
    if sglang_fused_experts is None:
        raise RuntimeError(
            "LINGBOT_MOE_EXPERT_BACKEND=sglang_triton requires SGLang MoE runtime"
        ) from SGLANG_MOE_IMPORT_ERROR
    if _SERVER_ARGS_READY:
        return
    _ensure_moe_server_args_attrs(SGLANG_MOE_SERVER_ARGS)
    if sglang_get_global_server_args is not None:
        try:
            server_args = sglang_get_global_server_args()
        except Exception:
            server_args = SGLANG_MOE_SERVER_ARGS
            if sglang_set_global_server_args_for_scheduler is not None:
                sglang_set_global_server_args_for_scheduler(server_args)
        _ensure_moe_server_args_attrs(server_args)
    _SERVER_ARGS_READY = True
