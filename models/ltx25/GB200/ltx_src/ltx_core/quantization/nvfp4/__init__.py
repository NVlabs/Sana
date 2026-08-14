"""NVFP4 Linear via ``ltx_kernels.nvfp4`` (cast + prequant policies).
The implementation modules (``linear``, ``convert``, …) import
``ltx_kernels.nvfp4`` at top level. This package stays importable without those
kernels; the gate fires when a policy builder is called (same pattern as
``ltx_core.quantization.blockwise``).
"""

from __future__ import annotations

from ltx_core.quantization.nvfp4.types import ActScale
from ltx_core.quantization.policy import QuantizationPolicy

__all__ = [
    "ActScale",
    "build_nvfp4_cast_policy",
    "build_nvfp4_prequant_policy",
]


def _import_impl():  # noqa: ANN202 - internal helper
    try:
        from ltx_kernels import nvfp4 as nvfp4_mod  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "ltx-kernels not built; NVFP4 quantization requires the nvfp4 extension. "
            "Build on a CUDA host: TORCH_CUDA_ARCH_LIST='10.0' "
            "uv sync --group kernels (or "
            "`uv pip install -e packages/ltx-kernels --no-build-isolation`)."
        ) from e
    if reason := nvfp4_mod.unavailable_reason():
        raise RuntimeError(reason)
    # Absolute sibling imports (avoid relative from . import X / importlib).
    from ltx_core.quantization.nvfp4 import convert, fuse, prequant  # noqa: PLC0415

    return convert, prequant, fuse


def build_nvfp4_cast_policy(*, act_scale: ActScale = ActScale.FIXED_1) -> QuantizationPolicy:
    """Cast policy: FV allowlist meta-swap + load-time BF16→NVFP4 ``SDOps``.
    Raises ``RuntimeError`` if ``ltx-kernels`` nvfp4 is not usable.
    """
    convert, _prequant, fuse = _import_impl()
    return QuantizationPolicy(
        sd_ops=convert.build_convert_sd_ops(),
        module_ops=convert.get_convert_module_ops(act_scale=act_scale),
        fuse_rule=fuse.nvfp4_fuse_rule,
    )


def build_nvfp4_prequant_policy(
    checkpoint_path: str,
    *,
    act_scale: ActScale = ActScale.STATIC,
) -> QuantizationPolicy:
    """Prequant policy: discover layers from ckpt + load-time pass-through SDOps.
    Default ``ActScale.STATIC`` loads calibrated ``.input_scale`` per Linear.
    Raises ``RuntimeError`` if ``ltx-kernels`` nvfp4 is not usable.
    """
    _convert, prequant, fuse = _import_impl()
    return QuantizationPolicy(
        sd_ops=prequant.build_prequant_sd_ops(checkpoint_path, act_scale=act_scale),
        module_ops=prequant.get_prequant_swap_module_ops(checkpoint_path, act_scale=act_scale),
        fuse_rule=fuse.nvfp4_fuse_rule,
    )
