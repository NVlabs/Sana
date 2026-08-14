"""Public API for blockwise FP8/FP6 quantization.
The implementation lives in :mod:`._impl`, which imports the compiled
``ltx_kernels.blockwise`` kernels at top level. This module deliberately defers
that import so that ``ltx_core.quantization.blockwise`` remains importable
without those kernels built; the gate fires only when one of the policy builders
is actually called.
"""

from ltx_core.quantization.policy import QuantizationPolicy

__all__ = ["build_fp6_policy", "build_fp8_policy"]


def _import_impl():  # noqa: ANN202 - internal helper
    try:
        from ltx_core.quantization.blockwise import _impl  # noqa: PLC0415

        return _impl
    except ImportError as e:
        raise RuntimeError(
            "ltx-kernels not built; blockwise FP8/FP6 quantization requires it. "
            "Build it on a CUDA host with `uv sync --group kernels` (or "
            "`uv pip install -e packages/ltx-kernels --no-build-isolation`) before "
            "calling build_fp8_policy() / build_fp6_policy()."
        ) from e


def build_fp8_policy() -> QuantizationPolicy:
    """Build a blockwise FP8 quantization policy. Raises ``RuntimeError`` if ``ltx-kernels`` is not built."""
    impl = _import_impl()
    return QuantizationPolicy(
        sd_ops=impl.build_sd_ops_fp8(),
        module_ops=(impl.build_module_ops_fp8(),),
        model_configurator=impl.BlockwiseFP8LTXModelConfigurator,
        fuse_rule=impl.fuse_rule_fp8,
    )


def build_fp6_policy() -> QuantizationPolicy:
    """Build a blockwise FP6 quantization policy. Raises ``RuntimeError`` if ``ltx-kernels`` is not built."""
    impl = _import_impl()
    return QuantizationPolicy(
        sd_ops=impl.build_sd_ops_fp6(),
        module_ops=(impl.build_module_ops_fp6(),),
        model_configurator=impl.BlockwiseFP6LTXModelConfigurator,
        fuse_rule=impl.fuse_rule_fp6,
    )
