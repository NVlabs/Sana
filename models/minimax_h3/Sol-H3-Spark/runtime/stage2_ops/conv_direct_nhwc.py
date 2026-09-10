"""Direct-NHWC pixel shuffle retaining the native upsample function code."""
from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from types import FunctionType


NATIVE_PATTERN = "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
NHWC_PATTERN = "b (c p1 p2 p3) d h w -> b (d p1) (h p2) (w p3) c"


@contextmanager
def installed_direct_upsample_nhwc(torch_module=None):
    """Scope the changed rearrange binding to a private copy of native globals.

    Only the measured B1/noncausal/nonresidual/prefer-NHWC route is supported.
    No Tensor/module-wide rearrange patch, extra output copy, or math rewrite.
    The caller must establish actual numerical parity and speed.
    """
    if torch_module is None:
        import torch as torch_module
    efficient = import_module("ltx_core.model.video_vae.memory_efficient_decode")
    original = efficient._upsample_forward_efficient
    if not isinstance(original, FunctionType):
        raise RuntimeError("direct NHWC requires the unwrapped native upsample function")
    native_rearrange = original.__globals__["rearrange"]
    receipt = {"implementation": "native_upsample_private_globals_direct_nhwc_shuffle",
        "upstream_source": efficient.__file__, "calls": 0, "layout_changes": 0,
        "layout_changes_definition": "native pixel shuffles materialized directly in NHWC",
        "stride_examples": [], "all_outputs_channels_last_3d": True,
        "causal": False, "residual": False, "restored": False}

    def direct_rearrange(x, pattern, **axes):
        if pattern != NATIVE_PATTERN or set(axes) != {"p1", "p2", "p3"}:
            raise RuntimeError("unexpected native pixel-shuffle pattern")
        value = native_rearrange(x, NHWC_PATTERN, **axes).permute(0, 4, 1, 2, 3)
        receipt["layout_changes"] += 1
        return value

    private_globals = {**original.__globals__, "rearrange": direct_rearrange}
    native = FunctionType(original.__code__, private_globals, original.__name__,
                          original.__defaults__, original.__closure__)
    native.__kwdefaults__ = original.__kwdefaults__

    @wraps(original)
    def upsample(block, x, causal, prefer_channels_last_3d=False):
        if (len(x.shape) != 5 or x.shape[0] != 1 or causal is not False
                or block.residual is not False or prefer_channels_last_3d is not True):
            raise RuntimeError("direct NHWC requires B1, noncausal, nonresidual, prefer_channels_last_3d=True")
        shape, stride = list(x.shape), list(x.stride())
        value = native(block, x, causal, prefer_channels_last_3d)
        nhwc = value.is_contiguous(memory_format=torch_module.channels_last_3d)
        receipt["all_outputs_channels_last_3d"] &= nhwc
        if not nhwc:
            raise RuntimeError("direct pixel-shuffle output is not channels_last_3d")
        receipt["calls"] += 1
        if len(receipt["stride_examples"]) < 4:
            receipt["stride_examples"].append({"input_shape": shape, "input_stride": stride,
                "output_shape": list(value.shape), "output_stride": list(value.stride())})
        return value

    efficient._upsample_forward_efficient = upsample
    try:
        yield receipt
    finally:
        efficient._upsample_forward_efficient = original
        receipt["restored"] = True
