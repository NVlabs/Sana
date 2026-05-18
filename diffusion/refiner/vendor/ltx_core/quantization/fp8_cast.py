import torch

from diffusion.refiner.vendor.ltx_core.loader.module_ops import ModuleOps
from diffusion.refiner.vendor.ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from diffusion.refiner.vendor.ltx_core.model.transformer.model import LTXModel

BLOCK_SIZE = 1024


def _fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    # Lazy import triton - only available on CUDA platforms
    import triton  # noqa: PLC0415

    from diffusion.refiner.vendor.ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    """
    if not with_stochastic_rounding:
        return weight.to(dtype)
    return _fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


class Fp8CastLinear(torch.nn.Linear):
    """nn.Linear storing weights in fp8, upcasting to input dtype during forward.
    Used via __class__ reassignment (not subclassing) so existing weight tensors
    are preserved in-place. Class-level forward is required for torch.compile
    compatibility — instance-level closure monkey-patches cause graph breaks.
    """

    _with_stochastic_rounding: bool
    _seed: int

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002, type: ignore[override]
        w_up = _upcast_and_round(self.weight, input.dtype, self._with_stochastic_rounding, self._seed)
        b_up = (
            _upcast_and_round(self.bias, input.dtype, self._with_stochastic_rounding, self._seed)
            if self.bias is not None
            else None
        )
        return torch.nn.functional.linear(input, w_up, b_up)


def _replace_fwd_with_upcast(layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0) -> None:
    """
    Intended to be applied via __class__ reassignment to existing nn.Linear
    instances so that their parameter and buffer tensors are preserved in-place,
    avoiding re-instantiation. Forward remains defined at the class level, which
    is required for torch.compile compatibility — instance-level closure
    monkey-patches cause graph breaks.
    """
    layer.__class__ = Fp8CastLinear
    layer._with_stochastic_rounding = with_stochastic_rounding
    layer._seed = seed


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the model's Linear layers to forward
    with upcast and optional stochastic rounding.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear)):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


TRANSFORMER_LINEAR_DOWNCAST_MAP = (
    SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.0.proj.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.0.proj.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.2.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.2.bias", operation=_naive_weight_or_bias_downcast
    )
)

UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: _amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    ModuleOps for upcasting the model's float8_e4m3fn weights and biases to the bfloat16 dtype
    and applying stochastic rounding during linear forward.
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=lambda model: _amend_forward_with_upcast(model, True, seed),
        )
