from pathlib import Path

import safetensors
import torch

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.kernels import TRITON_AVAILABLE
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.quantization.policy import QuantizationPolicy

BLOCK_SIZE = 1024


def fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused_add_round_launch requires Triton, which is not available on this platform. "
            "Callers should gate on ltx_core.loader.kernels.TRITON_AVAILABLE and use a "
            "deterministic-rounding fallback instead."
        )
    import triton  # noqa: PLC0415

    from ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

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
    Stochastic rounding is implemented via a Triton kernel. When Triton is not
    available (e.g., on Windows), this falls back to deterministic (nearest)
    rounding via ``weight.to(dtype)``.
    """
    if not with_stochastic_rounding or not TRITON_AVAILABLE or weight.device.type != "cuda":
        return weight.to(dtype)
    return fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


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
    instances. Forward remains defined at the class level, which is required for
    torch.compile compatibility — instance-level closure monkey-patches cause
    graph breaks.
    Also retypes ``weight`` and ``bias`` to fp8 so the meta param dtype matches
    the post-load tensor dtype (sd_ops downcasts checkpoint bf16 -> fp8 at load).
    Block streaming relies on this to derive pool buffer layout from the meta
    model without an eager checkpoint read.
    """
    layer.__class__ = Fp8CastLinear
    layer._with_stochastic_rounding = with_stochastic_rounding
    layer._seed = seed
    layer.weight = torch.nn.Parameter(
        torch.empty(layer.weight.shape, dtype=torch.float8_e4m3fn, device=layer.weight.device),
        requires_grad=layer.weight.requires_grad,
    )
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(
            torch.empty(layer.bias.shape, dtype=torch.float8_e4m3fn, device=layer.bias.device),
            requires_grad=layer.bias.requires_grad,
        )


# Module-name suffixes for the Linears that participate in fp8 cast. Used by
# both the upcast matcher and the sd_ops downcast map so the two cannot drift.
# - ``.to_q`` / ``.to_k`` / ``.to_v`` / ``.to_out.0`` have a leading dot so they
#   only match the attention Linears at ``...attnN.to_q`` etc.
# - ``ff.net.0.proj`` / ``ff.net.2`` are intentionally **dotless** so they match
#   both video FF (``...ff.net.0.proj``) and audio FF (``...audio_ff.net.0.proj``).
_FP8_CAST_KEY_PREFIX = "transformer_blocks."
_FP8_CAST_LINEAR_SUFFIXES: tuple[str, ...] = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _is_fp8_cast_linear(module_name: str) -> bool:
    """Return True if *module_name* names a Linear that should be fp8-cast."""
    if _FP8_CAST_KEY_PREFIX not in module_name:
        return False
    return any(module_name.endswith(suffix) for suffix in _FP8_CAST_LINEAR_SUFFIXES)


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the fp8-cast Linear layers (per
    :data:`_FP8_CAST_LINEAR_SUFFIXES`) to forward with upcast and optional
    stochastic rounding.
    Only the Linears whose weights are downcast by :data:`TRANSFORMER_LINEAR_DOWNCAST_MAP`
    are retyped. Linears outside that subset (e.g. ``to_gate_logits``) are left as
    plain ``nn.Linear`` so the meta-model param dtype matches the loaded checkpoint
    dtype.
    """
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and _is_fp8_cast_linear(name):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


def _build_transformer_linear_downcast_map() -> SDOps:
    """Build the sd_ops downcast map from the same suffix registry as the matcher."""
    ops = SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    for suffix in _FP8_CAST_LINEAR_SUFFIXES:
        ops = ops.with_kv_operation(
            key_prefix=_FP8_CAST_KEY_PREFIX,
            key_suffix=suffix + ".weight",
            operation=_naive_weight_or_bias_downcast,
        ).with_kv_operation(
            key_prefix=_FP8_CAST_KEY_PREFIX,
            key_suffix=suffix + ".bias",
            operation=_naive_weight_or_bias_downcast,
        )
    return ops


TRANSFORMER_LINEAR_DOWNCAST_MAP = _build_transformer_linear_downcast_map()

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


def fuse_cast_fp8_weight(
    delta_bf16: torch.Tensor,
    weight_fp8: torch.Tensor,
) -> torch.Tensor:
    """Return ``(delta_bf16 + dequantize(weight_fp8)).to(weight_fp8.dtype)``.
    CUDA with Triton uses stochastic rounding via the fused kernel; otherwise
    falls back to a deterministic bf16 add. ``delta_bf16`` is the bf16
    accumulator and is mutated in place.
    """
    if delta_bf16.dtype != torch.bfloat16:
        raise ValueError(f"delta_bf16 must be bfloat16, got {delta_bf16.dtype}")
    if str(weight_fp8.device).startswith("cuda") and TRITON_AVAILABLE:
        fused_add_round_launch(delta_bf16, weight_fp8, seed=0)
    else:
        delta_bf16.add_(weight_fp8.to(dtype=torch.bfloat16))
    return delta_bf16.to(dtype=weight_fp8.dtype)


def _fp8_cast_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Cast the dequantized FP8 weight + BF16 deltas back to ``weight.dtype``
    (FP8) via the fused-add-round kernel on CUDA.
    Only a subset of linears are FP8-downcast (see ``TRANSFORMER_LINEAR_DOWNCAST_MAP``);
    LoRAs may also target layers left in BF16 (e.g. audio ``add_q/k/v_proj``, cross-modal
    projections). For those, fall back to a plain BF16 fuse.
    """
    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return bf16_fuse_rule(key, weight, deltas, model_sd)
    return {key: fuse_cast_fp8_weight(deltas, weight)}


fp8_cast_fuse_rule = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_fp8_cast_fuse)


# Raw safetensors storage prefix shared by every diffusion-transformer
# parameter (and every prequant `*_scale` sibling). Verified against
# ltx-2.3-22b-{dev,distilled}-fp8.safetensors: 2924/2924 and 2992/2992 of
# the scale keys start with this exact prefix.
_RAW_DIFFUSION_MODEL_PREFIX = "model.diffusion_model."


def _read_scales(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Return ``{post_rename_param_key: scale_tensor}`` for every prequant
    ``*_scale`` sibling in *checkpoint_path*.
    Keys are returned in the post-rename form the loader will pass to the
    sd-op (e.g. ``transformer_blocks.0.attn1.to_q.weight``) -- the raw
    ``model.diffusion_model.`` prefix and the ``_scale`` suffix are both
    stripped. Catches both ``.weight_scale`` and ``.bias_scale``; the
    latter is absent in the current LTX-2.3 prequant checkpoints but
    accepted for forward compatibility.
    """
    out: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as h:
        raw_keys = h.keys()
        for k in raw_keys:
            if not k.endswith("_scale"):
                continue
            if not k.startswith(_RAW_DIFFUSION_MODEL_PREFIX):
                raise ValueError(
                    f"Scale key {k!r} does not start with the expected raw prefix {_RAW_DIFFUSION_MODEL_PREFIX!r}"
                )
            param_key = k.removeprefix(_RAW_DIFFUSION_MODEL_PREFIX).removesuffix("_scale")
            out[param_key] = h.get_tensor(k)
    return out


def _build_prequant_fold_sd_ops(scales: dict[str, torch.Tensor]) -> SDOps:
    """Build sd-ops that fold prequant ``*_scale`` siblings into their parent
    tensor at load time.
    *scales* is keyed by the **post-rename** param key (e.g.
    ``transformer_blocks.0.attn1.to_q.weight``); see :func:`_read_scales`.
    Four ``with_kv_operation`` entries (symmetric for ``.weight`` and ``.bias``):
    * ``.weight`` / ``.bias`` -> if a sibling scale exists in *scales*, fold;
      then delegate to ``TRANSFORMER_LINEAR_DOWNCAST_MAP`` (downcast covered
      Linears, pass everything else through). Without a scale, delegate
      directly.
    * ``.weight_scale`` / ``.bias_scale`` -> drop (the scale is consumed by
      the fold). Raises if the scale key doesn't correspond to a known
      entry in *scales* -- that means the file shipped a scale we didn't
      pre-register, which would silently desync the fold.
    """

    def _on_param(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        scale = scales.get(param_key)
        if scale is None:
            return TRANSFORMER_LINEAR_DOWNCAST_MAP.apply_to_key_value(param_key, value)
        scale = scale.to(device=value.device)
        if scale.ndim != 0:
            raise ValueError(f"Unsupported scale shape {tuple(scale.shape)} for {param_key}")
        bf16 = (value.to(torch.float32) * scale).to(torch.bfloat16)
        # Delegate the final fp8-vs-bf16 decision to the downcast map: Linears
        # outside the fp8 subset (e.g. to_gate_logits) stay bf16 to match the
        # plain nn.Linear that the upcast matcher leaves untouched.
        return TRANSFORMER_LINEAR_DOWNCAST_MAP.apply_to_key_value(param_key, bf16)

    def _drop_scale(scale_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        param_key = scale_key.removesuffix("_scale")
        if param_key not in scales:
            raise ValueError(
                f"Scale key {scale_key!r} has no matching entry in the prequant scales dict; "
                f"_read_scales and the loader's rename map have drifted"
            )
        return []

    # Register the drop ops first so the dict-membership sanity check is the
    # earliest sd-op that can fire on a scale key -- we crash on a stray scale
    # before any silently mismatched fold has a chance to land in the state
    # dict. Registration order is irrelevant for correctness (no overlap
    # between matchers) but communicates intent.
    return (
        SDOps("FP8_CAST_PREQUANT_AWARE")
        .with_kv_operation(key_suffix=".weight_scale", operation=_drop_scale)
        .with_kv_operation(key_suffix=".bias_scale", operation=_drop_scale)
        .with_kv_operation(key_suffix=".weight", operation=_on_param)
        .with_kv_operation(key_suffix=".bias", operation=_on_param)
    )


def build_policy(checkpoint_path: str | Path) -> QuantizationPolicy:
    """FP8 casting with upcasting during inference.
    *checkpoint_path* is required (mirroring ``fp8_scaled_mm.build_policy``).
    For prequantized fp8 checkpoints, sibling ``*_scale`` tensors (weight or
    bias) are folded into the parent at load time.
    """
    scales = _read_scales(checkpoint_path)
    return QuantizationPolicy(
        sd_ops=_build_prequant_fold_sd_ops(scales),
        module_ops=(UPCAST_DURING_INFERENCE,),
        fuse_rule=fp8_cast_fuse_rule,
    )
