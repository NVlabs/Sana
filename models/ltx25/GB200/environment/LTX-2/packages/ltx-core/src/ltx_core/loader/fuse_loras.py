from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict


class LoraProduct(NamedTuple):
    """A LoRA's ``A``, ``B`` factors and its strength scalar."""

    a: torch.Tensor
    b: torch.Tensor
    strength: float


#: Signature for a fuse callable used by :class:`FuseRule`.
#:
#: Args:
#:     key: The state-dict key being fused (e.g. ``"...layers.0.attn.q.weight"``).
#:     weight: The current value at ``key`` from ``model_sd``, on the fusion device.
#:     deltas: The pre-aggregated LoRA delta for ``key``, in ``aggregation_dtype``.
#:     model_sd: The full state dict, for rules that need companion keys
#:         (e.g. an existing ``.weight_scale``).
#:
#: Returns a dict of state-dict keys to overwrite -- at minimum ``{key: new_weight}``,
#: plus any companion keys (e.g. an updated ``.weight_scale``) the policy needs to
#: keep in sync.
FuseFn = Callable[[str, torch.Tensor, torch.Tensor, StateDict], dict[str, torch.Tensor]]


@dataclass(frozen=True)
class FuseRule:
    """Fuse an aggregated LoRA delta into one weight key.
    Each policy supplies its own rule (see ``QuantizationPolicy.fuse_rule``);
    ``fuse_lora_weights`` is policy-agnostic boilerplate around it.
    Attributes:
        aggregation_dtype: Dtype callers must pre-aggregate LoRA deltas in
            before invoking the rule.
        fuse_fn: Callable that applies the pre-aggregated deltas to the weight
            (and any companion keys) and returns a dict of keys to overwrite —
            at minimum ``{key: new_weight}``, plus any companion keys (e.g. an
            updated ``.weight_scale`` for scaled-FP8 layouts) the policy needs
            to keep in sync.
    """

    aggregation_dtype: torch.dtype
    fuse_fn: FuseFn

    def __call__(
        self,
        key: str,
        weight: torch.Tensor,
        deltas: torch.Tensor,
        model_sd: StateDict,
    ) -> dict[str, torch.Tensor]:
        return self.fuse_fn(key, weight, deltas, model_sd)


def _bf16_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,  # noqa: ARG001
) -> dict[str, torch.Tensor]:
    deltas.add_(weight)
    return {key: deltas.to(dtype=weight.dtype)}


bf16_fuse_rule = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_bf16_fuse)


def device_fuse_rule(target_device: torch.device, base_rule: FuseRule) -> FuseRule:
    """Return the fuse rule to use when fusing onto *target_device*.
    On MPS, swap the rule's aggregation dtype to fp32: the LoRA ``B@A`` then runs
    on the GPU (far faster than fusing on CPU) and fp32 sidesteps the bf16-on-MPS
    numerical unreliability that would otherwise force the slow CPU path. The
    rule's ``fuse_fn`` still casts the fused result back to the weight dtype.
    CUDA/CPU keep *base_rule* unchanged.
    """
    if target_device.type == "mps":
        return replace(base_rule, aggregation_dtype=torch.float32)
    return base_rule


def _fusion_device(target_device: torch.device) -> torch.device:
    """Device to run the fusion on: the target's own accelerator (CUDA/MPS), else
    CUDA when present (accelerating a CPU-resident fuse), else CPU. The caller
    moves the fused result back to the weight's device afterwards.
    """
    if target_device.type in ("cuda", "mps"):
        return target_device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def aggregate_lora_products(
    products: Iterable[LoraProduct],
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Accumulate ``sum((B * strength) @ A)`` across :class:`LoraProduct` items.
    The first product materializes a freshly-allocated aggregator via
    ``torch.matmul(B * strength, A).to(dtype)`` -- preserving the
    ``(B * strength) @ A`` rounding pattern. Subsequent products use
    ``addmm_`` to avoid allocating the full intermediate delta.
    Returns the aggregator, or ``None`` if ``products`` was empty.
    """
    aggregated: torch.Tensor | None = None
    for product in products:
        if aggregated is None:
            aggregated = torch.matmul(product.b * product.strength, product.a).to(dtype=dtype)
        else:
            aggregated.addmm_(product.b, product.a, alpha=product.strength)
    return aggregated


def fuse_lora_weights(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    fuse_rule: FuseRule = bf16_fuse_rule,
    preserve_input_device: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, fused_tensor)`` for each weight modified by at least one LoRA.
    The fusion math is delegated to ``fuse_rule``.
    Output dtypes are the rule's responsibility.
    When ``preserve_input_device`` is False, fused tensors are yielded on the device
    used for fusion; caller is responsible for moving them to their final
    destination.
    """
    rule = device_fuse_rule(model_sd.device, fuse_rule)
    fusion_device = _fusion_device(model_sd.device)
    for key in _affected_weight_keys(lora_sd_and_strengths):
        original_weight = model_sd.sd.get(key)
        if original_weight is None:
            continue

        products = _products_for_sd_key(lora_sd_and_strengths, key, rule.aggregation_dtype, fusion_device)
        deltas = aggregate_lora_products(products, rule.aggregation_dtype)
        if deltas is None:
            continue

        original_device = original_weight.device
        weight = original_weight.to(device=fusion_device)

        fused = rule(key, weight, deltas, model_sd)

        for k, v in fused.items():
            yield k, v.to(device=original_device) if preserve_input_device else v


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    fuse_rule: FuseRule = bf16_fuse_rule,
    destination_sd: StateDict | None = None,
    preserve_input_device: bool = True,
) -> StateDict:
    """Fuse LoRAs into ``model_sd`` and place the results in ``destination_sd``.
    When ``destination_sd`` is provided, the fused tensors are placed directly into it.
    When omitted, a fresh SD is allocated; untouched keys may alias ``model_sd``.
    Pass ``preserve_input_device=False`` to leave fused tensors on the fusion device (e.g. when
    ``model_sd`` is a CPU-retained clean cache and the caller will H2D untouched keys later).
    """
    fused_iter = fuse_lora_weights(
        model_sd,
        lora_sd_and_strengths,
        fuse_rule=fuse_rule,
        preserve_input_device=preserve_input_device,
    )
    if destination_sd is not None:
        for key, fused in fused_iter:
            destination_sd.sd[key] = fused
        return destination_sd

    fused = dict(fused_iter)
    # Untouched keys may alias ``model_sd`` (e.g. pinned retained cache); do not clone them.
    sd = {k: fused.get(k, v) for k, v in model_sd.sd.items()}
    return StateDict(sd, model_sd.device, model_sd.size, model_sd.dtype)


def _affected_weight_keys(lora_sd_and_strengths: list[LoraStateDictWithStrength]) -> set[str]:
    """Return the set of ``.weight`` keys touched by at least one LoRA in the list."""
    suffix = ".lora_A.weight"
    return {k[: -len(suffix)] + ".weight" for lsd, _ in lora_sd_and_strengths for k in lsd.sd if k.endswith(suffix)}


def _products_for_sd_key(
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    key: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[LoraProduct]:
    """Yield :class:`LoraProduct` items matching *key* across state-dict-backed LoRAs."""
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"
    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        a = lsd.sd[key_a].to(device=device, dtype=dtype, non_blocking=True)
        b = lsd.sd[key_b].to(device=device, dtype=dtype, non_blocking=True)
        yield LoraProduct(a, b, coef)
