# Copyright 2025 SGLang authors
#
# KWLFusions -- BUILD-time transform for operator-fusion policy metadata.
#
# The generic transform emits model-agnostic SGLANG_HQ_KWL_* strategy flags and
# backend policy metadata. A full LTX2 KWL replay bundle is model-specific and
# is emitted only when explicitly requested through kwl_adapter="ltx2".

from __future__ import annotations

from dataclasses import dataclass

from techniques.registry import register_transform
from techniques.technique import Seam
from techniques.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)

# the full-opt KWL bundle (matches apply_kwl in slurm_ltx23_fullopt_vs_baseline_24p.sh)
_KWL_FLAGS = (
    "SHARE_BLOCK0_SELF_ATTN",
    "SHARE_GUIDANCE_PREFIX",
    "FUSED_QK_ROPE",
    "FUSED_RMS_ADALN",
    "FUSED_ADALN",
    "FUSED_QKNORM_ROPE",
    "FUSED_DUAL_MODULATE",
    "FUSED_CA_DUAL_MODULATE",
    "FUSED_ADA_VALUES_ALL",
    "FUSED_RESIDUAL_GATE",
    "FUSED_FFN_PROJ_IN_GELU",
    "COMPILE_GATE_TO_OUT",
    "FUSED_AUDIO_QKVG",
    "ENABLE_FUSED_QKNORM_ROPE",
    "COMPILE_TILED_VAE",
)

_COMPILE_CAPTURE_REGIONS = {
    "COMPILE_GATE_TO_OUT": "gate_to_out",
    "COMPILE_TILED_VAE": "tiled_vae",
}


@dataclass(frozen=True)
class KWLBackendSelectionPlan:
    """Model-agnostic backend policy, not a backend kernel implementation."""

    component: str
    preferred_backend: str
    fallback_backend: str = "fa"
    policy_name: str = ""
    public_reference_families: tuple[str, ...] = ("FlashAttention", "CUTLASS")

    @property
    def policy_id(self) -> str:
        return self.policy_name or f"{self.component}_{self.preferred_backend}"

    def as_env(self) -> dict[str, str]:
        return {
            "SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS": (
                f"{self.component}={self.preferred_backend}"
            ),
            "SGLANG_HQ_KWL_BACKEND_SELECTION_POLICY": self.policy_id,
            "SGLANG_HQ_KWL_BACKEND_SELECTION_FALLBACK": self.fallback_backend,
            "SGLANG_HQ_KWL_BACKEND_SELECTION_PUBLIC_FAMILIES": ",".join(
                self.public_reference_families
            ),
            "SGLANG_HQ_KWL_BACKEND_SELECTION_BOUNDARY": "policy_not_kernel_port",
        }


@dataclass(frozen=True)
class KWLCompileCapturePlan:
    """Model-agnostic compile/capture-region policy with eager fallback."""

    flags: tuple[str, ...]
    fallback: str = "eager"
    public_reference_families: tuple[str, ...] = (
        "CUDA graph",
        "CUTLASS",
        "TransformerEngine",
    )

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(
            _COMPILE_CAPTURE_REGIONS[flag]
            for flag in self.flags
            if flag in _COMPILE_CAPTURE_REGIONS
        )

    def as_env(self) -> dict[str, str]:
        if not self.regions:
            return {}
        return {
            "SGLANG_HQ_KWL_COMPILE_CAPTURE_POLICY": "shape_stable_regions",
            "SGLANG_HQ_KWL_COMPILE_CAPTURE_REGIONS": ",".join(self.regions),
            "SGLANG_HQ_KWL_COMPILE_CAPTURE_FALLBACK": self.fallback,
            "SGLANG_HQ_KWL_COMPILE_CAPTURE_PUBLIC_FAMILIES": ",".join(
                self.public_reference_families
            ),
            "SGLANG_HQ_KWL_COMPILE_CAPTURE_BOUNDARY": "policy_not_kernel_port",
        }


def kwl_backend_selection_plan(
    *,
    component: str,
    preferred_backend: str,
    fallback_backend: str = "fa",
    policy_name: str = "",
) -> KWLBackendSelectionPlan:
    return KWLBackendSelectionPlan(
        component=str(component),
        preferred_backend=str(preferred_backend),
        fallback_backend=str(fallback_backend or "fa"),
        policy_name=str(policy_name or ""),
    )


def kwl_compile_capture_plan(flags: tuple[str, ...]) -> KWLCompileCapturePlan:
    return KWLCompileCapturePlan(flags=tuple(str(flag) for flag in flags))


@register_transform("kwl_fusions")
class KWLFusions(ModelTransform):
    """Enable the KWL operator-fusion bundle. ``flags`` overrides the default
    full-opt bundle (a subset of _KWL_FLAGS) for ablations."""

    name = "kwl_fusions"
    phase = TransformPhase.BUILD
    writes = frozenset({Seam.KERNEL_FUSION})  # SHARED -> never a false conflict

    def __init__(
        self,
        flags: tuple[str, ...] | None = None,
        attention_backend_component: str = "",
        attention_backend: str = "",
        attention_backend_fallback: str = "fa",
        backend_policy_name: str = "",
        kwl_adapter: str = "",
    ):
        self.kwl_adapter = str(kwl_adapter or "")
        self.flags = (
            tuple(flags)
            if flags is not None
            else _KWL_FLAGS
            if self.kwl_adapter == "ltx2"
            else ()
        )
        self.attention_backend_component = attention_backend_component
        self.attention_backend = attention_backend
        self.attention_backend_fallback = attention_backend_fallback
        self.backend_policy_name = backend_policy_name
        self.backend_selection_plan = (
            kwl_backend_selection_plan(
                component=self.attention_backend_component,
                preferred_backend=self.attention_backend,
                fallback_backend=self.attention_backend_fallback,
                policy_name=self.backend_policy_name,
            )
            if self.attention_backend_component and self.attention_backend
            else None
        )
        self.compile_capture_plan = kwl_compile_capture_plan(self.flags)

    def set_env(self, ctx: TransformContext) -> None:
        e = ctx.env
        e["SGLANG_HQ_VARIANT"] = "kwl"
        if self.kwl_adapter:
            e["SGLANG_HQ_KWL_ADAPTER"] = self.kwl_adapter
        for f in _KWL_FLAGS:
            e[f"SGLANG_HQ_KWL_{f}"] = "0"
        for f in self.flags:
            e[f"SGLANG_HQ_KWL_{f}"] = "1"
        if self.backend_selection_plan is not None:
            e.update(self.backend_selection_plan.as_env())
        e.update(self.compile_capture_plan.as_env())
