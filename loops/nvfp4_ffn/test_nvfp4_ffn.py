#!/usr/bin/env python3
"""Independent NVFP4 FFN transform test for the autovideo loop.

This is CPU-only. It verifies the efficiency transform contract and does not
import TransformerEngine or run NVFP4 GEMMs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from efficiency import Capability, ModelSpec, compose  # noqa: E402
from efficiency.nvfp4_profile import (  # noqa: E402
    format_int_ranges,
    select_profiled_nvfp4_layers,
)
from efficiency.transforms.nvfp4_ffn import NVFP4FFN  # noqa: E402


ok = 0
fail = 0


def check(name: str, condition: bool) -> None:
    global ok, fail
    if condition:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


def nvfp4_items(enabled: bool):
    return [NVFP4FFN()] if enabled else []


def main() -> int:
    print("[nvfp4_ffn] compose + transform env")
    spec = ModelSpec(
        name="NVFP4Fixture",
        capabilities=frozenset({Capability.SUPPORTS_NVFP4_LINEAR}),
        seq_dim=1,
    )
    check("fixture spec has NVFP4 capability", spec.has(Capability.SUPPORTS_NVFP4_LINEAR))

    plan = compose(nvfp4_items(True), spec)
    check("NVFP4FFN composes as one transform", len(plan.transforms) == 1)
    check("NVFP4FFN adds no runtime techniques", len(plan.techniques) == 0)

    env: dict[str, str] = {}
    returned = plan.apply_transforms(None, stage="stage2", env=env)
    check("apply_transforms preserves placeholder transformer", returned is None)

    expected = {
        "SGLANG_HQ_ENABLE_TE_NVFP4_FFN": "1",
        "SGLANG_HQ_NVFP4_MODULE_SCOPE": "video_ffn",
        "SGLANG_HQ_NVFP4_DISABLE_RHT": "1",
        "SGLANG_HQ_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "SGLANG_HQ_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION": "0",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU": "0",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "0",
        "SGLANG_HQ_NVFP4_PAD_M_TO": "16",
        "SGLANG_HQ_NVFP4_FALLBACK_POLICY": "bf16",
    }
    for key, value in expected.items():
        check(f"{key} set", env.get(key) == value)
    check(
        "generic NVFP4 transform does not emit LTX2 adapter env by default",
        not any(key.startswith("SGLANG_LTX2_TE_NVFP4_") for key in env),
    )

    variant_env: dict[str, str] = {}
    variant_plan = compose(
        [
            NVFP4FFN(
                module_scope="profiled_ffn_and_attention",
                disable_rht=False,
                disable_stochastic_rounding=False,
                disable_2d_quantization=False,
                row_scaled_activation=True,
                fused_proj_in_gelu=True,
                fused_proj_out_bias_gate=True,
                pad_m_to=32,
                fp4_gemm_backend="cudnn",
                dense_layers="0-1,30-31",
                dense_steps="0-2,32-34",
                fallback_policy="bf16_on_shape_miss",
            )
        ],
        spec,
    )
    variant_plan.apply_transforms(None, stage="stage2", env=variant_env)
    variant_expected = {
        "SGLANG_HQ_NVFP4_MODULE_SCOPE": "profiled_ffn_and_attention",
        "SGLANG_HQ_NVFP4_DISABLE_RHT": "0",
        "SGLANG_HQ_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "0",
        "SGLANG_HQ_NVFP4_DISABLE_2D_QUANTIZATION": "0",
        "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION": "1",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU": "1",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "1",
        "SGLANG_HQ_NVFP4_PAD_M_TO": "32",
        "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND": "cudnn",
        "SGLANG_HQ_NVFP4_DENSE_LAYERS": "0-1,30-31",
        "SGLANG_HQ_NVFP4_DENSE_STEPS": "0-2,32-34",
        "SGLANG_HQ_NVFP4_FALLBACK_POLICY": "bf16_on_shape_miss",
    }
    for key, value in variant_expected.items():
        check(f"variant {key} set", variant_env.get(key) == value)
    check(
        "variant generic NVFP4 transform does not emit LTX2 adapter env",
        not any(key.startswith("SGLANG_LTX2_TE_NVFP4_") for key in variant_env),
    )

    selection = select_profiled_nvfp4_layers(
        "0-1:0.05,2-29:1.0,30-31:0.05",
        keep_ratio=0.875,
    )
    check(
        "profile selector chooses hot middle layers",
        format_int_ranges(selection.profiled_layers) == "2-29",
    )
    check(
        "profile selector derives cold dense guards",
        format_int_ranges(selection.dense_layers) == "0-1,30-31",
    )

    profiled_env: dict[str, str] = {}
    compose(
        [
            NVFP4FFN(
                module_scope="profiled_hot_ffn",
                profile_layer_scores="0-1:0.05,2-29:1.0,30-31:0.05",
                profile_keep_ratio=0.875,
                fallback_policy="bf16_on_profile_miss",
            )
        ],
        spec,
    ).apply_transforms(None, stage="stage2", env=profiled_env)
    check(
        "profiled transform emits selector-derived profiled layers",
        profiled_env.get("SGLANG_HQ_NVFP4_PROFILED_LAYERS") == "2-29",
    )
    check(
        "profiled transform emits selector-derived dense guards",
        profiled_env.get("SGLANG_HQ_NVFP4_DENSE_LAYERS") == "0-1,30-31",
    )
    check(
        "profiled transform records model-agnostic profile source",
        profiled_env.get("SGLANG_HQ_NVFP4_PROFILE_SOURCE")
        == "manifest_layer_scores",
    )

    ltx2_env: dict[str, str] = {}
    compose(
        [
            NVFP4FFN(
                disable_rht=False,
                fused_proj_in_gelu=True,
                fused_proj_out_bias_gate=True,
                pad_m_to=32,
                te_adapter="ltx2",
            )
        ],
        spec,
    ).apply_transforms(None, stage="stage2", env=ltx2_env)
    ltx2_expected = {
        "SGLANG_LTX2_TE_NVFP4_VIDEO_FFN": "1",
        "SGLANG_LTX2_TE_NVFP4_DISABLE_RHT": "0",
        "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU": "1",
        "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "1",
        "SGLANG_LTX2_TE_NVFP4_PAD_M_TO": "32",
    }
    for key, value in ltx2_expected.items():
        check(f"explicit LTX2 adapter {key} set", ltx2_env.get(key) == value)

    env_off: dict[str, str] = {}
    compose(nvfp4_items(False), spec).apply_transforms(
        None, stage="stage2", env=env_off
    )
    check(
        "no-fp4 variant: primary NVFP4 env NOT set",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FFN" not in env_off,
    )
    check(
        "no-fp4 variant: LTX TE NVFP4 env NOT set",
        "SGLANG_LTX2_TE_NVFP4_VIDEO_FFN" not in env_off,
    )

    print(f"\n=== {ok} passed, {fail} failed ===")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
