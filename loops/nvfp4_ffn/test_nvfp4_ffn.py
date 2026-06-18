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

from efficiency import compose, get_model_spec  # noqa: E402
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
    spec = get_model_spec("Cosmos3")
    check("Cosmos3 spec registered", spec is not None and spec.name == "Cosmos3")

    plan = compose(nvfp4_items(True), spec)
    check("NVFP4FFN composes as one transform", len(plan.transforms) == 1)
    check("NVFP4FFN adds no runtime techniques", len(plan.techniques) == 0)

    env: dict[str, str] = {}
    returned = plan.apply_transforms(None, stage="stage2", env=env)
    check("apply_transforms preserves placeholder transformer", returned is None)

    expected = {
        "SGLANG_HQ_ENABLE_TE_NVFP4_FFN": "1",
        "SGLANG_LTX2_TE_NVFP4_VIDEO_FFN": "1",
        "SGLANG_HQ_NVFP4_MODULE_SCOPE": "video_ffn",
        "SGLANG_LTX2_TE_NVFP4_DISABLE_RHT": "1",
        "SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION": "0",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU": "0",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "0",
        "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU": "0",
        "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "0",
        "SGLANG_LTX2_TE_NVFP4_PAD_M_TO": "16",
        "SGLANG_HQ_NVFP4_FALLBACK_POLICY": "bf16",
    }
    for key, value in expected.items():
        check(f"{key} set", env.get(key) == value)

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
        "SGLANG_LTX2_TE_NVFP4_DISABLE_RHT": "0",
        "SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "0",
        "SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION": "0",
        "SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION": "1",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU": "1",
        "SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "1",
        "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU": "1",
        "SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE": "1",
        "SGLANG_LTX2_TE_NVFP4_PAD_M_TO": "32",
        "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND": "cudnn",
        "SGLANG_HQ_NVFP4_DENSE_LAYERS": "0-1,30-31",
        "SGLANG_HQ_NVFP4_DENSE_STEPS": "0-2,32-34",
        "SGLANG_HQ_NVFP4_FALLBACK_POLICY": "bf16_on_shape_miss",
    }
    for key, value in variant_expected.items():
        check(f"variant {key} set", variant_env.get(key) == value)

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
