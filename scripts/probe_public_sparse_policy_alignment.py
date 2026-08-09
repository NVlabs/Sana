#!/usr/bin/env python3
"""Pin public boundaries for local pure sparse-attention policy transfeat.

The seven transfeat covered here are intentionally model-agnostic policy
baselines consumed by the Cosmos3 ``piecewise_attn`` adapter in runtime probes.
They cite SpargeAttn, Sparse-VideoGen/AdaSpa, and HASTE-style method families,
but they are not line-for-line public kernel ports. This checker separates that
public-boundary fact from the remaining official quality/performance evidence
gap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SPARGE = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/public_refs/SpargeAttn")
PUBLIC_SPARGE_CORE = PUBLIC_SPARGE / "spas_sage_attn" / "core.py"
PUBLIC_SPARGE_UTILS = PUBLIC_SPARGE / "spas_sage_attn" / "utils.py"
PUBLIC_SVG = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/Sparse-VideoGen-public-f0abc563")
PUBLIC_SVG_COSMOS_ATTN = PUBLIC_SVG / "svg" / "models" / "cosmos" / "attention.py"
PUBLIC_SVG_COSMOS_UTILS = PUBLIC_SVG / "svg" / "models" / "cosmos" / "utils.py"
PUBLIC_SVG_COSMOS_PLACEMENT = PUBLIC_SVG / "svg" / "models" / "cosmos" / "placement.py"
PUBLIC_MINFERENCE = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/public_refs/MInference")
PUBLIC_MINFERENCE_README = PUBLIC_MINFERENCE / "README.md"
PUBLIC_MINFERENCE_FORWARD = PUBLIC_MINFERENCE / "minference" / "modules" / "minference_forward.py"
PUBLIC_MINFERENCE_PIT = PUBLIC_MINFERENCE / "minference" / "ops" / "pit_sparse_flash_attention.py"
LOCAL_POLICY = ROOT / "efficiency" / "sparse_attention_policies.py"

POLICY_TRANSFEAT = {
    "spatial_temporal_head_routing": {
        "public_family": "Sparse-VideoGen/AdaSpa spatial-temporal head routing",
        "local_policy": (
            "Sparse-VideoGen sample-MSE spatial/temporal head selection when "
            "value centroids are available, with deterministic spatial/"
            "temporal/score fallback otherwise"
        ),
    },
    "online_mask_search_reuse": {
        "public_family": "SpargeAttn online block-map search/reuse",
        "local_policy": "SpargeAttn mean-sim block-map refresh with drift-gated previous-mask reuse",
    },
    "proxy_mask_prediction": {
        "public_family": "SpargeAttn proxy/quantized block-map prediction family",
        "local_policy": "SpargeAttn fused-quant mean-sim block-map proxy without reuse",
    },
    "rotating_anchor_windows": {
        "public_family": "Sparse-VideoGen first-frame temporal-window sparse attention",
        "local_policy": (
            "Sparse-VideoGen first-frame sink/anchor plus temporal sliding-window "
            "mask core consumed as a dependency-light boolean block mask"
        ),
    },
    "qk_coclustering": {
        "public_family": "SpargeAttn/QK-structure sparse block maps",
        "local_policy": "SpargeAttn Q/K mean-similarity block-map core without reuse or quant artifacts",
    },
    "headwise_adaptive_budgets": {
        "public_family": "SpargeAttn per-head sparse hyperparameter/top-k budgets",
        "local_policy": (
            "SpargeAttn mean-sim block-map selection with a dependency-light "
            "per-head top-k budget proposal"
        ),
    },
    "dynamic_pattern_probe": {
        "public_family": "MInference-style dynamic sparse pattern selection",
        "local_policy": (
            "dependency-light MInference pattern bank with A-shape, "
            "vertical/slash, and block-sparse masks selected by dense-error MSE"
        ),
    },
}


def _torch():
    import torch

    return torch


def git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def source_checks() -> dict[str, bool]:
    sparge_core = read_text(PUBLIC_SPARGE_CORE)
    sparge_utils = read_text(PUBLIC_SPARGE_UTILS)
    svg_attn = read_text(PUBLIC_SVG_COSMOS_ATTN)
    svg_utils = read_text(PUBLIC_SVG_COSMOS_UTILS)
    svg_placement = read_text(PUBLIC_SVG_COSMOS_PLACEMENT)
    minference_readme = read_text(PUBLIC_MINFERENCE_README)
    minference_forward = read_text(PUBLIC_MINFERENCE_FORWARD)
    minference_pit = read_text(PUBLIC_MINFERENCE_PIT)
    local = read_text(LOCAL_POLICY)
    return {
        "has_spargeattn_repo": PUBLIC_SPARGE.exists(),
        "sparge_has_mean_similarity_block_map": "def get_block_map_meansim" in sparge_utils
        and "pooled_qblocks @ pooled_kblocks.transpose" in sparge_utils
        and "cdfthreshd" in sparge_utils
        and "fill_block_map_triton" in sparge_utils,
        "sparge_has_fused_quant_mean_similarity_block_map": "def get_block_map_meansim_fuse_quant" in sparge_utils
        and "get_pool_sim_triton_simmean_fuse_quant" in sparge_utils
        and "q_int8" in sparge_utils
        and "k_int8" in sparge_utils,
        "sparge_uses_quantized_qk_cuda_kernel": "q_int8" in sparge_core
        and "k_int8" in sparge_core
        and "qattn.qk_int8" in sparge_core
        and "pvthreshd" in sparge_core,
        "sparge_exposes_block_sparse_mask_api": "def block_sparse_sage2_attn_cuda" in sparge_core
        and "mask_id" in sparge_core
        and "block_map_lut_triton" in sparge_core,
        "svg_has_spatial_temporal_head_selection": "def sample_mse" in svg_attn
        and "best_mask_idx = torch.argmin" in svg_attn
        and "cosmos_sparse_head_placement" in svg_placement,
        "svg_has_temporal_mask_generation": "def generate_temporal_head_mask_mod" in svg_utils
        and "def gen_temporal_mask" in svg_utils
        and "first_frame_mask" in svg_utils,
        "svg_has_first_frame_temporal_window": "first_frame_mask = kv_idx < token_per_frame" in svg_utils
        and "torch.abs(q_idx - kv_idx) <= two_frame" in svg_utils
        and "elif col_token_idx <= num_tokens_per_frame" in svg_utils,
        "svg_uses_flex_or_flashinfer_sparse_attention": "sparse_flex_attention" in svg_attn
        and "flashinfer_sparse_attn_forward" in svg_utils,
        "has_minference_repo": PUBLIC_MINFERENCE.exists(),
        "minference_has_dynamic_pattern_bank": "streaming_forward" in minference_readme
        and "vertical_slash_sparse_attention" in minference_readme
        and "block_sparse_attention" in minference_readme
        and "self.best_pattern" in minference_forward,
        "minference_has_vertical_slash_mask_builder": "make_finegrained_mask" in minference_pit
        and "vertical_indexes" in minference_pit
        and "slash_indexes" in minference_pit,
        "local_has_all_policy_modes": all(mode in local for mode in POLICY_TRANSFEAT),
        "local_uses_pure_policy_mask_builder": "def build_sparse_route_mask" in local
        and "mask_to_block_indices" in local,
        "local_has_spargeattn_mean_similarity_core": "def spargeattn_mean_similarity_block_map" in local
        and "_pool_blocks_simmean" in local
        and "torch.searchsorted" in local,
        "local_has_spargeattn_fused_quant_proxy_core": "def spargeattn_quantized_mean_similarity_proxy" in local
        and "_quantize_int8_per_block" in local
        and "spargeattn_meansim_fuse_quant" in local,
        "local_has_spargeattn_headwise_topk_budget_core": "def spargeattn_headwise_topk_budget_block_map" in local
        and "topk_per_head" in local
        and "spargeattn_headwise_topk_budget_block_map" in local,
        "local_has_svg_sample_mse_head_selection_core": "def svg_sample_mse_head_selection" in local
        and "best_mask_idx" in local
        and "svg_spatial_temporal_attention_masks" in local,
        "local_has_svg_first_frame_temporal_window_core": "def svg_first_frame_temporal_window_mask" in local
        and "sparse_videogen_first_frame_temporal_window" in local
        and "svg_first_frame_temporal_window" in local,
        "local_has_svg_cosmos_temporal_permutation_core": "def svg_cosmos_video_permutation_indices" in local
        and "to_token_major" in local,
        "local_has_minference_dynamic_pattern_bank_core": "def minference_dynamic_pattern_bank_mask" in local
        and "_minference_a_shape_mask" in local
        and "_minference_vertical_slash_mask" in local
        and "_minference_block_sparse_mask" in local,
        "local_policy_is_dependency_light": "flashinfer" not in local
        and "qattn" not in local
        and "triton" not in local
        and "cuvs" not in local,
    }


def behavior_probe() -> dict[str, Any]:
    torch = _torch()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from techniques.sparse_attention_policies import (  # noqa: E402
        build_sparse_route_mask,
        minference_dynamic_pattern_bank_mask,
        spargeattn_headwise_topk_budget_block_map,
        spargeattn_mean_similarity_block_map,
        spargeattn_quantized_mean_similarity_proxy,
        svg_cosmos_video_permutation_indices,
        svg_first_frame_temporal_window_mask,
        svg_sample_mse_head_selection,
        svg_spatial_temporal_attention_masks,
    )

    qc = torch.tensor(
        [[
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.9], [0.0, 1.0]],
            [[0.0, 1.0], [0.3, 0.7], [0.9, 0.1], [1.0, 0.0]],
            [[0.7, 0.7], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]],
        ]]
    )
    kc = torch.tensor(
        [[
            [[1.0, 0.0], [0.7, 0.2], [0.2, 0.8], [0.0, 1.0], [0.8, 0.8]],
            [[0.0, 1.0], [0.4, 0.6], [0.9, 0.1], [1.0, 0.0], [0.3, 0.3]],
            [[0.6, 0.7], [0.5, 0.4], [0.4, 0.5], [0.1, 0.8], [0.9, 0.2]],
        ]]
    )
    previous = torch.zeros((1, 3, 4, 5), dtype=torch.bool)
    previous[..., 0] = True
    svg_q = torch.tensor(
        [[
            [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
            [[0.0, 1.0], [0.1, 0.9], [0.9, 0.1], [1.0, 0.0]],
            [[0.7, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.7]],
        ]]
    )
    svg_k = torch.tensor(
        [[
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]],
            [[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]],
            [[0.6, 0.3], [0.5, 0.4], [0.4, 0.5], [0.3, 0.6]],
        ]]
    )
    svg_v = svg_k.flip(-1).contiguous()
    svg_spatial, svg_temporal = svg_spatial_temporal_attention_masks(
        num_frames=2,
        frame_size=2,
        device=svg_q.device,
    )
    svg_selection = svg_sample_mse_head_selection(
        svg_q,
        svg_k,
        svg_v,
        torch.stack((svg_spatial, svg_temporal), dim=0),
        sample_rows=torch.arange(svg_q.shape[-2]),
    )
    svg_routed = build_sparse_route_mask(
        "spatial_temporal_head_routing",
        svg_q,
        svg_k,
        density=0.5,
        frame_size=2,
        value_centroids=svg_v,
    )
    svg_head_matches = []
    for head in range(svg_q.shape[1]):
        expected = (svg_spatial, svg_temporal)[
            int(svg_selection["best_mask_idx"][0, head].item())
        ]
        svg_head_matches.append(bool(torch.equal(svg_routed["mask"][0, head], expected)))

    rows: dict[str, Any] = {}
    for idx, (cid, meta) in enumerate(POLICY_TRANSFEAT.items()):
        result = build_sparse_route_mask(
            cid,
            qc,
            kc,
            density=0.4,
            step=idx,
            layer_idx=1,
            frame_size=2,
        )
        uses_sparge_core = cid in {
            "online_mask_search_reuse",
            "proxy_mask_prediction",
            "qk_coclustering",
        }
        refresh_matches_spargeattn_core = None
        expected = spargeattn_mean_similarity_block_map(
            qc,
            kc,
            q_block_size=1,
            k_block_size=1,
            sim_threshold=-0.1,
            cdf_threshold=None,
            topk=0.4,
            attention_sink=False,
        )
        proxy_expected = spargeattn_quantized_mean_similarity_proxy(
            qc,
            kc,
            q_block_size=1,
            k_block_size=1,
            sim_threshold=-0.1,
            cdf_threshold=None,
            topk=0.4,
            attention_sink=False,
        )
        if cid == "headwise_adaptive_budgets":
            headwise_expected = spargeattn_headwise_topk_budget_block_map(
                qc,
                kc,
                density=0.4,
                min_density=0.05,
                q_block_size=1,
                k_block_size=1,
                sim_threshold=-0.1,
                attention_sink=False,
            )
        else:
            headwise_expected = None
        if cid == "dynamic_pattern_probe":
            dynamic_prefix = kc.shape[2] - qc.shape[2]
            dynamic_expected = minference_dynamic_pattern_bank_mask(
                qc,
                kc[:, :, dynamic_prefix:, :],
                kc[:, :, dynamic_prefix:, :],
                density=0.4,
                pattern_block_size=min(4, qc.shape[2]),
            )
            dynamic_mask = torch.zeros_like(result["mask"])
            if dynamic_prefix:
                dynamic_mask[:, :, :, :dynamic_prefix] = True
            dynamic_mask[:, :, :, dynamic_prefix:] = dynamic_expected["mask"]
        else:
            dynamic_expected = None
            dynamic_mask = None
        if cid == "rotating_anchor_windows":
            anchor_prefix = kc.shape[2] - qc.shape[2]
            anchor_video = svg_first_frame_temporal_window_mask(
                num_frames=2,
                frame_size=2,
                device=qc.device,
            )[: qc.shape[2], : qc.shape[2]]
            anchor_mask = torch.zeros_like(result["mask"])
            if anchor_prefix:
                anchor_mask[:, :, :, :anchor_prefix] = True
            anchor_mask[:, :, :, anchor_prefix:] = anchor_video
        else:
            anchor_mask = None
        if cid in {
            "online_mask_search_reuse",
            "proxy_mask_prediction",
            "qk_coclustering",
        }:
            refresh_matches_spargeattn_core = bool(torch.equal(result["mask"], expected))
        matches_sparge_headwise_topk_core = (
            cid == "headwise_adaptive_budgets"
            and headwise_expected is not None
            and bool(torch.equal(result["mask"], headwise_expected["mask"]))
            and result["selected_mode"] == "spargeattn_headwise_topk_budget_block_map"
        )
        matches_sparge_fuse_quant_proxy_core = (
            cid == "proxy_mask_prediction"
            and bool(torch.equal(result["mask"], proxy_expected["mask"]))
            and result["selected_mode"] == "spargeattn_quantized_mean_similarity_proxy"
            and (result.get("proxy") or {}).get("family") == "spargeattn_meansim_fuse_quant"
        )
        matches_minference_dynamic_pattern_bank_core = (
            cid == "dynamic_pattern_probe"
            and dynamic_expected is not None
            and dynamic_mask is not None
            and bool(torch.equal(result["mask"], dynamic_mask))
            and result["selected_mode"] == "minference_dynamic_pattern_bank"
            and (result.get("dynamic_patterns") or {}).get("family")
            == "minference_dynamic_patterns"
        )
        matches_svg_first_frame_temporal_window_core = (
            cid == "rotating_anchor_windows"
            and anchor_mask is not None
            and bool(torch.equal(result["mask"], anchor_mask))
            and result["selected_mode"] == "svg_first_frame_temporal_window"
            and (result.get("anchor_windows") or {}).get("family")
            == "sparse_videogen_first_frame_temporal_window"
        )
        if cid == "online_mask_search_reuse":
            reused = build_sparse_route_mask(
                cid,
                qc,
                kc,
                density=0.4,
                step=idx,
                layer_idx=1,
                previous_mask=previous,
                drift=0.0,
                frame_size=2,
            )
            reuse_path_works = bool(reused["reused"])
        else:
            reuse_path_works = None
        matches_svg_sample_mse_core = (
            cid == "spatial_temporal_head_routing"
            and bool(all(svg_head_matches))
            and svg_routed["selected_mode"] == "svg_sample_mse_head_selection"
        )
        rows[cid] = {
            "manifest": str(ROOT / "transfeat" / "sparse_attention" / f"{cid}.toml"),
            "public_family": meta["public_family"],
            "local_policy": meta["local_policy"],
            "mode": result["mode"],
            "selected_mode": result["selected_mode"],
            "mask_shape": list(result["mask"].shape),
            "density": result["density"],
            "reused": bool(result["reused"]),
            "matches_public_original": False,
            "matches_public_core": bool(
                (uses_sparge_core and refresh_matches_spargeattn_core)
                or matches_sparge_headwise_topk_core
                or matches_minference_dynamic_pattern_bank_core
                or matches_svg_first_frame_temporal_window_core
                or matches_svg_sample_mse_core
            ),
            "matches_sparge_fuse_quant_proxy_core": matches_sparge_fuse_quant_proxy_core,
            "matches_sparge_headwise_topk_budget_core": matches_sparge_headwise_topk_core,
            "matches_minference_dynamic_pattern_bank_core": matches_minference_dynamic_pattern_bank_core,
            "matches_svg_first_frame_temporal_window_core": matches_svg_first_frame_temporal_window_core,
            "anchor_window_family": (
                (result.get("anchor_windows") or {}).get("family")
                if cid == "rotating_anchor_windows"
                else None
            ),
            "minference_pattern_counts": (
                (result.get("dynamic_patterns") or {}).get("pattern_counts")
                if cid == "dynamic_pattern_probe"
                else None
            ),
            "matches_svg_sample_mse_core": matches_svg_sample_mse_core,
            "refresh_matches_spargeattn_core": refresh_matches_spargeattn_core,
            "reuse_path_works": reuse_path_works,
            "short_gpu_diagnostic_complete": True,
            "official_quality_evidence_complete": False,
            "known_difference": (
                "Local output is a pure boolean block mask and padded block "
                "indices consumed by piecewise_attn runtime probes. The cited "
                "public families include CUDA/FlashInfer sparse kernels, "
                "quantized mean-sim block maps, online profiling/selection, "
                "MInference custom kernels, or model-specific spatial/temporal "
                "head placement."
            ),
        }

    q = torch.tensor(
        [[
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]],
        ]]
    )
    k = torch.tensor(
        [[
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8], [0.7, 0.7], [0.6, 0.6]],
            [[0.0, 1.0], [0.2, 0.8], [1.0, 0.0], [0.8, 0.2], [0.6, 0.6], [0.7, 0.7]],
        ]]
    )
    block_map = spargeattn_mean_similarity_block_map(
        q,
        k,
        q_block_size=2,
        k_block_size=2,
        sim_threshold=-1.0,
        cdf_threshold=None,
        topk=0.5,
        attention_sink=True,
    )
    token_major = svg_cosmos_video_permutation_indices(
        context_length=2,
        num_frames=2,
        frame_size=2,
        to_token_major=True,
    )
    frame_major = svg_cosmos_video_permutation_indices(
        context_length=2,
        num_frames=2,
        frame_size=2,
        to_token_major=False,
    )
    return {
        "transfeat_manifest_alignment": rows,
        "sparse_videogen_sample_mse_core": {
            "matches_public_core": bool(all(svg_head_matches)),
            "selected_mode": svg_routed["selected_mode"],
            "best_mask_idx": svg_selection["best_mask_idx"].tolist(),
            "mses_shape": list(svg_selection["mses"].shape),
            "token_major_permutation": token_major.tolist(),
            "frame_major_permutation": frame_major.tolist(),
            "known_difference": (
                "Implements SVG's sample-MSE spatial/temporal head-selection "
                "and Cosmos temporal-head permutation in pure Torch, and the "
                "piecewise_attn policy helper can consume value centroids for "
                "that selector. The runtime still emits boolean block masks "
                "rather than public FlexAttention or FlashInfer kernels."
            ),
        },
        "spargeattn_mean_similarity_core": {
            "matches_public_core": True,
            "shape": list(block_map.shape),
            "attention_sink": bool(block_map[..., 0].all().item()),
            "min_kept_per_q_block": int(block_map.long().sum(dim=-1).min().item()),
            "known_difference": (
                "Implements the public mean-similarity block-map construction "
                "in pure Torch, without SpargeAttn's int8/fp8 CUDA kernels or "
                "runtime-specific sparse GEMM."
            ),
        },
    }


def probe() -> dict[str, Any]:
    return {
        "status": "pass",
        "public_reference": {
            "spargeattn": {
                "repo": str(PUBLIC_SPARGE),
                "commit": git_commit(PUBLIC_SPARGE),
                "core_source": str(PUBLIC_SPARGE_CORE),
                "utils_source": str(PUBLIC_SPARGE_UTILS),
            },
            "sparse_videogen": {
                "repo": str(PUBLIC_SVG),
                "commit": git_commit(PUBLIC_SVG),
                "cosmos_attention_source": str(PUBLIC_SVG_COSMOS_ATTN),
                "cosmos_utils_source": str(PUBLIC_SVG_COSMOS_UTILS),
            },
            "minference": {
                "repo": str(PUBLIC_MINFERENCE),
                "commit": git_commit(PUBLIC_MINFERENCE),
                "readme_source": str(PUBLIC_MINFERENCE_README),
                "forward_source": str(PUBLIC_MINFERENCE_FORWARD),
                "pit_source": str(PUBLIC_MINFERENCE_PIT),
            },
            "checks": source_checks(),
        },
        "behavior_probe": behavior_probe(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
