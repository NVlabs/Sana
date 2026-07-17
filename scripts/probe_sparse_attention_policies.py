#!/usr/bin/env python3
"""Behavior probes for pure sparse-attention route policies.

This is algorithm-boundary evidence only. It proves the local policy layer can
construct masks, indices, per-head budgets, and reuse decisions without touching
Cosmos3 runtime glue or a GPU backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techniques.sparse_attention_policies import (  # noqa: E402
    SUPPORTED_SPARSE_ROUTE_POLICIES,
    build_sparse_route_mask,
    minference_dynamic_pattern_bank_mask,
    spargeattn_headwise_topk_budget_block_map,
    spargeattn_mean_similarity_block_map,
    spargeattn_quantized_mean_similarity_proxy,
    sparse_route_policy_config,
    sparse_videogen_sap_plan,
    sparse_videogen_identify_dynamic_map,
    sparse_videogen_permutation_indices,
    sparse_videogen_weighted_softmax,
    svg_cosmos_video_permutation_indices,
    svg_first_frame_temporal_window_mask,
    svg_sample_mse_head_selection,
    svg_spatial_temporal_attention_masks,
)


def assert_equal(name: str, got, expected) -> None:
    if got != expected:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def assert_true(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def fixtures() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    qc = torch.randn(1, 4, 6, 8)
    kc = torch.randn(1, 4, 8, 8)
    k_var = torch.linspace(0.5, 1.5, steps=8).view(1, 1, 8).expand(1, 4, 8)
    return qc, kc, k_var


def probe_mode(mode: str) -> dict[str, object]:
    qc, kc, k_var = fixtures()
    result = build_sparse_route_mask(
        mode,
        qc,
        kc,
        density=0.25,
        scale=0.5,
        k_var=k_var,
        step=1,
        layer_idx=2,
        frame_size=2,
    )
    mask = result["mask"]
    indices = result["indices"]
    assert_equal(f"{mode} mask shape", tuple(mask.shape), (1, 4, 6, 8))
    assert_equal(f"{mode} indices prefix shape", tuple(indices.shape[:3]), (1, 4, 6))
    assert_true(f"{mode} keeps at least one block per query", bool(mask.any(dim=-1).all()))
    assert_true(f"{mode} remains sparse", float(mask.float().mean().item()) < 1.0)
    if mode == "qk_coclustering":
        expected = spargeattn_mean_similarity_block_map(
            qc,
            kc,
            q_block_size=1,
            k_block_size=1,
            sim_threshold=-0.1,
            cdf_threshold=None,
            topk=0.25,
            attention_sink=False,
        )
        assert_true(
            "qk_coclustering follows SpargeAttn Q/K mean-sim block map",
            bool(torch.equal(mask, expected)),
        )
        assert_equal(
            "qk_coclustering selected mode",
            result["selected_mode"],
            "spargeattn_qk_mean_similarity_block_map",
        )
    if mode == "dynamic_pattern_probe":
        kv_prefix = kc.shape[2] - qc.shape[2]
        expected = minference_dynamic_pattern_bank_mask(
            qc,
            kc[:, :, kv_prefix:, :],
            kc[:, :, kv_prefix:, :],
            density=0.25,
            pattern_block_size=4,
        )
        expected_mask = torch.zeros_like(mask)
        expected_mask[:, :, :, :kv_prefix] = True
        expected_mask[:, :, :, kv_prefix:] = expected["mask"]
        assert_true(
            "dynamic_pattern_probe follows MInference pattern-bank selector",
            bool(torch.equal(mask, expected_mask)),
        )
        assert_equal(
            "dynamic_pattern_probe selected mode",
            result["selected_mode"],
            "minference_dynamic_pattern_bank",
        )
        assert_true(
            "dynamic_pattern_probe records pattern counts",
            bool((result.get("dynamic_patterns") or {}).get("pattern_counts")),
        )
    if mode == "rotating_anchor_windows":
        kv_prefix = kc.shape[2] - qc.shape[2]
        video = svg_first_frame_temporal_window_mask(
            num_frames=3,
            frame_size=2,
            device=qc.device,
        )[: qc.shape[2], : qc.shape[2]]
        expected_mask = torch.zeros_like(mask)
        expected_mask[:, :, :, :kv_prefix] = True
        expected_mask[:, :, :, kv_prefix:] = video
        assert_true(
            "rotating_anchor_windows follows SVG first-frame temporal-window core",
            bool(torch.equal(mask, expected_mask)),
        )
        assert_equal(
            "rotating_anchor_windows selected mode",
            result["selected_mode"],
            "svg_first_frame_temporal_window",
        )
        assert_equal(
            "rotating_anchor_windows records public-style family",
            (result.get("anchor_windows") or {}).get("family"),
            "sparse_videogen_first_frame_temporal_window",
        )
    return {
        "selected_mode": result["selected_mode"],
        "density": round(float(result["density"]), 6),
        "indices_shape": list(indices.shape),
        "row_keeps": sorted(set(mask.long().sum(dim=-1).flatten().tolist())),
        "dynamic_patterns": result.get("dynamic_patterns"),
        "anchor_windows": result.get("anchor_windows"),
    }


def probe_reuse() -> dict[str, object]:
    qc, kc, k_var = fixtures()
    first = build_sparse_route_mask(
        "online_mask_search_reuse",
        qc,
        kc,
        density=0.25,
        k_var=k_var,
        drift=1.0,
    )
    reused = build_sparse_route_mask(
        "online_mask_search_reuse",
        qc,
        kc,
        density=0.25,
        k_var=k_var,
        previous_mask=first["mask"],
        drift=0.001,
        reuse_threshold=0.05,
    )
    refreshed = build_sparse_route_mask(
        "online_mask_search_reuse",
        qc,
        kc,
        density=0.25,
        k_var=k_var,
        previous_mask=first["mask"],
        drift=0.5,
        reuse_threshold=0.05,
    )
    assert_true("low drift reuses previous mask", reused["reused"] is True)
    assert_true("high drift refreshes mask", refreshed["reused"] is False)
    assert_true(
        "reused mask is byte equal",
        bool(torch.equal(first["mask"], reused["mask"])),
    )
    return {
        "first_reused": first["reused"],
        "low_drift_reused": reused["reused"],
        "high_drift_reused": refreshed["reused"],
    }


def probe_headwise_budget() -> dict[str, object]:
    qc, kc, _ = fixtures()
    qc[:, 0] *= 4.0
    result = build_sparse_route_mask(
        "headwise_adaptive_budgets",
        qc,
        kc,
        density=0.25,
        min_density=0.125,
    )
    budgets = result["budgets"].flatten().tolist()
    assert_true("headwise budgets vary by head", len(set(budgets)) > 1)
    expected = spargeattn_headwise_topk_budget_block_map(
        qc,
        kc,
        density=0.25,
        min_density=0.125,
        q_block_size=1,
        k_block_size=1,
        sim_threshold=-0.1,
    )
    assert_equal(
        "headwise selected mode",
        result["selected_mode"],
        "spargeattn_headwise_topk_budget_block_map",
    )
    assert_true(
        "headwise route uses SpargeAttn per-head top-k block map",
        bool(torch.equal(result["mask"], expected["mask"])),
    )
    return {
        "budgets": budgets,
        "selected_mode": result["selected_mode"],
        "topk_per_head": expected["topk_per_head"].tolist(),
    }


def probe_spargeattn_mean_similarity_core() -> dict[str, object]:
    torch.manual_seed(19)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 10, 4)
    cdf_map = spargeattn_mean_similarity_block_map(
        q,
        k,
        q_block_size=2,
        k_block_size=2,
        sim_threshold=-1.0,
        cdf_threshold=0.85,
        attention_sink=True,
    )
    topk_map = spargeattn_mean_similarity_block_map(
        q,
        k,
        q_block_size=2,
        k_block_size=2,
        sim_threshold=-1.0,
        cdf_threshold=None,
        topk=0.4,
    )
    proxy = spargeattn_quantized_mean_similarity_proxy(
        q,
        k,
        q_block_size=2,
        k_block_size=2,
        sim_threshold=-1.0,
        cdf_threshold=None,
        topk=0.4,
    )
    causal_map = spargeattn_mean_similarity_block_map(
        q,
        k,
        is_causal=True,
        q_block_size=2,
        k_block_size=2,
        sim_threshold=-1.0,
        cdf_threshold=None,
        topk=1.0,
    )
    assert_equal("SpargeAttn CDF block-map shape", tuple(cdf_map.shape), (1, 2, 4, 5))
    assert_equal("SpargeAttn top-k block-map shape", tuple(topk_map.shape), (1, 2, 4, 5))
    assert_true("SpargeAttn attention sink keeps first K block", bool(cdf_map[..., 0].all()))
    assert_true(
        "SpargeAttn top-k keeps at least one K block per Q block",
        bool(topk_map.any(dim=-1).all()),
    )
    assert_true(
        "SpargeAttn fused-quant proxy preserves top-k block map",
        bool(torch.equal(proxy["mask"], topk_map)),
    )
    assert_equal("SpargeAttn proxy q int8 dtype", str(proxy["q_int8"].dtype), "torch.int8")
    assert_equal("SpargeAttn proxy k int8 dtype", str(proxy["k_int8"].dtype), "torch.int8")
    assert_equal("SpargeAttn proxy q-scale shape", tuple(proxy["q_scale"].shape), (1, 2, 4))
    assert_equal("SpargeAttn proxy k-scale shape", tuple(proxy["k_scale"].shape), (1, 2, 5))
    causal_allowed = torch.tril(torch.ones((4, 5), dtype=torch.bool))
    assert_true(
        "SpargeAttn causal block map does not keep future blocks",
        bool((causal_map & ~causal_allowed.view(1, 1, 4, 5)).any().item() is False),
    )
    return {
        "cdf_shape": list(cdf_map.shape),
        "topk_row_keeps": sorted(set(topk_map.long().sum(dim=-1).flatten().tolist())),
        "proxy_mask_matches_topk": bool(torch.equal(proxy["mask"], topk_map)),
        "proxy_q_scale_shape": list(proxy["q_scale"].shape),
        "proxy_k_scale_shape": list(proxy["k_scale"].shape),
        "attention_sink": bool(cdf_map[..., 0].all().item()),
        "causal_future_blocks": int((causal_map & ~causal_allowed.view(1, 1, 4, 5)).sum().item()),
    }


def probe_svg_sample_mse_head_selection() -> dict[str, object]:
    torch.manual_seed(23)
    num_frames = 3
    frame_size = 2
    seq_len = num_frames * frame_size
    query = torch.randn(1, 2, seq_len, 4)
    key = torch.randn(1, 2, seq_len, 4)
    value = torch.randn(1, 2, seq_len, 4)
    spatial, temporal = svg_spatial_temporal_attention_masks(
        num_frames=num_frames,
        frame_size=frame_size,
        device=query.device,
    )
    selection = svg_sample_mse_head_selection(
        query,
        key,
        value,
        torch.stack((spatial, temporal), dim=0),
        sample_rows=torch.arange(seq_len),
    )
    routed = build_sparse_route_mask(
        "spatial_temporal_head_routing",
        query,
        key,
        density=0.5,
        frame_size=frame_size,
        value_centroids=value,
    )
    best = selection["best_mask_idx"]
    assert_equal("SVG best-mask shape", tuple(best.shape), (1, 2))
    assert_equal(
        "SVG routed mask shape",
        tuple(routed["mask"].shape),
        (1, 2, seq_len, seq_len),
    )
    assert_equal(
        "SVG routed selected mode",
        routed["selected_mode"],
        "svg_sample_mse_head_selection",
    )
    for head in range(best.shape[1]):
        expected = (spatial, temporal)[int(best[0, head].item())]
        assert_true(
            f"SVG routed head {head} follows sample-MSE choice",
            bool(torch.equal(routed["mask"][0, head], expected)),
        )

    token_major = svg_cosmos_video_permutation_indices(
        context_length=2,
        num_frames=num_frames,
        frame_size=frame_size,
        to_token_major=True,
    )
    frame_major = svg_cosmos_video_permutation_indices(
        context_length=2,
        num_frames=num_frames,
        frame_size=frame_size,
        to_token_major=False,
    )
    assert_equal("SVG token-major permutation", token_major.tolist(), [0, 2, 4, 1, 3, 5, 6, 7])
    assert_equal("SVG frame-major inverse permutation", frame_major.tolist(), [0, 3, 1, 4, 2, 5, 6, 7])
    return {
        "best_mask_idx": best.tolist(),
        "mses_shape": list(selection["mses"].shape),
        "routed_density": round(float(routed["density"]), 6),
        "token_major_permutation": token_major.tolist(),
        "frame_major_permutation": frame_major.tolist(),
    }


def probe_policy_config() -> dict[str, object]:
    rows = {}
    for mode in sorted(SUPPORTED_SPARSE_ROUTE_POLICIES):
        cfg = sparse_route_policy_config(
            mode,
            sparsity=0.9,
            block_size=64,
            dense_fallback="fa",
        )
        rows[mode] = cfg.as_env()
        assert_equal(f"{mode} env policy", rows[mode]["SGLANG_HQ_SPARSE_ROUTE_POLICY"], mode)
    return rows


def probe_sparse_videogen_sap_plan() -> dict[str, object]:
    plan = sparse_videogen_sap_plan()
    config = plan.as_manifest_config()
    assert_equal("SAP route mode", config["route_mode"], "semantic_permutation")
    assert_equal("SAP backend", config["backend"], "sparse_video_gen_2_attn")
    assert_equal("SAP public q centroids", config["svg2_num_q_centroids"], 400)
    assert_equal("SAP public k centroids", config["svg2_num_k_centroids"], 1000)
    assert_equal("SAP public kmeans init iters", config["svg2_kmeans_iter_init"], 50)
    assert_equal("SAP public kmeans step iters", config["svg2_kmeans_iter_step"], 2)
    assert_true(
        "SAP algorithm stages include dynamic map",
        "identify_dynamic_map" in plan.algorithm_steps,
    )
    assert_true(
        "SAP algorithm stages include inverse permutation",
        "apply_inverse_permutation_triton" in plan.algorithm_steps,
    )
    env = plan.as_env()
    assert_equal(
        "SAP plan env family",
        env["SGLANG_HQ_SVG2_ALGORITHM_FAMILY"],
        "sparse_videogen_sap",
    )
    return {
        "config": config,
        "algorithm_steps": list(plan.algorithm_steps),
        "env": env,
    }


def probe_sparse_videogen_sap_core() -> dict[str, object]:
    torch.manual_seed(11)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 5, 4)
    q_sizes = torch.tensor([[[2, 3, 1], [1, 2, 3]]], dtype=torch.float32)
    k_sizes = torch.tensor([[[1, 4, 2, 1, 2], [2, 1, 3, 1, 1]]], dtype=torch.float32)
    dynamic_map = sparse_videogen_identify_dynamic_map(
        q,
        k,
        q_sizes,
        k_sizes,
        top_p_kmeans=0.9,
        min_kc_ratio=0.1,
    )
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (4.0 ** 0.5)
    probs = sparse_videogen_weighted_softmax(scores, k_sizes.unsqueeze(-2))
    assert_equal("SAP dynamic map shape", tuple(dynamic_map.shape), (1, 2, 3, 5))
    assert_true("SAP dynamic map keeps at least one cluster", dynamic_map.any(dim=-1).all())
    assert_true(
        "SAP weighted probabilities normalize",
        torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0])),
    )
    labels = torch.tensor([[[2, 0, 1, 0], [1, 1, 0, 2]]])
    sorted_indices = sparse_videogen_permutation_indices(labels)
    assert_equal(
        "SAP permutation sorts labels",
        sorted_indices.tolist(),
        [[[1, 3, 2, 0], [2, 0, 1, 3]]],
    )
    return {
        "dynamic_map_shape": list(dynamic_map.shape),
        "row_keep_counts": sorted(set(dynamic_map.long().sum(dim=-1).flatten().tolist())),
        "weighted_prob_row_sums": probs.sum(dim=-1).round(decimals=6).tolist(),
        "sorted_indices": sorted_indices.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    modes = [
        "score",
        "local",
        "spatial_temporal_head_routing",
        "proxy_mask_prediction",
        "rotating_anchor_windows",
        "qk_coclustering",
        "dynamic_pattern_probe",
    ]
    result = {
        "modes": {mode: probe_mode(mode) for mode in modes},
        "online_mask_search_reuse": probe_reuse(),
        "headwise_adaptive_budgets": probe_headwise_budget(),
        "spargeattn_mean_similarity_core": probe_spargeattn_mean_similarity_core(),
        "svg_sample_mse_head_selection": probe_svg_sample_mse_head_selection(),
        "sparse_videogen_sap_plan": probe_sparse_videogen_sap_plan(),
        "sparse_videogen_sap_core": probe_sparse_videogen_sap_core(),
        "policy_config": probe_policy_config(),
        "status": "pass",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
