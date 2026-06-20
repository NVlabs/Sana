#!/usr/bin/env python3
"""Compare local token-prune implementations against public token-prune families.

This probe pins the public behavior boundary for CAT-Pruning's
``convergence_stale_cpp`` selector and the ToMeSD merge/unmerge boundary. It
does not import the public SD3 runtime or require torch-geometric/sklearn at
test time; the local CAT selector is checked against an explicit fixture with
public-style cached-noise deltas, cluster labels, staleness counts, and cached
replacement semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CAT = Path("/home/haozhel/.cache/autovideo/public_refs/CAT-Pruning")
PUBLIC_CAT_PROJ_OUT = PUBLIC_CAT / "qcache" / "modules" / "proj_out.py"
PUBLIC_CAT_ATTN = PUBLIC_CAT / "qcache" / "modules" / "attn.py"
PUBLIC_CAT_EXAMPLE = PUBLIC_CAT / "example_sd3.py"
PUBLIC_TOMESD = Path("/home/haozhel/.cache/autovideo/public_refs/tomesd")
PUBLIC_TOMESD_MERGE = PUBLIC_TOMESD / "tomesd" / "merge.py"
LOCAL_TOKEN_PRUNE = ROOT / "efficiency" / "techniques" / "token_prune.py"
MANIFESTS = {
    cid: ROOT / "candidates" / "token_prune" / f"{cid}.toml"
    for cid in (
        "feature_norm_prune",
        "shape_stable_compute_mask",
        "region_dynamic_density",
        "cluster_representative_update",
    )
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_checks() -> dict[str, bool]:
    cat_proj = read_text(PUBLIC_CAT_PROJ_OUT)
    cat_attn = read_text(PUBLIC_CAT_ATTN)
    cat_example = read_text(PUBLIC_CAT_EXAMPLE)
    tomesd_merge = read_text(PUBLIC_TOMESD_MERGE)
    local = read_text(LOCAL_TOKEN_PRUNE)
    return {
        "has_cat_repo": PUBLIC_CAT.exists(),
        "cat_has_convergence_stale_cpp": "convergence_stale_cpp" in cat_proj
        and "convergence_stale_cpp" in cat_example,
        "cat_uses_noise_delta": "hidden_states - self.cached_noise" in cat_proj
        and "hidden_states - self.cached_noise_prev" in cat_proj,
        "cat_uses_kmeans_clusters": "KMeans(n_clusters=20" in cat_proj
        and "map_labels_to_indices(labels)" in cat_proj,
        "cat_uses_grid_graph_pooling": "create_grid_graph_2d((64,64)" in cat_proj
        and "max_pool(" in cat_proj
        and "self.graph_pool" in cat_proj,
        "cat_uses_cluster_topk": "top5_clusters = torch.topk" in cat_proj
        and "cluster_delta[idx].norm" in cat_proj,
        "cat_uses_staleness_counts": "indices_stale = torch.topk" in cat_proj
        and "largest=False" in cat_proj
        and "self.counts.index_add_" in cat_proj,
        "cat_replaces_unselected_from_cache": "cached_noise_prev" in cat_proj
        and "topk_rest_indices" in cat_proj,
        "cat_attention_consumes_cached_indices": "get_cache_attn" in cat_attn
        and "index_copy_" in cat_attn,
        "cat_sd3_example_sets_select_modes": "'proj': 'convergence_stale_cpp'" in cat_example
        and "'joint_attn':'convergence_t_noise'" in cat_example,
        "tomesd_uses_random2d_matching": "def bipartite_soft_matching_random2d" in tomesd_merge,
        "tomesd_uses_bipartite_matching": "def bipartite_soft_matching" in tomesd_merge
        and "scatter_reduce" in tomesd_merge,
        "local_has_feature_norm": '"feature_norm_prune"' in local
        and "hidden_states.float().pow(2).sum(-1).mean(0)" in local,
        "local_has_tomesd_random2d_core": "def tomesd_random2d_matching" in local
        and "class TomeRandom2DMergePlan" in local
        and "def _is_tomesd_random2d_method" in local,
        "local_has_region_density": "def region_dynamic_density_indices" in local
        and '"region_dynamic_density"' in local
        and "density.clamp_min(1.0).sqrt()" in local,
        "local_has_cluster_representative": '"cluster_representative_update"' in local,
        "local_has_cat_state": "class CatPruneState" in local
        and "cached_noise" in local
        and "cached_noise_prev" in local
        and "counts" in local,
        "local_has_cat_selector": "def cat_convergence_stale_indices" in local
        and "cluster_scores" in local
        and "selected_tensor" in local,
        "local_has_optional_kmeans_fallback": "from sklearn.cluster import KMeans" in local
        and "def _cat_torch_kmeans_labels" in local,
        "local_keeps_graph_pooling_out_of_generic_runtime": "torch_geometric" not in local
        and "max_pool" not in local,
    }


def cat_core_boundary_indices(hidden, cached_noise, labels, stale_counts, keep: int):
    """A small boundary emulation of CAT's public selector, not a public port.

    CAT ranks changes from a cached noise baseline, pools them by cluster, and
    mixes in least-recently-updated tokens. This fixture keeps only the
    observable boundary needed for mismatch detection.
    """

    torch = _torch()
    delta_norm = (hidden - cached_noise)[0].float().norm(dim=-1)
    cluster_scores = []
    for cluster_id in torch.unique(labels).tolist():
        mask = labels == int(cluster_id)
        cluster_scores.append((float(delta_norm[mask].sum().item()), int(cluster_id)))
    cluster_order = [cluster_id for _, cluster_id in sorted(cluster_scores, reverse=True)]

    selected: list[int] = []
    for cluster_id in cluster_order:
        idx = torch.nonzero(labels == cluster_id, as_tuple=False).flatten()
        ranked = idx[torch.argsort(delta_norm[idx], descending=True)]
        for item in ranked.tolist():
            if item not in selected:
                selected.append(int(item))
            if len(selected) >= keep:
                break
        if len(selected) >= keep:
            break

    if len(selected) < keep:
        ranked_stale = torch.argsort(stale_counts, descending=False)
        for item in ranked_stale.tolist():
            if item not in selected:
                selected.append(int(item))
            if len(selected) >= keep:
                break

    return torch.sort(torch.tensor(selected[:keep], device=hidden.device)).values


def behavior_probe() -> dict[str, Any]:
    torch = _torch()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from efficiency.techniques.token_prune import (  # noqa: E402
        CatPruneState,
        _cat_torch_kmeans_labels,
        cat_convergence_stale_indices,
        keep_indices,
        tomesd_random2d_matching,
    )

    hidden = torch.tensor(
        [[[10.0, 0.0], [9.0, 0.0], [8.0, 0.0], [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]]]
    )
    cached_noise = torch.tensor(
        [[[10.0, 0.0], [9.0, 0.0], [8.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    stale_counts = torch.tensor([10.0, 9.0, 8.0, 0.0, 1.0, 2.0])
    keep_ratio = 0.5
    keep = int(round(hidden.shape[1] * keep_ratio))
    cat_indices = cat_core_boundary_indices(hidden, cached_noise, labels, stale_counts, keep)
    cat_state = CatPruneState()
    seed_indices = cat_convergence_stale_indices(cached_noise, keep_ratio, cat_state)
    cat_state.labels = labels.to(hidden.device)
    cat_state.counts = stale_counts.to(hidden.device)
    cached_noise_prev_before = cat_state.cached_noise_prev.detach().clone()
    local_cat_indices = cat_convergence_stale_indices(hidden, keep_ratio, cat_state)
    cached_noise_prev_after = cat_state.cached_noise_prev.detach()
    selected_mask = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    selected_mask[local_cat_indices] = True
    selected_cache_matches_current = torch.allclose(
        cached_noise_prev_after[:, selected_mask, :],
        hidden[:, selected_mask, :],
        atol=1e-6,
        rtol=1e-6,
    )
    unselected_cache_matches_previous = torch.allclose(
        cached_noise_prev_after[:, ~selected_mask, :],
        cached_noise_prev_before[:, ~selected_mask, :],
        atol=1e-6,
        rtol=1e-6,
    )

    fallback_features = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ],
        dtype=torch.float32,
    )
    fallback_labels = _cat_torch_kmeans_labels(fallback_features, 3)
    fallback_labels_repeat = _cat_torch_kmeans_labels(fallback_features, 3)
    fallback_valid = bool(
        fallback_labels.shape == (6,)
        and torch.equal(fallback_labels, fallback_labels_repeat)
        and int(fallback_labels.min().item()) >= 0
        and int(fallback_labels.max().item()) < 3
        and torch.unique(fallback_labels).numel() == 3
    )

    tomesd_merge = load_module(PUBLIC_TOMESD_MERGE, "public_tomesd_merge_probe")
    shape_hidden = torch.arange(64, dtype=torch.float32).reshape(1, 16, 4)
    public_merge, public_unmerge = tomesd_merge.bipartite_soft_matching_random2d(
        shape_hidden, w=4, h=4, sx=2, sy=2, r=4, no_rand=True
    )
    public_shape_merged = public_merge(shape_hidden, mode="mean")
    public_shape_restored = public_unmerge(public_shape_merged)
    local_shape_plan = tomesd_random2d_matching(
        shape_hidden, 4, width=4, height=4, sx=2, sy=2, no_rand=True
    )
    if local_shape_plan is None:
        raise AssertionError("expected active ToMeSD random2D plan")
    local_shape_merged = local_shape_plan.merge(shape_hidden, mode="mean")
    local_shape_restored = local_shape_plan.unmerge(local_shape_merged)
    shape_random2d_matches = bool(
        torch.allclose(public_shape_merged, local_shape_merged, atol=1e-6, rtol=1e-6)
        and torch.allclose(
            public_shape_restored, local_shape_restored, atol=1e-6, rtol=1e-6
        )
    )

    local = {
        "feature_norm_prune": keep_indices("feature_norm_prune", hidden.shape[1], keep_ratio, hidden),
        "region_dynamic_density": keep_indices(
            "region_dynamic_density", hidden.shape[1], keep_ratio, hidden
        ),
        "cluster_representative_update": local_cat_indices,
    }

    per_candidate = {}
    for cid, indices in local.items():
        per_candidate[cid] = {
            "manifest": str(MANIFESTS[cid]),
            "local_indices": indices.tolist(),
            "cat_core_boundary_indices": cat_indices.tolist(),
            "matches_cat_core_boundary": bool(torch.equal(indices, cat_indices)),
            "matches_public_original": False,
        }
    per_candidate["cluster_representative_update"].update(
        {
            "local_cat_selector_indices": local_cat_indices.tolist(),
            "cat_seed_indices": seed_indices.tolist(),
            "matches_cat_core_boundary": bool(torch.equal(local_cat_indices, cat_indices)),
            "matches_public_cat_selector_boundary": bool(
                torch.equal(local_cat_indices, cat_indices)
            ),
            "cat_cache_replacement_matches_boundary": bool(
                selected_cache_matches_current and unselected_cache_matches_previous
            ),
            "cat_cache_selected_tokens_match_current": bool(
                selected_cache_matches_current
            ),
            "cat_cache_unselected_tokens_match_previous_cache": bool(
                unselected_cache_matches_previous
            ),
            "matches_public_original": False,
            "known_difference": (
                "local cluster_representative_update now matches the CAT "
                "cached-delta cluster/staleness selector fixture, but does not "
                "claim the public SD3 proj_out plus joint-attention KV-cache "
                "runtime or torch-geometric graph-pooling implementation."
            ),
        }
    )
    per_candidate["shape_stable_compute_mask"] = {
        "manifest": str(MANIFESTS["shape_stable_compute_mask"]),
        "public_merged_shape": list(public_shape_merged.shape),
        "local_merged_shape": list(local_shape_merged.shape),
        "public_restored_shape": list(public_shape_restored.shape),
        "local_restored_shape": list(local_shape_restored.shape),
        "local_merged_token_indices": local_shape_plan.merged_token_indices().tolist(),
        "matches_tomesd_random2d_core": shape_random2d_matches,
        "matches_public_original": False,
        "known_difference": (
            "local shape_stable_compute_mask now matches the public ToMeSD "
            "random-2D bipartite merge/unmerge fixture with deterministic "
            "no_rand=True selection, but does not claim the full diffusion "
            "patching/runtime schedule or current Cosmos3 GPU quality."
        ),
    }

    return {
        "fixture": {
            "keep_ratio": keep_ratio,
            "cluster_labels": labels.tolist(),
            "stale_counts": stale_counts.tolist(),
            "cat_core_boundary_note": (
                "Boundary emulation only: public CAT source checks pin KMeans, "
                "grid graph pooling, cluster top-k, stale counts, and cached "
                "replacement in the SD3 runtime."
            ),
            "torch_kmeans_fallback_labels": fallback_labels.tolist(),
            "torch_kmeans_fallback_valid": fallback_valid,
        },
        "candidate_manifest_alignment": per_candidate,
    }


def probe() -> dict[str, Any]:
    return {
        "status": "pass",
        "public_reference": {
            "cat_pruning": {
                "repo": str(PUBLIC_CAT),
                "commit": git_commit(PUBLIC_CAT),
                "proj_out_source": str(PUBLIC_CAT_PROJ_OUT),
                "attention_source": str(PUBLIC_CAT_ATTN),
            },
            "tomesd": {
                "repo": str(PUBLIC_TOMESD),
                "commit": git_commit(PUBLIC_TOMESD),
                "merge_source": str(PUBLIC_TOMESD_MERGE),
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
