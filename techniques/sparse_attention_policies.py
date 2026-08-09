# Copyright 2025 SGLang authors
#
# Model-agnostic sparse-attention routing policies.
#
# These helpers describe which key/value blocks a sparse-attention config
# wants to keep. They intentionally do not know about Cosmos3 modules, run
# scripts, or backend kernels. A model adapter may consume the returned mask or
# fixed-width indices; until then these functions are pure algorithm evidence,
# not GPU evidence.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_SPARSE_ROUTE_POLICIES = frozenset(
    {
        "score",
        "local",
        "spatial_temporal_head_routing",
        "online_mask_search_reuse",
        "proxy_mask_prediction",
        "rotating_anchor_windows",
        "qk_coclustering",
        "headwise_adaptive_budgets",
        "dynamic_pattern_probe",
        "semantic_permutation",
    }
)


@dataclass(frozen=True)
class SparseRoutePolicy:
    mode: str
    family: str
    density: float
    block_size: int
    dense_fallback: str
    requires_runtime_mask: bool
    supports_variable_head_budget: bool = False
    supports_mask_reuse: bool = False

    def as_env(self) -> dict[str, str]:
        return {
            "SGLANG_HQ_SPARSE_ROUTE_POLICY": self.mode,
            "SGLANG_HQ_SPARSE_ROUTE_FAMILY": self.family,
            "SGLANG_HQ_SPARSE_ROUTE_DENSITY": f"{self.density:.8g}",
            "SGLANG_HQ_SPARSE_ROUTE_BLOCK_SIZE": str(self.block_size),
            "SGLANG_HQ_SPARSE_ROUTE_REQUIRES_MASK": (
                "1" if self.requires_runtime_mask else "0"
            ),
            "SGLANG_HQ_SPARSE_ROUTE_VARIABLE_HEAD_BUDGET": (
                "1" if self.supports_variable_head_budget else "0"
            ),
            "SGLANG_HQ_SPARSE_ROUTE_MASK_REUSE": (
                "1" if self.supports_mask_reuse else "0"
            ),
        }


@dataclass(frozen=True)
class SparseVideoGenSAPPlan:
    """Pure Sparse-VideoGen SAP algorithm plan, without runtime glue.

    The public Cosmos SAP path is a sequence of k-means clustering, dynamic-map
    construction, semantic permutation, sparse attention, and inverse
    permutation. This object keeps those public hyperparameters and algorithm
    stages in the model-agnostic layer; Cosmos3 metadata/GQA/text-prefix/varlen
    wrappers remain runtime-consumer glue.
    """

    route_mode: str = "semantic_permutation"
    backend: str = "sparse_video_gen_2_attn"
    svg2_num_q_centroids: int = 400
    svg2_num_k_centroids: int = 1000
    svg2_top_p_kmeans: float = 0.9
    svg2_min_kc_ratio: float = 0.1
    svg2_kmeans_iter_init: int = 50
    svg2_kmeans_iter_step: int = 2
    svg2_zero_step_kmeans_init: bool = False
    svg2_first_layers_fp: float = 0.03
    svg2_first_times_fp: float = 0.3

    @property
    def algorithm_steps(self) -> tuple[str, ...]:
        return (
            "batch_kmeans_Euclid",
            "identify_dynamic_map",
            "permute_tensor_by_labels_triton",
            "dynamic_block_sparse_fwd_flashinfer",
            "apply_inverse_permutation_triton",
        )

    def as_manifest_config(self) -> dict[str, Any]:
        return {
            "component": "transformer",
            "route_mode": self.route_mode,
            "backend": self.backend,
            "svg2_num_q_centroids": self.svg2_num_q_centroids,
            "svg2_num_k_centroids": self.svg2_num_k_centroids,
            "svg2_top_p_kmeans": self.svg2_top_p_kmeans,
            "svg2_min_kc_ratio": self.svg2_min_kc_ratio,
            "svg2_kmeans_iter_init": self.svg2_kmeans_iter_init,
            "svg2_kmeans_iter_step": self.svg2_kmeans_iter_step,
            "svg2_first_layers_fp": self.svg2_first_layers_fp,
            "svg2_first_times_fp": self.svg2_first_times_fp,
        }

    def as_env(self) -> dict[str, str]:
        return {
            "SGLANG_HQ_SVG2_ALGORITHM_FAMILY": "sparse_videogen_sap",
            "SGLANG_HQ_SVG2_ALGORITHM_STEPS": ",".join(self.algorithm_steps),
            "SGLANG_HQ_SVG2_PUBLIC_COSMOS_DEFAULTS": "1",
        }


def sparse_videogen_sap_plan(
    *,
    num_q_centroids: int = 400,
    num_k_centroids: int = 1000,
    top_p_kmeans: float = 0.9,
    min_kc_ratio: float = 0.1,
    kmeans_iter_init: int = 50,
    kmeans_iter_step: int = 2,
    zero_step_kmeans_init: bool = False,
    first_layers_fp: float = 0.03,
    first_times_fp: float = 0.3,
    route_mode: str = "semantic_permutation",
    backend: str = "sparse_video_gen_2_attn",
) -> SparseVideoGenSAPPlan:
    return SparseVideoGenSAPPlan(
        route_mode=route_mode,
        backend=backend,
        svg2_num_q_centroids=int(num_q_centroids),
        svg2_num_k_centroids=int(num_k_centroids),
        svg2_top_p_kmeans=float(top_p_kmeans),
        svg2_min_kc_ratio=float(min_kc_ratio),
        svg2_kmeans_iter_init=int(kmeans_iter_init),
        svg2_kmeans_iter_step=int(kmeans_iter_step),
        svg2_zero_step_kmeans_init=bool(zero_step_kmeans_init),
        svg2_first_layers_fp=float(first_layers_fp),
        svg2_first_times_fp=float(first_times_fp),
    )


def sparse_videogen_weighted_softmax(scores: Any, weights: Any):
    """Weighted softmax used by Sparse-VideoGen's dynamic-map selector."""

    torch = _torch()
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    denom = torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return (weighted_exp / denom).to(input_dtype)


def sparse_videogen_identify_dynamic_map(
    query_centroids: Any,
    key_centroids: Any,
    q_cluster_sizes: Any,
    k_cluster_sizes: Any,
    *,
    top_p_kmeans: float,
    min_kc_ratio: float = 0.0,
):
    """Pure Torch port of Sparse-VideoGen's SAP dynamic-map selection.

    Inputs match the public `identify_dynamic_map` helper:
    query/key centroids are [B,H,Qc,D]/[B,H,Kc,D], cluster sizes are
    [B,H,Qc]/[B,H,Kc], and the output is a boolean [B,H,Qc,Kc] map.
    """

    torch = _torch()
    if query_centroids.ndim != 4 or key_centroids.ndim != 4:
        raise ValueError("query/key centroids must be [B,H,C,D]")
    if query_centroids.shape[:2] != key_centroids.shape[:2]:
        raise ValueError("query/key centroids must share batch and head axes")
    if query_centroids.shape[-1] != key_centroids.shape[-1]:
        raise ValueError("query/key centroid feature dimensions must match")
    bsz, heads, q_centroids, dim = query_centroids.shape
    k_centroids = key_centroids.shape[2]
    if tuple(q_cluster_sizes.shape) != (bsz, heads, q_centroids):
        raise ValueError("q_cluster_sizes must be [B,H,Qc]")
    if tuple(k_cluster_sizes.shape) != (bsz, heads, k_centroids):
        raise ValueError("k_cluster_sizes must be [B,H,Kc]")

    attn_scores = torch.matmul(
        query_centroids.float(),
        key_centroids.float().transpose(-2, -1),
    ) / (float(dim) ** 0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()
    weighted_probs = sparse_videogen_weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(weighted_probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > float(top_p_kmeans)
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False
    if float(min_kc_ratio) > 0:
        preserve_length = int(float(min_kc_ratio) * k_centroids)
        remove_indices[..., :preserve_length] = False
    sorted_clusters_to_keep = ~remove_indices
    dynamic_map = torch.zeros(
        bsz,
        heads,
        q_centroids,
        k_centroids,
        dtype=torch.bool,
        device=query_centroids.device,
    )
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map


def sparse_videogen_permutation_indices(labels: Any):
    """Return the stable label-sort indices used by SAP permutation."""

    torch = _torch()
    labels = torch.as_tensor(labels)
    if labels.ndim not in {2, 3}:
        raise ValueError("labels must be [B*H,N] or [B,H,N]")
    return torch.argsort(labels, dim=-1, stable=True)


def canonical_route_mode(route_mode: str) -> str:
    mode = str(route_mode or "score").strip().lower()
    aliases = {
        "feat_score": "score",
        "taylor": "score",
        "local_exact": "local",
        "anchor_windows": "rotating_anchor_windows",
        "head_routing": "spatial_temporal_head_routing",
        "online_mask_reuse": "online_mask_search_reuse",
        "proxy_mask": "proxy_mask_prediction",
        "headwise_budget": "headwise_adaptive_budgets",
        "dynamic_pattern": "dynamic_pattern_probe",
    }
    return aliases.get(mode, mode)


def sparse_route_policy_config(
    route_mode: str,
    *,
    sparsity: float,
    block_size: int,
    dense_fallback: str = "fa",
) -> SparseRoutePolicy:
    mode = canonical_route_mode(route_mode)
    density = min(1.0, max(0.0, 1.0 - float(sparsity)))
    families = {
        "score": "piecewise_score_topk",
        "local": "structured_local_window",
        "spatial_temporal_head_routing": "svg_head_role_routing",
        "online_mask_search_reuse": "adaspa_mask_search_reuse",
        "proxy_mask_prediction": "spargeattn_proxy_prediction",
        "rotating_anchor_windows": "sparse_videogen_first_frame_temporal_window",
        "qk_coclustering": "spargeattn_qk_meansim_block_map",
        "headwise_adaptive_budgets": "spargeattn_headwise_topk_budget",
        "dynamic_pattern_probe": "minference_dynamic_patterns",
        "semantic_permutation": "svg2_semantic_aware_permutation",
    }
    return SparseRoutePolicy(
        mode=mode,
        family=families.get(mode, "unknown_sparse_route"),
        density=density,
        block_size=int(block_size),
        dense_fallback=str(dense_fallback),
        requires_runtime_mask=mode not in {"score", "local"},
        supports_variable_head_budget=mode == "headwise_adaptive_budgets",
        supports_mask_reuse=mode == "online_mask_search_reuse",
    )


def _torch():
    import torch

    return torch


def _topk_count(num_kv_blocks: int, density: float) -> int:
    return max(1, min(num_kv_blocks, int(round(num_kv_blocks * float(density)))))


def _score_blocks(qc: Any, kc: Any, k_var: Any | None, scale: float, normalize: bool):
    torch = _torch()
    q = qc.float()
    k = kc.float()
    if normalize:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scores = torch.einsum("bhik,bhjk->bhij", q, k) * float(scale)
    if k_var is not None:
        scores = scores + torch.log(k_var.float().clamp_min(1e-8)).unsqueeze(-2)
    return scores


def _head_vector(value: Any, heads: int, device: Any):
    torch = _torch()
    if value is None:
        raise ValueError("head vector value must not be None")
    if isinstance(value, (float, int)):
        return torch.full((heads,), float(value), device=device)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0 or tensor.numel() == 1:
        return torch.full((heads,), float(tensor.reshape(()).item()), device=device)
    if tensor.ndim != 1 or tensor.numel() != heads:
        raise ValueError(
            f"expected scalar or {heads} per-head values, got shape {tuple(tensor.shape)}"
        )
    return tensor


def _pool_blocks_simmean(
    x: Any,
    *,
    block_size: int,
    sim_threshold: Any,
    subtract_mean: Any | None = None,
):
    """Torch implementation of SpargeAttn's block pooling + simmean gate."""

    torch = _torch()
    x = x.float()
    if x.ndim != 4:
        raise ValueError(f"expected [B,H,N,D] tensor, got shape {tuple(x.shape)}")
    bsz, heads, seqlen, dim = x.shape
    block_size = max(1, int(block_size))
    num_blocks = (seqlen + block_size - 1) // block_size
    thresholds = _head_vector(sim_threshold, heads, x.device)
    pooled = torch.empty((bsz, heads, num_blocks, dim), dtype=x.dtype, device=x.device)
    similar = torch.empty((bsz, heads, num_blocks), dtype=torch.bool, device=x.device)
    mean = subtract_mean.float() if subtract_mean is not None else None
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(seqlen, start + block_size)
        chunk = x[:, :, start:end, :]
        if mean is not None:
            chunk = chunk - mean
        pooled[:, :, block_idx, :] = chunk.mean(dim=-2)
        normed = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        grams = torch.matmul(normed, normed.transpose(-1, -2))
        mean_sim = grams.mean(dim=(-1, -2))
        similar[:, :, block_idx] = mean_sim > thresholds.view(1, heads)
    return pooled, similar


def _causal_block_mask(
    num_q_blocks: int,
    num_k_blocks: int,
    q_block_size: int,
    k_block_size: int,
    device: Any,
):
    torch = _torch()
    q_idx = torch.arange(num_q_blocks, device=device).view(num_q_blocks, 1)
    k_idx = torch.arange(num_k_blocks, device=device).view(1, num_k_blocks)
    return k_idx < ((q_idx + 1).float() * float(q_block_size) / float(k_block_size))


def _safe_softmax(scores: Any):
    torch = _torch()
    all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    safe_scores = torch.where(all_masked, torch.zeros_like(scores), scores)
    probs = torch.softmax(safe_scores, dim=-1)
    return torch.where(all_masked, torch.zeros_like(probs), probs)


def _scatter_sorted_prefix(sorted_indices: Any, num_to_select: Any):
    torch = _torch()
    bsz, heads, num_q, num_k = sorted_indices.shape
    final_map = torch.zeros(
        (bsz, heads, num_q, num_k), dtype=torch.bool, device=sorted_indices.device
    )
    for b in range(bsz):
        for h in range(heads):
            for q_idx in range(num_q):
                keep = int(num_to_select[b, h, q_idx].item())
                keep = max(1, min(num_k, keep))
                final_map[b, h, q_idx, sorted_indices[b, h, q_idx, :keep]] = True
    return final_map


def _quantize_int8_per_block(
    x: Any,
    *,
    block_size: int,
    subtract_mean: Any | None = None,
):
    torch = _torch()
    x = x.float()
    if x.ndim != 4:
        raise ValueError(f"expected [B,H,N,D] tensor, got shape {tuple(x.shape)}")
    bsz, heads, seqlen, _ = x.shape
    block_size = max(1, int(block_size))
    num_blocks = (seqlen + block_size - 1) // block_size
    quant = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((bsz, heads, num_blocks), dtype=torch.float32, device=x.device)
    mean = subtract_mean.float() if subtract_mean is not None else None
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(seqlen, start + block_size)
        chunk = x[:, :, start:end, :]
        if mean is not None:
            chunk = chunk - mean
        scale = (chunk.abs().amax(dim=(-1, -2)) / 127.0).clamp_min(1e-6)
        scales[:, :, block_idx] = scale
        quant[:, :, start:end, :] = (
            chunk / scale.view(bsz, heads, 1, 1)
        ).round().clamp(-127, 127).to(torch.int8)
    return quant, scales


def spargeattn_mean_similarity_block_map(
    q: Any,
    k: Any,
    *,
    is_causal: bool = False,
    q_block_size: int = 128,
    k_block_size: int = 64,
    sim_threshold: float | Any = 0.1,
    cdf_threshold: float | Any | None = 0.9,
    topk: float | Any | None = None,
    attention_sink: bool = False,
    smooth_k: bool = False,
):
    """Pure Torch core of SpargeAttn's mean-similarity block-map selection.

    This mirrors the public `get_block_map_meansim*` algorithm without the
    Triton/CUDA quantization and sparse-GEMM kernels: pool Q/K token blocks,
    mark low self-similarity blocks as dense fallback, select high-probability
    K blocks by CDF or top-k over pooled QK scores, apply optional causal masking,
    and optionally keep the first K block as an attention sink.
    """

    torch = _torch()
    if (cdf_threshold is None) == (topk is None):
        raise ValueError("exactly one of cdf_threshold or topk must be set")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"expected q/k [B,H,N,D], got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q/k batch, head, and dim must match, got {tuple(q.shape)} and {tuple(k.shape)}"
        )

    heads = q.shape[1]
    sim_threshold = _head_vector(sim_threshold, heads, q.device)
    k_mean = k.float().mean(dim=-2, keepdim=True) if smooth_k else None
    pooled_q, sim_q = _pool_blocks_simmean(
        q, block_size=q_block_size, sim_threshold=sim_threshold
    )
    pooled_k, sim_k = _pool_blocks_simmean(
        k,
        block_size=k_block_size,
        sim_threshold=sim_threshold,
        subtract_mean=k_mean,
    )
    num_q = pooled_q.shape[-2]
    num_k = pooled_k.shape[-2]
    sim_k_expanded = sim_k.unsqueeze(-2).expand(-1, -1, num_q, -1)
    sim_q_expanded = sim_q.unsqueeze(-1).expand(-1, -1, -1, num_k)
    scores = torch.matmul(pooled_q, pooled_k.transpose(-1, -2)) * (
        q.shape[-1] ** -0.5
    )
    scores = scores.masked_fill(~sim_k_expanded, -torch.inf)

    if is_causal:
        causal = _causal_block_mask(
            num_q, num_k, q_block_size, k_block_size, q.device
        )
        scores = scores.masked_fill(~causal.view(1, 1, num_q, num_k), -torch.inf)
    else:
        causal = None

    probs = _safe_softmax(scores)
    sorted_score = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    if cdf_threshold is not None:
        threshold = _head_vector(cdf_threshold, heads, q.device).view(1, heads, 1, 1)
        threshold = threshold.expand(q.shape[0], -1, num_q, 1).contiguous()
        num_to_select = torch.searchsorted(cdf.contiguous(), threshold, right=True).squeeze(-1)
    else:
        topk_vec = _head_vector(topk, heads, q.device).view(1, heads, 1)
        num_to_select = (topk_vec * num_k).to(torch.int64).expand(q.shape[0], -1, num_q)

    final_map = _scatter_sorted_prefix(sorted_score.indices, num_to_select)
    final_map = final_map | (~sim_k_expanded) | (~sim_q_expanded)
    if causal is not None:
        final_map = final_map & causal.view(1, 1, num_q, num_k)
    if attention_sink and num_k:
        final_map[:, :, :, 0] = True
    return final_map


def spargeattn_quantized_mean_similarity_proxy(
    q: Any,
    k: Any,
    *,
    is_causal: bool = False,
    q_block_size: int = 128,
    k_block_size: int = 64,
    sim_threshold: float | Any = 0.1,
    cdf_threshold: float | Any | None = 0.9,
    topk: float | Any | None = None,
    attention_sink: bool = False,
    smooth_k: bool = False,
) -> dict[str, Any]:
    """Pure Torch boundary for SpargeAttn's fused mean-sim + quant path.

    Public SpargeAttn's `get_block_map_meansim_fuse_quant` produces the same
    mean-similarity block map as `get_block_map_meansim` while also preparing
    per-block int8 Q/K tensors and scales for the CUDA sparse kernel. This
    helper keeps those algorithmic artifacts dependency-light; runtime adapters
    may consume only the returned mask until a matching sparse kernel exists.
    """

    k_mean = k.float().mean(dim=-2, keepdim=True) if smooth_k else None
    mask = spargeattn_mean_similarity_block_map(
        q,
        k,
        is_causal=is_causal,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sim_threshold=sim_threshold,
        cdf_threshold=cdf_threshold,
        topk=topk,
        attention_sink=attention_sink,
        smooth_k=smooth_k,
    )
    q_int8, q_scale = _quantize_int8_per_block(q, block_size=q_block_size)
    k_int8, k_scale = _quantize_int8_per_block(
        k,
        block_size=k_block_size,
        subtract_mean=k_mean,
    )
    return {
        "mask": mask,
        "q_int8": q_int8,
        "k_int8": k_int8,
        "q_scale": q_scale,
        "k_scale": k_scale,
    }


def _topk_mask(scores: Any, keep_counts: int | Any):
    torch = _torch()
    bsz, heads, nq, nk = scores.shape
    mask = torch.zeros((bsz, heads, nq, nk), dtype=torch.bool, device=scores.device)
    if isinstance(keep_counts, int):
        keep = max(1, min(nk, keep_counts))
        idx = torch.topk(scores, k=keep, dim=-1, sorted=False).indices
        mask.scatter_(-1, idx, True)
        return mask

    for b in range(bsz):
        for h in range(heads):
            keep = max(1, min(nk, int(keep_counts[b, h].item())))
            idx = torch.topk(scores[b, h], k=keep, dim=-1, sorted=False).indices
            mask[b, h].scatter_(-1, idx, True)
    return mask


def _local_row_indices(q_idx: int, nq: int, nk: int, keep: int, offset: int, device: Any):
    torch = _torch()
    center = 0 if nq <= 1 else int(round(q_idx * (nk - 1) / max(1, nq - 1)))
    center = (center + int(offset)) % nk
    half = keep // 2
    start = center - half
    values = [(start + i) % nk for i in range(keep)]
    return torch.tensor(sorted(set(values)), device=device, dtype=torch.long)


def _local_mask(nq: int, nk: int, keep: int, device: Any, offset: int = 0):
    torch = _torch()
    mask = torch.zeros((nq, nk), dtype=torch.bool, device=device)
    for q_idx in range(nq):
        mask[q_idx, _local_row_indices(q_idx, nq, nk, keep, offset, device)] = True
    return mask


def _rotating_anchor_mask(
    nq: int,
    nk: int,
    keep: int,
    *,
    step: int,
    layer_idx: int,
    device: Any,
):
    torch = _torch()
    anchor_count = max(1, min(nk, keep // 4 or 1))
    local_keep = max(1, keep - anchor_count)
    mask = _local_mask(nq, nk, local_keep, device)
    stride = max(1, nk // anchor_count)
    offset = (int(step) + int(layer_idx)) % nk
    anchors = (torch.arange(anchor_count, device=device) * stride + offset) % nk
    mask[:, anchors.long()] = True
    return mask


def svg_first_frame_temporal_window_mask(
    *,
    num_frames: int,
    frame_size: int,
    multiplier: float = 2.0,
    device: Any | None = None,
):
    """Sparse-VideoGen first-frame anchor plus temporal window mask.

    Public SVG temporal heads keep the first frame as an attention sink/anchor
    and add a sliding temporal window. This helper keeps that mask construction
    dependency-light and separate from FlashInfer/FlexAttention runtime glue.
    """

    torch = _torch()
    num_frames = max(1, int(num_frames))
    frame_size = max(1, int(frame_size))
    video_len = num_frames * frame_size
    window = max(1, int(round(float(multiplier) * frame_size)))
    idx = torch.arange(video_len, device=device)
    mask = (idx[:, None] - idx[None, :]).abs() <= window
    mask[:, :frame_size] = True
    return mask


def _minference_a_shape_mask(
    n: int,
    *,
    init_keep: int,
    local_keep: int,
    device: Any,
    is_causal: bool = False,
):
    torch = _torch()
    n = max(1, int(n))
    init_keep = max(1, min(n, int(init_keep)))
    local_keep = max(1, min(n, int(local_keep)))
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask[:, :init_keep] = True
    for q_idx in range(n):
        if is_causal:
            start = max(0, q_idx - local_keep + 1)
            end = q_idx + 1
        else:
            half = local_keep // 2
            start = max(0, q_idx - half)
            end = min(n, start + local_keep)
            start = max(0, end - local_keep)
        mask[q_idx, start:end] = True
    if is_causal:
        arange = torch.arange(n, device=device)
        mask.logical_and_(arange[:, None] >= arange[None, :])
    return mask


def _minference_vertical_slash_mask(
    q: Any,
    k: Any,
    *,
    vertical_keep: int,
    slash_keep: int,
    last_q: int = 64,
    is_causal: bool = False,
):
    torch = _torch()
    if q.shape != k.shape:
        raise ValueError("MInference vertical/slash masks require equal q/k shapes")
    bsz, heads, n, dim = q.shape
    n = int(n)
    last = max(1, min(n, int(last_q)))
    vertical_keep = max(1, min(n, int(vertical_keep)))
    slash_keep = max(1, min(2 * n - 1, int(slash_keep)))

    q_tail = q[:, :, -last:, :].float()
    scores = torch.matmul(q_tail, k.float().transpose(-2, -1)) / (float(dim) ** 0.5)
    q_pos = torch.arange(n - last, n, device=q.device).view(last, 1)
    k_pos = torch.arange(n, device=q.device).view(1, n)
    if is_causal:
        scores = scores.masked_fill(k_pos.view(1, 1, 1, n) > q_pos.view(1, 1, last, 1), -torch.inf)
    weights = _safe_softmax(scores)

    vertical_scores = weights.sum(dim=-2)
    vertical_idx = torch.topk(vertical_scores, k=vertical_keep, dim=-1).indices
    vertical_mask = torch.zeros((bsz, heads, n, n), dtype=torch.bool, device=q.device)
    vertical_mask.scatter_(
        -1,
        vertical_idx.unsqueeze(-2).expand(bsz, heads, n, vertical_keep),
        True,
    )

    offset_index = (q_pos - k_pos + (n - 1)).long()
    diag_scores = torch.zeros((bsz, heads, 2 * n - 1), dtype=weights.dtype, device=q.device)
    diag_scores.scatter_add_(
        -1,
        offset_index.view(1, 1, last * n).expand(bsz, heads, last * n),
        weights.reshape(bsz, heads, last * n),
    )
    if is_causal:
        diag_scores[:, :, : n - 1] = -torch.inf
    slash_indices = torch.topk(diag_scores, k=slash_keep, dim=-1).indices
    slash_offsets = slash_indices - (n - 1)
    all_offsets = (
        torch.arange(n, device=q.device).view(n, 1)
        - torch.arange(n, device=q.device).view(1, n)
    )
    slash_mask = (
        all_offsets.view(1, 1, n, n, 1)
        == slash_offsets.view(bsz, heads, 1, 1, slash_keep)
    ).any(dim=-1)
    mask = vertical_mask | slash_mask
    if is_causal:
        mask.logical_and_(all_offsets.view(1, 1, n, n) >= 0)
    return mask


def _minference_block_sparse_mask(
    q: Any,
    k: Any,
    *,
    block_keep: int,
    pattern_block_size: int,
    is_causal: bool = False,
):
    torch = _torch()
    if q.shape != k.shape:
        raise ValueError("MInference block-sparse masks require equal q/k shapes")
    bsz, heads, n, dim = q.shape
    block_size = max(1, int(pattern_block_size))
    num_blocks = (n + block_size - 1) // block_size
    block_keep = max(1, min(num_blocks, int(block_keep)))
    q_pool = torch.empty((bsz, heads, num_blocks, dim), dtype=torch.float32, device=q.device)
    k_pool = torch.empty_like(q_pool)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(n, start + block_size)
        q_pool[:, :, block_idx, :] = q[:, :, start:end, :].float().mean(dim=-2)
        k_pool[:, :, block_idx, :] = k[:, :, start:end, :].float().mean(dim=-2)
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) / (float(dim) ** 0.5)
    if is_causal:
        block_ids = torch.arange(num_blocks, device=q.device)
        scores = scores.masked_fill(block_ids.view(1, 1, 1, num_blocks) > block_ids.view(1, 1, num_blocks, 1), -torch.inf)
    block_idx = torch.topk(scores, k=block_keep, dim=-1).indices
    block_map = torch.zeros((bsz, heads, num_blocks, num_blocks), dtype=torch.bool, device=q.device)
    block_map.scatter_(-1, block_idx, True)
    token_blocks = torch.arange(n, device=q.device) // block_size
    mask = torch.zeros((bsz, heads, n, n), dtype=torch.bool, device=q.device)
    for q_idx in range(n):
        for k_idx in range(n):
            mask[:, :, q_idx, k_idx] = block_map[
                :, :, token_blocks[q_idx], token_blocks[k_idx]
            ]
    if is_causal:
        arange = torch.arange(n, device=q.device)
        mask.logical_and_(arange.view(1, 1, n, 1) >= arange.view(1, 1, 1, n))
    return mask


def minference_dynamic_pattern_bank_mask(
    q: Any,
    k: Any,
    value: Any | None = None,
    *,
    density: float,
    pattern_block_size: int = 4,
    last_q: int = 64,
    is_causal: bool = False,
):
    """Pure MInference-style dynamic pattern bank and per-head selector.

    Public MInference searches/assigns heads across streaming A-shape,
    vertical/slash, and block-sparse patterns, then builds sparse indices online.
    This helper keeps the dependency-light algorithm boundary: construct those
    three pattern masks, compare each sparse reconstruction to dense attention,
    and select the lowest-error pattern per batch/head. It does not port the
    public LLM-specific CUDA kernels or causal-only runtime assumptions.
    """

    torch = _torch()
    if q.shape != k.shape:
        raise ValueError(
            f"MInference dynamic pattern bank requires equal q/k shapes, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if value is None:
        value = k
    if value.shape != q.shape:
        raise ValueError(
            f"value must match q/k shape for dynamic pattern selection, got {tuple(value.shape)}"
        )
    bsz, heads, n, dim = q.shape
    target_keep = _topk_count(n, density)
    init_keep = max(1, target_keep // 4)
    local_keep = max(1, target_keep - init_keep)
    vertical_keep = max(1, target_keep // 2)
    slash_keep = max(1, target_keep - vertical_keep)
    pattern_block_size = max(1, min(n, int(pattern_block_size)))
    block_keep = _topk_count((n + pattern_block_size - 1) // pattern_block_size, density)

    a_shape = _minference_a_shape_mask(
        n,
        init_keep=init_keep,
        local_keep=local_keep,
        device=q.device,
        is_causal=is_causal,
    ).view(1, 1, n, n).expand(bsz, heads, n, n)
    vertical_slash = _minference_vertical_slash_mask(
        q,
        k,
        vertical_keep=vertical_keep,
        slash_keep=slash_keep,
        last_q=last_q,
        is_causal=is_causal,
    )
    block_sparse = _minference_block_sparse_mask(
        q,
        k,
        block_keep=block_keep,
        pattern_block_size=pattern_block_size,
        is_causal=is_causal,
    )
    config_masks = torch.stack((a_shape, vertical_slash, block_sparse), dim=0)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (float(dim) ** 0.5)
    if is_causal:
        arange = torch.arange(n, device=q.device)
        scores = scores.masked_fill(arange.view(1, 1, 1, n) > arange.view(1, 1, n, 1), -torch.inf)
    dense_hidden = torch.matmul(_safe_softmax(scores), value.float())
    mses = torch.empty((3, bsz, heads), dtype=torch.float32, device=q.device)
    for pattern_idx, pattern_mask in enumerate(config_masks):
        sparse_scores = scores.masked_fill(~pattern_mask, -torch.inf)
        sparse_hidden = torch.matmul(_safe_softmax(sparse_scores), value.float())
        mses[pattern_idx] = (sparse_hidden - dense_hidden).pow(2).mean(dim=(2, 3))
    best = torch.argmin(mses, dim=0)
    mask = torch.zeros((bsz, heads, n, n), dtype=torch.bool, device=q.device)
    for pattern_idx in range(config_masks.shape[0]):
        mask = torch.where(
            (best == pattern_idx).view(bsz, heads, 1, 1),
            config_masks[pattern_idx],
            mask,
        )
    pattern_names = ("a_shape", "vertical_slash", "block_sparse")
    pattern_counts = {
        name: int((best == idx).sum().item())
        for idx, name in enumerate(pattern_names)
    }
    return {
        "mask": mask,
        "mses": mses,
        "best_pattern_idx": best,
        "pattern_names": pattern_names,
        "pattern_counts": pattern_counts,
        "target_keep": int(target_keep),
        "vertical_keep": int(vertical_keep),
        "slash_keep": int(slash_keep),
        "block_keep": int(block_keep),
    }


def _layout_mask(
    *,
    role: str,
    nq: int,
    nk: int,
    keep: int,
    frame_size: int,
    device: Any,
):
    torch = _torch()
    if frame_size <= 0:
        return _local_mask(nq, nk, keep, device)
    mask = torch.zeros((nq, nk), dtype=torch.bool, device=device)
    frames = max(1, (nk + frame_size - 1) // frame_size)
    for q_idx in range(nq):
        q_frame = min(frames - 1, q_idx // frame_size)
        spatial = q_idx % frame_size
        if role == "temporal":
            config = [
                frame * frame_size + spatial
                for frame in range(frames)
                if frame * frame_size + spatial < nk
            ]
            if not config:
                config = [min(nk - 1, q_idx)]
            order = sorted(config, key=lambda idx: abs((idx // frame_size) - q_frame))
            chosen = order[:keep]
        else:
            start = q_frame * frame_size
            end = min(nk, start + frame_size)
            local = _local_row_indices(
                q_idx - start,
                max(1, end - start),
                max(1, end - start),
                min(keep, max(1, end - start)),
                0,
                device,
            )
            chosen = (local + start).tolist()
        mask[q_idx, torch.tensor(chosen, device=device, dtype=torch.long)] = True
    return mask


def svg_spatial_temporal_attention_masks(
    *,
    num_frames: int,
    frame_size: int,
    multiplier: float = 2.0,
    device: Any | None = None,
):
    """Sparse-VideoGen style spatial/temporal mask pair for video tokens.

    The public Cosmos SVG path builds a frame-major local/first-frame-sink mask,
    then derives the temporal mask by reinterpreting the video tokens in
    token-major order. This helper keeps that pure mask construction separate
    from FlashInfer/FlexAttention and Cosmos3 runtime glue.
    """

    torch = _torch()
    num_frames = max(1, int(num_frames))
    frame_size = max(1, int(frame_size))
    video_len = num_frames * frame_size
    window = max(1, int(round(float(multiplier) * frame_size)))
    spatial = torch.zeros((video_len, video_len), dtype=torch.bool, device=device)
    spatial[:, :frame_size] = True
    for q_idx in range(video_len):
        start = max(0, q_idx - window + 1)
        end = min(video_len, q_idx + window)
        spatial[q_idx, start:end] = True
    temporal = (
        spatial.reshape(frame_size, num_frames, frame_size, num_frames)
        .permute(1, 0, 3, 2)
        .reshape(video_len, video_len)
        .contiguous()
    )
    return spatial, temporal


def svg_sample_mse_head_selection(
    query: Any,
    key: Any,
    value: Any,
    attention_masks: Any,
    *,
    sample_rows: Any | None = None,
    sample_mse_max_row: int | None = None,
):
    """Pure Torch Sparse-VideoGen sample-MSE head mask selector.

    Public SVG samples Q rows, computes dense attention as the reference, then
    chooses between spatial and temporal sparse masks per batch/head by minimum
    MSE. This mirrors that decision core without random sampling side effects or
    backend-specific sparse kernels.
    """

    torch = _torch()
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,H,N,D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            f"query/key/value shapes must match, got {tuple(query.shape)}, "
            f"{tuple(key.shape)}, {tuple(value.shape)}"
        )
    masks = torch.as_tensor(attention_masks, dtype=torch.bool, device=query.device)
    if masks.ndim != 3 or masks.shape[1:] != query.shape[-2:-1] + key.shape[-2:-1]:
        raise ValueError(
            "attention_masks must be [M,N,N] matching query/key sequence length"
        )

    seq_len = query.shape[-2]
    if sample_rows is None:
        max_row = seq_len if sample_mse_max_row is None else min(seq_len, int(sample_mse_max_row))
        rows = torch.arange(max_row, device=query.device)
    else:
        rows = torch.as_tensor(sample_rows, dtype=torch.long, device=query.device)
        rows = rows[(rows >= 0) & (rows < seq_len)]
    if rows.numel() == 0:
        raise ValueError("sample_rows must select at least one valid row")

    dim = query.shape[-1]
    sampled_q = query[:, :, rows, :]
    qk_scores = torch.matmul(sampled_q.float(), key.float().transpose(-2, -1)) / (
        float(dim) ** 0.5
    )
    dense_weights = torch.softmax(qk_scores, dim=-1)
    dense_hidden = torch.matmul(dense_weights, value.float())
    mses = torch.empty(
        (masks.shape[0], query.shape[0], query.shape[1]),
        dtype=query.float().dtype,
        device=query.device,
    )
    for mask_idx, mask in enumerate(masks):
        sampled_mask = mask.index_select(0, rows)
        masked_scores = qk_scores.masked_fill(~sampled_mask.view(1, 1, rows.numel(), seq_len), -torch.inf)
        weights = _safe_softmax(masked_scores)
        hidden = torch.matmul(weights, value.float())
        mses[mask_idx] = (hidden - dense_hidden).pow(2).mean(dim=(2, 3))
    return {"mses": mses, "best_mask_idx": torch.argmin(mses, dim=0)}


def svg_cosmos_video_permutation_indices(
    *,
    context_length: int,
    num_frames: int,
    frame_size: int,
    to_token_major: bool = True,
    device: Any | None = None,
):
    """Permutation used by Sparse-VideoGen Cosmos temporal-head placement."""

    torch = _torch()
    context_length = max(0, int(context_length))
    num_frames = max(1, int(num_frames))
    frame_size = max(1, int(frame_size))
    video_len = num_frames * frame_size
    if to_token_major:
        video = [
            frame * frame_size + patch
            for patch in range(frame_size)
            for frame in range(num_frames)
        ]
    else:
        video = [
            patch * num_frames + frame
            for frame in range(num_frames)
            for patch in range(frame_size)
        ]
    tail = list(range(video_len, video_len + context_length))
    return torch.tensor(video + tail, dtype=torch.long, device=device)


def _headwise_budgets(qc: Any, kc: Any, density: float, min_density: float):
    torch = _torch()
    nk = kc.shape[2]
    base = _topk_count(nk, density)
    q_energy = qc.float().pow(2).mean(dim=(2, 3))
    k_energy = kc.float().pow(2).mean(dim=(2, 3))
    energy = q_energy + k_energy
    rel = energy / energy.mean(dim=1, keepdim=True).clamp_min(1e-6)
    scaled = (float(base) * rel.clamp(0.5, 1.5)).round().long()
    min_keep = _topk_count(nk, min_density)
    return scaled.clamp(min=min_keep, max=nk)


def spargeattn_headwise_topk_budget_block_map(
    q: Any,
    k: Any,
    *,
    density: float,
    min_density: float = 0.05,
    is_causal: bool = False,
    q_block_size: int = 1,
    k_block_size: int = 1,
    sim_threshold: float | Any = -0.1,
    attention_sink: bool = False,
    smooth_k: bool = False,
):
    """SpargeAttn mean-sim block map with per-head top-k budgets.

    Public SpargeAttn exposes per-head sparse hyperparameters (`cdfthreshd`,
    `simthreshd1`, `pvthreshd`, and `topk`) after tuning. This dependency-light
    helper keeps that public block-map boundary: a per-head top-k vector is
    passed into the same mean-similarity block selection core instead of using a
    local raw-score top-k mask. The budget proposal remains model-agnostic and
    can be replaced by offline profile/tune data without changing the public
    block-map core.
    """

    nk = k.shape[2]
    budgets = _headwise_budgets(q, k, density, min_density)
    topk_per_head = budgets.float().mean(dim=0) / max(1, nk)
    topk_per_head = topk_per_head.clamp(
        min=float(min_density),
        max=1.0,
    )
    mask = spargeattn_mean_similarity_block_map(
        q,
        k,
        is_causal=is_causal,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sim_threshold=sim_threshold,
        cdf_threshold=None,
        topk=topk_per_head,
        attention_sink=attention_sink,
        smooth_k=smooth_k,
    )
    return {
        "mask": mask,
        "budgets": budgets,
        "topk_per_head": topk_per_head,
    }


def mask_to_block_indices(mask: Any):
    """Convert a boolean [B,H,NQ,NK] mask to padded int32 block indices."""

    torch = _torch()
    counts = mask.long().sum(dim=-1)
    max_keep = max(1, int(counts.max().item()))
    out = torch.zeros(
        (*mask.shape[:-1], max_keep), dtype=torch.int32, device=mask.device
    )
    for b in range(mask.shape[0]):
        for h in range(mask.shape[1]):
            for q_idx in range(mask.shape[2]):
                idx = torch.nonzero(mask[b, h, q_idx], as_tuple=False).flatten()
                if idx.numel() == 0:
                    idx = torch.zeros((1,), dtype=torch.long, device=mask.device)
                if idx.numel() < max_keep:
                    pad = idx[-1:].expand(max_keep - idx.numel())
                    idx = torch.cat([idx, pad], dim=0)
                out[b, h, q_idx] = idx[:max_keep].to(torch.int32)
    return out


def build_sparse_route_mask(
    route_mode: str,
    qc: Any,
    kc: Any,
    *,
    density: float,
    scale: float = 1.0,
    k_var: Any | None = None,
    step: int = 0,
    layer_idx: int = 0,
    previous_mask: Any | None = None,
    drift: float | None = None,
    reuse_threshold: float = 0.05,
    frame_size: int = 0,
    min_density: float = 0.05,
    value_centroids: Any | None = None,
) -> dict[str, Any]:
    """Build a pure sparse-route mask.

    Inputs use block centroids with shape [B, H, NQ, D] and [B, H, NK, D].
    The returned mask is [B, H, NQ, NK]. The companion indices are padded for
    fixed-width block-sparse kernels, but the mask is the durable algorithmic
    artifact.
    """

    torch = _torch()
    mode = canonical_route_mode(route_mode)
    if mode not in SUPPORTED_SPARSE_ROUTE_POLICIES:
        mode = "score"
    nq = qc.shape[2]
    nk = kc.shape[2]
    keep = _topk_count(nk, density)
    scores = _score_blocks(qc, kc, k_var, scale, normalize=False)
    reused = False
    budgets = torch.full((qc.shape[0], qc.shape[1]), keep, device=qc.device, dtype=torch.long)
    selected_mode = mode
    proxy_metadata = None
    dynamic_metadata = None
    anchor_metadata = None

    if mode == "online_mask_search_reuse":
        if (
            previous_mask is not None
            and tuple(previous_mask.shape) == (qc.shape[0], qc.shape[1], nq, nk)
            and float(drift if drift is not None else 0.0) <= float(reuse_threshold)
        ):
            mask = previous_mask.to(device=qc.device, dtype=torch.bool)
            reused = True
        else:
            mask = spargeattn_mean_similarity_block_map(
                qc,
                kc,
                q_block_size=1,
                k_block_size=1,
                sim_threshold=-0.1,
                cdf_threshold=None,
                topk=float(density),
                attention_sink=False,
            )
    elif mode == "proxy_mask_prediction":
        proxy = spargeattn_quantized_mean_similarity_proxy(
            qc,
            kc,
            q_block_size=1,
            k_block_size=1,
            sim_threshold=-0.1,
            cdf_threshold=None,
            topk=float(density),
            attention_sink=False,
        )
        mask = proxy["mask"]
        selected_mode = "spargeattn_quantized_mean_similarity_proxy"
        proxy_metadata = {
            "family": "spargeattn_meansim_fuse_quant",
            "q_int8_shape": list(proxy["q_int8"].shape),
            "k_int8_shape": list(proxy["k_int8"].shape),
            "q_scale_shape": list(proxy["q_scale"].shape),
            "k_scale_shape": list(proxy["k_scale"].shape),
        }
    elif mode == "qk_coclustering":
        mask = spargeattn_mean_similarity_block_map(
            qc,
            kc,
            q_block_size=1,
            k_block_size=1,
            sim_threshold=-0.1,
            cdf_threshold=None,
            topk=float(density),
            attention_sink=False,
        )
        selected_mode = "spargeattn_qk_mean_similarity_block_map"
    elif mode == "rotating_anchor_windows":
        if nk >= nq and frame_size > 0:
            frame_blocks = max(1, int(frame_size))
            num_frames = max(1, (nq + frame_blocks - 1) // frame_blocks)
            kv_prefix_blocks = nk - nq
            video = svg_first_frame_temporal_window_mask(
                num_frames=num_frames,
                frame_size=frame_blocks,
                multiplier=2.0,
                device=qc.device,
            )[:nq, :nq]
            mask = torch.zeros(
                (qc.shape[0], qc.shape[1], nq, nk),
                dtype=torch.bool,
                device=qc.device,
            )
            if kv_prefix_blocks:
                mask[:, :, :, :kv_prefix_blocks] = True
            mask[:, :, :, kv_prefix_blocks : kv_prefix_blocks + nq] = video
            selected_mode = "svg_first_frame_temporal_window"
            anchor_metadata = {
                "family": "sparse_videogen_first_frame_temporal_window",
                "num_frames": int(num_frames),
                "frame_size": int(frame_blocks),
                "kv_prefix_blocks": int(kv_prefix_blocks),
                "multiplier": 2.0,
            }
        else:
            base = _rotating_anchor_mask(
                nq,
                nk,
                keep,
                step=step,
                layer_idx=layer_idx,
                device=qc.device,
            )
            mask = base.view(1, 1, nq, nk).expand(qc.shape[0], qc.shape[1], nq, nk).clone()
            selected_mode = "rotating_anchor_windows_fallback"
    elif mode == "spatial_temporal_head_routing":
        if (
            value_centroids is not None
            and nk >= nq
            and frame_size > 0
        ):
            frame_blocks = max(1, int(frame_size))
            num_frames = max(1, (nq + frame_blocks - 1) // frame_blocks)
            kv_prefix_blocks = nk - nq
            video_kc = kc[:, :, kv_prefix_blocks : kv_prefix_blocks + nq, :]
            video_vc = value_centroids[:, :, kv_prefix_blocks : kv_prefix_blocks + nq, :]
            spatial, temporal = svg_spatial_temporal_attention_masks(
                num_frames=num_frames,
                frame_size=frame_blocks,
                device=qc.device,
            )
            masks = torch.stack((spatial[:nq, :nq], temporal[:nq, :nq]), dim=0)
            selection = svg_sample_mse_head_selection(
                qc,
                video_kc,
                video_vc,
                masks,
                sample_rows=torch.arange(nq, device=qc.device),
            )
            best = selection["best_mask_idx"]
            mask = torch.zeros(
                (qc.shape[0], qc.shape[1], nq, nk),
                dtype=torch.bool,
                device=qc.device,
            )
            if kv_prefix_blocks:
                mask[:, :, :, :kv_prefix_blocks] = True
            for b in range(qc.shape[0]):
                for h in range(qc.shape[1]):
                    mask[
                        b,
                        h,
                        :,
                        kv_prefix_blocks : kv_prefix_blocks + nq,
                    ] = masks[int(best[b, h].item())]
            selected_mode = "svg_sample_mse_head_selection"
        else:
            mask = torch.zeros((qc.shape[0], qc.shape[1], nq, nk), dtype=torch.bool, device=qc.device)
            for h in range(qc.shape[1]):
                role = "temporal" if h % 3 == 0 else "spatial" if h % 3 == 1 else "score"
                if role == "score":
                    mask[:, h] = _topk_mask(scores[:, h : h + 1], keep)[:, 0]
                else:
                    role_mask = _layout_mask(
                        role=role,
                        nq=nq,
                        nk=nk,
                        keep=keep,
                        frame_size=frame_size,
                        device=qc.device,
                    )
                    mask[:, h] = role_mask.view(1, nq, nk).expand(qc.shape[0], nq, nk)
    elif mode == "headwise_adaptive_budgets":
        headwise = spargeattn_headwise_topk_budget_block_map(
            qc,
            kc,
            density=density,
            min_density=min_density,
            q_block_size=1,
            k_block_size=1,
            sim_threshold=-0.1,
            attention_sink=False,
        )
        mask = headwise["mask"]
        budgets = headwise["budgets"]
        selected_mode = "spargeattn_headwise_topk_budget_block_map"
    elif mode == "dynamic_pattern_probe":
        if nk >= nq:
            kv_prefix_blocks = nk - nq
            video_kc = kc[:, :, kv_prefix_blocks : kv_prefix_blocks + nq, :]
            if (
                value_centroids is not None
                and tuple(value_centroids.shape[:3]) == tuple(kc.shape[:3])
            ):
                video_vc = value_centroids[
                    :, :, kv_prefix_blocks : kv_prefix_blocks + nq, :
                ]
            else:
                video_vc = video_kc
            pattern_block_size = max(1, min(4, nq))
            dynamic = minference_dynamic_pattern_bank_mask(
                qc,
                video_kc,
                video_vc,
                density=density,
                pattern_block_size=pattern_block_size,
                last_q=64,
                is_causal=False,
            )
            mask = torch.zeros(
                (qc.shape[0], qc.shape[1], nq, nk),
                dtype=torch.bool,
                device=qc.device,
            )
            if kv_prefix_blocks:
                mask[:, :, :, :kv_prefix_blocks] = True
            mask[:, :, :, kv_prefix_blocks : kv_prefix_blocks + nq] = dynamic["mask"]
            selected_mode = "minference_dynamic_pattern_bank"
            dynamic_metadata = {
                "family": "minference_dynamic_patterns",
                "pattern_names": list(dynamic["pattern_names"]),
                "pattern_counts": dynamic["pattern_counts"],
                "target_keep": dynamic["target_keep"],
                "vertical_keep": dynamic["vertical_keep"],
                "slash_keep": dynamic["slash_keep"],
                "block_keep": dynamic["block_keep"],
                "used_value_centroids": value_centroids is not None,
            }
        else:
            mask = _topk_mask(scores, keep)
            selected_mode = "score_fallback"
    elif mode == "local":
        base = _local_mask(nq, nk, keep, qc.device)
        mask = base.view(1, 1, nq, nk).expand(qc.shape[0], qc.shape[1], nq, nk).clone()
    else:
        mask = _topk_mask(scores, keep)
        selected_mode = "score"

    return {
        "mode": mode,
        "selected_mode": selected_mode,
        "mask": mask,
        "indices": mask_to_block_indices(mask),
        "budgets": budgets,
        "reused": reused,
        "density": float(mask.float().mean().item()),
        "proxy": proxy_metadata,
        "dynamic_patterns": dynamic_metadata,
        "anchor_windows": anchor_metadata,
    }
