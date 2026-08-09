# Copyright 2025 SGLang authors
#
# TokenPrune -- the generic, model-agnostic token-pruning technique.
#
# This is the SINGLE source of truth for the scoring/selection logic shared by
# the LTX2 stage-1/stage-2 prune. The model-specific bits (which token span is
# prunable, and per-rank-local selection under SP) live in the ModelSpec, not
# here.
#
# Method: at active steps, score each prunable token, keep the top-K =
# round(N * keep_ratio) by score, run the transformer blocks on ONLY those K
# tokens (gather in before_blocks), then scatter the K-token result back to the
# full N and fill the dropped tokens with a compensation hidden state -- the
# previous step's ('prev') or zero. keep_ratio >= 1 is byte-identical baseline.

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from techniques.registry import register_technique
    from techniques.schedule import Schedule, as_schedule
    from techniques.technique import (
        Capability,
        Phase,
        Seam,
        Technique,
        TechniqueContext,
    )
except ModuleNotFoundError:  # pragma: no cover - runtime mirror import path
    from sglang.multimodal_gen.runtime.efficiency.registry import register_technique
    from sglang.multimodal_gen.runtime.efficiency.schedule import Schedule, as_schedule
    from sglang.multimodal_gen.runtime.efficiency.technique import (
        Capability,
        Phase,
        Seam,
        Technique,
        TechniqueContext,
    )


def _torch():
    import torch

    return torch


def _uniform_indices(num_tokens: int, keep: int, device):
    """Deterministic ascending uniform subset: idx[i] = floor(i * N / K)."""
    torch = _torch()
    arange = torch.arange(keep, device=device, dtype=torch.long)
    return ((arange * num_tokens) // keep).clamp_(max=num_tokens - 1)


def _is_tome_method(method: str) -> bool:
    return method in ("tome", "tome_merge_restore", "merge_restore")


def _is_tomesd_random2d_method(method: str) -> bool:
    return method in (
        "shape_stable",
        "shape_stable_compute_mask",
        "tomesd_random2d",
        "tomesd_random2d_merge_restore",
    )


def _is_cat_method(method: str) -> bool:
    return method in (
        "cat",
        "cat_prune",
        "cat_convergence_stale_cpp",
        "cluster_representative_update",
    )


@dataclass
class CatPruneState:
    """State for CAT-Pruning's convergence/staleness selector.

    CAT's public SD3 implementation keeps an initial noise/cache tensor,
    selects changed clusters, mixes in stale tokens, and reuses cached values for
    unselected tokens. This object stores that model-agnostic control state; the
    model decides where the selected token set is consumed.
    """

    cached_noise: object = None
    cached_noise_prev: object = None
    counts: object = None
    cached_indices: object = None
    labels: object = None
    cluster_visits: object = None
    calls: int = 0


def _cat_positional_features(token_features):
    torch = _torch()
    feats = token_features.float()
    num_tokens = feats.shape[0]
    device = feats.device
    idx = torch.arange(num_tokens, device=device, dtype=feats.dtype)
    width = max(1, int(round(num_tokens**0.5)))
    y = (idx // width) / max(1, (num_tokens + width - 1) // width - 1)
    x = (idx % width) / max(1, width - 1)
    summary = torch.stack(
        [
            feats.mean(dim=-1),
            feats.norm(dim=-1),
            feats.abs().amax(dim=-1),
            feats.float().var(dim=-1, unbiased=False),
        ],
        dim=-1,
    )
    return torch.cat([0.05 * summary, y[:, None], x[:, None]], dim=-1)


def _cat_torch_kmeans_labels(features, num_clusters: int, iters: int = 8):
    torch = _torch()
    num_tokens = features.shape[0]
    device = features.device
    num_clusters = max(1, min(int(num_clusters), num_tokens))
    if num_clusters >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    init = _uniform_indices(num_tokens, num_clusters, device)
    centers = features.index_select(0, init).float().clone()
    labels = torch.zeros(num_tokens, device=device, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(features.float(), centers)
        labels = dist.argmin(dim=-1)
        new_centers = centers.clone()
        for cid in range(num_clusters):
            mask = labels == cid
            if bool(mask.any()):
                new_centers[cid] = features[mask].float().mean(dim=0)
        if bool(torch.equal(new_centers, centers)):
            break
        centers = new_centers
    return labels


def _cat_kmeans_labels(features, num_clusters: int):
    torch = _torch()
    num_clusters = max(1, min(int(num_clusters), features.shape[0]))
    try:
        from sklearn.cluster import KMeans  # type: ignore

        labels = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(
            features.detach().cpu().float().numpy()
        ).labels_
        return torch.as_tensor(labels, device=features.device, dtype=torch.long)
    except Exception:
        return _cat_torch_kmeans_labels(features, num_clusters)


def _cat_append_unique(selected: list[int], transfeat, limit: int) -> None:
    seen = set(selected)
    for item in transfeat.detach().flatten().tolist():
        value = int(item)
        if value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= limit:
            break


def cat_convergence_stale_indices(
    hidden_states,
    keep_ratio: float,
    state: CatPruneState,
    *,
    max_clusters: int = 20,
    batch_index: int = 1,
):
    """Select token indices using CAT-Pruning's cluster/staleness boundary.

    This is a dependency-light controller for the public CAT
    ``convergence_stale_cpp`` selector. When sklearn is installed it uses
    public-style KMeans with ``random_state=0``; otherwise it falls back to a
    deterministic PyTorch KMeans. Graph-pooling and SD3 joint-attention cache
    updates remain model-specific and are not claimed here.
    """

    torch = _torch()
    num_tokens = hidden_states.shape[1]
    device = hidden_states.device
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    full = torch.arange(num_tokens, device=device, dtype=torch.long)
    if keep >= num_tokens:
        return full

    if (
        state.cached_noise is None
        or tuple(state.cached_noise.shape) != tuple(hidden_states.shape)
    ):
        state.cached_noise = hidden_states.detach().clone()
        state.cached_noise_prev = hidden_states.detach().clone()
        state.counts = torch.zeros(num_tokens, device=device, dtype=torch.float32)
        state.cached_indices = None
        state.labels = None
        state.cluster_visits = None
        state.calls = 0
        return full

    ref_batch = min(max(0, int(batch_index)), hidden_states.shape[0] - 1)
    delta = (hidden_states - state.cached_noise)[ref_batch].float()
    delta_norm = delta.norm(dim=-1)

    if state.labels is None or state.labels.numel() != num_tokens:
        features = _cat_positional_features(hidden_states[ref_batch])
        state.labels = _cat_kmeans_labels(features, max_clusters)
        num_clusters = int(state.labels.max().item()) + 1
        state.cluster_visits = torch.zeros(
            num_clusters, device=device, dtype=torch.float32
        )
    labels = state.labels.to(device=device)
    num_clusters = int(labels.max().item()) + 1

    if state.counts is None or state.counts.numel() != num_tokens:
        state.counts = torch.zeros(num_tokens, device=device, dtype=torch.float32)
    else:
        state.counts = state.counts.to(device=device, dtype=torch.float32)
        if state.cached_indices is not None and state.cached_indices.numel() > 0:
            state.counts.mul_(0.5)
            state.counts.index_add_(
                0,
                state.cached_indices.to(device=device, dtype=torch.long),
                torch.ones(state.cached_indices.numel(), device=device),
            )

    cluster_scores = torch.full(
        (num_clusters,), -math.inf, device=device, dtype=delta_norm.dtype
    )
    for cid in range(num_clusters):
        mask = labels == cid
        if bool(mask.any()):
            cluster_scores[cid] = delta_norm[mask].sum()
    cluster_order = torch.argsort(cluster_scores, descending=True)

    first_selection = state.cached_indices is None
    cluster_budget = keep if first_selection else max(1, int(round(keep * 0.25)))
    selected: list[int] = []
    for rank, cid_tensor in enumerate(cluster_order):
        if len(selected) >= cluster_budget:
            break
        cid = int(cid_tensor.item())
        idx = torch.nonzero(labels == cid, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if first_selection:
            per_cluster = max(1, keep // 10)
        else:
            per_cluster = max(1, int(keep * 0.25 / 5))
            if rank < 3:
                per_cluster = max(1, per_cluster * (3 - rank))
        ranked_delta = idx[torch.argsort(delta_norm[idx], descending=True)]
        _cat_append_unique(selected, ranked_delta[:per_cluster], cluster_budget)
        if not first_selection and rank != 0 and len(selected) < cluster_budget:
            stale_local = idx[torch.argsort(state.counts[idx], descending=False)]
            _cat_append_unique(
                selected,
                stale_local[: max(1, per_cluster // 2)],
                cluster_budget,
            )

    if not first_selection and len(selected) < keep:
        stale_budget = keep - len(selected)
        stale = torch.argsort(state.counts, descending=False)
        _cat_append_unique(selected, stale[:stale_budget], keep)

    if len(selected) < keep:
        ranked_delta = torch.argsort(delta_norm, descending=True)
        _cat_append_unique(selected, ranked_delta, keep)
    if len(selected) < keep:
        _cat_append_unique(selected, full, keep)

    selected_tensor = torch.tensor(selected[:keep], device=device, dtype=torch.long)
    selected_tensor = torch.sort(selected_tensor).values
    if state.cluster_visits is not None and selected_tensor.numel() > 0:
        state.cluster_visits.index_add_(
            0,
            labels[selected_tensor],
            torch.ones(selected_tensor.numel(), device=device),
        )

    if state.cached_noise_prev is None or tuple(state.cached_noise_prev.shape) != tuple(
        hidden_states.shape
    ):
        state.cached_noise_prev = state.cached_noise.detach().clone()
    replaced = state.cached_noise_prev.to(
        device=device, dtype=hidden_states.dtype
    ).clone()
    replaced[:, selected_tensor, :] = hidden_states[:, selected_tensor, :]
    state.cached_noise_prev = replaced.detach()
    state.cached_indices = selected_tensor.detach()
    state.calls += 1
    return selected_tensor


@dataclass(frozen=True)
class TomeMergePlan:
    """Public ToMe-style balanced bipartite token merge plan."""

    unm_idx: object
    src_idx: object
    dst_idx: object
    num_tokens: int
    removed: int
    distill_token: bool = False

    def merged_token_indices(self):
        torch = _torch()
        batch = self.unm_idx.shape[0]
        device = self.unm_idx.device
        unm = (2 * self.unm_idx.squeeze(-1)).to(device=device)
        dst_tokens = self.num_tokens // 2
        dst = (2 * torch.arange(dst_tokens, device=device, dtype=torch.long) + 1)
        dst = dst.unsqueeze(0).expand(batch, -1)
        if self.distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def merge(self, x, mode: str = "mean"):
        torch = _torch()
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        batch, src_tokens, channels = src.shape
        unm = src.gather(
            dim=-2,
            index=self.unm_idx.expand(batch, src_tokens - self.removed, channels),
        )
        src = src.gather(
            dim=-2,
            index=self.src_idx.expand(batch, self.removed, channels),
        )
        dst = dst.scatter_reduce(
            -2,
            self.dst_idx.expand(batch, self.removed, channels),
            src,
            reduce=mode,
        )
        if self.distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(self, x):
        torch = _torch()
        unm_len = self.unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        batch, _, channels = unm.shape
        src = dst.gather(
            dim=-2,
            index=self.dst_idx.expand(batch, self.removed, channels),
        )
        out = torch.zeros(
            batch,
            self.num_tokens,
            channels,
            device=x.device,
            dtype=x.dtype,
        )
        out[..., 1::2, :] = dst
        out.scatter_(
            dim=-2,
            index=(2 * self.unm_idx).expand(batch, unm_len, channels),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=(2 * self.src_idx).expand(batch, self.removed, channels),
            src=src,
        )
        return out


@dataclass(frozen=True)
class TomeRandom2DMergePlan:
    """Public ToMeSD random-2D bipartite merge plan."""

    a_idx: object
    b_idx: object
    unm_idx: object
    src_idx: object
    dst_idx: object
    num_tokens: int
    removed: int

    def merged_token_indices(self):
        batch = self.a_idx.shape[0]
        device = self.a_idx.device
        channels = 1
        unm = self.a_idx.expand(batch, self.a_idx.shape[1], channels).gather(
            dim=1, index=self.unm_idx
        )
        return torch_cat_indices(unm.squeeze(-1), self.b_idx.squeeze(-1))

    def _split(self, x):
        gather = _gather
        batch, num_tokens, channels = x.shape
        src = gather(
            x,
            dim=1,
            index=self.a_idx.expand(batch, num_tokens - self.b_idx.shape[1], channels),
        )
        dst = gather(
            x,
            dim=1,
            index=self.b_idx.expand(batch, self.b_idx.shape[1], channels),
        )
        return src, dst

    def merge(self, x, mode: str = "mean"):
        src, dst = self._split(x)
        batch, src_tokens, channels = src.shape
        unm = _gather(
            src,
            dim=-2,
            index=self.unm_idx.expand(batch, src_tokens - self.removed, channels),
        )
        src = _gather(
            src,
            dim=-2,
            index=self.src_idx.expand(batch, self.removed, channels),
        )
        dst = dst.scatter_reduce(
            -2,
            self.dst_idx.expand(batch, self.removed, channels),
            src,
            reduce=mode,
        )
        torch = _torch()
        return torch.cat([unm, dst], dim=1)

    def unmerge(self, x):
        torch = _torch()
        batch = x.shape[0]
        unm_len = self.unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, channels = unm.shape
        src = _gather(
            dst,
            dim=-2,
            index=self.dst_idx.expand(batch, self.removed, channels),
        )
        out = torch.zeros(
            batch,
            self.num_tokens,
            channels,
            device=x.device,
            dtype=x.dtype,
        )
        out.scatter_(dim=-2, index=self.b_idx.expand(batch, self.b_idx.shape[1], channels), src=dst)
        out.scatter_(
            dim=-2,
            index=_gather(self.a_idx.expand(batch, self.a_idx.shape[1], 1), dim=1, index=self.unm_idx).expand(batch, unm_len, channels),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=_gather(self.a_idx.expand(batch, self.a_idx.shape[1], 1), dim=1, index=self.src_idx).expand(batch, self.removed, channels),
            src=src,
        )
        return out


def torch_cat_indices(left, right):
    torch = _torch()
    return torch.cat([left, right], dim=1)


def _gather(input_tensor, dim: int, index):
    torch = _torch()
    if input_tensor.device.type == "mps" and input_tensor.shape[-1] == 1:
        return torch.gather(
            input_tensor.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1),
        ).squeeze(-1)
    return torch.gather(input_tensor, dim, index)


def _factor_grid(num_tokens: int) -> tuple[int, int]:
    root = int(math.sqrt(num_tokens))
    for h in range(root, 0, -1):
        if num_tokens % h == 0:
            return num_tokens // h, h
    return num_tokens, 1


def tome_bipartite_soft_matching(
    metric,
    remove: int,
    *,
    class_token: bool = False,
    distill_token: bool = False,
) -> TomeMergePlan | None:
    """Build the public ToMe balanced bipartite merge/unmerge plan.

    ``metric`` is ``[B, tokens, channels]``. ``remove`` is capped at ToMe's
    public maximum of half the unprotected tokens.
    """

    torch = _torch()
    protected = int(class_token) + int(distill_token)
    num_tokens = metric.shape[1]
    remove = min(int(remove), (num_tokens - protected) // 2)
    if remove <= 0:
        return None

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
        scores = src_metric @ dst_metric.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., remove:, :]
        src_idx = edge_idx[..., :remove, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    return TomeMergePlan(
        unm_idx=unm_idx,
        src_idx=src_idx,
        dst_idx=dst_idx,
        num_tokens=num_tokens,
        removed=remove,
        distill_token=distill_token,
    )


def tomesd_random2d_matching(
    metric,
    remove: int,
    *,
    width: int | None = None,
    height: int | None = None,
    sx: int = 2,
    sy: int = 2,
    no_rand: bool = True,
    generator=None,
) -> TomeRandom2DMergePlan | None:
    """Build the public ToMeSD random-2D merge/unmerge plan.

    This mirrors ToMeSD's ``bipartite_soft_matching_random2d`` in pure Torch.
    ``no_rand=True`` is the deterministic shape-stable default; callers may
    pass a generator to reproduce the randomized public variant.
    """

    torch = _torch()
    batch, num_tokens, _ = metric.shape
    if remove <= 0:
        return None
    if width is None or height is None:
        width, height = _factor_grid(num_tokens)
    if int(width) * int(height) != num_tokens:
        raise ValueError(
            f"ToMeSD random2d grid must cover all tokens, got width={width} "
            f"height={height} tokens={num_tokens}"
        )
    width = int(width)
    height = int(height)
    sx = max(1, min(int(sx), width))
    sy = max(1, min(int(sy), height))
    hsy, wsx = height // sy, width // sx
    if hsy <= 0 or wsx <= 0:
        return None

    with torch.no_grad():
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            device = getattr(generator, "device", metric.device) if generator is not None else metric.device
            rand_idx = torch.randint(
                sy * sx,
                size=(hsy, wsx, 1),
                device=device,
                generator=generator,
            ).to(metric.device)

        idx_buffer_view = torch.zeros(
            hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64
        )
        idx_buffer_view.scatter_(
            dim=2,
            index=rand_idx,
            src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype),
        )
        idx_buffer_view = (
            idx_buffer_view.view(hsy, wsx, sy, sx)
            .transpose(1, 2)
            .reshape(hsy * sy, wsx * sx)
        )
        if (hsy * sy) < height or (wsx * sx) < width:
            idx_buffer = torch.zeros(height, width, device=metric.device, dtype=torch.int64)
            idx_buffer[: (hsy * sy), : (wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]
        b_idx = rand_idx[:, :num_dst, :]

        def split(x):
            channels = x.shape[-1]
            src = _gather(x, dim=1, index=a_idx.expand(batch, num_tokens - num_dst, channels))
            dst = _gather(x, dim=1, index=b_idx.expand(batch, num_dst, channels))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = split(metric)
        remove = min(a.shape[1], int(remove))
        if remove <= 0:
            return None
        scores = a @ b.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., remove:, :]
        src_idx = edge_idx[..., :remove, :]
        dst_idx = _gather(node_idx[..., None], dim=-2, index=src_idx)

    return TomeRandom2DMergePlan(
        a_idx=a_idx,
        b_idx=b_idx,
        unm_idx=unm_idx,
        src_idx=src_idx,
        dst_idx=dst_idx,
        num_tokens=num_tokens,
        removed=remove,
    )


def _topk_indices(scores, keep: int):
    torch = _torch()
    return torch.sort(torch.topk(scores, keep, largest=True).indices).values


def feature_norm_prune_indices(hidden_states, keep_ratio: float):
    """Token indices for the feature-L2 pruning baseline."""

    num_tokens = hidden_states.shape[1]
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    if keep >= num_tokens:
        torch = _torch()
        return torch.arange(num_tokens, device=hidden_states.device, dtype=torch.long)
    scores = hidden_states.float().pow(2).sum(-1).mean(0)
    return _topk_indices(scores, keep)


def region_dynamic_density_indices(hidden_states, keep_ratio: float):
    """Token indices for the density-balanced feature-norm baseline."""

    torch = _torch()
    num_tokens = hidden_states.shape[1]
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=hidden_states.device, dtype=torch.long)
    scores = hidden_states.float().pow(2).sum(-1).mean(0)
    window = max(1, int(round(num_tokens**0.5)))
    region = torch.arange(num_tokens, device=hidden_states.device) // window
    region_count = torch.bincount(region, minlength=int(region.max()) + 1).float()
    density = region_count[region].to(device=hidden_states.device, dtype=scores.dtype)
    return _topk_indices(scores / density.clamp_min(1.0).sqrt(), keep)


def cat_cluster_stale_indices(
    hidden_states,
    keep_ratio: float,
    state: CatPruneState,
    *,
    max_clusters: int = 20,
    batch_index: int = 1,
):
    """Explicit wrapper for the CAT-style cluster/staleness selector."""

    return cat_convergence_stale_indices(
        hidden_states,
        keep_ratio,
        state,
        max_clusters=max_clusters,
        batch_index=batch_index,
    )


def tome_merge_restore_plan(hidden_states, keep_ratio: float) -> TomeMergePlan | None:
    """Explicit wrapper for the public ToMe merge/unmerge plan."""

    num_tokens = hidden_states.shape[1]
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    return tome_bipartite_soft_matching(hidden_states.float(), num_tokens - keep)


def tomesd_random2d_merge_restore_plan(
    hidden_states,
    keep_ratio: float,
    *,
    width: int | None = None,
    height: int | None = None,
    sx: int = 2,
    sy: int = 2,
    no_rand: bool = True,
    generator=None,
) -> TomeRandom2DMergePlan | None:
    """Explicit wrapper for the public ToMeSD random-2D merge/unmerge plan."""

    num_tokens = hidden_states.shape[1]
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    return tomesd_random2d_matching(
        hidden_states.float(),
        num_tokens - keep,
        width=width,
        height=height,
        sx=sx,
        sy=sy,
        no_rand=no_rand,
        generator=generator,
    )


def keep_indices(
    method: str,
    num_tokens: int,
    keep_ratio: float,
    hidden_states,
    prev_velocity=None,
):
    """Ascending kept-token indices over a [B, S, C] (segment) tensor.

    Content-aware methods score each token (averaged over batch) and take the
    top-K; uniform/random are content-blind. Ascending order keeps the
    attention sequence monotone. This is the merged LTX2/Cosmos3 scoring core.
    """
    torch = _torch()
    device = hidden_states.device
    keep = max(1, min(num_tokens, int(round(num_tokens * keep_ratio))))
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    if method in ("velocity", "vel") and prev_velocity is not None:
        scores = prev_velocity.float().pow(2).sum(-1).mean(0)
    elif method in ("feat_norm", "feature_norm_prune", "feat", "norm", "feat_l2"):
        return feature_norm_prune_indices(hidden_states, keep_ratio)
    elif method in ("shape_stable", "shape_stable_compute_mask"):
        return _uniform_indices(num_tokens, keep, device)
    elif method == "feat_l1":
        scores = hidden_states.float().abs().sum(-1).mean(0)
    elif method in ("feat_linf", "feat_max"):
        scores = hidden_states.float().abs().amax(-1).mean(0)
    elif method == "feat_var":
        scores = hidden_states.float().var(-1).mean(0)
    elif _is_tome_method(method):
        feats = hidden_states.float().mean(0)
        norm = feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        feats = feats / norm
        left = torch.zeros(num_tokens, device=device, dtype=feats.dtype)
        right = torch.zeros(num_tokens, device=device, dtype=feats.dtype)
        if num_tokens > 1:
            sim = (feats[:-1] * feats[1:]).sum(-1)
            left[1:] = sim
            right[:-1] = sim
        scores = 1.0 - torch.maximum(left, right)
        scores[0] = scores[-1] = scores.max().clamp_min(1.0)
    elif method in ("region_density", "region_dynamic_density"):
        return region_dynamic_density_indices(hidden_states, keep_ratio)
    elif method in ("cluster_representative", "cluster_representative_update"):
        feats = hidden_states.float().mean(0)
        center = feats.mean(0, keepdim=True)
        spread = (feats - center).pow(2).sum(-1)
        norm = feats.pow(2).sum(-1)
        scores = spread + 0.01 * norm
    elif method in ("random", "rand"):
        gen = torch.Generator(device=device).manual_seed(42)
        return torch.sort(torch.randperm(num_tokens, generator=gen, device=device)[:keep]).values
    else:  # uniform / unknown -> content-blind even stride
        return _uniform_indices(num_tokens, keep, device)

    return _topk_indices(scores, keep)


@register_technique("token_prune")
class TokenPrune(Technique):
    """Generic mid-loop token pruning.

    Parameters
    ----------
    keep_ratio : Schedule[float] | float  -- fraction kept (>=1 => OFF/identity).
    method     : token-scoring method (feat_norm, feat_l1, uniform, ...).
    compensation : 'prev' (reuse previous step's dropped-token hidden) or 'zero'.
    enabled    : Schedule[bool] -- typically at_steps("1-2", ...) to prune only
                 select steps; step 0 always runs full to seed the 'prev' buffer.
    """

    name = "token_prune"
    phase = Phase.PRE_BLOCKS  # gather pre-loop; scatter is the paired POST work
    reads = frozenset({Seam.HIDDEN_STATES})
    writes = frozenset({Seam.TOKEN_SET, Seam.HIDDEN_STATES})
    required_capabilities = frozenset({Capability.PRUNABLE_TOKENS})

    def __init__(
        self,
        keep_ratio: "Schedule | float" = 1.0,
        method: str = "feat_norm",
        compensation: str = "prev",
        enabled: "Schedule | bool" = True,
    ):
        super().__init__(enabled=enabled)
        self.keep_ratio = as_schedule(keep_ratio)
        self.method = method
        self.compensation = compensation

    def is_active(self, ctx: TechniqueContext) -> bool:
        return super().is_active(ctx) and self.keep_ratio.at(ctx.step, ctx.stage) < 1.0

    def before_blocks(self, ctx: TechniqueContext, hidden):
        """Gather the prunable segment down to K tokens. carry = (start, end,
        keep_idx, full_S, prev_full) for after_blocks to scatter back."""
        spec = ctx.spec
        start, end = spec.segment(hidden, ctx)
        seg = hidden[:, start:end, :] if spec.seq_dim == 1 else hidden  # [B, S, C]
        full_S = seg.shape[1]
        ratio = float(self.keep_ratio.at(ctx.step, ctx.stage))

        is_tome = _is_tome_method(self.method)
        is_tomesd_random2d = _is_tomesd_random2d_method(self.method)
        is_cat = _is_cat_method(self.method)
        prev_full = ctx.scratch.get(ctx.cache_key)
        # step 0 (or first active step with no prev) runs FULL to seed 'prev'.
        if (
            prev_full is None
            and self.compensation == "prev"
            and not is_tome
            and not is_tomesd_random2d
            and not is_cat
        ):
            ctx.scratch[ctx.cache_key] = seg.detach()
            return hidden, None

        if is_cat:
            state_key = ("cat_prune", ctx.cache_key, self.method)
            state = ctx.scratch.get(state_key)
            if state is None:
                state = CatPruneState()
                ctx.scratch[state_key] = state
            idx = cat_cluster_stale_indices(seg, ratio, state)
            if idx.shape[0] >= full_S:
                ctx.scratch[ctx.cache_key] = seg.detach()
                return hidden, None
            kept = seg.index_select(1, idx)
            torch = _torch()
            new_hidden = torch.cat([hidden[:, :start, :], kept, hidden[:, end:, :]], dim=1)
            return new_hidden, (start, end, idx, full_S, prev_full)

        if is_tome:
            plan = tome_merge_restore_plan(seg, ratio)
            if plan is None:
                ctx.scratch[ctx.cache_key] = seg.detach()
                return hidden, None
            merged = plan.merge(seg, mode="mean")
            torch = _torch()
            new_hidden = torch.cat([hidden[:, :start, :], merged, hidden[:, end:, :]], dim=1)
            return new_hidden, ("tome", start, end, plan)

        if is_tomesd_random2d:
            plan = tomesd_random2d_merge_restore_plan(seg, ratio, no_rand=True)
            if plan is None:
                ctx.scratch[ctx.cache_key] = seg.detach()
                return hidden, None
            merged = plan.merge(seg, mode="mean")
            torch = _torch()
            new_hidden = torch.cat([hidden[:, :start, :], merged, hidden[:, end:, :]], dim=1)
            return new_hidden, ("tome", start, end, plan)

        idx = keep_indices(self.method, full_S, ratio, seg)
        kept = seg.index_select(1, idx)
        torch = _torch()
        new_hidden = torch.cat([hidden[:, :start, :], kept, hidden[:, end:, :]], dim=1)
        return new_hidden, (start, end, idx, full_S, prev_full)

    def after_blocks(self, ctx: TechniqueContext, hidden, carry):
        """Scatter the K-token result back to full S; fill dropped tokens with
        the compensation hidden; refresh the 'prev' buffer."""
        spec = ctx.spec
        if carry is None:
            # full (seed) step: store the full segment as next step's 'prev'.
            start, end = spec.segment(hidden, ctx)
            ctx.scratch[ctx.cache_key] = hidden[:, start:end, :].detach()
            return hidden

        if isinstance(carry, tuple) and carry and carry[0] == "tome":
            _, start, end, plan = carry
            merged_len = end - start - plan.removed
            merged_out = hidden[:, start : start + merged_len, :]
            restored = plan.unmerge(merged_out)
            ctx.scratch[ctx.cache_key] = restored.detach()
            torch = _torch()
            return torch.cat(
                [hidden[:, :start, :], restored, hidden[:, start + merged_len :, :]],
                dim=1,
            )

        start, end, idx, full_S, prev_full = carry
        kept_len = idx.shape[0]
        kept_out = hidden[:, start : start + kept_len, :]
        B, _, C = kept_out.shape
        if self.compensation == "zero" or prev_full is None:
            full = kept_out.new_zeros((B, full_S, C))
        else:
            full = prev_full.to(dtype=kept_out.dtype, device=kept_out.device).clone()
        full[:, idx, :] = kept_out
        ctx.scratch[ctx.cache_key] = full.detach()
        torch = _torch()
        return torch.cat([hidden[:, :start, :], full, hidden[:, start + kept_len :, :]], dim=1)
