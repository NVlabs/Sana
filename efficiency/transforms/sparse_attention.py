# Copyright 2025 SGLang authors
#
# SparseAttention -- BUILD-time transform that installs a sparse attention
# backend for a component. The algorithm-facing params live here; model-specific
# runtime code is responsible for translating latent/text shapes into backend
# metadata.

from __future__ import annotations

from efficiency.sparse_attention_policies import (
    sparse_route_policy_config,
    sparse_videogen_sap_plan,
)
from efficiency.registry import register_transform
from efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)
from efficiency.technique import Capability, Seam


@register_transform("sparse_attention")
class SparseAttention(ModelTransform):
    """Install sparse video attention on the given component(s).

    ``backend`` selects the concrete runtime backend while the remaining params
    describe the algorithm. PISA uses the ``piecewise_*`` keys. Sparse VideoGen2
    semantic permutation uses the ``svg2_*`` keys consumed by its metadata
    builder in the model runtime.
    """

    name = "sparse_attention"
    phase = TransformPhase.BUILD
    writes = frozenset({Seam.ATTENTION_BACKEND})
    required_capabilities = frozenset({Capability.SWAPPABLE_ATTENTION})

    def __init__(
        self,
        sparsity: float = 0.9,
        block_size: int = 64,
        frame_size: int = 0,
        only_video_self: bool = True,
        component: str = "transformer_2",
        stage1_dense: bool = False,
        dense_steps: int = 0,
        route_mode: str = "score",
        route_bias: bool = True,
        allow_qk_mismatch: bool = False,
        allow_gqa: bool = False,
        dense_fallback: str = "fa",
        stage2_dense_layers: str = "0-1",
        backend: str | None = None,
        svg2_num_q_centroids: int = 300,
        svg2_num_k_centroids: int = 1000,
        svg2_top_p_kmeans: float = 0.9,
        svg2_min_kc_ratio: float = 0.1,
        svg2_kmeans_iter_init: int = 50,
        svg2_kmeans_iter_step: int = 2,
        svg2_zero_step_kmeans_init: bool = False,
        svg2_first_layers_fp: float = 0.03,
        svg2_first_times_fp: float = 0.2,
    ):
        self.sparsity = sparsity
        self.block_size = block_size
        self.frame_size = frame_size
        self.only_video_self = only_video_self
        self.component = component
        self.stage1_dense = stage1_dense
        self.dense_steps = dense_steps
        self.route_mode = route_mode
        self.route_bias = route_bias
        self.allow_qk_mismatch = allow_qk_mismatch
        self.allow_gqa = allow_gqa
        self.dense_fallback = dense_fallback
        self.stage2_dense_layers = stage2_dense_layers
        self.backend = backend or (
            "sparse_video_gen_2_attn"
            if route_mode == "semantic_permutation"
            else "piecewise_attn"
        )
        self.svg2_num_q_centroids = svg2_num_q_centroids
        self.svg2_num_k_centroids = svg2_num_k_centroids
        self.svg2_top_p_kmeans = svg2_top_p_kmeans
        self.svg2_min_kc_ratio = svg2_min_kc_ratio
        self.svg2_kmeans_iter_init = svg2_kmeans_iter_init
        self.svg2_kmeans_iter_step = svg2_kmeans_iter_step
        self.svg2_zero_step_kmeans_init = svg2_zero_step_kmeans_init
        self.svg2_first_layers_fp = svg2_first_layers_fp
        self.svg2_first_times_fp = svg2_first_times_fp
        self.svg2_plan = None
        if self.route_mode == "semantic_permutation":
            self.svg2_plan = sparse_videogen_sap_plan(
                route_mode=self.route_mode,
                backend=self.backend,
                num_q_centroids=self.svg2_num_q_centroids,
                num_k_centroids=self.svg2_num_k_centroids,
                top_p_kmeans=self.svg2_top_p_kmeans,
                min_kc_ratio=self.svg2_min_kc_ratio,
                kmeans_iter_init=self.svg2_kmeans_iter_init,
                kmeans_iter_step=self.svg2_kmeans_iter_step,
                zero_step_kmeans_init=self.svg2_zero_step_kmeans_init,
                first_layers_fp=self.svg2_first_layers_fp,
                first_times_fp=self.svg2_first_times_fp,
            )
        self.policy = sparse_route_policy_config(
            self.route_mode,
            sparsity=float(self.sparsity),
            block_size=int(self.block_size),
            dense_fallback=self.dense_fallback,
        )

    def set_env(self, ctx: TransformContext) -> None:
        e = ctx.env
        # Other components stay dense (fa); the chosen component receives the
        # algorithm backend. Cosmos3 uses "transformer", LTX2 stage-2 uses
        # "transformer_2".
        backends = {"transformer": "fa", "transformer_2": "fa"}
        backends[self.component] = self.backend
        e["SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS"] = ",".join(
            f"{k}={v}" for k, v in backends.items()
        )
        if self.backend == "sparse_video_gen_2_attn":
            cfg = (
                f"sparse_attention_backend={self.backend},"
                f"sparse_attention_route_mode={self.route_mode},"
                f"sparse_attention_policy_family={self.policy.family},"
                f"svg2_num_q_centroids={self.svg2_num_q_centroids},"
                f"svg2_num_k_centroids={self.svg2_num_k_centroids},"
                f"svg2_top_p_kmeans={self.svg2_top_p_kmeans},"
                f"svg2_min_kc_ratio={self.svg2_min_kc_ratio},"
                f"svg2_kmeans_iter_init={self.svg2_kmeans_iter_init},"
                f"svg2_kmeans_iter_step={self.svg2_kmeans_iter_step},"
                f"svg2_zero_step_kmeans_init={str(self.svg2_zero_step_kmeans_init).lower()},"
                f"svg2_first_layers_fp={self.svg2_first_layers_fp},"
                f"svg2_first_times_fp={self.svg2_first_times_fp}"
            )
        else:
            cfg = (
                f"piecewise_sparsity={self.sparsity},"
                f"piecewise_block_size={self.block_size},"
                f"piecewise_frame_size={self.frame_size},"
                f"piecewise_only_video_self_attention={str(self.only_video_self).lower()},"
                f"piecewise_stage1_schedule={str(self.stage1_dense).lower()},"
                f"piecewise_stage1_dense_steps={self.dense_steps},"
                f"piecewise_stage2_dense_layers={self.stage2_dense_layers},"
                f"piecewise_approx_remainder=true,"
                f"piecewise_route_mode={self.route_mode},"
                f"piecewise_route_bias={str(self.route_bias).lower()},"
                f"piecewise_allow_qk_mismatch={str(self.allow_qk_mismatch).lower()},"
                f"piecewise_allow_gqa={str(self.allow_gqa).lower()},"
                f"piecewise_policy_family={self.policy.family},"
                f"piecewise_dense_fallback={self.dense_fallback}"
            )
        e["SGLANG_HQ_ATTENTION_BACKEND_CONFIG"] = cfg
        e["SGLANG_HQ_SPARSE_ATTENTION_BACKEND"] = self.backend
        e.update(self.policy.as_env())
        if self.svg2_plan is not None:
            e.update(self.svg2_plan.as_env())
