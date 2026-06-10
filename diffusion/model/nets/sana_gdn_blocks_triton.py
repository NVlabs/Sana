"""Triton-backed GDN attention blocks.

These classes keep the baseline module structure and state-dict keys while
routing supported GDN paths through fused Triton kernels. Context-parallel
main-branch training uses fused prep/output kernels around the distributed CP
scan.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion.distributed.context_parallel.config import cp_enabled
from diffusion.model.nets.sana_camctrl_blocks import (
    _maybe_drop_cam_branch,
    _prepare_ray_apply_fns,
)
from diffusion.model.nets.sana_gdn_blocks import (
    BidirectionalGDN,
    ChunkCausalGDN,
)
from diffusion.model.nets.sana_gdn_camctrl_blocks import (
    BidirectionalGDNUCPESinglePathLiteLA,
    ChunkCausalGDNUCPESinglePathLiteLA,
)
from diffusion.model.ops.fused_cam_gdn import (
    _invert_SE3,
    _prepare_ucpe_rope_tables,
    _process_camera_conditions_raymats_only,
    cam_prep_func,
    cam_prep_func_with_grad,
    cam_scan_func,
    cam_scan_func_with_grad,
)
from diffusion.model.ops.fused_gdn import (
    fused_bigdn_forward_with_grad,
    fused_bigdn_func,
    fused_qk_inv_rms,
    prepare_rope_tables,
)
from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise, cam_scan_pair_chunkwise
from diffusion.utils.chunk_utils import (
    is_chunk_causal_request,
    normalize_chunk_index,
    size1_chunk_position_indices,
)
from diffusion.model.registry import ATTENTION_BLOCKS


def _mask_reverse_gates_for_chunk_boundaries(
    beta: torch.Tensor,
    decay: torch.Tensor,
    valid_chunk_index: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero reverse-scan gates where chunk-local anti-causal state must reset."""
    interior = [i for i in valid_chunk_index if 0 < i < beta.shape[2]]
    size1_positions = size1_chunk_position_indices(valid_chunk_index)
    bwd_zero_positions = sorted(set(interior) | set(size1_positions))
    if not bwd_zero_positions:
        return beta, decay

    beta_bwd = beta.clone()
    decay_bwd = decay.clone()
    beta_bwd[:, :, bwd_zero_positions, :] = 0.0
    decay_bwd[:, :, bwd_zero_positions] = 0.0
    return beta_bwd, decay_bwd


@ATTENTION_BLOCKS.register_module()
class ChunkCausalGDNTriton(ChunkCausalGDN):
    """Chunk-causal GDN with a fused Triton scan.

    Subclasses :class:`ChunkCausalGDN` and only overrides :meth:`__init__`
    (to accept ``use_autograd_kernel``) and :meth:`forward`.  All sub-modules
    (``qkv``, ``proj``, ``q_norm``, ``k_norm``, ``conv_k``, ``beta_proj``,
    ``gate_proj``, ``A_log``, ``dt_bias``, ``output_gate``) and helpers
    (``_apply_temporal_short_conv``, ``_compute_frame_gates``,
    ``_apply_output_gate``) are inherited unchanged so checkpoints are 100%
    compatible.

    When ``use_autograd_kernel=True``, the fused-kernel call switches to
    :func:`fused_bigdn_forward_with_grad` (autograd-enabled).  Chunk-causal
    ``*_bwd`` masking is fully supported in autograd mode: the block builds
    masked ``beta_bwd`` / ``decay_bwd`` clones (zeroed at interior chunk
    boundaries) exactly as in the inference path and forwards them to the
    autograd wrapper, which routes the reverse-direction beta/decay
    gradient back through the caller's clone+mask op so anti-causal state
    resets at chunk boundaries while keeping the autograd graph correct.
    """

    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = use_autograd_kernel

    def _forward_cp_scan_triton_ag(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Context-parallel chunk-causal GDN using fused prep/output kernels.

        The forward branch scans the local shard directly. The reverse branch
        materializes the exclusive anti-causal recurrence by flipping Q and
        flip-and-shifting K/V/beta/decay across CP-rank boundaries, then runs
        the same fused CP raw path with separate Q-side and K-side RoPE tables.

        Args:
            x: ``(B, N_local, C)`` CP-local input slice.
            mask: Unused.
            HW: ``(T_local, H, W)`` token layout for THIS rank.
            rotary_emb: CP-local RoPE complex frequencies.
            block_mask: Unused.
            apply_output_gate: When False, return raw attention output
                before gate + projection.
            chunk_size / chunk_split_strategy / chunk_index: Chunk-causal
                boundary specification (in GLOBAL frame coords).

        Returns:
            ``(B, N_local, C)`` after attention + (optional) output gate
            + projection.
        """
        import torch.distributed as dist

        from diffusion.distributed.context_parallel.config import get_cp_group
        from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
        from diffusion.model.ops.fused_gdn_cp import cp_fused_gdn_chunkwise_raw_autograd
        from diffusion.utils.chunk_utils import normalize_chunk_index

        del mask, block_mask  # unused on this path

        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "ChunkCausalGDNTriton CP-Triton path does not support " "frame_valid_mask (training-only feature)."
            )
        if self.conv_q is not None or self.conv_v is not None:
            raise NotImplementedError(
                "ChunkCausalGDNTriton CP-Triton path supports k_conv_only="
                "True; got conv_q or conv_v which would require additional "
                "Triton paths."
            )

        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        if N != T * S:
            raise ValueError(f"N={N} != T*S={T * S} for HW={HW}.")
        if C != H * D:
            raise ValueError(f"C={C} != heads*dim={H * D}.")

        cp_group = get_cp_group()
        if cp_group is None:
            raise RuntimeError(
                "ChunkCausalGDNTriton._forward_cp_scan_triton_ag called but " "CP group is not initialized."
            )

        # ---- 1. QKV projection on the CP-local slice. ---------------------
        qkv = self.qkv(x).reshape(B, N, 3, H, D)

        # ---- 2. Chunk-causal short conv on K (in-place writeback). --------
        if self.conv_k is not None:
            k_raw = qkv[:, :, 1].contiguous().reshape(B, N, C)
            conv_kwargs: dict = dict(
                chunk_size=chunk_size,
                chunk_index=chunk_index,
                chunk_split_strategy=chunk_split_strategy,
            )
            _ci_g = kwargs.get("chunk_index_global", None)
            if _ci_g is not None:
                conv_kwargs["chunk_index_global"] = _ci_g
            k_conv = self._apply_temporal_short_conv(k_raw, self.conv_k, HW, **conv_kwargs)
            qkv = qkv.clone()
            qkv[:, :, 1] = k_conv.reshape(B, N, H, D)

        # ---- 3. Frame gates (precomputed when shared with cam branch). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.contiguous()
        decay = decay.contiguous()

        # ---- 4. Full-channel RMSNorm weights + norm_eps. -----------------
        if not isinstance(self.q_norm, nn.Identity):
            q_nw = self.q_norm.weight.float().contiguous()
            k_nw = self.k_norm.weight.float().contiguous()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        else:
            q_nw = None
            k_nw = None
            norm_eps = 1e-5

        # ---- 5. CP-local RoPE tables. ------------------------------------
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, x.device)

        # ---- 6. K scale via the model's configured mode. -----------------
        k_scale = self._key_scale(S)

        # ---- 7. dot_precision: fp32 inputs → IEEE fp32 bridge (2);
        # bf16/fp16 → TF32 bf16 bridge (0).
        dot_precision = 2 if x.dtype == torch.float32 else 0

        # Forward branch.
        res = cp_fused_gdn_chunkwise_raw_autograd(
            qkv,
            beta,
            decay,
            q_nw,
            k_nw,
            rope_cos,
            rope_sin,
            F=T,
            S=S,
            group=cp_group,
            k_scale=k_scale,
            norm_eps=norm_eps,
            eps=self.eps,
            dot_precision=dot_precision,
            reverse_rank_order=False,
            truncate_to_active=None,
        )
        num_fwd, den_fwd = res.num, res.den

        # Reverse branch.
        cp_world = dist.get_world_size(cp_group)
        cp_rank_local = dist.get_rank(cp_group)
        if chunk_size is None and chunk_index is None:
            decay_bwd_input = decay
            input_mask = torch.ones(B, 1, T, 1, 1, device=x.device, dtype=qkv.dtype)
        else:
            T_global = T * cp_world
            global_offset = cp_rank_local * T
            valid_chunk_index_global, _ = normalize_chunk_index(
                chunk_index,
                T_global,
                chunk_size,
                chunk_split_strategy,
            )
            boundaries = [
                idx - global_offset
                for idx in valid_chunk_index_global
                if global_offset <= idx < global_offset + T and idx != 0
            ]
            decay_bwd_input = decay.clone()
            if boundaries:
                decay_bwd_input[:, :, boundaries] = 0.0
            input_mask = torch.ones(B, 1, T, 1, 1, device=x.device, dtype=qkv.dtype)
            if boundaries:
                input_mask[:, :, boundaries] = 0.0

        def _cp_flip_and_shift(tensors: list[torch.Tensor], shift_vals: list[float]) -> list[torch.Tensor]:
            is_last = cp_rank_local == cp_world - 1
            results = []
            for tensor, sv in zip(tensors, shift_vals):
                first_frame = tensor[:, :, :1, ...].contiguous()
                haloed = cp_halo_exchange(first_frame, left_size=0, right_size=1, dim=2, group=cp_group)
                boundary = haloed[:, :, 1:2, ...]
                if is_last and sv != 0.0:
                    boundary = boundary.mul(0.0).add(sv)
                T_loc = tensor.shape[2]
                flipped = torch.flip(tensor, dims=[2])
                body = flipped[:, :, : T_loc - 1, ...]
                results.append(torch.cat([boundary, body], dim=2))
            return results

        def _bnhd_to_frame(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 2, 3, 1).reshape(B, H, D, T, S).permute(0, 1, 3, 2, 4).contiguous()

        def _frame_to_bnhd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 2, 4, 1, 3).reshape(B, T * S, H, D).contiguous()

        q_raw_f = _bnhd_to_frame(qkv[:, :, 0])
        k_raw_f = _bnhd_to_frame(qkv[:, :, 1]) * input_mask
        v_raw_f = _bnhd_to_frame(qkv[:, :, 2]) * input_mask
        q_bwd_f = torch.flip(q_raw_f, dims=[2])
        k_bwd_f, v_bwd_f = _cp_flip_and_shift([k_raw_f, v_raw_f], [0.0, 0.0])
        qkv_bwd = torch.stack(
            [
                _frame_to_bnhd(q_bwd_f),
                _frame_to_bnhd(k_bwd_f),
                _frame_to_bnhd(v_bwd_f),
            ],
            dim=2,
        )

        beta_f = beta.unsqueeze(3)
        decay_f_bwd = decay_bwd_input.view(B, H, T, 1, 1)
        beta_bwd_f, decay_bwd_f = _cp_flip_and_shift([beta_f, decay_f_bwd], [0.0, 1.0])
        beta_bwd = beta_bwd_f.squeeze(3).contiguous()
        decay_bwd = decay_bwd_f.squeeze(-1).squeeze(-1).contiguous()

        def _rope_to_frame(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(T, S, D).permute(0, 2, 1).reshape(1, 1, T, D, S).contiguous()

        def _rope_from_frame(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(T, D, S).permute(0, 2, 1).reshape(T * S, D).contiguous()

        rope_cos_f = _rope_to_frame(rope_cos)
        rope_sin_f = _rope_to_frame(rope_sin)
        rope_cos_q = _rope_from_frame(torch.flip(rope_cos_f, dims=[2]))
        rope_sin_q = _rope_from_frame(torch.flip(rope_sin_f, dims=[2]))
        rope_cos_k_f, rope_sin_k_f = _cp_flip_and_shift([rope_cos_f, rope_sin_f], [1.0, 0.0])
        rope_cos_k = _rope_from_frame(rope_cos_k_f)
        rope_sin_k = _rope_from_frame(rope_sin_k_f)

        res_bwd = cp_fused_gdn_chunkwise_raw_autograd(
            qkv_bwd,
            beta_bwd,
            decay_bwd,
            q_nw,
            k_nw,
            rope_cos_k,
            rope_sin_k,
            F=T,
            S=S,
            group=cp_group,
            k_scale=k_scale,
            norm_eps=norm_eps,
            eps=self.eps,
            dot_precision=dot_precision,
            reverse_rank_order=True,
            truncate_to_active=None,
            rope_cos_q=rope_cos_q,
            rope_sin_q=rope_sin_q,
        )
        num_bwd_flipped = res_bwd.num.reshape(B, T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()
        den_bwd_flipped = res_bwd.den.reshape(B, H, T, S).unsqueeze(3).contiguous()
        num_bwd_eager = torch.flip(num_bwd_flipped, dims=[2])  # (B, H, T, D, S)
        den_bwd_eager = torch.flip(den_bwd_flipped, dims=[2])  # (B, H, T, 1, S)

        num_fwd_5d = num_fwd.reshape(B, T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()
        den_fwd_5d = den_fwd.reshape(B, H, T, S).unsqueeze(3).contiguous()

        total_num = num_fwd_5d.float() + num_bwd_eager.float()
        total_den = den_fwd_5d.float() + den_bwd_eager.float()

        out = total_num / (total_den + self.eps)  # (B, H, T, D, S)
        if getattr(self, "fp32_attention", True) and x.dtype != torch.float32:
            out = out.to(x.dtype)

        out = out.permute(0, 1, 3, 2, 4).reshape(B, self.heads, D, N)
        out = out.permute(0, 3, 1, 2).reshape(B, N, C)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        chunk_size: int | None = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: list[int] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if HW is None:
            raise ValueError("ChunkCausalGDNTriton requires HW=(T, H, W).")
        if cp_enabled():
            from diffusion.distributed.context_parallel.config import (
                get_cp_triton_block_fusion,
            )

            if not get_cp_triton_block_fusion():
                raise NotImplementedError(
                    "ChunkCausalGDNTriton context-parallel execution requires "
                    "train.extra.cp.triton_block_fusion=true."
                )
            return self._forward_cp_scan_triton_ag(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=apply_output_gate,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                **kwargs,
            )
        del mask, block_mask  # unused in the chunk-causal Triton path
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "ChunkCausalGDNTriton does not support frame_valid_mask " "(training-only feature)."
            )
        if self.conv_q is not None or self.conv_v is not None:
            raise NotImplementedError(
                "ChunkCausalGDNTriton supports k_conv_only=True; got conv_q "
                "or conv_v which would require additional Triton paths."
            )

        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        if N != T * S:
            raise ValueError(f"N={N} != T*S={T * S} for HW={HW}.")
        if C != H * D:
            raise ValueError(f"C={C} != heads*dim={H * D}.")

        # ---- 1. QKV projection -> (B, N, 3, H, D), kept contiguous. -------
        qkv = self.qkv(x).reshape(B, N, 3, H, D)

        # ---- 2. Chunk-causal short conv on K (in-place writeback). --------
        # ``qkv[:, :, 1]`` is a strided view of the contiguous ``qkv`` buffer.
        # ``copy_`` mutates that view in-place, avoiding a torch.stack repack.
        if self.conv_k is not None:
            k_raw = qkv[:, :, 1].contiguous().reshape(B, N, C)
            k_conv = self._apply_temporal_short_conv(
                k_raw,
                self.conv_k,
                HW,
                chunk_size=chunk_size,
                chunk_index=chunk_index,
                chunk_split_strategy=chunk_split_strategy,
            )
            qkv[:, :, 1].copy_(k_conv.reshape(B, N, H, D))

        # ---- 3. Frame gates (precomputed when shared with cam branch). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.contiguous()
        decay = decay.contiguous()

        # ---- 4. Backward-direction chunk-boundary masking. ----------------
        # The forward (causal) scan keeps full context across chunks; the
        # backward (anti-causal) scan must reset state at every interior
        # chunk boundary. Zeroing decay zeros the state-carry; zeroing beta
        # zeros the new-info update. Done in PyTorch so the kernel itself
        # stays oblivious to chunk structure.
        #
        # SIZE-1 CHUNK SKIP: Frame-positions belonging to size-1
        # (singleton) chunks have no intra-chunk lookahead, so the
        # anti-causal scan should contribute nothing.  We zero
        # ``beta_bwd`` *and* ``decay_bwd`` at every position inside a
        # size-1 chunk (the union of interior boundaries, the very
        # first frame when chunk-0 has size 1, and any other singleton
        # chunk position).  The fused Triton kernel then produces 0
        # anti-causal contribution at those positions, matching the
        # design intent of ``cond_chunk_mode='frame_causal'`` where
        # cond positions are fully causal.
        valid_chunk_index, _ = normalize_chunk_index(chunk_index, T, chunk_size, chunk_split_strategy)
        beta_bwd, decay_bwd = _mask_reverse_gates_for_chunk_boundaries(beta, decay, valid_chunk_index)
        if beta_bwd is beta:
            beta_bwd = None
            decay_bwd = None

        # ---- 5. Full-channel RMSNorm weights. -----------------------------
        if not isinstance(self.q_norm, nn.Identity):
            q_nw = self.q_norm.weight.float().contiguous()
            k_nw = self.k_norm.weight.float().contiguous()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        else:
            q_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            k_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            norm_eps = 1e-5

        # ---- 6. Fused Q+K inverse-RMS (single Triton launch). -------------
        q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=norm_eps)

        # ---- 7. Expanded RoPE cos/sin tables (N, D). ----------------------
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, x.device)

        # ---- 8. K scale absorbs both the Q/K^T variance and the spatial
        # mean-pool of the ReLU-kernel features over S frames. -------------
        k_scale = (D**-0.5) * (S**-0.5)

        # ---- 9. Fused bidirectional Triton scan. --------------------------
        if getattr(self, "use_autograd_kernel", False):
            # Autograd path: full-channel RMSNorm + bidirectional scan with
            # autograd bookkeeping. The autograd kernel computes inv_rms
            # internally, so q_inv_rms / k_inv_rms are unused here.
            # Chunk-causal *_bwd masking is plumbed through to the
            # reverse-direction kernel call; gradients are routed back
            # through the (autograd-tracked) clone+mask chain on the
            # caller side so the anti-causal scan resets state at every
            # interior chunk boundary.
            out = fused_bigdn_forward_with_grad(
                qkv,
                beta,
                decay,
                q_nw,
                k_nw,
                rope_cos,
                rope_sin,
                F=T,
                S=S,
                k_scale=k_scale,
                norm_eps=norm_eps,
                eps=self.eps,
                beta_bwd=beta_bwd,
                decay_bwd=decay_bwd,
            )
        else:
            out = fused_bigdn_func(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_norm_weight=q_nw,
                k_norm_weight=k_nw,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                beta=beta,
                decay=decay,
                F=T,
                S=S,
                k_scale=k_scale,
                eps=self.eps,
                beta_bwd=beta_bwd,
                decay_bwd=decay_bwd,
            )  # (B, N, H, D)

        # ---- 10. Output gate + projection. --------------------------------
        out = out.reshape(B, N, C)
        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out


@ATTENTION_BLOCKS.register_module()
class ChunkCausalGDNUCPESinglePathLiteLATriton(ChunkCausalGDNUCPESinglePathLiteLA):
    """Camera-controlled chunk-causal GDN with Triton main branch.

    Inherits the entire camera branch (``_forward_cam_branch``), the dual-
    branch wrapper (``forward``), the chunk-aware ``_apply_temporal_short_conv``
    routing, and every checkpoint key from
    :class:`ChunkCausalGDNUCPESinglePathLiteLA`.  The **only** behavioural
    delta is a single class-attribute hook that the parent's
    :meth:`forward` already consults:

        ``_main_chunk_causal_class`` — class whose ``forward`` is invoked for
        the main GDN scan when a multi-chunk schedule is active.  Switching
        from :class:`ChunkCausalGDN` to :class:`ChunkCausalGDNTriton` swaps
        the entire multi-stage scan to the fused Triton kernel while
        leaving the camera branch bit-identical.

    The ``use_autograd_kernel`` flag is stored on this instance and consulted
    inside :meth:`ChunkCausalGDNTriton.forward` (called via the
    ``_main_chunk_causal_class`` dispatch) — the dispatch passes ``self``,
    so the flag is visible to the main-branch forward.  The cam branch is
    inherited as-is (torch path); use :class:`ChunkCausalGDNUCPESinglePathLiteLABothTriton`
    for a Triton cam branch with autograd support.
    """

    _main_chunk_causal_class = ChunkCausalGDNTriton

    # This class does not inherit from ChunkCausalGDNTriton directly.
    _forward_cp_scan_triton_ag = ChunkCausalGDNTriton._forward_cp_scan_triton_ag

    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = use_autograd_kernel

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if HW is not None:
            precomputed_gates = self._compute_frame_gates(x, HW)
        else:
            precomputed_gates = None

        main_raw = ChunkCausalGDNTriton.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            precomputed_gates=precomputed_gates,
            **kwargs,
        )

        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions,
            kwargs.get("cam_branch_drop_prob", 0.0),
            self.training,
            x.device,
        )
        if camera_conditions is not None:
            if HW is None:
                raise ValueError("HW (T, H, W) must be provided for UCPE camera branch.")
            cam_raw = self._forward_cam_branch(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                chunk_size=chunk_size,
                precomputed_gates=precomputed_gates,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(self.proj.weight.dtype))


@ATTENTION_BLOCKS.register_module()
class ChunkCausalGDNUCPESinglePathLiteLABothTriton(ChunkCausalGDNUCPESinglePathLiteLATriton):
    """Chunk-causal GDN with **both** main and camera branches on Triton.

    Subclasses :class:`ChunkCausalGDNUCPESinglePathLiteLATriton` (which
    already rewires the main GDN scan through the fused kernel) and
    overrides only :meth:`_forward_cam_branch` to dispatch the camera branch
    through the fused cam-branch prep + scan kernels in
    :mod:`diffusion.model.ops.fused_cam_gdn`.

    All sub-modules and state-dict keys are inherited unchanged, so an
    existing :class:`ChunkCausalGDNUCPESinglePathLiteLA` checkpoint loads
    cleanly with zero conversion.

    Set ``use_autograd_kernel=True`` (inherited from parent ``__init__``) to
    enable autograd-mode kernels for both branches.  In autograd mode the
    cam branch dispatches through :func:`cam_prep_func_with_grad` +
    :func:`cam_scan_func_with_grad` (torch-recompute backward fallback);
    the main branch goes through :func:`fused_bigdn_forward_with_grad`,
    which now supports chunk-causal ``*_bwd`` masking (interior chunk
    boundaries are routed through the autograd wrapper's separate
    ``beta_bwd`` / ``decay_bwd`` gradient slots so the anti-causal scan
    resets state at every chunk boundary).

    Context-parallel training dispatches to the CP-Triton camera branch
    when block fusion is enabled, otherwise it falls back to the inherited
    eager camera branch.  ``frame_valid_mask`` and Q/V short convolutions
    are still rejected.  Calls without any chunk schedule (both
    ``chunk_size>=T`` and no ``chunk_index``) are also rejected because the
    per-chunk backward scan needs at least one boundary; callers wanting
    fully bidirectional attention should route through
    ``BidirectionalGDNUCPESinglePathLiteLA`` instead.  When ``chunk_index``
    is provided (e.g. staircase cold-start phase 0 with
    ``T=2, chunk_index=[0,1]``), the kernel respects it via
    :func:`normalize_chunk_index` and the length-1 fast-path produces the
    correct frame-causal output for cond positions.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Triton-fused chunk-causal camera branch.

        Matches the reference ``_forward_cam_branch`` signature, inputs,
        and return shape ``(B, N, cam_dim)``. The pipeline is:

            1. QKV linear (torch) + short conv on K (torch, reusing the
               parent's chunk-aware routing).
            2. Build UCPE projmats ``P``, ``P_T``, ``P_inv`` from
               ``camera_conditions`` via
               :func:`_process_camera_conditions_raymats_only` + SE(3) inverse.
            3. Slice ``rotary_emb`` to the cam-branch ``new_t/new_h/new_w``
               segments (same formulas as ``prepare_prope_fns_ucpe``) and
               convert to interleaved ``(N, D/2)`` cos/sin tables.
            4. Fused Triton prep kernel — RMSNorm + ReLU + K-scale + UCPE
               4x4 + RoPE in one pass, emits ``inflation_sq`` for Dynamic
               Beta Discounting.
            5. Adjust ``beta`` via the inflation-squared factor (mirrors the
               torch path).
            6. Forward scan (``reverse=False``) over the global sequence,
               then per-chunk backward scan (``reverse=True``) over each
               segment of ``valid_chunk_index``.
            7. Apply inverse UCPE (``apply_fn_o``) in torch.
        """
        if cp_enabled():
            from diffusion.distributed.context_parallel.config import (
                get_cp_triton_block_fusion,
            )

            if not get_cp_triton_block_fusion():
                raise NotImplementedError(
                    "ChunkCausalGDNUCPESinglePathLiteLABothTriton context-parallel "
                    "execution requires train.extra.cp.triton_block_fusion=true."
                )
            return self._forward_cam_branch_cp_triton_ag(x, HW, camera_conditions, rotary_emb, **kwargs)
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "ChunkCausalGDNUCPESinglePathLiteLABothTriton does not "
                "support frame_valid_mask (training-only feature)."
            )
        if self.conv_q_cam is not None or self.conv_v_cam is not None:
            raise NotImplementedError(
                "ChunkCausalGDNUCPESinglePathLiteLABothTriton requires "
                "k_conv_only=True (conv_q_cam / conv_v_cam must be None)."
            )

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        chunk_size = kwargs.get("chunk_size", None)
        chunk_index = kwargs.get("chunk_index", None)
        chunk_split_strategy = kwargs.get("chunk_split_strategy", "uniform")
        # The kernel requires either chunk_size<T (uniform multi-chunk
        # schedule) or an explicit chunk_index. Length-1 chunks reset the
        # reverse scan at that frame, so the position remains purely causal.
        if not is_chunk_causal_request(chunk_size, T, chunk_index):
            raise NotImplementedError(
                "ChunkCausalGDNUCPESinglePathLiteLABothTriton requires either "
                "chunk_size<T (multi-chunk schedule) or an explicit chunk_index. "
                f"Got chunk_size={chunk_size}, T={T}, chunk_index={chunk_index}."
            )

        # ---- 1. QKV linear + short conv on K -----------------------------
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = torch.nn.functional.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW, **kwargs)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

        # ---- 2. UCPE P, P_T, P_inv (inline, ignoring prope_fns cache) ----
        # We deliberately ignore ``kwargs["prope_fns"]`` here — introspecting
        # a cached closure for its internal matrices is brittle, and
        # recomputing via ``_process_camera_conditions_raymats_only`` +
        # SE(3) inverse costs ~0.9 ms per block which is acceptable.
        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        raymats = raymats.reshape(B, -1, 4, 4)
        P = raymats
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        # ---- 3. Sliced cam-branch RoPE + interleaved tables --------------
        if rotary_emb is not None:
            head_dim = D_head
            orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
            orig_h_size = head_dim // 6
            new_head_dim = head_dim // 2
            new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
            new_h_size = new_head_dim // 6
            new_w_size = new_head_dim // 6
            t_part = rotary_emb[..., :new_t_size]
            h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
            w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
            rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
            rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
        else:
            rotary_emb_cam = None
            rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
            rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

        # ---- 4. Fused Triton prep kernel ---------------------------------
        q_norm_w = self.q_norm_cam.weight.float().contiguous()
        k_norm_w = self.k_norm_cam.weight.float().contiguous()
        k_scale = (D_head**-0.5) * (S**-0.5)
        norm_eps_val = float(
            getattr(
                self.q_norm_cam,
                "eps",
                getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
            )
        )
        prep_fn = cam_prep_func_with_grad if getattr(self, "use_autograd_kernel", False) else cam_prep_func
        q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = prep_fn(
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            proj_q=P_T,
            proj_kv=P_inv,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            k_scale=k_scale,
            norm_eps=norm_eps_val,
        )
        inflation_sq = inflation_sq.view(B, H_heads, 1, N)

        # ---- 5. Gates + beta discounting --------------------------------
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, H_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # ---- 6. fp32 cast + broadcast beta to (B, H, F, S) --------------
        if getattr(self, "fp32_attention", True):
            q_cam_trans = q_cam_trans.float()
            k_cam_trans = k_cam_trans.float()
            v_cam_trans = v_cam_trans.float()
            beta = beta.float()
            decay = decay.float()
        if beta.ndim == 3:
            beta = beta.unsqueeze(-1).expand(B, H_heads, T, S).contiguous()
        else:
            assert beta.shape == (B, H_heads, T, S), f"beta shape {beta.shape}"
            beta = beta.contiguous()
        decay = decay.contiguous()

        q_cam_trans = q_cam_trans.contiguous()
        k_cam_trans = k_cam_trans.contiguous()
        v_cam_trans = v_cam_trans.contiguous()

        # ---- 7. Camera scan. --------------------------------------------
        scan_fn = cam_scan_func_with_grad if getattr(self, "use_autograd_kernel", False) else cam_scan_func
        valid_chunk_index, _ = normalize_chunk_index(chunk_index, T, chunk_size, chunk_split_strategy)
        beta_bwd, decay_bwd = _mask_reverse_gates_for_chunk_boundaries(beta, decay, valid_chunk_index)
        if getattr(self, "use_autograd_kernel", False):
            out_fwd = scan_fn(q_cam_trans, k_cam_trans, v_cam_trans, beta, decay, reverse=False)
            out_bwd = scan_fn(q_cam_trans, k_cam_trans, v_cam_trans, beta_bwd, decay_bwd, reverse=True)
            out = out_fwd + out_bwd
        else:
            out = cam_scan_pair_chunkwise(q_cam_trans, k_cam_trans, v_cam_trans, beta, decay, beta_bwd, decay_bwd)

        # ---- 8. Cast back to input dtype, then inverse UCPE -------------
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        _, _, apply_fn_o = _prepare_ray_apply_fns(
            head_dim=D_head,
            P=P,
            P_T=P_T,
            P_inv=P_inv,
            rotary_emb=rotary_emb_cam,
        )
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
        return out

    def _forward_cam_branch_cp_triton_ag(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """CP-aware Triton-fused chunk-causal camera branch.

        The forward branch uses :func:`cam_prep_func_with_grad` followed by
        :func:`cp_fused_cam_gdn_num_autograd`. The backward branch keeps
        chunk-local anti-causal semantics by running each rank's local chunks
        independently, so no cross-rank communication is needed for the
        reverse scan.

        Limitations / rejections:

        * ``frame_valid_mask`` (training-only feature) -> NotImplementedError.
        * Q/V short conv (``conv_q_cam`` / ``conv_v_cam``) -> NotImplementedError.
        * Multi-chunk schedule required (chunk_size < T_global or
          explicit chunk_index). Single-chunk falls through to eager
          parent (matches inherited chunk-causal semantics).

        Args:
            x: ``(B, N_local, in_dim)`` CP-local input slice.
            HW: ``(T_local, H, W)`` token layout for THIS rank.
            camera_conditions: ``(B, T_local, 20)`` UCPE inputs.
            rotary_emb: CP-local RoPE complex frequencies.

        Returns:
            ``(B, N_local, cam_dim)`` post-inverse-UCPE camera output.
        """
        import torch.distributed as dist

        from diffusion.distributed.context_parallel.distributed_scan import get_local_scan_cls
        from diffusion.distributed.context_parallel.config import get_cp_group
        from diffusion.model.nets.sana_gdn_blocks import flip_and_shift
        from diffusion.model.ops.frame_gdn.api import _build_transition_matrices
        from diffusion.model.ops.fused_gdn_cp import cp_fused_cam_gdn_num_autograd

        # ---- Guards. ----
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "ChunkCausalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch does not support frame_valid_mask (training-only feature)."
            )
        if self.conv_q_cam is not None or self.conv_v_cam is not None:
            raise NotImplementedError(
                "ChunkCausalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch requires k_conv_only=True (conv_q_cam / "
                "conv_v_cam must be None)."
            )

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        cp_group = get_cp_group()
        if cp_group is None:
            raise RuntimeError("_forward_cam_branch_cp_triton_ag called but CP group is not initialized.")
        cp_world = dist.get_world_size(cp_group)
        cp_rank_local = dist.get_rank(cp_group)

        chunk_size = kwargs.get("chunk_size", None)
        chunk_index = kwargs.get("chunk_index", None)
        chunk_split_strategy = kwargs.get("chunk_split_strategy", "uniform")
        T_global = T * cp_world
        if not is_chunk_causal_request(chunk_size, T_global, chunk_index):
            # No chunk schedule -> bidirectional camera path is handled
            # by the eager parent's fallback delegation. We mirror that
            # contract: when the request degenerates to bidirectional,
            # delegate to the bidi parent rather than carry chunk-causal
            # backward semantics that don't apply.
            return ChunkCausalGDNUCPESinglePathLiteLA._forward_cam_branch(
                self, x, HW, camera_conditions, rotary_emb, **kwargs
            )

        # ---- 1. QKV linear + short conv on K (mirrors non-CP cam branch). ----
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = torch.nn.functional.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW, **kwargs)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

        # ---- 2. UCPE P, P_T, P_inv (CP-local segment). ----
        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        raymats = raymats.reshape(B, -1, 4, 4)
        P = raymats
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        # ---- 3. Sliced cam-branch RoPE + interleaved tables (CP-local). ----
        if rotary_emb is not None:
            head_dim = D_head
            orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
            orig_h_size = head_dim // 6
            new_head_dim = head_dim // 2
            new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
            new_h_size = new_head_dim // 6
            new_w_size = new_head_dim // 6
            t_part = rotary_emb[..., :new_t_size]
            h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
            w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
            rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
            rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
        else:
            rotary_emb_cam = None
            rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
            rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

        # ---- 4. Fused Triton prep kernel (RMSNorm+ReLU+K-scale+UCPE+RoPE). ----
        q_norm_w = self.q_norm_cam.weight.float().contiguous()
        k_norm_w = self.k_norm_cam.weight.float().contiguous()
        k_scale = (D_head**-0.5) * (S**-0.5)
        norm_eps_val = float(
            getattr(
                self.q_norm_cam,
                "eps",
                getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
            )
        )
        # Always use autograd-enabled prep in the CP-Triton path so the
        # training graph stays connected back to qkv / norm weights.
        prep_fn = cam_prep_func_with_grad
        q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = prep_fn(
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            proj_q=P_T,
            proj_kv=P_inv,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            k_scale=k_scale,
            norm_eps=norm_eps_val,
        )
        inflation_sq = inflation_sq.view(B, H_heads, 1, N)

        # ---- 5. Gates + beta discounting (camera inflation-sq). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, H_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # ---- 6. fp32 cast + broadcast beta to (B, H, F, S). ----
        if getattr(self, "fp32_attention", True):
            q_cam_trans = q_cam_trans.float()
            k_cam_trans = k_cam_trans.float()
            v_cam_trans = v_cam_trans.float()
            beta = beta.float()
            decay = decay.float()
        if beta.ndim == 3:
            beta_bhfs = beta.unsqueeze(-1).expand(B, H_heads, T, S).contiguous()
        else:
            assert beta.shape == (B, H_heads, T, S), f"beta shape {beta.shape}"
            beta_bhfs = beta.contiguous()
        decay = decay.contiguous()

        q_cam_trans = q_cam_trans.contiguous()
        k_cam_trans = k_cam_trans.contiguous()
        v_cam_trans = v_cam_trans.contiguous()

        # ---- 7. Forward scan (CP-correct via cp_fused_cam_gdn_num_autograd). ----
        out_fwd, _ = cp_fused_cam_gdn_num_autograd(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta_bhfs,
            decay,
            F=T,
            S=S,
            group=cp_group,
            reverse_rank_order=False,
            truncate_to_active=None,
        )

        # ---- 8. Backward scan (per-chunk local). ----
        # Each rank runs the backward independently on its local chunks.
        # Chunk-isolated backward semantics make this correct without
        # cross-rank communication.
        global_offset = cp_rank_local * T
        valid_chunk_index_global, _ = normalize_chunk_index(
            chunk_index,
            T_global,
            chunk_size,
            chunk_split_strategy,
        )
        boundaries = [
            idx - global_offset
            for idx in valid_chunk_index_global
            if global_offset <= idx < global_offset + T and idx != 0
        ]

        local_chunk_ends = [0]
        for b in boundaries:
            if 0 < b < T:
                local_chunk_ends.append(b)
        local_chunk_ends.append(T)
        local_chunk_ends = sorted(set(local_chunk_ends))

        def _to_time_bwd(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4)

        q_rot_T = _to_time_bwd(q_cam_trans)
        k_rot_T = _to_time_bwd(k_cam_trans)
        v_T = _to_time_bwd(v_cam_trans)

        out_bwd_chunks: list[torch.Tensor] = []
        for ci in range(len(local_chunk_ends) - 1):
            cs = local_chunk_ends[ci]
            ce = local_chunk_ends[ci + 1]
            chunk_len = ce - cs

            # Size-1 chunk: anti-causal scan contributes zero (matches
            # eager frame-causal design).
            if chunk_len == 1:
                out_bwd_chunks.append(q_rot_T.new_zeros(B, H_heads, D_head, 1, S))
                continue

            q_chunk = q_rot_T[:, :, cs:ce]
            k_chunk = k_rot_T[:, :, cs:ce]
            v_chunk = v_T[:, :, cs:ce]
            if beta_bhfs.ndim == 4:
                beta_chunk = beta_bhfs[:, :, cs:ce, :]
            else:
                beta_chunk = beta_bhfs[:, :, cs:ce]
            decay_chunk = decay[:, :, cs:ce]

            q_bwd_c = torch.flip(q_chunk, dims=[2])
            k_bwd_c = flip_and_shift(k_chunk, dim=2, shift_val=0.0)
            v_bwd_c = flip_and_shift(v_chunk, dim=2, shift_val=0.0)
            beta_bwd_c = flip_and_shift(beta_chunk, dim=2, shift_val=0.0)
            decay_bwd_c = flip_and_shift(decay_chunk, dim=2, shift_val=1.0)

            beta_bwd_c = beta_bwd_c.unsqueeze(3)
            decay_bwd_c = decay_bwd_c.view(B, H_heads, chunk_len, 1, 1)
            identity = torch.eye(D_head, device=x.device, dtype=q_bwd_c.dtype).reshape(1, 1, 1, D_head, D_head)
            W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd = _build_transition_matrices(
                k_bwd_c,
                v_bwd_c,
                k_bwd_c,
                beta_bwd_c,
                decay_bwd_c,
                identity,
                B * H_heads,
                chunk_len,
                D_head,
            )
            W_z_bwd = torch.zeros_like(W_z_bwd)
            U_z_bwd = torch.zeros_like(U_z_bwd)
            local_scan = get_local_scan_cls(W_kv_bwd.is_cuda)
            S_kv_bwd, _ = local_scan.apply(W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd)
            S_kv_bwd = S_kv_bwd.view(B, H_heads, chunk_len, D_head, D_head)
            out_bwd_c = torch.matmul(S_kv_bwd, q_bwd_c)
            out_bwd_c = torch.flip(out_bwd_c, dims=[2]).permute(0, 1, 3, 2, 4).contiguous()
            out_bwd_chunks.append(out_bwd_c)

        out_bwd_cat = torch.cat(out_bwd_chunks, dim=3)
        out_bwd_flat = out_bwd_cat.reshape(B, H_heads, D_head, T * S)

        # ---- 9. Single-path combine: num_fwd + num_bwd (no divide). ----
        out = out_fwd + out_bwd_flat

        # ---- 10. Cast back, then inverse UCPE + projection. ----
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        _, _, apply_fn_o = _prepare_ray_apply_fns(
            head_dim=D_head,
            P=P,
            P_T=P_T,
            P_inv=P_inv,
            rotary_emb=rotary_emb_cam,
        )
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
        return out


# =========================================================================
#  Bidirectional variants
# =========================================================================
#
# The bidirectional path is strictly simpler than chunk-causal: neither the
# short convolution nor the GDN scan need to reset at chunk boundaries.
# We therefore:
#
#   * reuse the parent's bidirectional ``_apply_temporal_short_conv``
#     (forward + backward causal conv + average);
#   * call ``fused_bigdn_func`` with **no** ``*_bwd`` overrides, so the
#     kernel scans the full sequence in both directions;
#   * for the camera branch, run a single global forward scan and a single
#     global reverse scan — no per-chunk backward loop.


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDNTriton(BidirectionalGDN):
    """Bidirectional GDN with a fused Triton scan (inference + opt-in autograd).

    Subclasses :class:`BidirectionalGDN` and only overrides :meth:`__init__`
    (to accept ``use_autograd_kernel``) and :meth:`forward`.  Every learned
    sub-module (``qkv``, ``proj``, ``q_norm``, ``k_norm``, ``conv_k``,
    ``beta_proj``, ``gate_proj``, ``A_log``, ``dt_bias``, ``output_gate``)
    and helper (``_apply_temporal_short_conv``, ``_compute_frame_gates``,
    ``_apply_output_gate``) is inherited unchanged so existing checkpoints
    load with zero conversion.

    When ``use_autograd_kernel=True`` the fused-kernel call switches to
    :func:`fused_bigdn_forward_with_grad` (autograd-enabled, identical
    forward, real Triton backward kernel for the main branch).
    """

    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = use_autograd_kernel

    def _forward_cp_scan_triton_ag(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        """Context-parallel bidirectional GDN using fused prep/output kernels.

        The reverse branch materializes the exclusive anti-causal recurrence
        by flipping Q and flip-and-shifting K/V/beta/decay across CP ranks,
        then runs the same fused CP raw path with separate Q-side and K-side
        RoPE tables.

        Args:
            x: ``(B, N_local, C)`` CP-local input slice.
            mask: Unused (API symmetry).
            HW: ``(T_local, H, W)`` token layout for THIS rank.
            rotary_emb: CP-local RoPE complex frequencies.
            block_mask: Unused (API symmetry).
            apply_output_gate: When False, return raw attention output
                before gate + projection.

        Returns:
            ``(B, N_local, C)`` after attention + (optional) output gate
            + projection.
        """
        import torch.distributed as dist

        from diffusion.distributed.context_parallel.config import get_cp_group
        from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
        from diffusion.model.ops.fused_gdn_cp import cp_fused_gdn_chunkwise_raw_autograd

        del mask, block_mask  # unused on this path

        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNTriton CP-Triton path does not support " "frame_valid_mask (training-only feature)."
            )
        if self.conv_q is not None or self.conv_v is not None:
            raise NotImplementedError(
                "BidirectionalGDNTriton CP-Triton path supports k_conv_only="
                "True; got conv_q or conv_v which would require additional "
                "Triton paths."
            )
        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        if N != T * S:
            raise ValueError(f"N={N} != T*S={T * S} for HW={HW}.")
        if C != H * D:
            raise ValueError(f"C={C} != heads*dim={H * D}.")

        cp_group = get_cp_group()
        if cp_group is None:
            raise RuntimeError(
                "BidirectionalGDNTriton._forward_cp_scan_triton_ag called but " "CP group is not initialized."
            )

        # ---- 1. QKV projection on the CP-local slice. ---------------------
        qkv = self.qkv(x).reshape(B, N, 3, H, D)

        # ---- 2. Bidirectional short conv on K (parent method). ----
        if self.conv_k is not None:
            k_raw = qkv[:, :, 1].contiguous().reshape(B, N, C)
            k_conv = self._apply_temporal_short_conv(k_raw, self.conv_k, HW)
            qkv = qkv.clone()
            qkv[:, :, 1] = k_conv.reshape(B, N, H, D)

        # ---- 3. Frame gates (precomputed when shared with cam branch). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.contiguous()
        decay = decay.contiguous()

        # ---- 4. Full-channel RMSNorm weights + norm_eps. -----------------
        if not isinstance(self.q_norm, nn.Identity):
            q_nw = self.q_norm.weight.float().contiguous()
            k_nw = self.k_norm.weight.float().contiguous()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        else:
            q_nw = None
            k_nw = None
            norm_eps = 1e-5

        # ---- 5. CP-local RoPE tables. ------------------------------------
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, x.device)

        # ---- 6. K scale via the model's documented mode. -----------------
        k_scale = self._key_scale(S)

        # ---- 7. dot_precision: fp32 inputs -> IEEE fp32 bridge (2);
        # bf16/fp16 -> TF32 bf16 bridge (0).
        dot_precision = 2 if x.dtype == torch.float32 else 0

        # Forward branch.
        res = cp_fused_gdn_chunkwise_raw_autograd(
            qkv,
            beta,
            decay,
            q_nw,
            k_nw,
            rope_cos,
            rope_sin,
            F=T,
            S=S,
            group=cp_group,
            k_scale=k_scale,
            norm_eps=norm_eps,
            eps=self.eps,
            dot_precision=dot_precision,
            reverse_rank_order=False,
            truncate_to_active=None,
        )
        num_fwd, den_fwd = res.num, res.den

        # Reverse branch.
        cp_world = dist.get_world_size(cp_group)
        cp_rank_local = dist.get_rank(cp_group)

        def _cp_flip_and_shift(tensors: list[torch.Tensor], shift_vals: list[float]) -> list[torch.Tensor]:
            is_last = cp_rank_local == cp_world - 1
            results = []
            for tensor, sv in zip(tensors, shift_vals):
                first_frame = tensor[:, :, :1, ...].contiguous()
                haloed = cp_halo_exchange(first_frame, left_size=0, right_size=1, dim=2, group=cp_group)
                boundary = haloed[:, :, 1:2, ...]
                if is_last and sv != 0.0:
                    boundary = boundary.mul(0.0).add(sv)
                T_loc = tensor.shape[2]
                flipped = torch.flip(tensor, dims=[2])
                body = flipped[:, :, : T_loc - 1, ...]
                results.append(torch.cat([boundary, body], dim=2))
            return results

        def _bnhd_to_frame(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 2, 3, 1).reshape(B, H, D, T, S).permute(0, 1, 3, 2, 4).contiguous()

        def _frame_to_bnhd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 2, 4, 1, 3).reshape(B, T * S, H, D).contiguous()

        q_raw_f = _bnhd_to_frame(qkv[:, :, 0])
        k_raw_f = _bnhd_to_frame(qkv[:, :, 1])
        v_raw_f = _bnhd_to_frame(qkv[:, :, 2])
        q_bwd_f = torch.flip(q_raw_f, dims=[2])
        k_bwd_f, v_bwd_f = _cp_flip_and_shift([k_raw_f, v_raw_f], [0.0, 0.0])
        qkv_bwd = torch.stack(
            [
                _frame_to_bnhd(q_bwd_f),
                _frame_to_bnhd(k_bwd_f),
                _frame_to_bnhd(v_bwd_f),
            ],
            dim=2,
        )

        beta_f = beta.unsqueeze(3)
        decay_f = decay.view(B, H, T, 1, 1)
        beta_bwd_f, decay_bwd_f = _cp_flip_and_shift([beta_f, decay_f], [0.0, 1.0])
        beta_bwd = beta_bwd_f.squeeze(3).contiguous()
        decay_bwd = decay_bwd_f.squeeze(-1).squeeze(-1).contiguous()

        def _rope_to_frame(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(T, S, D).permute(0, 2, 1).reshape(1, 1, T, D, S).contiguous()

        def _rope_from_frame(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(T, D, S).permute(0, 2, 1).reshape(T * S, D).contiguous()

        rope_cos_f = _rope_to_frame(rope_cos)
        rope_sin_f = _rope_to_frame(rope_sin)
        rope_cos_q = _rope_from_frame(torch.flip(rope_cos_f, dims=[2]))
        rope_sin_q = _rope_from_frame(torch.flip(rope_sin_f, dims=[2]))
        rope_cos_k_f, rope_sin_k_f = _cp_flip_and_shift([rope_cos_f, rope_sin_f], [1.0, 0.0])
        rope_cos_k = _rope_from_frame(rope_cos_k_f)
        rope_sin_k = _rope_from_frame(rope_sin_k_f)

        res_bwd = cp_fused_gdn_chunkwise_raw_autograd(
            qkv_bwd,
            beta_bwd,
            decay_bwd,
            q_nw,
            k_nw,
            rope_cos_k,
            rope_sin_k,
            F=T,
            S=S,
            group=cp_group,
            k_scale=k_scale,
            norm_eps=norm_eps,
            eps=self.eps,
            dot_precision=dot_precision,
            reverse_rank_order=True,
            truncate_to_active=None,
            rope_cos_q=rope_cos_q,
            rope_sin_q=rope_sin_q,
        )
        num_bwd_flipped = res_bwd.num.reshape(B, T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()
        den_bwd_flipped = res_bwd.den.reshape(B, H, T, S).unsqueeze(3).contiguous()
        num_bwd_eager = torch.flip(num_bwd_flipped, dims=[2])  # (B, H, T, D, S)
        den_bwd_eager = torch.flip(den_bwd_flipped, dims=[2])  # (B, H, T, 1, S)

        num_fwd_5d = num_fwd.reshape(B, T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()
        den_fwd_5d = den_fwd.reshape(B, H, T, S).unsqueeze(3).contiguous()

        total_num = num_fwd_5d.float() + num_bwd_eager.float()
        total_den = den_fwd_5d.float() + den_bwd_eager.float()

        out = total_num / (total_den + self.eps)  # (B, H, T, D, S)
        if getattr(self, "fp32_attention", True) and x.dtype != torch.float32:
            out = out.to(x.dtype)

        out = out.permute(0, 1, 3, 2, 4).reshape(B, self.heads, D, N)
        out = out.permute(0, 3, 1, 2).reshape(B, N, C)

        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        apply_output_gate: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        if HW is None:
            raise ValueError("BidirectionalGDNTriton requires HW=(T, H, W).")
        if cp_enabled():
            from diffusion.distributed.context_parallel.config import (
                get_cp_triton_block_fusion,
            )

            if not get_cp_triton_block_fusion():
                raise NotImplementedError(
                    "BidirectionalGDNTriton context-parallel execution requires "
                    "train.extra.cp.triton_block_fusion=true."
                )
            return self._forward_cp_scan_triton_ag(
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=apply_output_gate,
                **kwargs,
            )
        del mask, block_mask  # unused in the bidirectional Triton path
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNTriton does not support frame_valid_mask (training-only feature)."
            )
        if self.conv_q is not None or self.conv_v is not None:
            raise NotImplementedError("BidirectionalGDNTriton requires k_conv_only=True; got conv_q or conv_v.")

        B, N, C = x.shape
        T, H_s, W_s = HW
        S = H_s * W_s
        H, D = self.heads, self.dim
        if N != T * S:
            raise ValueError(f"N={N} != T*S={T * S} for HW={HW}.")
        if C != H * D:
            raise ValueError(f"C={C} != heads*dim={H * D}.")

        # ---- 1. QKV projection -> (B, N, 3, H, D), kept contiguous. -------
        qkv = self.qkv(x).reshape(B, N, 3, H, D)

        # ---- 2. Bidirectional short conv on K (parent method).  ----------
        # ``BidirectionalGDN._apply_temporal_short_conv`` runs the causal
        # conv forward + backward then averages, giving a symmetric filter
        # with one set of weights.  Inherited unchanged.
        if self.conv_k is not None:
            k_raw = qkv[:, :, 1].contiguous().reshape(B, N, C)
            k_conv = self._apply_temporal_short_conv(k_raw, self.conv_k, HW)
            qkv[:, :, 1].copy_(k_conv.reshape(B, N, H, D))

        # ---- 3. Frame gates (precomputed when shared with cam branch). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)
        beta = beta.contiguous()
        decay = decay.contiguous()

        # ---- 4. Full-channel RMSNorm weights. -----------------------------
        if not isinstance(self.q_norm, nn.Identity):
            q_nw = self.q_norm.weight.float().contiguous()
            k_nw = self.k_norm.weight.float().contiguous()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
        else:
            q_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            k_nw = torch.ones(C, device=x.device, dtype=torch.float32)
            norm_eps = 1e-5

        # ---- 5. Fused Q+K inverse-RMS (single Triton launch). -------------
        q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=norm_eps)

        # ---- 6. Expanded RoPE cos/sin tables (N, D). ---------------------
        rope_cos, rope_sin = prepare_rope_tables(rotary_emb, N, D, x.device)

        # ---- 7. K scale absorbs Q/K^T variance + spatial mean-pool. -----
        k_scale = (D**-0.5) * (S**-0.5)

        # ---- 8. Fused bidirectional Triton scan over the full sequence. --
        # No ``*_bwd`` overrides: the kernel's ``reverse=True`` path already
        # implements the exclusive (t+1..T) reverse recurrence, matching the
        # torch ``flip_and_shift`` semantics used in ``BidirectionalGDN``.
        if getattr(self, "use_autograd_kernel", False):
            # Autograd path: the wrapper recomputes inv-RMS internally so q/k
            # norm backward flows naturally; full-sequence bidirectional only.
            out = fused_bigdn_forward_with_grad(
                qkv,
                beta,
                decay,
                q_nw,
                k_nw,
                rope_cos,
                rope_sin,
                F=T,
                S=S,
                k_scale=k_scale,
                norm_eps=norm_eps,
                eps=self.eps,
            )
        else:
            out = fused_bigdn_func(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_norm_weight=q_nw,
                k_norm_weight=k_nw,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                beta=beta,
                decay=decay,
                F=T,
                S=S,
                k_scale=k_scale,
                eps=self.eps,
            )  # (B, N, H, D)

        # ---- 9. Output gate + projection. --------------------------------
        out = out.reshape(B, N, C)
        if apply_output_gate:
            out = self._apply_output_gate(out, x)
            out = self.proj(out.to(self.proj.weight.dtype))
        return out


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDNUCPESinglePathLiteLATriton(BidirectionalGDNUCPESinglePathLiteLA):
    """Bidirectional UCPE camera-controlled GDN with a Triton main branch.

    Inherits the entire camera branch (``_forward_cam_branch``),
    ``_prepare_cam_qkv``, every sub-module and every checkpoint key from
    :class:`BidirectionalGDNUCPESinglePathLiteLA`.  The **only** behavioural
    delta is that the main-branch GDN scan dispatches through
    :class:`BidirectionalGDNTriton.forward` instead of the inherited
    :class:`BidirectionalGDN.forward`.

    Because ``_GDNUCPEBase.forward`` routes the main branch via
    ``super().forward(...)`` — which MRO-resolves to
    :class:`BidirectionalGDN`, not our Triton variant — we re-implement the
    dual-branch forward here to explicitly call
    ``BidirectionalGDNTriton.forward(self, ...)``.  The body is otherwise
    bit-identical to the parent's ``forward``.

    The ``use_autograd_kernel`` flag is stored on this instance and consulted
    inside :meth:`BidirectionalGDNTriton.forward` (the dispatch passes
    ``self``, so the flag is visible to the main-branch forward).  The cam
    branch is the inherited torch path; use
    :class:`BidirectionalGDNUCPESinglePathLiteLABothTriton` for a fully
    Triton + autograd-aware cam branch.
    """

    # This class does not inherit from BidirectionalGDNTriton directly.
    _forward_cp_scan_triton_ag = BidirectionalGDNTriton._forward_cp_scan_triton_ag

    def __init__(self, *args, use_autograd_kernel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autograd_kernel = use_autograd_kernel

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        HW: tuple[int, int, int] | None = None,
        rotary_emb: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        camera_conditions: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Pre-compute shared gates once for both branches.
        if HW is not None:
            precomputed_gates = self._compute_frame_gates(x, HW)
        else:
            precomputed_gates = None

        # Main branch — Triton-fused bidirectional scan.
        main_raw = BidirectionalGDNTriton.forward(
            self,
            x,
            mask=mask,
            HW=HW,
            rotary_emb=rotary_emb,
            block_mask=block_mask,
            apply_output_gate=False,
            chunk_size=chunk_size,
            precomputed_gates=precomputed_gates,
            **kwargs,
        )

        # Camera branch (inherited torch implementation).
        cam_contrib: torch.Tensor | int = 0
        camera_conditions = _maybe_drop_cam_branch(
            camera_conditions,
            kwargs.get("cam_branch_drop_prob", 0.0),
            self.training,
            x.device,
        )
        if camera_conditions is not None:
            if HW is None:
                raise ValueError("HW (T, H, W) must be provided for UCPE camera branch.")
            cam_raw = self._forward_cam_branch(
                x,
                HW,
                camera_conditions,
                rotary_emb,
                chunk_size=chunk_size,
                precomputed_gates=precomputed_gates,
                **kwargs,
            )
            cam_contrib = self.out_proj_cam(cam_raw)

        combined = main_raw + cam_contrib
        combined = self._apply_output_gate(combined, x)
        return self.proj(combined.to(self.proj.weight.dtype))


@ATTENTION_BLOCKS.register_module()
class BidirectionalGDNUCPESinglePathLiteLABothTriton(BidirectionalGDNUCPESinglePathLiteLATriton):
    """Bidirectional UCPE camera-controlled GDN with **both** branches on Triton.

    Subclasses :class:`BidirectionalGDNUCPESinglePathLiteLATriton` (which
    already rewires the main GDN scan) and replaces
    :meth:`_forward_cam_branch` with a fused Triton camera pipeline:

        1. Torch QKV linear + bidirectional short conv on K.
        2. UCPE ``P / P_T / P_inv`` from ``camera_conditions``.
        3. Sliced cam-branch RoPE → interleaved ``(N, D/2)`` cos/sin tables.
        4. Fused prep kernel (RMSNorm + ReLU + K-scale + UCPE 4x4 + RoPE),
           emitting ``inflation_sq`` for Dynamic Beta Discounting.
        5. Beta discounting via ``inflation_sq`` (mirrors torch path).
        6. Fused forward scan (``reverse=False``) over the full sequence.
        7. Fused reverse scan (``reverse=True``) over the full sequence —
           the kernel applies flip-and-shift internally, so no per-chunk
           loop is needed.
        8. Inverse UCPE (``apply_fn_o``) in torch.

    State-dict keys are identical to
    :class:`BidirectionalGDNUCPESinglePathLiteLA`.

    Set ``use_autograd_kernel=True`` (inherited from
    :class:`BidirectionalGDNUCPESinglePathLiteLATriton`) to enable autograd
    mode for both branches: the main branch goes through
    :func:`fused_bigdn_forward_with_grad` and the cam branch through
    :func:`cam_prep_func_with_grad` + :func:`cam_scan_func_with_grad`
    (torch-recompute backward fallback).  Forward cost is unchanged.
    """

    def _forward_cam_branch(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        # ---- Guards: no CP, k_conv_only=True (apply in either mode). ----
        if cp_enabled():
            from diffusion.distributed.context_parallel.config import (
                get_cp_triton_block_fusion,
            )

            if not get_cp_triton_block_fusion():
                raise NotImplementedError(
                    "BidirectionalGDNUCPESinglePathLiteLABothTriton context-parallel "
                    "execution requires train.extra.cp.triton_block_fusion=true."
                )
            return self._forward_cam_branch_cp_triton_ag(x, HW, camera_conditions, rotary_emb, **kwargs)
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton does not "
                "support frame_valid_mask (training-only feature)."
            )
        if self.conv_q_cam is not None or self.conv_v_cam is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton requires "
                "k_conv_only=True (conv_q_cam / conv_v_cam must be None)."
            )

        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        # ---- 1. QKV linear + bidirectional short conv on K ---------------
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = torch.nn.functional.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            # Parent routing (BidirectionalGDN) gives the bidirectional
            # forward+backward causal conv + average.
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

        # ---- 2. UCPE P, P_T, P_inv (inline; skip cached prope_fns). -----
        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        raymats = raymats.reshape(B, -1, 4, 4)
        P = raymats
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        # ---- 3. Sliced cam-branch RoPE + interleaved tables. ------------
        if rotary_emb is not None:
            head_dim = D_head
            orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
            orig_h_size = head_dim // 6
            new_head_dim = head_dim // 2
            new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
            new_h_size = new_head_dim // 6
            new_w_size = new_head_dim // 6
            t_part = rotary_emb[..., :new_t_size]
            h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
            w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
            rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
            rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
        else:
            rotary_emb_cam = None
            rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
            rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

        # ---- 4. Fused Triton prep kernel --------------------------------
        q_norm_w = self.q_norm_cam.weight.float().contiguous()
        k_norm_w = self.k_norm_cam.weight.float().contiguous()
        k_scale = (D_head**-0.5) * (S**-0.5)
        norm_eps_val = float(
            getattr(
                self.q_norm_cam,
                "eps",
                getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
            )
        )
        prep_fn = cam_prep_func_with_grad if getattr(self, "use_autograd_kernel", False) else cam_prep_func
        q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = prep_fn(
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            proj_q=P_T,
            proj_kv=P_inv,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            k_scale=k_scale,
            norm_eps=norm_eps_val,
        )
        inflation_sq = inflation_sq.view(B, H_heads, 1, N)

        # ---- 5. Gates + beta discounting -------------------------------
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, H_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # ---- 6. fp32 cast + broadcast beta to (B, H, F, S) -------------
        if getattr(self, "fp32_attention", True):
            q_cam_trans = q_cam_trans.float()
            k_cam_trans = k_cam_trans.float()
            v_cam_trans = v_cam_trans.float()
            beta = beta.float()
            decay = decay.float()
        if beta.ndim == 3:
            beta = beta.unsqueeze(-1).expand(B, H_heads, T, S).contiguous()
        else:
            assert beta.shape == (B, H_heads, T, S), f"beta shape {beta.shape}"
            beta = beta.contiguous()
        decay = decay.contiguous()

        q_cam_trans = q_cam_trans.contiguous()
        k_cam_trans = k_cam_trans.contiguous()
        v_cam_trans = v_cam_trans.contiguous()

        # ---- 7. Fused bidirectional scan. ------------------------------
        if getattr(self, "use_autograd_kernel", False):
            # Keep the autograd path on the explicit scan calls so gradients
            # continue to flow through the existing CamScanFunction wrapper.
            scan_fn = cam_scan_func_with_grad
            out_fwd = scan_fn(q_cam_trans, k_cam_trans, v_cam_trans, beta, decay, reverse=False)
            out_bwd = scan_fn(q_cam_trans, k_cam_trans, v_cam_trans, beta, decay, reverse=True)
            out = out_fwd + out_bwd
        else:
            out = cam_scan_bidi_chunkwise(q_cam_trans, k_cam_trans, v_cam_trans, beta, decay)

        # ---- 8. Cast back to input dtype, then inverse UCPE. -----------
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        _, _, apply_fn_o = _prepare_ray_apply_fns(
            head_dim=D_head,
            P=P,
            P_T=P_T,
            P_inv=P_inv,
            rotary_emb=rotary_emb_cam,
        )
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
        return out

    def _forward_cam_branch_cp_triton_ag(
        self,
        x: torch.Tensor,
        HW: tuple[int, int, int],
        camera_conditions: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Bidirectional CP-aware Triton-fused camera branch.

        * **Forward branch** uses Triton-fused
          :func:`cam_prep_func_with_grad` (RMSNorm + ReLU + K-scale +
          UCPE-projmat + RoPE) followed by the autograd-aware
          :func:`cp_fused_cam_gdn_num_autograd` (pure-PyTorch transition
          build + :func:`cp_frame_gdn_scan` + num-only output projection).
        * **Backward branch** uses eager ``_cp_flip_and_shift`` +
          :func:`_build_transition_matrices` +
          :func:`cp_frame_gdn_scan(reverse=True)` + matmul output.

        * NO chunk boundary masking on the backward branch.
        * NO per-chunk local non-CP backward loop -- a single global CP
          reverse scan is used (the eager bidi cam path).
        * Bidirectional camera runs full sequence end-to-end.
        """
        import torch.distributed as dist

        from diffusion.distributed.context_parallel.config import get_cp_group
        from diffusion.distributed.context_parallel.distributed_scan import cp_frame_gdn_scan
        from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
        from diffusion.model.ops.frame_gdn.api import _build_transition_matrices
        from diffusion.model.ops.fused_gdn_cp import cp_fused_cam_gdn_num_autograd

        # ---- Guards: training-only / Q-V conv rejections. ----
        if kwargs.get("frame_valid_mask", None) is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch does not support frame_valid_mask (training-only feature)."
            )
        if self.conv_q_cam is not None or self.conv_v_cam is not None:
            raise NotImplementedError(
                "BidirectionalGDNUCPESinglePathLiteLABothTriton CP-Triton "
                "cam branch requires k_conv_only=True (conv_q_cam / "
                "conv_v_cam must be None)."
            )
        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        dtype_orig = x.dtype
        H_heads = self.cam_heads
        D_head = self.cam_head_dim

        cp_group = get_cp_group()
        if cp_group is None:
            raise RuntimeError("_forward_cam_branch_cp_triton_ag (bidi) called but CP group is not initialized.")

        # ---- 1. QKV linear + bidirectional short conv on K (CP-local). ----
        qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
        qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
        qkv_cam = torch.nn.functional.linear(x, qkv_w, qkv_b)
        q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)

        if self.conv_k_cam is not None:
            # Parent (BidirectionalGDN._apply_temporal_short_conv) handles
            # CP halo exchange internally.
            k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW)

        q_raw = q_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        k_raw = k_raw.contiguous().view(B, N, H_heads, D_head).contiguous()
        v_raw = v_raw.contiguous().view(B, N, H_heads, D_head).contiguous()

        # ---- 2. UCPE P, P_T, P_inv (CP-local segment). ----
        raymats = _process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
        raymats = raymats.reshape(B, -1, 4, 4)
        P = raymats
        P_T = P.transpose(-1, -2).contiguous()
        P_inv = _invert_SE3(P).contiguous()

        # ---- 3. Sliced cam-branch RoPE + interleaved tables (CP-local). ----
        if rotary_emb is not None:
            head_dim = D_head
            orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
            orig_h_size = head_dim // 6
            new_head_dim = head_dim // 2
            new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
            new_h_size = new_head_dim // 6
            new_w_size = new_head_dim // 6
            t_part = rotary_emb[..., :new_t_size]
            h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
            w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
            rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
            rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, N, D_head // 2, x.device)
        else:
            rotary_emb_cam = None
            rope_cos = torch.ones(N, D_head // 2, device=x.device, dtype=torch.float32)
            rope_sin = torch.zeros(N, D_head // 2, device=x.device, dtype=torch.float32)

        # ---- 4. Fused Triton prep kernel (RMSNorm+ReLU+K-scale+UCPE+RoPE). ----
        q_norm_w = self.q_norm_cam.weight.float().contiguous()
        k_norm_w = self.k_norm_cam.weight.float().contiguous()
        k_scale = (D_head**-0.5) * (S**-0.5)
        norm_eps_val = float(
            getattr(
                self.q_norm_cam,
                "eps",
                getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
            )
        )
        # Always use autograd-enabled prep so the training graph stays
        # connected back to qkv / norm weights.
        prep_fn = cam_prep_func_with_grad
        q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = prep_fn(
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            proj_q=P_T,
            proj_kv=P_inv,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            k_scale=k_scale,
            norm_eps=norm_eps_val,
        )
        inflation_sq = inflation_sq.view(B, H_heads, 1, N)

        # ---- 5. Gates + beta discounting (camera inflation-sq). ----
        precomputed_gates = kwargs.get("precomputed_gates", None)
        if precomputed_gates is not None:
            beta, decay = precomputed_gates
        else:
            beta, decay = self._compute_frame_gates(x, HW)

        inflation_sq_spatial = inflation_sq.view(B, H_heads, T, S)
        frame_inflation_sq = inflation_sq_spatial.mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        # ---- 6. fp32 cast + broadcast beta to (B, H, F, S). ----
        if getattr(self, "fp32_attention", True):
            q_cam_trans = q_cam_trans.float()
            k_cam_trans = k_cam_trans.float()
            v_cam_trans = v_cam_trans.float()
            beta = beta.float()
            decay = decay.float()
        if beta.ndim == 3:
            beta_bhfs = beta.unsqueeze(-1).expand(B, H_heads, T, S).contiguous()
        else:
            assert beta.shape == (B, H_heads, T, S), f"beta shape {beta.shape}"
            beta_bhfs = beta.contiguous()
        decay = decay.contiguous()

        q_cam_trans = q_cam_trans.contiguous()
        k_cam_trans = k_cam_trans.contiguous()
        v_cam_trans = v_cam_trans.contiguous()

        # ---- 7. Forward branch. ----
        out_fwd, _ = cp_fused_cam_gdn_num_autograd(
            q_cam_trans,
            k_cam_trans,
            v_cam_trans,
            beta_bhfs,
            decay,
            F=T,
            S=S,
            group=cp_group,
            reverse_rank_order=False,
            truncate_to_active=None,
        )  # (B, H, D, N_local)

        # ---- 8. Backward branch. ----
        # Reshape (B, H, D, N) -> frame layout (B, H, T, D, S) so we can
        # reuse _build_transition_matrices.
        BH = B * H_heads

        def _to_frame(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, H_heads, D_head, T, S).permute(0, 1, 3, 2, 4).contiguous()

        q_rot_f = _to_frame(q_cam_trans)
        k_rot_f = _to_frame(k_cam_trans)
        v_f = _to_frame(v_cam_trans)
        beta_f = beta_bhfs.unsqueeze(3)  # (B, H, T, 1, S)
        decay_f = decay.view(B, H_heads, T, 1, 1)
        I = torch.eye(D_head, device=x.device, dtype=q_rot_f.dtype).reshape(1, 1, 1, D_head, D_head)

        # Distributed flip+shift across CP ranks (no chunk-boundary
        # masking; bidi cam is full-sequence).
        cp_world = dist.get_world_size(cp_group)
        cp_rank_local = dist.get_rank(cp_group)

        def _cp_flip_and_shift(tensors: list[torch.Tensor], shift_vals: list[float]) -> list[torch.Tensor]:
            is_last = cp_rank_local == cp_world - 1
            results = []
            for tensor, sv in zip(tensors, shift_vals):
                first_frame = tensor[:, :, :1, ...].contiguous()
                haloed = cp_halo_exchange(first_frame, left_size=0, right_size=1, dim=2, group=cp_group)
                boundary = haloed[:, :, 1:2, ...]
                if is_last and sv != 0.0:
                    boundary = boundary.mul(0.0).add(sv)
                T_loc = tensor.shape[2]
                flipped = torch.flip(tensor, dims=[2])
                body = flipped[:, :, : T_loc - 1, ...]
                results.append(torch.cat([boundary, body], dim=2))
            return results

        q_rot_bwd_f = torch.flip(q_rot_f, dims=[2])
        k_rot_bwd_f, v_bwd_f, beta_bwd_f, decay_bwd_f = _cp_flip_and_shift(
            [k_rot_f, v_f, beta_f, decay_f],
            [0.0, 0.0, 0.0, 1.0],
        )

        # Camera single-path passes k_rot in BOTH k_f and k_rot_f slots
        # (mirrors sana_gdn_camctrl_blocks.py:1442-1452 -- single-path
        # uses rotated keys only).
        W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd = _build_transition_matrices(
            k_rot_bwd_f,
            v_bwd_f,
            k_rot_bwd_f,
            beta_bwd_f,
            decay_bwd_f,
            I,
            BH,
            T,
            D_head,
        )
        # Zero Z component -- single-path numerator-only has no
        # denominator. Matches sana_gdn_camctrl_blocks.py:1453-1454.
        W_z_bwd = torch.zeros_like(W_z_bwd)
        U_z_bwd = torch.zeros_like(U_z_bwd)

        S_kv_bwd, _ = cp_frame_gdn_scan(W_kv_bwd, U_kv_bwd, W_z_bwd, U_z_bwd, cp_group, reverse=True)
        S_kv_bwd = S_kv_bwd.view(B, H_heads, T, D_head, D_head)
        # Num-only output projection: out = S_kv_bwd @ q_rot_bwd_f.
        out_bwd_flipped_5d = torch.matmul(S_kv_bwd, q_rot_bwd_f)  # (B, H, T, D, S)
        out_bwd_5d = torch.flip(out_bwd_flipped_5d, dims=[2])  # back to original frame order
        # (B, H, T, D, S) -> (B, H, D, T, S) -> (B, H, D, N).
        out_bwd = out_bwd_5d.permute(0, 1, 3, 2, 4).reshape(B, H_heads, D_head, N).contiguous()

        # ============================================================
        #  COMBINE: out = out_fwd + out_bwd (num-only, no divide).
        # ============================================================
        out = out_fwd + out_bwd

        # ---- 9. Cast back, then inverse UCPE + projection. ----
        if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
            out = out.to(dtype_orig)

        _, _, apply_fn_o = _prepare_ray_apply_fns(
            head_dim=D_head,
            P=P,
            P_T=P_T,
            P_inv=P_inv,
            rotary_emb=rotary_emb_cam,
        )
        out = apply_fn_o(out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        out = out.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
        return out
