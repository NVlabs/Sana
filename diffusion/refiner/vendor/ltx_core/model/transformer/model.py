from enum import Enum

import torch

from diffusion.refiner.vendor.ltx_core.guidance.perturbations import BatchedPerturbationConfig
from diffusion.refiner.vendor.ltx_core.model.transformer.adaln import (
    AdaLayerNormSingle,
    adaln_embedding_coefficient,
)
from diffusion.refiner.vendor.ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from diffusion.refiner.vendor.ltx_core.model.transformer.modality import Modality
from diffusion.refiner.vendor.ltx_core.model.transformer.rope import LTXRopeType
from diffusion.refiner.vendor.ltx_core.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
)
from diffusion.refiner.vendor.ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from diffusion.refiner.vendor.ltx_core.utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if self._enable_gradient_checkpointing and self.training:
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

            # Optional: capture block outputs for distillation
            # Single block: _capture_block_idx (legacy)
            capture_idx = getattr(self, "_capture_block_idx", None)
            if capture_idx is not None and idx == capture_idx:
                self._captured_block_output = video.x if video is not None else None
            # Multiple blocks: _capture_block_indices (set of ints)
            capture_indices = getattr(self, "_capture_block_indices", None)
            if capture_indices is not None and idx in capture_indices:
                if not hasattr(self, "_captured_block_outputs"):
                    self._captured_block_outputs = {}
                self._captured_block_outputs[idx] = video.x if video is not None else None

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        # Teacher-forcing two-pass: set _tf_context before calling forward().
        # Runs inside FSDP wrapper so root params (patchify_proj, adaln) are materialized.
        tf_context = getattr(self, "_tf_context", None)
        if tf_context is not None:
            return self._forward_teacher_forcing(video, tf_context, perturbations), None

        vx, ax = self.velocity_model(video, audio, perturbations)
        # Under context parallel, vx is already sharded along the seq dim. The
        # original modality.latent / modality.timesteps are still full-length —
        # slice them to match vx before the element-wise to_denoised math.
        from diffusion.refiner.ltx_core.context_parallel import (
            cp_enabled,
            shard_along_seq,
        )

        if cp_enabled() and vx is not None:
            v_lat = shard_along_seq(video.latent, dim=1)
            v_ts = (
                shard_along_seq(video.timesteps, dim=1)
                if video.timesteps.dim() >= 2 and video.timesteps.shape[1] > 1
                else video.timesteps
            )
        else:
            v_lat = video.latent if video is not None else None
            v_ts = video.timesteps if video is not None else None
        if cp_enabled() and ax is not None:
            a_lat = shard_along_seq(audio.latent, dim=1)
            a_ts = (
                shard_along_seq(audio.timesteps, dim=1)
                if audio.timesteps.dim() >= 2 and audio.timesteps.shape[1] > 1
                else audio.timesteps
            )
        else:
            a_lat = audio.latent if audio is not None else None
            a_ts = audio.timesteps if audio is not None else None
        denoised_video = to_denoised(v_lat, vx, v_ts) if vx is not None else None
        denoised_audio = to_denoised(a_lat, ax, a_ts) if ax is not None else None
        return denoised_video, denoised_audio

    def _forward_teacher_forcing(
        self,
        target_video: Modality,
        context_video: Modality,
        perturbations: BatchedPerturbationConfig,
    ) -> torch.Tensor:
        """Two-pass forward: context (no_grad, σ=0) → KV → target (grad, σ=σ_t).

        Called from forward() when _tf_context is set, so FSDP root params are
        already materialized.  Each block is also individually FSDP-sharded.
        """
        vm = self.velocity_model
        preprocessor = vm.video_args_preprocessor
        if hasattr(preprocessor, "simple_preprocessor"):
            preprocessor = preprocessor.simple_preprocessor

        ctx_args = preprocessor.prepare(context_video)
        tgt_args = preprocessor.prepare(target_video)

        use_ckpt = getattr(vm, "_enable_gradient_checkpointing", False) and vm.training

        for idx, block in enumerate(vm.transformer_blocks):
            if use_ckpt:
                from dataclasses import replace as _replace

                from diffusion.refiner.vendor.ltx_core.model.transformer.transformer_args import TransformerArgs

                def _make_fn(_block):
                    def _fn(
                        tgt_x,
                        tgt_ctx,
                        tgt_cm,
                        tgt_ts,
                        tgt_ets,
                        tgt_pe,
                        tgt_sam,
                        tgt_pt,
                        ctx_x,
                        ctx_ctx,
                        ctx_cm,
                        ctx_ts,
                        ctx_ets,
                        ctx_pe,
                        ctx_sam,
                        ctx_pt,
                    ):
                        _ctx = TransformerArgs(
                            x=ctx_x,
                            context=ctx_ctx,
                            context_mask=ctx_cm,
                            timesteps=ctx_ts,
                            embedded_timestep=ctx_ets,
                            positional_embeddings=ctx_pe,
                            cross_positional_embeddings=None,
                            cross_scale_shift_timestep=None,
                            cross_gate_timestep=None,
                            enabled=True,
                            prompt_timestep=ctx_pt,
                            self_attention_mask=ctx_sam,
                        )
                        _tgt = TransformerArgs(
                            x=tgt_x,
                            context=tgt_ctx,
                            context_mask=tgt_cm,
                            timesteps=tgt_ts,
                            embedded_timestep=tgt_ets,
                            positional_embeddings=tgt_pe,
                            cross_positional_embeddings=None,
                            cross_scale_shift_timestep=None,
                            cross_gate_timestep=None,
                            enabled=True,
                            prompt_timestep=tgt_pt,
                            self_attention_mask=tgt_sam,
                        )
                        with torch.no_grad():
                            _block.attn1._tf_capture_kv = True
                            _ctx_out, _ = _block(video=_ctx, audio=None, perturbations=perturbations)
                            _kv = _block.attn1._tf_cached_kv
                            _block.attn1._tf_capture_kv = False
                        _block.attn1._tf_kv_prefix = _kv
                        _tgt_out, _ = _block(video=_tgt, audio=None, perturbations=perturbations)
                        _block.attn1._tf_kv_prefix = None
                        return _tgt_out.x, _ctx_out.x

                    return _fn

                tgt_x_new, ctx_x_new = torch.utils.checkpoint.checkpoint(
                    _make_fn(block),
                    tgt_args.x,
                    tgt_args.context,
                    tgt_args.context_mask,
                    tgt_args.timesteps,
                    tgt_args.embedded_timestep,
                    tgt_args.positional_embeddings,
                    tgt_args.self_attention_mask,
                    tgt_args.prompt_timestep,
                    ctx_args.x,
                    ctx_args.context,
                    ctx_args.context_mask,
                    ctx_args.timesteps,
                    ctx_args.embedded_timestep,
                    ctx_args.positional_embeddings,
                    ctx_args.self_attention_mask,
                    ctx_args.prompt_timestep,
                    use_reentrant=False,
                )
                tgt_args = _replace(tgt_args, x=tgt_x_new)
                ctx_args = _replace(ctx_args, x=ctx_x_new)
            else:
                with torch.no_grad():
                    block.attn1._tf_capture_kv = True
                    ctx_args, _ = block(video=ctx_args, audio=None, perturbations=perturbations)
                    cached_kv = block.attn1._tf_cached_kv
                    block.attn1._tf_capture_kv = False
                block.attn1._tf_kv_prefix = cached_kv
                tgt_args, _ = block(video=tgt_args, audio=None, perturbations=perturbations)
                block.attn1._tf_kv_prefix = None
                del cached_kv

        vx = vm._process_output(
            vm.scale_shift_table,
            vm.norm_out,
            vm.proj_out,
            tgt_args.x,
            tgt_args.embedded_timestep,
        )
        return to_denoised(target_video.latent, vx, target_video.timesteps)
