# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Diffusers-backed LTX-2 refiner used by Sana-WM inference.

The Sana-WM refiner checkpoint is a standard LTX-2 transformer plus text
connectors. Diffusers already owns those modules, but its public transformer
forward always runs the audio stream and does not expose the streaming
sink/current video self-attention mask that this refiner was trained with.

This wrapper keeps the custom surface narrow: load diffusers components, encode
the prompt through Gemma + ``LTX2TextConnectors``, and run a video-only forward
through the diffusers transformer blocks. The only local attention code is the
streaming sink/current split, implemented with diffusers attention modules
without materializing the full sequence-by-sequence mask.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time
import types

import torch
from torch import nn
import torch.nn.functional as F

STAGE_2_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


class DiffusersLTX2Refiner(nn.Module):
    """Small Sana-WM adapter around diffusers LTX-2 modules."""

    def __init__(
        self,
        refiner_root: str | Path,
        gemma_root: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        text_max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.refiner_root = Path(refiner_root)
        self.gemma_root = Path(gemma_root)
        self.dtype = dtype
        self.device = torch.device(device)
        self.text_max_sequence_length = int(text_max_sequence_length)
        self._te_nvfp4_requested = _env_flag("SANA_WM_REFINER_NVFP4")
        self._te_nvfp4_recipe = None
        self._te_nvfp4_converted = False
        self._self_qkv_fused = False
        self._attention_backend = os.environ.get("SANA_WM_REFINER_ATTN_BACKEND", "").strip()
        self._uniform_timestep_cache: dict[
            tuple[int, int, float, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self.transformer, self.connectors = self._load_diffusers_components()

    def _load_diffusers_components(self) -> tuple[nn.Module, nn.Module]:
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        cache_path = self._prepared_transformer_cache_path()
        if cache_path is not None and cache_path.is_file():
            t0 = time.perf_counter()
            print(f"[refiner-cache] loading prepared transformer from {cache_path}", flush=True)
            try:
                transformer = torch.load(cache_path, map_location=self.device, weights_only=False).eval()
                self._te_nvfp4_converted = bool(self._te_nvfp4_requested)
                self._self_qkv_fused = _env_flag("SANA_WM_REFINER_FUSE_SELF_QKV")
                self._te_nvfp4_recipe = self._make_nvfp4_recipe() if self._te_nvfp4_converted else None
                print(f"[refiner-cache] loaded prepared transformer in {time.perf_counter() - t0:.1f}s", flush=True)
            except Exception as exc:
                print(f"[refiner-cache] failed to load {cache_path}: {exc}; rebuilding", flush=True)
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    self.refiner_root,
                    subfolder="transformer",
                    torch_dtype=self.dtype,
                ).eval()
        else:
            transformer = LTX2VideoTransformer3DModel.from_pretrained(
                self.refiner_root,
                subfolder="transformer",
                torch_dtype=self.dtype,
            ).eval()
        if (
            not self._te_nvfp4_requested
            and os.environ.get("SANA_WM_REFINER_FP8_STORAGE", "").lower() in {"1", "true", "yes", "on"}
        ):
            skip_patterns = None
            extra_skip_patterns = _env_tuple("SANA_WM_REFINER_FP8_SKIP_PATTERNS")
            if extra_skip_patterns:
                from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

                skip_patterns = tuple(dict.fromkeys((*DEFAULT_SKIP_MODULES_PATTERN, *extra_skip_patterns)))
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=self.dtype,
                skip_modules_pattern=skip_patterns,
            )
        connectors = LTX2TextConnectors.from_pretrained(
            self.refiner_root,
            subfolder="connectors",
            torch_dtype=self.dtype,
        ).eval()
        return transformer, connectors

    def _make_nvfp4_recipe(self):
        import transformer_engine.common.recipe as te_recipe

        return te_recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
        )

    def _prepared_transformer_cache_path(self) -> Path | None:
        root = _prepared_module_cache_root()
        if root is None or not self._te_nvfp4_requested:
            return None
        payload = {
            "kind": "refiner_transformer_prepared_v2",
            "refiner_root": _path_fingerprint(self.refiner_root / "transformer"),
            "dtype": str(self.dtype),
            "torch": torch.__version__,
            "refiner_nvfp4": os.environ.get("SANA_WM_REFINER_NVFP4", ""),
            "refiner_nvfp4_skip_patterns": os.environ.get("SANA_WM_REFINER_NVFP4_SKIP_PATTERNS", ""),
            "refiner_fuse_self_qkv": os.environ.get("SANA_WM_REFINER_FUSE_SELF_QKV", ""),
            "te_cpu_staging": os.environ.get("SANA_WM_TE_NVFP4_CPU_STAGING", ""),
        }
        try:
            import transformer_engine

            payload["transformer_engine"] = getattr(transformer_engine, "__version__", "unknown")
        except Exception:
            payload["transformer_engine"] = "unavailable"
        return root / "refiner" / f"{_prepared_module_cache_hash(payload)}.pt"

    def _save_prepared_transformer_cache(self) -> None:
        if os.environ.get("SANA_WM_PREPARED_MODULE_CACHE_SAVE", "1").strip().lower() in {
            "",
            "0",
            "false",
            "no",
            "off",
        }:
            return
        cache_path = self._prepared_transformer_cache_path()
        if cache_path is None or cache_path.is_file():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        t0 = time.perf_counter()
        print(f"[refiner-cache] saving prepared transformer to {cache_path}", flush=True)
        restore = _strip_local_callables_for_pickle(self.transformer)
        if restore:
            print(f"[refiner-cache] stripped {len(restore)} init-only callables before save", flush=True)
        try:
            torch.save(self.transformer, tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            print(f"[refiner-cache] failed to save {cache_path}: {exc}", flush=True)
        finally:
            _restore_stripped_pickle_values(restore)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        if cache_path.is_file():
            print(f"[refiner-cache] saved prepared transformer in {time.perf_counter() - t0:.1f}s", flush=True)

    def prepare_transformer_nvfp4(self) -> None:
        """Lazily replace eligible refiner Linear layers with TE NVFP4 Linear modules."""
        self._prepare_self_qkv_fusion()
        if not self._te_nvfp4_requested or self._te_nvfp4_converted:
            return

        recipe = self._make_nvfp4_recipe()
        converted, skipped = _replace_linear_with_te_nvfp4(
            self.transformer,
            recipe=recipe,
            params_dtype=self.dtype,
            skip_patterns=tuple(
                dict.fromkeys(
                    (
                        "^proj_in$",
                        "^proj_out$",
                        "(^|\\.)audio_",
                        "audio_to_video",
                        "video_to_audio",
                        "av_cross_attn",
                        "caption_projection",
                        "time_embed",
                        *_env_tuple("SANA_WM_REFINER_NVFP4_SKIP_PATTERNS"),
                    )
                )
            ),
        )
        if converted <= 0:
            raise RuntimeError(f"SANA_WM_REFINER_NVFP4=1 converted no Linear layers; skipped={skipped}.")
        self._te_nvfp4_recipe = recipe
        self._te_nvfp4_converted = True
        _empty_cuda_cache()
        self._save_prepared_transformer_cache()

    def _prepare_self_qkv_fusion(self) -> None:
        if self._self_qkv_fused or not _env_flag("SANA_WM_REFINER_FUSE_SELF_QKV"):
            return
        converted = _fuse_refiner_self_qkv(self.transformer)
        if converted <= 0:
            raise RuntimeError("SANA_WM_REFINER_FUSE_SELF_QKV=1 fused no self-attention QKV modules.")
        self._self_qkv_fused = True
        print(f"[refiner-fuse-qkv] fused {converted} self-attention QKV groups", flush=True)

    def offload_video_unused_audio_modules(self, device: torch.device | str = "cpu") -> None:
        """Keep LTX-2 audio-only branches off GPU for this wrapper's video-only forward."""
        _offload_video_unused_audio_modules(self.transformer, device)
        _empty_cuda_cache()

    def move_video_modules(self, device: torch.device | str) -> None:
        """Move only the modules and direct parameters used by the video-only forward."""
        _move_ltx2_video_modules_to(self.transformer, device)
        _empty_cuda_cache()

    def _nvfp4_autocast(self):
        if not self._te_nvfp4_converted:
            return nullcontext()
        import transformer_engine.pytorch as te

        return te.fp8_autocast(enabled=True, fp8_recipe=self._te_nvfp4_recipe)

    def _attention_backend_context(self):
        if not self._attention_backend:
            return nullcontext()
        from diffusers.models.attention_dispatch import attention_backend

        return attention_backend(self._attention_backend)

    def _uniform_timestep_tensors(
        self,
        *,
        batch_size: int,
        seq_len: int,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_value = float(sigma)
        if not _env_flag("SANA_WM_REFINER_TIMESTEP_CACHE"):
            raw_sigma = torch.full(
                (int(batch_size), int(seq_len), 1), sigma_value, dtype=torch.float32, device=self.device
            )
            model_timestep = raw_sigma.squeeze(-1) * float(self.transformer.config.timestep_scale_multiplier)
            return model_timestep, raw_sigma
        key = (int(batch_size), int(seq_len), sigma_value, str(self.device))
        cached = self._uniform_timestep_cache.get(key)
        if cached is not None:
            return cached
        raw_sigma = torch.full((int(batch_size), int(seq_len), 1), sigma_value, dtype=torch.float32, device=self.device)
        model_timestep = raw_sigma.squeeze(-1) * float(self.transformer.config.timestep_scale_multiplier)
        cached = (model_timestep, raw_sigma)
        self._uniform_timestep_cache[key] = cached
        return cached

    @torch.inference_mode()
    def refine_latents(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        seed: int = 42,
        progress: bool = True,
        block_size: int | None = None,
        kv_max_frames: int = 11,
        sigmas: tuple[float, ...] = STAGE_2_DISTILLED_SIGMA_VALUES,
    ) -> torch.Tensor:
        """Run the LTX-2 refiner and return refined VAE latents.

        When ``block_size`` is ``None`` (default), uses the legacy single-shot
        path that denoises all current frames jointly. When ``block_size`` is
        set (canonical: 3), runs the chunk-causal AR recipe with sliding-window
        attention over ``[source_sink + recent_history + active_block]``,
        matching tian's ``run_reforcing_inference`` contract — the model was
        trained to refine ``block_size`` frames at a time with clean prior
        context, and feeding the full sequence at once is out-of-distribution.

        Args:
            sana_latent: ``(B, C, F, H, W)`` stage-1 latent.
            prompt: text prompt.
            fps: video frame rate (drives LTX-2 RoPE temporal scaling).
            sink_size: how many leading raw ``z_sana`` frames to anchor as the
                attention sink (canonical: 1).
            seed: noise seed for the FM endpoint.
            progress: show a tqdm bar.
            block_size: latent frames per AR block (canonical: 3). ``None``
                disables AR mode.
            kv_max_frames: maximum context+active frames retained in the
                sliding window when AR mode is active (canonical: 11 =
                1 sink + 10 recent).
            sigmas: descending Euler schedule terminating at 0.0 (canonical
                3-step distilled: ``(0.909375, 0.725, 0.421875, 0.0)``).
        """
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(f"Stage-1 latent has {sana_latent.shape[2]} frames but sink_size={sink_size}.")

        self.transformer.to("cpu")
        _empty_cuda_cache()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(prompt)

        self.move_video_modules(self.device)
        self.offload_video_unused_audio_modules("cpu")
        self.prepare_transformer_nvfp4()
        z = sana_latent.to(device=self.device, dtype=self.dtype)
        sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=self.device)
        start_sigma = float(sigmas_t[0])

        if block_size is not None:
            return self._refine_latents_ar(
                z=z,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                fps=fps,
                sigmas=sigmas_t,
                source_sink_frames=int(sink_size),
                block_size=int(block_size),
                kv_max_frames=int(kv_max_frames),
                seed=int(seed),
                progress=bool(progress),
            )

        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        eps = torch.randn(current.shape, generator=generator, device=self.device, dtype=self.dtype)
        noisy = (1.0 - start_sigma) * current + start_sigma * eps

        iterator = range(len(sigmas_t) - 1)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="refiner", unit="step")

        for step_index in iterator:
            sigma = sigmas_t[step_index]
            denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
            )
            noisy_tokens = _pack_latents(
                noisy,
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = noisy_tokens.float() + velocity * (sigmas_t[step_index + 1] - sigma).float()
            noisy = _unpack_latents(
                next_tokens.to(self.dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )

        return torch.cat([sink, noisy], dim=2)

    @torch.inference_mode()
    def _refine_latents_ar(
        self,
        *,
        z: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        sigmas: torch.Tensor,
        source_sink_frames: int,
        block_size: int,
        kv_max_frames: int,
        seed: int,
        progress: bool,
    ) -> torch.Tensor:
        """Chunk-causal AR refinement — thin wrapper around ``RefinerChunkRunner``.

        Implements the canonical ``rf_shifted_sink`` KV-cache contract end-to-end:

        1. Pre-capture **pre-RoPE** sink K/V from raw ``z_sana[:source_sink_frames]``
           at σ=0 (``_kv_cache_capture`` hook). The sink frames themselves are
           **never refined** — they sit unchanged in the output volume.
        2. AR blocks cover frames ``[source_sink_frames, T_full)`` in
           ``block_size``-frame chunks. For each block:
           - Initialize ``x_t = (1-σ₀)·z_sana_block + σ₀·ε`` (single eps per block).
           - 3-step deterministic Euler. Each step injects the per-layer prefix
             ``{sink_k_pre, sink_v, sink_pe, history_k, history_v}`` where
             ``sink_pe`` is rebuilt at ``sink_rope_offset = active_start -
             history_frames - source_sink_frames`` so the sink slides to sit
             immediately before the bounded working cache (official RF layout).
           - Capture **post-RoPE** K/V from the refined block under the same
             prefix (``_tf_capture_kv`` hook); append to ``history_kv_post`` and
             trim to ``kv_max_frames - source_sink_frames``.

        For the chunk-pipelined interactive path, build a ``RefinerChunkRunner``
        directly and feed one block at a time as stage-1 yields it.

        The returned tensor has the same shape ``(B, C, T_full, H, W)`` as
        ``z``; the first ``source_sink_frames`` slots carry the raw sink
        latents unchanged, the rest carry the refined output.
        """
        runner = RefinerChunkRunner(
            self,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            fps=fps,
            sigmas=sigmas,
            source_sink_frames=int(source_sink_frames),
            block_size=int(block_size),
            kv_max_frames=int(kv_max_frames),
            seed=int(seed),
            spatial_shape=(int(z.shape[3]), int(z.shape[4])),
        )

        T_full = z.shape[2]
        sink_size = int(source_sink_frames)
        # Output keeps the raw sink prefix verbatim; AR blocks fill frames
        # [sink_size, T_full).
        output = z.clone()
        n_active = max(T_full - sink_size, 0)
        n_blocks = (n_active + block_size - 1) // block_size if n_active > 0 else 0
        iterator = range(n_blocks)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="refiner-ar", unit="block")

        for block_idx in iterator:
            block_start = sink_size + block_idx * block_size
            block_end = min(block_start + block_size, T_full)
            clean_block = z[:, :, block_start:block_end]
            refined = runner.refine_block(
                block_idx=block_idx,
                clean_block=clean_block,
                block_start=block_start,
                block_end=block_end,
                sink_seed_frames=(z[:, :, :sink_size] if block_idx == 0 else None),
            )
            output[:, :, block_start:block_end] = refined

        return output

    def _predict_x0_active_block(
        self,
        *,
        active: torch.Tensor,                  # (B, C, N_active, H, W) at σ_cur
        active_positions: list[int],
        sigma_cur: float,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        kv_prefix_per_layer: list[dict[str, object]] | None,
        active_video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        capture_post_kv: bool = False,
        capture_layer_mask: list[bool] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """Forward through the transformer on the ACTIVE BLOCK ONLY and return x0.

        The active block's Q attends to ``[prefix, current]`` K/V via the
        ``_tf_kv_prefix`` hook on every self-attention block. All active tokens
        carry the same ``sigma_cur`` (matching tian's per-block uniform σ).
        """
        latent_tokens = _pack_latents(
            active,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch_size, seq_len, _ = latent_tokens.shape
        # Use a per-token uniform sigma for the active block.
        model_timestep, raw_sigma = self._uniform_timestep_tensors(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            sigma=float(sigma_cur),
        )

        video_rotary_emb = active_video_rotary_emb
        if video_rotary_emb is None:
            video_rotary_emb = _build_rotary_emb_for_absolute_positions(
                transformer=self.transformer,
                batch_size=batch_size,
                frame_positions=active_positions,
                height=int(active.shape[3]),
                width=int(active.shape[4]),
                device=self.device,
                fps=float(fps),
            )
        # Replace the per-frame uniform-σ adaLN time embedding with the active
        # block's mean sigma (= sigma_cur here), mirroring tian's prompt_sigma
        # `mean_active` mode.
        _set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        if capture_post_kv:
            _set_capture_flag_on_blocks(self.transformer, "post_rope", enable=True, layer_mask=capture_layer_mask)
        try:
            velocity = self._forward_video_only_with_rope(
                hidden_states=latent_tokens,
                encoder_hidden_states=prompt_embeds,
                timestep=model_timestep,
                encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb,
                n_context_tokens=0,
            )
        finally:
            if capture_post_kv:
                _set_capture_flag_on_blocks(self.transformer, "post_rope", enable=False)
            _clear_kv_prefix_on_blocks(self.transformer)
        captured_kv = (
            _collect_captured_kv_from_blocks(self.transformer, "post_rope", layer_mask=capture_layer_mask)
            if capture_post_kv
            else None
        )

        # FM x0 prediction: x_t - σ_cur · v.
        denoised_tokens = latent_tokens.float() - velocity.float() * raw_sigma
        denoised = _unpack_latents(
            denoised_tokens.to(self.dtype),
            num_frames=int(active.shape[2]),
            height=int(active.shape[3]),
            width=int(active.shape[4]),
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        if captured_kv is not None:
            return denoised, captured_kv
        return denoised

    @torch.inference_mode()
    def _capture_block_kv(
        self,
        *,
        clean_block: torch.Tensor,         # (B, C, N, H, W) treated as σ=0 (clean) input
        frame_positions: list[int],
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        capture_mode: str,                 # "pre_rope" or "post_rope"
        kv_prefix_per_layer: list[dict[str, object]] | None,
        capture_layer_mask: list[bool] | None = None,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """Run one forward at σ=0 with capture hooks; return per-layer (K, V).

        ``capture_mode='pre_rope'`` uses the ``_kv_cache_capture`` hook (stored
        before RoPE so a future window can re-RoPE the sink to its shifted
        offset). ``capture_mode='post_rope'`` uses ``_tf_capture_kv`` (stored
        with RoPE already baked at the block's absolute positions, ready to
        concatenate into the next window's prefix).
        """
        latent_tokens = _pack_latents(
            clean_block,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        batch_size, seq_len, _ = latent_tokens.shape
        model_timestep = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)

        if video_rotary_emb is None:
            video_rotary_emb = _build_rotary_emb_for_absolute_positions(
                transformer=self.transformer,
                batch_size=batch_size,
                frame_positions=frame_positions,
                height=int(clean_block.shape[3]),
                width=int(clean_block.shape[4]),
                device=self.device,
                fps=float(fps),
            )

        stop_after_layer = None
        stop_after_capture_kv_layer = None
        if capture_layer_mask is not None and not all(capture_layer_mask):
            stop_after_layer = max(idx for idx, keep in enumerate(capture_layer_mask) if keep)
        if _env_flag("SANA_WM_REFINER_CAPTURE_KV_ONLY_LAST"):
            if capture_layer_mask is None:
                stop_after_capture_kv_layer = len(self.transformer.transformer_blocks) - 1
            else:
                stop_after_capture_kv_layer = max(idx for idx, keep in enumerate(capture_layer_mask) if keep)
            stop_after_layer = None

        _set_kv_prefix_on_blocks(self.transformer, kv_prefix_per_layer)
        _set_capture_flag_on_blocks(self.transformer, capture_mode, enable=True, layer_mask=capture_layer_mask)
        try:
            _ = self._forward_video_only_with_rope(
                hidden_states=latent_tokens,
                encoder_hidden_states=prompt_embeds,
                timestep=model_timestep,
                encoder_attention_mask=prompt_attention_mask,
                video_rotary_emb=video_rotary_emb,
                n_context_tokens=0,
                skip_output_projection=True,
                stop_after_layer=stop_after_layer,
                stop_after_capture_kv_layer=stop_after_capture_kv_layer,
            )
        finally:
            _set_capture_flag_on_blocks(self.transformer, capture_mode, enable=False)
            _clear_kv_prefix_on_blocks(self.transformer)

        return _collect_captured_kv_from_blocks(self.transformer, capture_mode, layer_mask=capture_layer_mask)

    @torch.inference_mode()
    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(self.gemma_root)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).eval()
        text_encoder.to(self.device)
        text_backbone = getattr(text_encoder, "model", text_encoder)
        outputs = text_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)
        del text_encoder, text_backbone, outputs
        _empty_cuda_cache()

        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=self.device,
            padding_side=tokenizer.padding_side,
        ).to(dtype=self.dtype)

        del hidden_states
        _empty_cuda_cache()

        self.connectors.to(self.device)
        connector_prompt_embeds, _, connector_attention_mask = self.connectors(prompt_embeds, attention_mask)
        self.connectors.to("cpu")
        del prompt_embeds, attention_mask
        _empty_cuda_cache()

        return connector_prompt_embeds.to(device=self.device, dtype=self.dtype), connector_attention_mask.to(
            device=self.device
        )

    def _predict_current_x0(
        self,
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
    ) -> torch.Tensor:
        full_latent = torch.cat([sink, noisy_current], dim=2)
        batch_size, _, num_frames, height, width = full_latent.shape
        latent_tokens = _pack_latents(
            full_latent,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        n_context_tokens = _pack_latents(
            sink,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        ).shape[1]

        raw_timestep = torch.zeros(batch_size, latent_tokens.shape[1], 1, dtype=torch.float32, device=self.device)
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        model_timestep = raw_timestep.squeeze(-1) * float(self.transformer.config.timestep_scale_multiplier)

        velocity = self._forward_video_only(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            n_context_tokens=n_context_tokens,
        )
        denoised = latent_tokens.float() - velocity.float() * raw_timestep
        return denoised[:, n_context_tokens:, :].to(self.dtype)

    def _forward_video_only_with_rope(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        n_context_tokens: int,
        skip_output_projection: bool = False,
        stop_after_layer: int | None = None,
        stop_after_capture_kv_layer: int | None = None,
    ) -> torch.Tensor:
        """Shared body of ``_forward_video_only`` that takes a pre-built RoPE.

        Used by the AR refinement path where each block forward needs custom
        per-frame absolute positions in the source video.
        """
        transformer = self.transformer
        batch_size = hidden_states.size(0)
        seq_len = int(hidden_states.shape[1])
        profiler = None
        if _refiner_layer_profile_enabled():
            forward_kind = "capture" if skip_output_projection else "predict"
            prefix_tokens = _current_refiner_prefix_tokens(transformer)
            profiler = _RefinerLayerCudaProfiler(
                enabled=True,
                device=self.device,
                label=f"{forward_kind} seq={seq_len} prefix={prefix_tokens}",
            )

        with _profile_section(profiler, "mask_prepare"):
            encoder_attention_mask = _prepare_encoder_attention_mask(encoder_attention_mask, hidden_states.dtype)

        with _profile_section(profiler, "proj_in"):
            hidden_states = transformer.proj_in(hidden_states)
        with _profile_section(profiler, "time_embed"):
            temb, embedded_timestep = transformer.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        with _profile_section(profiler, "caption_projection"):
            if _has_cross_attention_kv_cache(transformer):
                encoder_hidden_states = None
            else:
                encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        with self._attention_backend_context(), self._nvfp4_autocast():
            for layer_idx, block in enumerate(transformer.transformer_blocks):
                capture_kv_only = (
                    stop_after_capture_kv_layer is not None
                    and layer_idx >= int(stop_after_capture_kv_layer)
                )
                hidden_states = _forward_video_block(
                    block=block,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    video_rotary_emb=video_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    n_context_tokens=n_context_tokens,
                    profiler=profiler,
                    capture_kv_only=capture_kv_only,
                )
                if capture_kv_only:
                    break
                if stop_after_layer is not None and layer_idx >= int(stop_after_layer):
                    break

        if skip_output_projection:
            if profiler is not None:
                profiler.finish()
            return hidden_states

        with _profile_section(profiler, "proj_out"):
            scale_shift_values = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            hidden_states = transformer.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = transformer.proj_out(hidden_states)
        if profiler is not None:
            profiler.finish()
        return hidden_states

    def _forward_video_only(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        n_context_tokens: int,
    ) -> torch.Tensor:
        transformer = self.transformer
        batch_size = hidden_states.size(0)
        seq_len = int(hidden_states.shape[1])
        profiler = None
        if _refiner_layer_profile_enabled():
            profiler = _RefinerLayerCudaProfiler(
                enabled=True,
                device=self.device,
                label=f"legacy seq={seq_len} prefix={int(n_context_tokens)}",
            )

        with _profile_section(profiler, "mask_prepare"):
            encoder_attention_mask = _prepare_encoder_attention_mask(encoder_attention_mask, hidden_states.dtype)

        with _profile_section(profiler, "rope"):
            video_coords = transformer.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
            video_rotary_emb = transformer.rope(video_coords, device=hidden_states.device)

        with _profile_section(profiler, "proj_in"):
            hidden_states = transformer.proj_in(hidden_states)
        with _profile_section(profiler, "time_embed"):
            temb, embedded_timestep = transformer.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        with _profile_section(profiler, "caption_projection"):
            if _has_cross_attention_kv_cache(transformer):
                encoder_hidden_states = None
            else:
                encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        with self._attention_backend_context(), self._nvfp4_autocast():
            for block in transformer.transformer_blocks:
                hidden_states = _forward_video_block(
                    block=block,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    video_rotary_emb=video_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    n_context_tokens=n_context_tokens,
                    profiler=profiler,
                )

        with _profile_section(profiler, "proj_out"):
            scale_shift_values = transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            hidden_states = transformer.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = transformer.proj_out(hidden_states)
        if profiler is not None:
            profiler.finish()
        return hidden_states


class RefinerChunkRunner:
    """Stateful per-AR-block driver for ``DiffusersLTX2Refiner``.

    Owns the rolling KV state that the chunk-causal AR recipe accumulates as
    refiner blocks complete:

    * ``_sink_kv_pre``: per-layer pre-RoPE K/V captured from the first
      ``source_sink_frames`` raw stage-1 latents at σ=0. Lazily filled on the
      first call to :meth:`refine_block` (the orchestrator only has the first
      stage-1 chunk in hand by then).
    * ``_history_kv_post``: per-layer post-RoPE K/V of every refined block
      already produced, trimmed to ``kv_max_frames - source_sink_frames``
      frames so the sliding window stays bounded.
    * ``_history_frames``: number of frames currently in
      ``_history_kv_post`` (drives token-level trim).

    The numerical contract is identical to a single in-place call to
    ``_refine_latents_ar``: same RNG-seeded epsilon stream consumed
    block-by-block, same ``rf_shifted_sink`` per-window prefix dict, same
    3-step deterministic Euler, same post-RoPE capture under that prefix. The
    orchestrator can therefore call :meth:`refine_block` once per stage-1 chunk
    without changing inference semantics, and concurrently launch the
    downstream causal-VAE decode on a separate CUDA stream while the next
    block's refinement runs on the refiner stream.
    """

    def __init__(
        self,
        refiner: "DiffusersLTX2Refiner",
        *,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        fps: float,
        sigmas: torch.Tensor,
        source_sink_frames: int,
        block_size: int,
        kv_max_frames: int,
        seed: int,
        spatial_shape: tuple[int, int],
        n_active_frames: int | None = None,
        latent_channels: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self._refiner = refiner
        self._prompt_embeds = prompt_embeds
        self._prompt_attention_mask = prompt_attention_mask
        self._fps = float(fps)
        self._sigmas = sigmas
        self._sigma_values = [float(v) for v in sigmas.detach().float().cpu()]
        self._sigma_pairs = list(zip(self._sigma_values[:-1], self._sigma_values[1:]))
        self._sigma_max = self._sigma_values[0]
        self._n_steps = int(sigmas.numel() - 1)
        self._source_sink_frames = int(source_sink_frames)
        self._block_size = int(block_size)
        self._kv_max_frames = int(kv_max_frames)
        self._max_history_frames = int(kv_max_frames) - int(source_sink_frames)
        self._device = refiner.device
        self._dtype = refiner.dtype
        self._generator = torch.Generator(device=self._device).manual_seed(int(seed))
        self._kv_cache_storage_dtype = _resolve_kv_cache_storage_dtype()

        transformer = refiner.transformer
        self._n_layers = len(transformer.transformer_blocks)
        H, W = spatial_shape
        self._H, self._W = int(H), int(W)
        self._tokens_per_frame = (
            int(H // transformer.config.patch_size)
            * int(W // transformer.config.patch_size)
            * int(transformer.config.patch_size_t)
        )
        self._precomputed_eps_blocks: list[torch.Tensor] | None = None
        if (
            _env_flag("SANA_WM_REFINER_PREGENERATE_NOISE")
            and n_active_frames is not None
            and latent_channels is not None
        ):
            n_active = int(n_active_frames)
            channels = int(latent_channels)
            batch = int(batch_size)
            n_blocks = (n_active + self._block_size - 1) // self._block_size if n_active > 0 else 0
            self._precomputed_eps_blocks = []
            for block_idx in range(n_blocks):
                active_len = min(self._block_size, n_active - block_idx * self._block_size)
                self._precomputed_eps_blocks.append(
                    torch.randn(
                        (batch, channels, active_len, self._H, self._W),
                        generator=self._generator,
                        device=self._device,
                        dtype=self._dtype,
                    )
                )
            print(f"[refiner-noise] precomputed {len(self._precomputed_eps_blocks)} eps blocks", flush=True)
        if _env_flag("SANA_WM_REFINER_CROSS_ATTN_KV_CACHE"):
            with refiner._nvfp4_autocast():
                _set_cross_attention_kv_cache(refiner.transformer, prompt_embeds, prompt_attention_mask)
        else:
            _clear_cross_attention_kv_cache(refiner.transformer)

        self._sink_kv_pre: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._history_kv_post: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self._n_layers
        self._history_frames: int = 0
        self._history_layer_mask = _refiner_history_layer_mask(self._n_layers)
        self._exact_capture_layer_mask = _refiner_exact_capture_layer_mask(
            self._n_layers,
            default_mask=self._history_layer_mask,
        )
        if not all(self._history_layer_mask):
            kept = sum(1 for keep in self._history_layer_mask if keep)
            print(
                f"[refiner-history] recent history enabled on {kept}/{self._n_layers} layers",
                flush=True,
            )
        if self._exact_capture_layer_mask != self._history_layer_mask:
            kept = sum(1 for keep in self._exact_capture_layer_mask if keep)
            print(
                f"[refiner-history] exact post-capture on {kept}/{self._n_layers} layers",
                flush=True,
            )

    @torch.inference_mode()
    def pre_capture_sink(self, sink_seed_frames: torch.Tensor) -> None:
        """Capture the source-sink K/V before the first active refiner block.

        The sink is just the conditioning latent frame and does not depend on
        stage-1 sampling. Scheduling this on the refiner stream lets it overlap
        with stage-1 chunk 0 while preserving the exact same cached K/V that
        ``refine_block`` would have produced lazily.
        """
        if self._sink_kv_pre is not None:
            return
        if sink_seed_frames is None:
            raise ValueError("pre_capture_sink requires sink_seed_frames.")
        if sink_seed_frames.shape[2] != self._source_sink_frames:
            raise ValueError(
                f"sink_seed_frames has {sink_seed_frames.shape[2]} frames "
                f"but source_sink_frames={self._source_sink_frames}."
            )
        source_sink = sink_seed_frames.contiguous()
        self._sink_kv_pre = [
            _store_kv_pair(pair, self._kv_cache_storage_dtype)
            for pair in self._refiner._capture_block_kv(
                clean_block=source_sink,
                frame_positions=list(range(self._source_sink_frames)),
                prompt_embeds=self._prompt_embeds,
                prompt_attention_mask=self._prompt_attention_mask,
                fps=self._fps,
                capture_mode="pre_rope",
                kv_prefix_per_layer=None,
            )
        ]

    @torch.inference_mode()
    def refine_block(
        self,
        *,
        block_idx: int,
        clean_block: torch.Tensor,
        block_start: int,
        block_end: int,
        sink_seed_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine one AR block; advance internal KV state.

        Args:
            block_idx: 0-based block index in the AR schedule. Used only for
                bookkeeping; positional state derives from ``block_start``.
            clean_block: ``(B, C, active_len, H, W)`` clean stage-1 latents
                covering frames ``[block_start, block_end)``. The active block
                is what actually gets refined; sink frames live outside the
                active range and are passed via ``sink_seed_frames`` on the
                first call.
            block_start: absolute latent-frame index of the active block's
                first frame (drives the ``rf_shifted_sink`` RoPE offset).
                Must be >= ``source_sink_frames`` so the sink doesn't overlap
                the active region.
            block_end: absolute latent-frame index just past the active block.
            sink_seed_frames: ``(B, C, source_sink_frames, H, W)`` raw sink
                latents used once on the very first ``refine_block`` call to
                pre-capture the pre-RoPE sink K/V at ``sigma=0`` with frame
                positions ``[0, source_sink_frames)``. Required on the first
                call; ignored thereafter. The orchestrator owns these — they
                are typically the first ``source_sink_frames`` of stage-1's
                first chunk.

        Returns:
            ``(B, C, active_len, H, W)`` refined latents for this block.
        """
        refiner = self._refiner
        device = self._device
        profiler = _RefinerCudaProfiler(enabled=_refiner_profile_enabled(), device=device, block_idx=int(block_idx))
        B = int(clean_block.shape[0])
        active_len = block_end - block_start
        if block_start < self._source_sink_frames:
            raise ValueError(
                f"block_start={block_start} overlaps the source sink "
                f"(source_sink_frames={self._source_sink_frames})."
            )

        # 1) On the first call: pre-capture PRE-RoPE sink K/V from the supplied
        # raw sink latents at sigma=0 with absolute positions [0, sink_size).
        if self._sink_kv_pre is None:
            with profiler.section("sink_capture"):
                if sink_seed_frames is None:
                    raise ValueError(
                        "First refine_block call requires sink_seed_frames "
                        "(raw stage-1 sink latents)."
                    )
                if sink_seed_frames.shape[2] != self._source_sink_frames:
                    raise ValueError(
                        f"sink_seed_frames has {sink_seed_frames.shape[2]} frames "
                        f"but source_sink_frames={self._source_sink_frames}."
                    )
                self.pre_capture_sink(sink_seed_frames)

        # 2) Build per-window kv_prefix dict per layer.
        with profiler.section("prefix_build"):
            sink_rope_offset_history = block_start - self._history_frames - self._source_sink_frames
            sink_rope_offset_no_history = block_start - self._source_sink_frames
            sink_pe_history = _build_rotary_emb_for_absolute_positions(
                transformer=refiner.transformer,
                batch_size=B,
                frame_positions=list(
                    range(sink_rope_offset_history, sink_rope_offset_history + self._source_sink_frames)
                ),
                height=self._H,
                width=self._W,
                device=device,
                fps=self._fps,
            )
            sink_pe_no_history = sink_pe_history
            if sink_rope_offset_no_history != sink_rope_offset_history:
                sink_pe_no_history = _build_rotary_emb_for_absolute_positions(
                    transformer=refiner.transformer,
                    batch_size=B,
                    frame_positions=list(
                        range(sink_rope_offset_no_history, sink_rope_offset_no_history + self._source_sink_frames)
                    ),
                    height=self._H,
                    width=self._W,
                    device=device,
                    fps=self._fps,
                )
            kv_prefix_per_layer: list[dict[str, object]] = []
            preconcat_prefix = _env_flag("SANA_WM_REFINER_PRECONCAT_PREFIX")
            empty_cache_before_prefix = _env_flag("SANA_WM_REFINER_EMPTY_CACHE_BEFORE_PREFIX")
            for layer_idx in range(self._n_layers):
                hk = self._history_kv_post[layer_idx]
                use_history = bool(self._history_layer_mask[layer_idx] and hk is not None and hk[0].shape[1] > 0)
                sink_pe = sink_pe_history if use_history else sink_pe_no_history
                prefix: dict[str, object] = {
                    "mode": "rf_shifted_sink",
                    "sink_k_pre": self._sink_kv_pre[layer_idx][0],
                    "sink_v": self._sink_kv_pre[layer_idx][1],
                    "sink_pe": sink_pe,
                    "history_k": (hk[0] if use_history else None),
                    "history_v": (hk[1] if use_history else None),
                }
                if preconcat_prefix:
                    prefix_k_parts: list[torch.Tensor] = []
                    prefix_v_parts: list[torch.Tensor] = []
                    sink_k_pre, sink_v = self._sink_kv_pre[layer_idx]
                    if sink_k_pre.shape[1] > 0 and sink_v.shape[1] > 0:
                        attn = refiner.transformer.transformer_blocks[layer_idx].attn1
                        sink_k = _apply_refiner_rotary(attn, sink_k_pre.to(self._dtype), sink_pe)
                        prefix_k_parts.append(sink_k)
                        prefix_v_parts.append(sink_v.to(self._dtype))
                    if use_history:
                        prefix_k_parts.append(hk[0].to(self._dtype))
                        prefix_v_parts.append(hk[1].to(self._dtype))
                    if prefix_k_parts:
                        if empty_cache_before_prefix and device.type == "cuda":
                            torch.cuda.empty_cache()
                        prefix_k = torch.cat(prefix_k_parts, dim=1)
                        prefix_v = torch.cat(prefix_v_parts, dim=1)
                        prefix["prefix_k"] = prefix_k
                        prefix["prefix_v"] = prefix_v
                kv_prefix_per_layer.append(prefix)

        # 3) FM endpoint at sigma=sigma0: single epsilon per block.
        with profiler.section("noise_init"):
            eps = None
            if self._precomputed_eps_blocks is not None and int(block_idx) < len(self._precomputed_eps_blocks):
                candidate_eps = self._precomputed_eps_blocks[int(block_idx)]
                if tuple(candidate_eps.shape) == tuple(clean_block.shape):
                    eps = candidate_eps
            if eps is None:
                eps = torch.randn(clean_block.shape, generator=self._generator, device=device, dtype=self._dtype)
            x_t = ((1.0 - self._sigma_max) * clean_block.float() + self._sigma_max * eps.float()).to(self._dtype)

        with profiler.section("active_rope"):
            active_positions = list(range(int(block_start), int(block_end)))
            active_video_rotary_emb = _build_rotary_emb_for_absolute_positions(
                transformer=refiner.transformer,
                batch_size=B,
                frame_positions=active_positions,
                height=self._H,
                width=self._W,
                device=device,
                fps=self._fps,
            )
        fast_kv_capture = _refiner_fast_kv_capture_mode()
        reuse_final_predict_kv = fast_kv_capture == "last_predict" and not _refiner_fast_kv_needs_clean_block(
            int(block_idx)
        )
        fill_missing_predict_kv = fast_kv_capture == "fill_missing"
        captured_kv_post: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        n_sigma_pairs = len(self._sigma_pairs)
        for step_idx, (sigma_cur, sigma_next) in enumerate(self._sigma_pairs):
            with profiler.section(f"denoise_step{step_idx}"):
                capture_predict_kv = bool((reuse_final_predict_kv or fill_missing_predict_kv) and step_idx == n_sigma_pairs - 1)
                pred_result = refiner._predict_x0_active_block(
                    active=x_t,
                    active_positions=active_positions,
                    sigma_cur=sigma_cur,
                    prompt_embeds=self._prompt_embeds,
                    prompt_attention_mask=self._prompt_attention_mask,
                    fps=self._fps,
                    kv_prefix_per_layer=kv_prefix_per_layer,
                    active_video_rotary_emb=active_video_rotary_emb,
                    capture_post_kv=capture_predict_kv,
                    capture_layer_mask=self._history_layer_mask,
                )
                if isinstance(pred_result, tuple):
                    pred_x0, captured_kv_post = pred_result
                    if fill_missing_predict_kv and captured_kv_post is not None:
                        captured_kv_post = [
                            None
                            if self._exact_capture_layer_mask[layer_idx]
                            else (
                                _store_kv_pair(pair, self._kv_cache_storage_dtype)
                                if pair is not None
                                else None
                            )
                            for layer_idx, pair in enumerate(captured_kv_post)
                        ]
                else:
                    pred_x0 = pred_result
                if sigma_cur <= 1.0e-6:
                    x_t = pred_x0.to(self._dtype)
                else:
                    ratio = sigma_next / sigma_cur
                    x_t = (ratio * x_t.float() + (1.0 - ratio) * pred_x0.float()).to(self._dtype)
                pred_x0 = None

        if self._max_history_frames <= 0:
            with profiler.section("history_update"):
                self._history_frames = 0
                for layer_idx in range(self._n_layers):
                    self._history_kv_post[layer_idx] = None
            profiler.finish()
            return x_t

        # 4) Capture POST-RoPE K/V for this refined block under the same prefix.
        with profiler.section("post_capture"):
            if reuse_final_predict_kv:
                if captured_kv_post is None:
                    raise RuntimeError("SANA_WM_REFINER_FAST_KV_CAPTURE=last_predict did not capture post-RoPE K/V.")
                block_kv_post = captured_kv_post
            else:
                if _refiner_empty_cache_before_capture() and device.type == "cuda":
                    torch.cuda.empty_cache()
                block_kv_post = refiner._capture_block_kv(
                    clean_block=x_t,
                    frame_positions=active_positions,
                    prompt_embeds=self._prompt_embeds,
                    prompt_attention_mask=self._prompt_attention_mask,
                    fps=self._fps,
                    capture_mode="post_rope",
                    kv_prefix_per_layer=kv_prefix_per_layer,
                    capture_layer_mask=self._exact_capture_layer_mask,
                    video_rotary_emb=active_video_rotary_emb,
                )
                if fill_missing_predict_kv:
                    if captured_kv_post is None:
                        raise RuntimeError("SANA_WM_REFINER_FAST_KV_CAPTURE=fill_missing did not capture fallback K/V.")
                    block_kv_post = [
                        exact_pair if self._exact_capture_layer_mask[layer_idx] else captured_kv_post[layer_idx]
                        for layer_idx, exact_pair in enumerate(block_kv_post)
                    ]
        with profiler.section("history_update"):
            for layer_idx in range(self._n_layers):
                if not self._history_layer_mask[layer_idx]:
                    self._history_kv_post[layer_idx] = None
                    continue
                raw_pair = block_kv_post[layer_idx]
                if raw_pair is None:
                    raise RuntimeError(f"Missing post-RoPE K/V capture for history layer {layer_idx}.")
                raw_k, raw_v = raw_pair
                new_k = _store_kv_tensor(raw_k, self._kv_cache_storage_dtype)
                new_v = _store_kv_tensor(raw_v, self._kv_cache_storage_dtype)
                block_kv_post[layer_idx] = (new_k, new_v)
                old = self._history_kv_post[layer_idx]
                if old is None:
                    if self._max_history_frames > 0 and active_len > self._max_history_frames:
                        keep_tokens = self._max_history_frames * self._tokens_per_frame
                        self._history_kv_post[layer_idx] = (new_k[:, -keep_tokens:], new_v[:, -keep_tokens:])
                    else:
                        self._history_kv_post[layer_idx] = (new_k, new_v)
                else:
                    if self._max_history_frames > 0:
                        keep_old_frames = max(0, self._max_history_frames - active_len)
                        keep_old_tokens = keep_old_frames * self._tokens_per_frame
                        old = (
                            old[0][:, -keep_old_tokens:] if keep_old_tokens > 0 else old[0][:, :0],
                            old[1][:, -keep_old_tokens:] if keep_old_tokens > 0 else old[1][:, :0],
                        )
                    self._history_kv_post[layer_idx] = (
                        torch.cat([old[0], new_k], dim=1),
                        torch.cat([old[1], new_v], dim=1),
                    )
                raw_k = None
                raw_v = None
            self._history_frames += active_len

            if self._max_history_frames > 0 and self._history_frames > self._max_history_frames:
                keep_tokens = self._max_history_frames * self._tokens_per_frame
                for layer_idx in range(self._n_layers):
                    hk = self._history_kv_post[layer_idx]
                    if hk is not None:
                        self._history_kv_post[layer_idx] = (hk[0][:, -keep_tokens:], hk[1][:, -keep_tokens:])
                self._history_frames = self._max_history_frames

        profiler.finish()
        return x_t


def _build_rotary_emb_for_absolute_positions(
    *,
    transformer: nn.Module,
    batch_size: int,
    frame_positions: list[int],
    height: int,
    width: int,
    device: torch.device,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplement ``LTX2VideoRotaryPosEmbed.prepare_video_coords`` with explicit per-frame positions.

    The default helper assumes contiguous ``torch.arange(num_frames)`` which is
    fine for bidirectional inference; the sliding-window AR refiner needs to
    keep each frame's absolute index in the source video so RoPE captures the
    correct temporal phase across the sink + recent + active window.
    """
    rope = transformer.rope
    patch_size_t = int(rope.patch_size_t)
    patch_size = int(rope.patch_size)
    f_positions = torch.tensor(frame_positions, dtype=torch.float32, device=device)
    if patch_size_t > 1:
        # Each patch covers ``patch_size_t`` latent frames; pick the start of each patch.
        f_positions = f_positions[::patch_size_t]
    n_f = int(f_positions.shape[0])
    grid_h = torch.arange(start=0, end=height, step=patch_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(start=0, end=width, step=patch_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(f_positions, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # [3, N_F, N_H, N_W]

    patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
    patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)
    latent_coords = torch.stack([grid, patch_ends], dim=-1)
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    scale_tensor = torch.tensor(rope.scale_factors, device=device)
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
    pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return rope(pixel_coords, device=device)


def _forward_video_block(
    *,
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    temb: torch.Tensor,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    encoder_attention_mask: torch.Tensor | None,
    n_context_tokens: int,
    profiler: _RefinerLayerCudaProfiler | None = None,
    capture_kv_only: bool = False,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)

    if profiler is None:
        norm_hidden_states = block.norm1(hidden_states)
        num_ada_params = block.scale_shift_table.shape[0]
        ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        if capture_kv_only:
            _capture_streaming_self_attention_kv(
                attn=block.attn1,
                hidden_states=norm_hidden_states,
                query_rotary_emb=video_rotary_emb,
            )
            return hidden_states

        attn_hidden_states = _streaming_self_attention(
            attn=block.attn1,
            hidden_states=norm_hidden_states,
            query_rotary_emb=video_rotary_emb,
            n_context_tokens=n_context_tokens,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        norm_hidden_states = block.norm2(hidden_states)
        cross_kv_cache = getattr(block.attn2, "_sana_cross_attn_kv_cache", None)
        if cross_kv_cache is None:
            attn_hidden_states = block.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                query_rotary_emb=None,
                attention_mask=encoder_attention_mask,
            )
        else:
            attn_hidden_states = _cross_attention_with_cached_kv(block.attn2, norm_hidden_states, cross_kv_cache)
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + block.ff(norm_hidden_states) * gate_mlp
        return hidden_states

    with _profile_section(profiler, "norm_adaln"):
        norm_hidden_states = block.norm1(hidden_states)
        num_ada_params = block.scale_shift_table.shape[0]
        ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    with _profile_section(profiler, "self_attn"):
        if capture_kv_only:
            _capture_streaming_self_attention_kv(
                attn=block.attn1,
                hidden_states=norm_hidden_states,
                query_rotary_emb=video_rotary_emb,
            )
            return hidden_states
        else:
            attn_hidden_states = _streaming_self_attention(
                attn=block.attn1,
                hidden_states=norm_hidden_states,
                query_rotary_emb=video_rotary_emb,
                n_context_tokens=n_context_tokens,
            )
            hidden_states = hidden_states + attn_hidden_states * gate_msa

    with _profile_section(profiler, "cross_attn"):
        norm_hidden_states = block.norm2(hidden_states)
        cross_kv_cache = getattr(block.attn2, "_sana_cross_attn_kv_cache", None)
        if cross_kv_cache is None:
            attn_hidden_states = block.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                query_rotary_emb=None,
                attention_mask=encoder_attention_mask,
            )
        else:
            attn_hidden_states = _cross_attention_with_cached_kv(block.attn2, norm_hidden_states, cross_kv_cache)
        hidden_states = hidden_states + attn_hidden_states

    with _profile_section(profiler, "ffn"):
        norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + block.ff(norm_hidden_states) * gate_mlp
    return hidden_states


def _streaming_self_attention(
    *,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    n_context_tokens: int,
) -> torch.Tensor:
    """LTX-2 self-attention with sink/current streaming mask + AR KV-cache hooks.

    Two modes are layered on top of vanilla diffusers self-attention, selected by
    ``n_context_tokens`` and per-block hook attributes (set by the AR refiner):

    * ``n_context_tokens > 0`` (legacy single-shot path) — sink queries attend
      sink only, current queries attend ``[sink + current]`` via two SDPA calls.

    * ``n_context_tokens == 0`` (AR mode) — Q comes from the active block only;
      the per-block ``_tf_kv_prefix`` dict (``rf_shifted_sink``) supplies the
      pre-RoPE sink K/V (re-RoPE'd here with its sliding offset PE) and the
      post-RoPE recent-history K/V, concatenated before SDPA. The
      ``_kv_cache_capture`` and ``_tf_capture_kv`` hooks record K/V into the
      module for the AR orchestrator to read back.
    """
    from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

    gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

    fused_qkv = getattr(attn, "_sana_fused_qkv", None)
    if fused_qkv is not None:
        query, key, value = fused_qkv(hidden_states)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # KV-cache capture / inject hooks for ``rf_shifted_sink`` AR refinement.
    # Mirrors tian's ``diffusion/vendors/ltx/ltx_core/model/transformer/attention.py``:
    # - ``_kv_cache_capture`` saves PRE-RoPE (post-norm) K/V so a future window
    #   can re-apply RoPE at its shifted sink offset.
    # - ``_tf_capture_kv`` saves POST-RoPE K/V so the next window can directly
    #   concatenate the recent history.
    # - ``_tf_kv_prefix`` (a dict with ``mode='rf_shifted_sink'``) prepends a
    #   re-RoPE'd sink + already-post-RoPE recent history before SDPA.
    if getattr(attn, "_kv_cache_capture", False):
        attn._cached_kv_pre = (_capture_kv_tensor(key), _capture_kv_tensor(value))

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

    if getattr(attn, "_tf_capture_kv", False):
        attn._cached_kv_post = (_capture_kv_tensor(key), _capture_kv_tensor(value))

    tf_prefix = getattr(attn, "_tf_kv_prefix", None)
    if isinstance(tf_prefix, dict) and tf_prefix.get("mode") == "rf_shifted_sink":
        prefix_k = tf_prefix.get("prefix_k")
        prefix_v = tf_prefix.get("prefix_v")
        if prefix_k is not None and prefix_v is not None:
            key = torch.cat([prefix_k.to(key.dtype), key], dim=1)
            value = torch.cat([prefix_v.to(value.dtype), value], dim=1)
        else:
            prefix_k_parts: list[torch.Tensor] = []
            prefix_v_parts: list[torch.Tensor] = []
            sink_k_pre = tf_prefix.get("sink_k_pre")
            sink_v = tf_prefix.get("sink_v")
            if sink_k_pre is not None and sink_v is not None and sink_k_pre.shape[1] > 0:
                sink_pe = tf_prefix.get("sink_pe")
                if sink_pe is None:
                    raise RuntimeError("rf_shifted_sink prefix requires a sink_pe RoPE tuple.")
                sink_k_pre_dt = sink_k_pre.to(key.dtype)
                if attn.rope_type == "interleaved":
                    sink_k = apply_interleaved_rotary_emb(sink_k_pre_dt, sink_pe)
                else:
                    sink_k = apply_split_rotary_emb(sink_k_pre_dt, sink_pe)
                prefix_k_parts.append(sink_k)
                prefix_v_parts.append(sink_v.to(value.dtype))
            history_k = tf_prefix.get("history_k")
            history_v = tf_prefix.get("history_v")
            if history_k is not None and history_v is not None and history_k.shape[1] > 0:
                prefix_k_parts.append(history_k.to(key.dtype))
                prefix_v_parts.append(history_v.to(value.dtype))
            if prefix_k_parts:
                key = torch.cat([*prefix_k_parts, key], dim=1)
                value = torch.cat([*prefix_v_parts, value], dim=1)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    processor = attn.processor
    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)

    # AR mode (n_context_tokens == 0): Q from active block attends to the
    # injected prefix + current K/V in one SDPA call. Legacy single-shot
    # mode keeps the sink-self / current-cross split.
    if n_context_tokens <= 0 or n_context_tokens >= query.shape[1]:
        hidden_states = _refiner_attention(
            query,
            key,
            value,
            backend=backend,
            parallel_config=parallel_config,
        )
    else:
        context_hidden_states = _refiner_attention(
            query[:, :n_context_tokens],
            key[:, :n_context_tokens],
            value[:, :n_context_tokens],
            backend=backend,
            parallel_config=parallel_config,
        )
        current_hidden_states = _refiner_attention(
            query[:, n_context_tokens:],
            key,
            value,
            backend=backend,
            parallel_config=parallel_config,
        )
        hidden_states = torch.cat([context_hidden_states, current_hidden_states], dim=1)

    hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

    if gate_logits is not None:
        hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
        gates = 2.0 * torch.sigmoid(gate_logits)
        hidden_states = hidden_states * gates.unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def _capture_streaming_self_attention_kv(
    *,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Capture the current layer self-attention K/V without computing attention output."""
    from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

    fused_qkv = getattr(attn, "_sana_fused_qkv", None)
    if fused_qkv is not None:
        _, key, value = fused_qkv(hidden_states)
    else:
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    key = attn.norm_k(key)

    if getattr(attn, "_kv_cache_capture", False):
        attn._cached_kv_pre = (_capture_kv_tensor(key), _capture_kv_tensor(value))

    if attn.rope_type == "interleaved":
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

    if getattr(attn, "_tf_capture_kv", False):
        attn._cached_kv_post = (_capture_kv_tensor(key), _capture_kv_tensor(value))


def _apply_refiner_rotary(
    attn: nn.Module,
    tensor: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

    if attn.rope_type == "interleaved":
        return apply_interleaved_rotary_emb(tensor, rotary_emb)
    if attn.rope_type == "split":
        return apply_split_rotary_emb(tensor, rotary_emb)
    raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")


def _refiner_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: object,
    parallel_config: object,
) -> torch.Tensor:
    kernel = _refiner_self_attn_kernel()
    if kernel in {"flash_attn", "flash-attn", "fa2"}:
        return _flash_attn_func()(query, key, value, dropout_p=0.0, causal=False)
    if kernel in {"sdpa", "torch_sdpa", "pytorch_sdpa"}:
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return hidden_states.transpose(1, 2)
    if kernel and kernel not in {"default", "dispatch", "diffusers", "0", "off"}:
        raise ValueError(f"Unsupported SANA_WM_REFINER_SELF_ATTN_KERNEL={kernel!r}.")
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    return dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=backend,
        parallel_config=parallel_config,
    )


def _refiner_self_attn_kernel() -> str:
    return os.environ.get("SANA_WM_REFINER_SELF_ATTN_KERNEL", "").strip().lower()


_FLASH_ATTN_FUNC = None


def _flash_attn_func():
    global _FLASH_ATTN_FUNC
    if _FLASH_ATTN_FUNC is None:
        from flash_attn import flash_attn_func

        _FLASH_ATTN_FUNC = flash_attn_func
    return _FLASH_ATTN_FUNC


def _set_cross_attention_kv_cache(
    transformer: nn.Module,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor | None,
) -> None:
    blocks = transformer.transformer_blocks
    if not blocks:
        return
    batch_size = int(prompt_embeds.shape[0])
    hidden_dim = int(blocks[0].attn2.to_k.in_features)
    encoder_hidden_states = transformer.caption_projection(prompt_embeds)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_dim)
    encoder_attention_mask = _prepare_encoder_attention_mask(prompt_attention_mask, encoder_hidden_states.dtype)

    for block in blocks:
        attn = block.attn2
        cross_hidden = encoder_hidden_states
        if getattr(attn, "norm_cross", False):
            cross_hidden = attn.norm_encoder_hidden_states(cross_hidden)

        key = attn.to_k(cross_hidden)
        value = attn.to_v(cross_hidden)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        inner_dim = int(key.shape[-1])
        head_dim = inner_dim // int(attn.heads)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        attn._sana_cross_attn_kv_cache = (key.detach(), value.detach(), encoder_attention_mask)


def _clear_cross_attention_kv_cache(transformer: nn.Module) -> None:
    for block in transformer.transformer_blocks:
        if hasattr(block.attn2, "_sana_cross_attn_kv_cache"):
            block.attn2._sana_cross_attn_kv_cache = None


def _has_cross_attention_kv_cache(transformer: nn.Module) -> bool:
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        return False
    return getattr(blocks[0].attn2, "_sana_cross_attn_kv_cache", None) is not None


def _cross_attention_with_cached_kv(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
) -> torch.Tensor:
    key, value, attention_mask = cache
    residual = hidden_states
    input_ndim = hidden_states.ndim

    spatial_norm = getattr(attn, "spatial_norm", None)
    if spatial_norm is not None:
        hidden_states = spatial_norm(hidden_states, None)

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    else:
        batch_size = int(hidden_states.shape[0])
        channel = height = width = None

    if attention_mask is not None:
        source_length = int(key.shape[2])
        prepare_attention_mask = getattr(attn, "prepare_attention_mask", None)
        if prepare_attention_mask is not None:
            attn_mask = prepare_attention_mask(attention_mask, source_length, batch_size)
            attn_mask = attn_mask.view(batch_size, attn.heads, -1, attn_mask.shape[-1])
        elif attention_mask.ndim == 3:
            attn_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            attn_mask = attention_mask[:, None, None, :]
        else:
            attn_mask = attention_mask
    else:
        attn_mask = None

    group_norm = getattr(attn, "group_norm", None)
    if group_norm is not None:
        hidden_states = group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    head_dim = int(key.shape[-1])
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    hidden_states = F.scaled_dot_product_attention(
        query,
        key.to(query.dtype),
        value.to(query.dtype),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + residual
    return hidden_states / float(getattr(attn, "rescale_output_factor", 1.0))


def _set_kv_prefix_on_blocks(
    transformer: nn.Module,
    kv_prefix_per_layer: list[dict[str, object]] | None,
) -> None:
    """Mirror tian's ``_inject_kv_prefix``: attach a per-layer prefix dict to each ``attn1``."""
    blocks = transformer.transformer_blocks
    if kv_prefix_per_layer is None:
        _clear_kv_prefix_on_blocks(transformer)
        return
    if len(kv_prefix_per_layer) != len(blocks):
        raise RuntimeError(
            f"kv_prefix_per_layer has {len(kv_prefix_per_layer)} entries but transformer has {len(blocks)} blocks."
        )
    for block, prefix in zip(blocks, kv_prefix_per_layer):
        block.attn1._tf_kv_prefix = prefix


def _clear_kv_prefix_on_blocks(transformer: nn.Module) -> None:
    for block in transformer.transformer_blocks:
        block.attn1._tf_kv_prefix = None


def _set_capture_flag_on_blocks(
    transformer: nn.Module,
    mode: str,
    *,
    enable: bool,
    layer_mask: list[bool] | None = None,
) -> None:
    """Toggle ``_kv_cache_capture`` (pre-RoPE) or ``_tf_capture_kv`` (post-RoPE) per block."""
    if mode == "pre_rope":
        attr = "_kv_cache_capture"
        clear_attr = "_cached_kv_pre"
    elif mode == "post_rope":
        attr = "_tf_capture_kv"
        clear_attr = "_cached_kv_post"
    else:
        raise ValueError(f"capture_mode must be 'pre_rope' or 'post_rope', got {mode!r}")
    blocks = transformer.transformer_blocks
    if layer_mask is not None and len(layer_mask) != len(blocks):
        raise RuntimeError(f"layer_mask has {len(layer_mask)} entries but transformer has {len(blocks)} blocks.")
    for layer_idx, block in enumerate(blocks):
        enabled = bool(enable and (layer_mask is None or layer_mask[layer_idx]))
        setattr(block.attn1, attr, enabled)
        if enabled:
            # Clear any previous capture so the next forward writes a fresh value.
            if hasattr(block.attn1, clear_attr):
                setattr(block.attn1, clear_attr, None)


def _collect_captured_kv_from_blocks(
    transformer: nn.Module,
    mode: str,
    layer_mask: list[bool] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    attr = "_cached_kv_pre" if mode == "pre_rope" else "_cached_kv_post"
    blocks = transformer.transformer_blocks
    if layer_mask is not None and len(layer_mask) != len(blocks):
        raise RuntimeError(f"layer_mask has {len(layer_mask)} entries but transformer has {len(blocks)} blocks.")
    out: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for layer_idx, block in enumerate(blocks):
        if layer_mask is not None and not layer_mask[layer_idx]:
            out.append(None)
            if hasattr(block.attn1, attr):
                setattr(block.attn1, attr, None)
            continue
        cached = getattr(block.attn1, attr, None)
        if cached is None:
            raise RuntimeError(
                f"Expected {attr!r} on attn1 after capture forward, but found None."
            )
        out.append(cached)
        # Release the reference so the orchestrator owns the only handle.
        setattr(block.attn1, attr, None)
    return out


def _pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: str | torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim, _ = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype

    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]

    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized_hidden_states = normalized_hidden_states * scale_factor
    normalized_hidden_states = normalized_hidden_states.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, normalized_hidden_states.shape[-1])
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    return normalized_hidden_states.to(dtype=original_dtype)


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, _, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _prepare_encoder_attention_mask(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2:
        return mask
    if bool(torch.all(mask)):
        return None
    return ((1 - mask.to(dtype)) * -10000.0).unsqueeze(1)


def _resolve_kv_cache_storage_dtype() -> torch.dtype | None:
    raw = os.environ.get("SANA_WM_REFINER_KV_CACHE_DTYPE", "").strip().lower()
    if not raw or raw in {"bf16", "bfloat16", "none", "off", "0"}:
        return None
    if raw in {"fp8", "fp8_e4m3", "fp8_e4m3fn", "float8_e4m3fn", "e4m3"}:
        return torch.float8_e4m3fn
    if raw in {"fp8_e5m2", "float8_e5m2", "e5m2"}:
        return torch.float8_e5m2
    raise ValueError(f"Unsupported SANA_WM_REFINER_KV_CACHE_DTYPE={raw!r}.")


def _refiner_fast_kv_capture_mode() -> str:
    raw = os.environ.get("SANA_WM_REFINER_FAST_KV_CAPTURE", "").strip().lower()
    if not raw or raw in {"clean", "exact", "off", "0"}:
        return "clean"
    # Reuses K/V from the final denoise prediction. This avoids the extra
    # post-refine capture forward, but the cached history is approximate.
    if raw in {"last_predict", "reuse_last_predict", "final_predict"}:
        return "last_predict"
    # Hybrid mode: run exact post-capture only for
    # SANA_WM_REFINER_EXACT_CAPTURE_LAYERS, and fill the remaining history
    # layers from the final denoise prediction K/V. This is approximate only
    # for layers outside the exact-capture mask.
    if raw in {"fill_missing", "fill-missing", "hybrid_fill", "hybrid"}:
        return "fill_missing"
    raise ValueError(f"Unsupported SANA_WM_REFINER_FAST_KV_CAPTURE={raw!r}.")


def _refiner_fast_kv_needs_clean_block(block_idx: int) -> bool:
    raw = os.environ.get("SANA_WM_REFINER_FAST_KV_CLEAN_INTERVAL", "").strip()
    if not raw:
        return False
    interval = int(raw)
    if interval <= 0:
        return False
    # Keep block 0 exact so the sink/first active history starts clean, then
    # refresh periodically to bound drift in long videos.
    return block_idx == 0 or ((block_idx + 1) % interval == 0)


def _refiner_history_layer_mask(n_layers: int) -> list[bool]:
    raw_layers = os.environ.get("SANA_WM_REFINER_HISTORY_LAYERS", "").strip()
    if raw_layers:
        mask = [False] * int(n_layers)
        for item in raw_layers.split(","):
            item = item.strip()
            if not item:
                continue
            if item.lower() == "last":
                mask[-1] = True
                continue
            if "-" in item:
                start_raw, end_raw = item.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if start < 0:
                    start += n_layers
                if end < 0:
                    end += n_layers
                for idx in range(max(0, start), min(n_layers - 1, end) + 1):
                    mask[idx] = True
                continue
            idx = int(item)
            if idx < 0:
                idx += n_layers
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"SANA_WM_REFINER_HISTORY_LAYERS index {item!r} outside 0..{n_layers - 1}.")
            mask[idx] = True
        if not any(mask):
            raise ValueError("SANA_WM_REFINER_HISTORY_LAYERS selected no layers.")
        return mask

    stride_raw = os.environ.get("SANA_WM_REFINER_HISTORY_LAYER_STRIDE", "").strip()
    if not stride_raw:
        return [True] * int(n_layers)
    stride = int(stride_raw)
    if stride <= 1:
        return [True] * int(n_layers)
    offset = int(os.environ.get("SANA_WM_REFINER_HISTORY_LAYER_OFFSET", "0"))
    mask = [((idx - offset) % stride == 0) for idx in range(int(n_layers))]
    if _env_flag_default_true("SANA_WM_REFINER_HISTORY_KEEP_LAST"):
        mask[-1] = True
    if not any(mask):
        mask[-1] = True
    return mask


def _refiner_exact_capture_layer_mask(n_layers: int, *, default_mask: list[bool]) -> list[bool]:
    raw_layers = os.environ.get("SANA_WM_REFINER_EXACT_CAPTURE_LAYERS", "").strip()
    if not raw_layers:
        return list(default_mask)
    mask = [False] * int(n_layers)
    for item in raw_layers.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "last":
            mask[-1] = True
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start < 0:
                start += n_layers
            if end < 0:
                end += n_layers
            for idx in range(max(0, start), min(n_layers - 1, end) + 1):
                mask[idx] = True
            continue
        idx = int(item)
        if idx < 0:
            idx += n_layers
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"SANA_WM_REFINER_EXACT_CAPTURE_LAYERS index {item!r} outside 0..{n_layers - 1}.")
        mask[idx] = True
    if not any(mask):
        raise ValueError("SANA_WM_REFINER_EXACT_CAPTURE_LAYERS selected no layers.")
    return mask


def _refiner_empty_cache_before_capture() -> bool:
    raw = os.environ.get("SANA_WM_REFINER_EMPTY_CACHE_BEFORE_CAPTURE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _refiner_profile_enabled() -> bool:
    return _env_flag("SANA_WM_REFINER_PROFILE")


def _refiner_layer_profile_enabled() -> bool:
    return _env_flag("SANA_WM_REFINER_LAYER_PROFILE")


class _RefinerCudaProfiler:
    """Tiny env-gated CUDA event profiler for one refiner AR block."""

    def __init__(self, *, enabled: bool, device: torch.device, block_idx: int) -> None:
        self.enabled = bool(enabled and device.type == "cuda")
        self.device = device
        self.block_idx = int(block_idx)
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._block_start: torch.cuda.Event | None = None
        self._block_end: torch.cuda.Event | None = None
        if self.enabled:
            stream = torch.cuda.current_stream(device)
            self._block_start = torch.cuda.Event(enable_timing=True)
            self._block_end = torch.cuda.Event(enable_timing=True)
            self._block_start.record(stream)

    def section(self, name: str):
        if not self.enabled:
            return nullcontext()
        return _RefinerCudaProfileSection(self, name)

    def _record_section(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._events.append((name, start, end))

    def finish(self) -> None:
        if not self.enabled:
            return
        stream = torch.cuda.current_stream(self.device)
        assert self._block_start is not None and self._block_end is not None
        self._block_end.record(stream)
        self._block_end.synchronize()

        totals_ms: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, start, end in self._events:
            elapsed_ms = float(start.elapsed_time(end))
            totals_ms[name] = totals_ms.get(name, 0.0) + elapsed_ms
            counts[name] = counts.get(name, 0) + 1

        block_total_ms = float(self._block_start.elapsed_time(self._block_end))
        parts = [f"block_total={block_total_ms / 1000.0:.6f}s"]
        for name, elapsed_ms in totals_ms.items():
            count_suffix = f"x{counts[name]}" if counts[name] != 1 else ""
            parts.append(f"{name}={elapsed_ms / 1000.0:.6f}s{count_suffix}")
        print(f"[refiner-profile] block={self.block_idx} " + " ".join(parts), flush=True)


class _RefinerCudaProfileSection:
    def __init__(self, profiler: _RefinerCudaProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = str(name)
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def __enter__(self):
        stream = torch.cuda.current_stream(self._profiler.device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record(stream)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._start is not None and self._end is not None
        stream = torch.cuda.current_stream(self._profiler.device)
        self._end.record(stream)
        self._profiler._record_section(self._name, self._start, self._end)
        return False


def _current_refiner_prefix_tokens(transformer: nn.Module) -> int:
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        return 0
    prefix = getattr(blocks[0].attn1, "_tf_kv_prefix", None)
    if not isinstance(prefix, dict):
        return 0
    prefix_k = prefix.get("prefix_k")
    if isinstance(prefix_k, torch.Tensor):
        return int(prefix_k.shape[1])
    total = 0
    sink_k_pre = prefix.get("sink_k_pre")
    if isinstance(sink_k_pre, torch.Tensor):
        total += int(sink_k_pre.shape[1])
    history_k = prefix.get("history_k")
    if isinstance(history_k, torch.Tensor):
        total += int(history_k.shape[1])
    return total


def _profile_section(profiler: "_RefinerLayerCudaProfiler | None", name: str):
    if profiler is None:
        return nullcontext()
    return profiler.section(name)


class _RefinerLayerCudaProfiler:
    """Env-gated CUDA event profiler for one transformer forward."""

    def __init__(self, *, enabled: bool, device: torch.device, label: str) -> None:
        self.enabled = bool(enabled and device.type == "cuda")
        self.device = device
        self.label = str(label)
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        if self.enabled:
            stream = torch.cuda.current_stream(device)
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record(stream)

    def section(self, name: str):
        if not self.enabled:
            return nullcontext()
        return _RefinerLayerCudaProfileSection(self, name)

    def _record_section(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._events.append((name, start, end))

    def finish(self) -> None:
        if not self.enabled:
            return
        stream = torch.cuda.current_stream(self.device)
        assert self._start is not None and self._end is not None
        self._end.record(stream)
        self._end.synchronize()
        totals_ms: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, start, end in self._events:
            elapsed_ms = float(start.elapsed_time(end))
            totals_ms[name] = totals_ms.get(name, 0.0) + elapsed_ms
            counts[name] = counts.get(name, 0) + 1
        total_ms = float(self._start.elapsed_time(self._end))
        parts = [f"total={total_ms / 1000.0:.6f}s"]
        for name, elapsed_ms in totals_ms.items():
            count_suffix = f"x{counts[name]}" if counts[name] != 1 else ""
            parts.append(f"{name}={elapsed_ms / 1000.0:.6f}s{count_suffix}")
        print(f"[refiner-layer-profile] {self.label} " + " ".join(parts), flush=True)


class _RefinerLayerCudaProfileSection:
    def __init__(self, profiler: _RefinerLayerCudaProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = str(name)
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def __enter__(self):
        stream = torch.cuda.current_stream(self._profiler.device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record(stream)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._start is not None and self._end is not None
        stream = torch.cuda.current_stream(self._profiler.device)
        self._end.record(stream)
        self._profiler._record_section(self._name, self._start, self._end)
        return False


def _store_kv_tensor(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    if dtype is None:
        return tensor
    return tensor.to(dtype)


def _store_kv_pair(
    pair: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (_store_kv_tensor(pair[0], dtype), _store_kv_tensor(pair[1], dtype))


def _capture_kv_tensor(tensor: torch.Tensor) -> torch.Tensor:
    captured = tensor.detach()
    if _env_flag("SANA_WM_REFINER_NO_CLONE_CAPTURED_KV"):
        return captured
    return captured.clone()


def _env_tuple(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_flag_default_true(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"", "0", "false", "no", "off"}


def _prepared_module_cache_root() -> Path | None:
    if os.environ.get("SANA_WM_PREPARED_MODULE_CACHE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    root = os.environ.get("SANA_WM_PREPARED_MODULE_CACHE_DIR", "").strip()
    return Path(root).expanduser() if root else Path.home() / ".cache" / "sana_wm_prepared_modules"


def _prepared_module_cache_hash(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def _path_fingerprint(path: str | Path) -> dict[str, object]:
    raw = str(path)
    try:
        resolved = Path(raw).expanduser().resolve()
    except Exception:
        return {"path": raw}
    if resolved.is_dir():
        markers = []
        for rel in ("config.json", "diffusion_pytorch_model.safetensors", "model.safetensors"):
            item = resolved / rel
            try:
                stat = item.stat()
            except OSError:
                continue
            markers.append((rel, int(stat.st_size), int(stat.st_mtime_ns)))
        return {"path": str(resolved), "markers": markers}
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved)}
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _is_local_callable_for_pickle(value: object) -> bool:
    if isinstance(value, types.MethodType):
        value = value.__func__
    if not isinstance(value, types.FunctionType):
        return False
    qualname = getattr(value, "__qualname__", "")
    return "<locals>" in qualname or getattr(value, "__name__", "") == "<lambda>"


def _strip_local_callables_for_pickle(root: object) -> list[tuple[object, object, object, str]]:
    """Temporarily remove TE init closures that are not used after construction."""

    restore: list[tuple[object, object, object, str]] = []
    seen: set[int] = set()
    leaf_types = (str, bytes, int, float, bool, type(None), Path, torch.device, torch.dtype)

    def set_value(owner: object, key: object, old_value: object, new_value: object, kind: str) -> None:
        if kind == "dict":
            owner[key] = new_value
        elif kind == "list":
            owner[key] = new_value
        else:
            setattr(owner, str(key), new_value)
        restore.append((owner, key, old_value, kind))

    def scrub_value(value: object) -> tuple[object, bool]:
        if _is_local_callable_for_pickle(value):
            return None, True
        if hasattr(value, "_replace") and hasattr(value, "init_fn"):
            updates = {}
            if _is_local_callable_for_pickle(getattr(value, "init_fn", None)):
                updates["init_fn"] = None
            if _is_local_callable_for_pickle(getattr(value, "get_rng_state_tracker", None)):
                updates["get_rng_state_tracker"] = None
            if updates:
                return value._replace(**updates), True
        return value, False

    def walk(obj: object) -> None:
        if isinstance(obj, leaf_types) or isinstance(obj, (torch.Tensor, nn.Parameter)):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                new_value, changed = scrub_value(value)
                if changed:
                    set_value(obj, key, value, new_value, "dict")
                else:
                    walk(value)
            return
        if isinstance(obj, list):
            for index, value in enumerate(list(obj)):
                new_value, changed = scrub_value(value)
                if changed:
                    set_value(obj, index, value, new_value, "list")
                else:
                    walk(value)
            return
        if isinstance(obj, tuple):
            return

        try:
            items = list(vars(obj).items())
        except TypeError:
            return
        for key, value in items:
            if key.startswith("__"):
                continue
            new_value, changed = scrub_value(value)
            if changed:
                set_value(obj, key, value, new_value, "attr")
            elif key not in {"_parameters", "_buffers"}:
                walk(value)

    walk(root)
    return restore


def _restore_stripped_pickle_values(restore: list[tuple[object, object, object, str]]) -> None:
    for owner, key, value, kind in reversed(restore):
        if kind == "dict":
            owner[key] = value
        elif kind == "list":
            owner[key] = value
        else:
            setattr(owner, str(key), value)


def _te_module_name_variants(name: str) -> tuple[str, ...]:
    if not _env_flag("SANA_WM_TE_NVFP4_NORMALIZE_MODULE_NAMES"):
        return (name,)
    variants = {name}
    stripped = name
    while stripped.startswith("_orig_mod."):
        stripped = stripped[len("_orig_mod.") :]
        variants.add(stripped)
    variants.add(name.replace("._orig_mod.", "."))
    variants.add(name.replace("_orig_mod.", ""))
    return tuple(dict.fromkeys(item for item in variants if item))


def _te_name_matches(patterns: tuple[str, ...], name: str) -> bool:
    return any(re.search(pattern, candidate) for pattern in patterns for candidate in _te_module_name_variants(name))


class _FusedQKVLinear(nn.Module):
    def __init__(self, to_q: nn.Linear, to_k: nn.Linear, to_v: nn.Linear) -> None:
        super().__init__()
        if to_q.in_features != to_k.in_features or to_q.in_features != to_v.in_features:
            raise ValueError("Cannot fuse QKV with mismatched input dimensions.")
        device = to_q.weight.device
        dtype = to_q.weight.dtype
        out_features = to_q.out_features + to_k.out_features + to_v.out_features
        use_bias = to_q.bias is not None or to_k.bias is not None or to_v.bias is not None
        self.linear = nn.Linear(to_q.in_features, out_features, bias=use_bias, device=device, dtype=dtype)
        self._splits = (to_q.out_features, to_k.out_features, to_v.out_features)
        with torch.no_grad():
            self.linear.weight.copy_(torch.cat([to_q.weight, to_k.weight, to_v.weight], dim=0))
            if self.linear.bias is not None:
                bias_parts = []
                for src in (to_q, to_k, to_v):
                    if src.bias is None:
                        bias_parts.append(torch.zeros(src.out_features, device=device, dtype=dtype))
                    else:
                        bias_parts.append(src.bias)
                self.linear.bias.copy_(torch.cat(bias_parts, dim=0))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.linear(hidden_states).split(self._splits, dim=-1)


def _fuse_refiner_self_qkv(transformer: nn.Module) -> int:
    converted = 0
    for block in getattr(transformer, "transformer_blocks", ()):
        attn = getattr(block, "attn1", None)
        if attn is None or getattr(attn, "_sana_fused_qkv", None) is not None:
            continue
        to_q = getattr(attn, "to_q", None)
        to_k = getattr(attn, "to_k", None)
        to_v = getattr(attn, "to_v", None)
        if not all(isinstance(module, nn.Linear) for module in (to_q, to_k, to_v)):
            continue
        fused = _FusedQKVLinear(to_q, to_k, to_v)
        fused.train(bool(to_q.training or to_k.training or to_v.training))
        attn._sana_fused_qkv = fused
        attn.to_q = nn.Identity()
        attn.to_k = nn.Identity()
        attn.to_v = nn.Identity()
        converted += 1
    return converted


def _replace_linear_with_te_nvfp4(
    module: nn.Module,
    *,
    recipe,
    params_dtype: torch.dtype,
    skip_patterns: tuple[str, ...],
    include_patterns: tuple[str, ...] | None = None,
    prefix: str = "",
) -> tuple[int, int]:
    import transformer_engine.pytorch as te

    converted = 0
    skipped = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if _te_name_matches(skip_patterns, child_prefix):
            skipped += 1
            continue
        if isinstance(child, nn.Linear):
            if include_patterns is not None and not _te_name_matches(include_patterns, child_prefix):
                skipped += 1
                continue
            if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                skipped += 1
                continue
            use_cpu_staging = _env_flag("SANA_WM_TE_NVFP4_CPU_STAGING")
            child_training = child.training
            has_bias = child.bias is not None
            params_dtype_for_replacement = (
                child.weight.dtype
                if child.weight.dtype in {torch.float16, torch.bfloat16, torch.float32}
                else params_dtype
            )
            if use_cpu_staging:
                old_weight = child.weight.detach().to("cpu", copy=True)
                old_bias = child.bias.detach().to("cpu", copy=True) if child.bias is not None else None
                setattr(module, name, nn.Identity())
                del child
                gc.collect()
                _empty_cuda_cache()
            else:
                old_weight = child.weight.detach()
                old_bias = child.bias.detach() if child.bias is not None else None
            try:
                ctx = te.fp8_model_init(
                    enabled=True,
                    recipe=recipe,
                    preserve_high_precision_init_val=False,
                )
            except TypeError:
                ctx = te.fp8_model_init(enabled=True, recipe=recipe)
            with ctx:
                replacement = te.Linear(
                    old_weight.shape[1],
                    old_weight.shape[0],
                    bias=has_bias,
                    params_dtype=params_dtype_for_replacement,
                    device=str(torch.device("cuda", torch.cuda.current_device())),
                )
            replacement.train(child_training)
            with torch.no_grad():
                replacement.weight.copy_(old_weight.to(device=replacement.weight.device))
                if old_bias is not None:
                    replacement.bias.copy_(old_bias.to(device=replacement.bias.device))
            if use_cpu_staging:
                del old_weight, old_bias
                _empty_cuda_cache()
            setattr(module, name, replacement)
            converted += 1
            continue
        child_converted, child_skipped = _replace_linear_with_te_nvfp4(
            child,
            recipe=recipe,
            params_dtype=params_dtype,
            skip_patterns=skip_patterns,
            include_patterns=include_patterns,
            prefix=child_prefix,
        )
        converted += child_converted
        skipped += child_skipped
    return converted, skipped


def _offload_video_unused_audio_modules(transformer: nn.Module, device: torch.device | str) -> None:
    for name in (
        "audio_proj_in",
        "audio_caption_projection",
        "audio_time_embed",
        "av_cross_attn_video_scale_shift",
        "av_cross_attn_audio_scale_shift",
        "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_v2a_gate",
        "audio_rope",
        "cross_attn_rope",
        "cross_attn_audio_rope",
        "audio_norm_out",
        "audio_proj_out",
    ):
        child = getattr(transformer, name, None)
        if isinstance(child, nn.Module):
            child.to(device)
    for block in getattr(transformer, "transformer_blocks", ()):
        for name in (
            "audio_norm1",
            "audio_attn1",
            "audio_norm2",
            "audio_attn2",
            "audio_to_video_norm",
            "audio_to_video_attn",
            "video_to_audio_norm",
            "video_to_audio_attn",
            "audio_norm3",
            "audio_ff",
        ):
            child = getattr(block, name, None)
            if isinstance(child, nn.Module):
                child.to(device)


def _move_ltx2_video_modules_to(transformer: nn.Module, device: torch.device | str) -> None:
    for name in ("proj_in", "caption_projection", "time_embed", "rope", "norm_out", "proj_out"):
        child = getattr(transformer, name, None)
        if isinstance(child, nn.Module):
            child.to(device)
    _move_tensor_attr(transformer, "scale_shift_table", device)
    for block in getattr(transformer, "transformer_blocks", ()):
        _move_tensor_attr(block, "scale_shift_table", device)
        for name in ("norm1", "attn1", "norm2", "attn2", "norm3", "ff"):
            child = getattr(block, name, None)
            if isinstance(child, nn.Module):
                child.to(device)


def _move_tensor_attr(module: nn.Module, name: str, device: torch.device | str) -> None:
    value = getattr(module, name, None)
    if isinstance(value, nn.Parameter):
        if value.device != torch.device(device):
            setattr(module, name, nn.Parameter(value.to(device), requires_grad=value.requires_grad))
    elif isinstance(value, torch.Tensor) and value.device != torch.device(device):
        setattr(module, name, value.to(device))


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
