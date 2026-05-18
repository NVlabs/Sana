from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch


@dataclass
class BlockTailCache:
    video_tail: torch.Tensor | None
    audio_tail: torch.Tensor | None


class LTX2BlockStateReusePatch(AbstractContextManager["LTX2BlockStateReusePatch"]):
    """Runtime monkey patch for block-level hidden-state reuse across windows.

    This patch wraps each discovered `LTX2VideoTransformerBlock.forward` call and
    performs two operations:

    1. Before the block forward: replace the prefix hidden states with the
       cached tail hidden states from the previous window at the same denoising
       step and block index.
    2. After the block forward: cache the current window tail hidden states for
       the same step and block index.

    This is intentionally implemented as a runtime patch so we do not have to
    modify the installed diffusers package on disk.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.transformer = transformer
        self._patched_blocks: list[tuple[int, torch.nn.Module, Any]] = []
        self._active = False
        self._current_step_index: int | None = None
        self._current_video_prefix_tokens = 0
        self._current_audio_prefix_tokens = 0
        self._previous_window_cache: list[dict[int, BlockTailCache]] | None = None
        self._current_window_cache: list[dict[int, BlockTailCache]] | None = None

    def __enter__(self) -> LTX2BlockStateReusePatch:
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()
        return False

    def install(self) -> None:
        if self._patched_blocks:
            return
        block_index = 0
        for module in self.transformer.modules():
            if module.__class__.__name__ != "LTX2VideoTransformerBlock":
                continue
            original_forward = module.forward

            def _wrapped_forward(
                block_self, *args, __orig_forward=original_forward, __block_index=block_index, **kwargs
            ):
                local_args = list(args)
                if self._active and self._current_step_index is not None and len(local_args) >= 2:
                    hidden_states = local_args[0]
                    audio_hidden_states = local_args[1]
                    prev_step_cache = None
                    if self._previous_window_cache is not None and 0 <= int(self._current_step_index) < len(
                        self._previous_window_cache
                    ):
                        prev_step_cache = self._previous_window_cache[int(self._current_step_index)].get(
                            int(__block_index)
                        )
                    if prev_step_cache is not None:
                        if (
                            self._current_video_prefix_tokens > 0
                            and prev_step_cache.video_tail is not None
                            and hidden_states.shape[1] >= self._current_video_prefix_tokens
                        ):
                            hidden_states = hidden_states.clone()
                            hidden_states[:, : self._current_video_prefix_tokens] = prev_step_cache.video_tail.to(
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                            )
                            local_args[0] = hidden_states
                        if (
                            self._current_audio_prefix_tokens > 0
                            and prev_step_cache.audio_tail is not None
                            and audio_hidden_states is not None
                            and audio_hidden_states.shape[1] >= self._current_audio_prefix_tokens
                        ):
                            audio_hidden_states = audio_hidden_states.clone()
                            audio_hidden_states[:, : self._current_audio_prefix_tokens] = prev_step_cache.audio_tail.to(
                                device=audio_hidden_states.device,
                                dtype=audio_hidden_states.dtype,
                            )
                            local_args[1] = audio_hidden_states

                output = __orig_forward(*local_args, **kwargs)

                if self._active and self._current_step_index is not None:
                    if not isinstance(output, tuple) or len(output) < 2:
                        raise RuntimeError(
                            "LTX2 block hidden-state reuse patch expected block forward to return "
                            "(hidden_states, audio_hidden_states)."
                        )
                    hidden_states_out = output[0]
                    audio_hidden_states_out = output[1]
                    video_tail = None
                    audio_tail = None
                    if self._current_video_prefix_tokens > 0:
                        video_tail = hidden_states_out[:, -self._current_video_prefix_tokens :].detach().cpu()
                    if self._current_audio_prefix_tokens > 0 and audio_hidden_states_out is not None:
                        audio_tail = audio_hidden_states_out[:, -self._current_audio_prefix_tokens :].detach().cpu()
                    if self._current_window_cache is not None:
                        self._current_window_cache[int(self._current_step_index)][int(__block_index)] = BlockTailCache(
                            video_tail=video_tail,
                            audio_tail=audio_tail,
                        )
                return output

            module.forward = MethodType(_wrapped_forward, module)
            self._patched_blocks.append((block_index, module, original_forward))
            block_index += 1

        if not self._patched_blocks:
            raise RuntimeError("Could not find any LTX2VideoTransformerBlock modules to patch.")

    def uninstall(self) -> None:
        for _, module, original_forward in reversed(self._patched_blocks):
            module.forward = original_forward
        self._patched_blocks.clear()

    def begin_window(
        self,
        *,
        num_inference_steps: int,
        overlap_num_frames: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        audio_num_frames: int,
        previous_window_cache: list[dict[int, BlockTailCache]] | None,
    ) -> None:
        video_prefix_tokens = max(0, int(overlap_num_frames) * int(latent_height) * int(latent_width))
        if int(latent_num_frames) > 0:
            audio_prefix_tokens = max(
                0,
                int(round(float(audio_num_frames) * float(overlap_num_frames) / float(latent_num_frames))),
            )
        else:
            audio_prefix_tokens = 0
        self._active = True
        self._current_step_index = None
        self._current_video_prefix_tokens = int(video_prefix_tokens)
        self._current_audio_prefix_tokens = int(audio_prefix_tokens)
        self._previous_window_cache = previous_window_cache
        self._current_window_cache = [dict() for _ in range(int(num_inference_steps))]

    def set_step_index(self, step_index: int) -> None:
        self._current_step_index = int(step_index)

    def finish_window(self) -> list[dict[int, BlockTailCache]]:
        current = self._current_window_cache or []
        self._active = False
        self._current_step_index = None
        self._current_video_prefix_tokens = 0
        self._current_audio_prefix_tokens = 0
        self._previous_window_cache = None
        self._current_window_cache = None
        return current
