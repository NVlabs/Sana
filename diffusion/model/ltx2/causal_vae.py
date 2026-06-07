# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LTX-2 Video VAE with causal encoding/decoding and streaming cache support.

This module implements a 3D variational autoencoder for video compression with support for:
- Causal temporal convolution (future frames depend only on past frames)
- Streaming decode with persistent feature cache for memory-efficient long video processing
- Bidirectional encoding/decoding mode

Key classes:
- AutoencoderKLCausalLTX2Video: Main VAE model with cache management
- DecoderCacheManager: Manages decoder cache state for streaming inference
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook


# =============================================================================
# Utility Functions
# =============================================================================

def _shape_of(x) -> Optional[tuple]:
    """Get shape of tensor if it is a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return None


def _compute_conv_output_size(in_size: int, k: int, s: int, p: int, d: int) -> int:
    """Compute output size of a convolution operation."""
    return max((in_size + 2 * p - d * (k - 1) - 1) // s + 1, 0)


# =============================================================================
# Cache Management Classes
# =============================================================================

@dataclass
class DecoderCacheState:
    """State container for decoder streaming cache.
    
    Attributes:
        feat_map: Per-layer feature cache for causal convolution
        is_first_chunk: Whether this is the first chunk in the sequence
        prev_latent_tail: Previous chunk's latent tail for context
        cache_mode: Current cache mode (causal or not)
    """
    feat_map: list = field(default_factory=list)
    is_first_chunk: bool = True
    prev_latent_tail: Optional[torch.Tensor] = None
    cache_mode: Optional[bool] = None


class DecoderCacheManager:
    """Manages decoder cache state for streaming video decoding.
    
    This class encapsulates all decoder cache logic, providing a clean interface
    for cache operations used during streaming decode of long videos.
    
    Example:
        >>> manager = DecoderCacheManager()
        >>> manager.clear()  # Reset cache
        >>> # In streaming loop:
        >>> z_chunk = manager.prepend_context(z_chunk, prepend_frames=1)
        >>> decoded = decoder(z_chunk, feat_cache=manager.feat_map, feat_idx=[0])
        >>> decoded = manager.trim_output(decoded, prepend_frames=1, chunk_frames=z.shape[2])
        >>> manager.update_tail(z_chunk, prepend_frames=1)
    """
    
    def __init__(self):
        self._state = DecoderCacheState()
    
    @property
    def feat_map(self) -> list:
        """Get the feature cache map."""
        return self._state.feat_map
    
    @property
    def is_first_chunk(self) -> bool:
        """Check if this is the first chunk."""
        return self._state.is_first_chunk
    
    @property
    def prev_latent_tail(self) -> Optional[torch.Tensor]:
        """Get previous latent tail."""
        return self._state.prev_latent_tail
    
    @property
    def cache_mode(self) -> Optional[bool]:
        """Get current cache mode."""
        return self._state.cache_mode
    
    def clear(self) -> None:
        """Clear all cache state."""
        self._state = DecoderCacheState()
    
    def validate_mode(self, causal: bool) -> None:
        """Validate and update cache mode.
        
        Args:
            causal: Desired causal mode
            
        Raises:
            ValueError: If mode conflicts with existing cache
        """
        if self._state.cache_mode is None:
            self._state.cache_mode = causal
        elif self._state.cache_mode != causal:
            # Mode mismatch - clear cache to avoid mixing states
            self.clear()
            self._state.cache_mode = causal
    
    def prepend_context(
        self, 
        z: torch.Tensor, 
        prepend_prev_latent_frames: int,
        temporal_compression_ratio: int
    ) -> torch.Tensor:
        """Prepend previous chunk's latent tail as left context.
        
        Args:
            z: Current latent chunk [B, C, T, H, W]
            prepend_prev_latent_frames: Number of frames to prepend
            temporal_compression_ratio: Ratio for converting latent to sample frames
            
        Returns:
            Latent tensor with context prepended
        """
        if prepend_prev_latent_frames <= 0:
            return z
            
        prev_tail = self._state.prev_latent_tail
        if prev_tail is None:
            # First chunk: repeat first frame
            left_ctx = z[:, :, :1, :, :].repeat(1, 1, prepend_prev_latent_frames, 1, 1)
        else:
            # Use previous chunk's tail
            left_ctx = prev_tail.to(z.device)
            if left_ctx.shape[2] > prepend_prev_latent_frames:
                left_ctx = left_ctx[:, :, -prepend_prev_latent_frames:, :, :]
            if left_ctx.shape[2] < prepend_prev_latent_frames:
                fill = z[:, :, :1, :, :].repeat(1, 1, prepend_prev_latent_frames - left_ctx.shape[2], 1, 1)
                left_ctx = torch.cat([fill, left_ctx], dim=2)
        
        return torch.cat([left_ctx, z], dim=2)
    
    def trim_output(
        self,
        decoded: torch.Tensor,
        prepend_prev_latent_frames: int,
        chunk_latent_frames: int,
        temporal_compression_ratio: int
    ) -> torch.Tensor:
        """Trim decoder output to remove context frames.
        
        Args:
            decoded: Full decoder output [B, C, T, H, W]
            prepend_prev_latent_frames: Number of prepended latent frames
            chunk_latent_frames: Number of latent frames in current chunk
            temporal_compression_ratio: Ratio for converting latent to sample frames
            
        Returns:
            Trimmed output tensor
        """
        drop_left_t = prepend_prev_latent_frames * temporal_compression_ratio
        
        if self._state.is_first_chunk:
            # First chunk: keep all frames from context start
            keep_t = (chunk_latent_frames - 1) * temporal_compression_ratio + 1
            self._state.is_first_chunk = False
        else:
            # Subsequent chunks: keep only new frames
            keep_t = chunk_latent_frames * temporal_compression_ratio
        
        return decoded[:, :, drop_left_t:drop_left_t + keep_t, :, :]
    
    def update_tail(self, z: torch.Tensor, prepend_prev_latent_frames: int) -> None:
        """Update the previous latent tail for next chunk.
        
        Args:
            z: Current latent chunk before context prepending
            prepend_prev_latent_frames: Number of frames to save as tail
        """
        if prepend_prev_latent_frames > 0:
            self._state.prev_latent_tail = z[:, :, -prepend_prev_latent_frames:, :, :].clone()

# =============================================================================
# Normalization Layers
# =============================================================================

class PerChannelRMSNorm(nn.Module):
    """Per-pixel (per-location) RMS normalization layer.

    For each element along the chosen dimension, this layer normalizes the tensor by the root-mean-square of its values
    across that dimension:

        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)

    Args:
        channel_dim: Dimension along which to compute the RMS (typically channels).
        eps: Small constant added for numerical stability.
    """

    def __init__(self, channel_dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, x: torch.Tensor, channel_dim: Optional[int] = None) -> torch.Tensor:
        """Apply RMS normalization along the configured dimension."""
        channel_dim = channel_dim or self.channel_dim
        mean_sq = torch.mean(x**2, dim=self.channel_dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


# =============================================================================
# Causal Convolution Layer
# =============================================================================

class LTX2VideoCausalConv3d(nn.Module):
    """Causal 3D convolution for video processing.
    
    Like LTXCausalConv3d, but whether causal inference is performed can be
    specified at runtime via the `causal` parameter.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Temporal, height, width kernel size (int or tuple)
        stride: Stride for convolution (int or tuple)
        dilation: Dilation for convolution (int or tuple)
        groups: Number of groups for grouped convolution
        spatial_padding_mode: Padding mode for spatial dimensions
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with optional causal convolution and feature caching.
        
        Args:
            hidden_states: Input tensor [B, C, T, H, W]
            causal: Whether to use causal convolution
            feat_cache: Cache for temporal feature reuse
            feat_idx: Current index into feat_cache (mutated in-place)
            
        Returns:
            Output tensor after convolution
        """
        time_kernel_size = self.kernel_size[0]
        input_shape = _shape_of(hidden_states)
        
        # Handle empty temporal chunks
        if hidden_states.shape[2] == 0:
            return self._handle_empty_chunk(hidden_states, causal, feat_cache, feat_idx, time_kernel_size)

        if causal:
            hidden_states = self._apply_causal_padding(
                hidden_states, feat_cache, feat_idx, time_kernel_size, input_shape
            )
        else:
            hidden_states = self._apply_bidirectional_padding(hidden_states, time_kernel_size)

        hidden_states = self.conv(hidden_states)
        return hidden_states
    
    def _handle_empty_chunk(
        self,
        hidden_states: torch.Tensor,
        causal: bool,
        feat_cache: Optional[list],
        feat_idx: Optional[list],
        time_kernel_size: int
    ) -> torch.Tensor:
        """Handle empty temporal chunks by reserving cache slots."""
        cache_len = max(time_kernel_size - 1, 0)
        if causal and feat_cache is not None and feat_idx is not None and cache_len > 0:
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)
            feat_idx[0] += 1

        batch = hidden_states.shape[0]
        out_h = _compute_conv_output_size(
            hidden_states.shape[3],
            self.conv.kernel_size[1],
            self.conv.stride[1],
            self.conv.padding[1],
            self.conv.dilation[1],
        )
        out_w = _compute_conv_output_size(
            hidden_states.shape[4],
            self.conv.kernel_size[2],
            self.conv.stride[2],
            self.conv.padding[2],
            self.conv.dilation[2],
        )
        return hidden_states.new_empty(batch, self.conv.out_channels, 0, out_h, out_w)
    
    def _apply_causal_padding(
        self,
        hidden_states: torch.Tensor,
        feat_cache: Optional[list],
        feat_idx: Optional[list],
        time_kernel_size: int,
        input_shape: Optional[tuple]
    ) -> torch.Tensor:
        """Apply causal padding using cached history."""
        cache_len = max(time_kernel_size - 1, 0)
        if feat_cache is not None and feat_idx is not None and cache_len > 0:
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)

            prefix = feat_cache[idx]
            current_cache = hidden_states[:, :, -cache_len:, :, :].clone()
            
            # Handle short chunks by preserving history
            if isinstance(prefix, torch.Tensor) and current_cache.shape[2] < cache_len:
                needed = cache_len - current_cache.shape[2]
                carry = prefix.to(hidden_states.device)
                carry = carry[:, :, -needed:, :, :]
                if carry.shape[2] < needed:
                    fill = carry[:, :, :1, :, :].repeat((1, 1, needed - carry.shape[2], 1, 1))
                    carry = torch.cat([fill, carry], dim=2)
                current_cache = torch.cat([carry, current_cache], dim=2)

            # Prepare prefix from cache or fallback
            if prefix is not None:
                prefix = prefix.to(hidden_states.device)
                if prefix.shape[2] > cache_len:
                    prefix = prefix[:, :, -cache_len:, :, :]
                if prefix.shape[2] < cache_len:
                    # Extend by repeating oldest cached frame
                    fallback = prefix[:, :, :1, :, :].repeat((1, 1, cache_len - prefix.shape[2], 1, 1))
                    prefix = torch.concatenate([fallback, prefix], dim=2)
            else:
                # No cache: use first frame as fallback
                prefix = hidden_states[:, :, :1, :, :].repeat((1, 1, cache_len, 1, 1))

            hidden_states = torch.concatenate([prefix, hidden_states], dim=2)
            feat_cache[idx] = current_cache
            feat_idx[0] += 1
        else:
            # No cache: use causal padding
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, cache_len, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        
        return hidden_states
    
    def _apply_bidirectional_padding(self, hidden_states: torch.Tensor, time_kernel_size: int) -> torch.Tensor:
        """Apply symmetric padding for bidirectional mode."""
        pad_size = (time_kernel_size - 1) // 2
        pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_size, 1, 1))
        pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad_size, 1, 1))
        return torch.concatenate([pad_left, hidden_states, pad_right], dim=2)


# =============================================================================
# ResNet Block
# =============================================================================

class LTX2VideoResnetBlock3d(nn.Module):
    """A 3D ResNet block used in the LTX 2.0 audiovisual model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. If None, defaults to `in_channels`.
        dropout: Dropout rate.
        eps: Epsilon value for normalization layers.
        elementwise_affine: Whether to enable elementwise affinity in normalization.
        non_linearity: Activation function to use.
        conv_shortcut: Whether to use a convolution shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = PerChannelRMSNorm()
        self.conv1 = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm2 = PerChannelRMSNorm()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTX2VideoCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            # LTX 2.0 uses a normal nn.Conv3d here rather than LTXVideoCausalConv3d
            self.conv_shortcut = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass through ResNet block."""
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


# =============================================================================
# Downsampler and Upsampler
# =============================================================================

class LTXVideoDownsampler3d(nn.Module):
    """3D downsampling layer for spatiotemporal reduction.
    
    Uses pixel unshuffle pattern for downsampling with causal temporal handling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (in_channels * stride[0] * stride[1] * stride[2]) // out_channels

        out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])

        self.conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with temporal grouping state management."""
        input_shape = _shape_of(hidden_states)
        prefix_len = max(self.stride[0] - 1, 0)
        
        if prefix_len > 0 and feat_cache is not None and feat_idx is not None:
            hidden_states = self._manage_temporal_grouping(hidden_states, feat_cache, feat_idx, prefix_len)
        else:
            hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)

        # Handle empty temporal dimension after grouping
        if hidden_states.shape[2] == 0:
            return self._handle_empty_output(hidden_states, feat_cache, feat_idx)

        # Apply pixel unshuffle downsampling
        residual = self._compute_residual(hidden_states)
        
        hidden_states = self.conv(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
        hidden_states = self._unshuffle_output(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
    
    def _manage_temporal_grouping(
        self,
        hidden_states: torch.Tensor,
        feat_cache: list,
        feat_idx: list,
        prefix_len: int
    ) -> torch.Tensor:
        """Manage temporal grouping state for chunked processing."""
        idx = feat_idx[0]
        while len(feat_cache) <= idx:
            feat_cache.append(None)

        state = feat_cache[idx]
        if state is None or not isinstance(state, dict):
            state = {"prefixed": False, "pending": None}

        sequence = hidden_states
        if not state["prefixed"]:
            prefix = hidden_states[:, :, :1, :, :].repeat((1, 1, prefix_len, 1, 1))
            sequence = torch.cat([prefix, sequence], dim=2)
            state["prefixed"] = True

        if state["pending"] is not None:
            pending = state["pending"].to(sequence.device)
            sequence = torch.cat([pending, sequence], dim=2)

        stride_t = self.stride[0]
        usable_t = (sequence.shape[2] // stride_t) * stride_t
        state["pending"] = sequence[:, :, usable_t:, :, :].clone() if usable_t < sequence.shape[2] else None
        hidden_states = sequence[:, :, :usable_t, :, :]
        feat_cache[idx] = state
        feat_idx[0] += 1
        
        return hidden_states
    
    def _handle_empty_output(
        self,
        hidden_states: torch.Tensor,
        feat_cache: Optional[list],
        feat_idx: Optional[list]
    ) -> torch.Tensor:
        """Handle empty output when temporal dimension is 0."""
        if feat_cache is not None and feat_idx is not None:
            # Reserve cache slot for conv
            idx = feat_idx[0]
            while len(feat_cache) <= idx:
                feat_cache.append(None)
            feat_idx[0] += 1
            
        out_channels = self.conv.out_channels * self.stride[0] * self.stride[1] * self.stride[2]
        out_h = hidden_states.shape[3] // self.stride[1]
        out_w = hidden_states.shape[4] // self.stride[2]
        return hidden_states.new_empty(
            hidden_states.shape[0],
            out_channels,
            0,
            out_h,
            out_w,
        )
    
    def _compute_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute residual connection for pixel unshuffle."""
        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        return residual.mean(dim=2)
    
    def _unshuffle_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply pixel unshuffle to output."""
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return hidden_states


class LTXVideoUpsampler3d(nn.Module):
    """3D upsampling layer for spatiotemporal expansion.
    
    Uses pixel shuffle pattern for upsampling with causal temporal handling.
    """
    
    def __init__(
        self,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor

        out_channels = (in_channels * stride[0] * stride[1] * stride[2]) // upscale_factor

        self.conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with trim state management."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        input_shape = _shape_of(hidden_states)
        trim_t = max(self.stride[0] - 1, 0)
        
        # Determine trim amount based on cache state
        if trim_t > 0 and feat_cache is not None and feat_idx is not None:
            trim_start = self._manage_trim_state(feat_cache, feat_idx, trim_t)
        else:
            trim_start = trim_t

        # Compute residual if enabled
        if self.residual:
            residual = self._compute_residual(hidden_states, trim_start)

        hidden_states = self.conv(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)
        hidden_states = self._shuffle_output(hidden_states, trim_start)

        if self.residual:
            hidden_states = hidden_states + residual

        return hidden_states
    
    def _manage_trim_state(self, feat_cache: list, feat_idx: list, trim_t: int) -> int:
        """Manage trim state for removing temporal padding."""
        idx = feat_idx[0]
        while len(feat_cache) <= idx:
            feat_cache.append(None)
        state = feat_cache[idx]
        if state is None or not isinstance(state, dict):
            state = {"trim_applied": False}
        trim_start = trim_t if not state["trim_applied"] else 0
        state["trim_applied"] = True
        feat_cache[idx] = state
        feat_idx[0] += 1
        return trim_start
    
    def _compute_residual(self, hidden_states: torch.Tensor, trim_start: int) -> torch.Tensor:
        """Compute residual connection."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        residual = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
        residual = residual.repeat(1, repeats, 1, 1, 1)
        return residual[:, :, trim_start:]
    
    def _shuffle_output(self, hidden_states: torch.Tensor, trim_start: int) -> torch.Tensor:
        """Apply pixel shuffle and trim to output."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states[:, :, trim_start:]




# =============================================================================
# Encoder/Decoder Blocks
# =============================================================================

class LTX2VideoDownBlock3D(nn.Module):
    """Down block used in the LTXVideo model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. If None, defaults to `in_channels`.
        num_layers: Number of resnet layers.
        dropout: Dropout rate.
        resnet_eps: Epsilon value for normalization layers.
        resnet_act_fn: Activation function to use.
        spatio_temporal_scale: Whether to use downsampling layer.
        downsample_type: Type of downsampling ("conv", "spatial", "temporal", "spatiotemporal")
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        downsample_type: str = "conv",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()

            if downsample_type == "conv":
                self.downsamplers.append(
                    LTX2VideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(1, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 1, 1),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator, causal)
            else:
                hidden_states = resnet(
                    hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        return hidden_states


class LTX2VideoMidBlock3d(nn.Module):
    """Middle block used in the LTXVideo model.

    Args:
        in_channels: Number of input channels.
        num_layers: Number of resnet layers.
        dropout: Dropout rate.
        resnet_eps: Epsilon value for normalization layers.
        resnet_act_fn: Activation function to use.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator, causal)
            else:
                hidden_states = resnet(
                    hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

        return hidden_states


class LTX2VideoUpBlock3d(nn.Module):
    """Up block used in the LTXVideo model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. If None, defaults to `in_channels`.
        num_layers: Number of resnet layers.
        dropout: Dropout rate.
        resnet_eps: Epsilon value for normalization layers.
        resnet_act_fn: Activation function to use.
        spatio_temporal_scale: Whether to use upsampling layer.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        upsample_residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
        upsample_stride: tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = LTX2VideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList(
                [
                    LTXVideoUpsampler3d(
                        out_channels * upscale_factor,
                        stride=upsample_stride,
                        residual=upsample_residual,
                        upscale_factor=upscale_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(
                hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
            )

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator, causal)
            else:
                hidden_states = resnet(
                    hidden_states, temb, generator, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx
                )

        return hidden_states


# =============================================================================
# Encoder
# =============================================================================

class LTX2VideoEncoder3d(nn.Module):
    """Encoder layer for encoding video samples to latent representation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of latent channels.
        block_out_channels: Number of output channels for each block.
        spatio_temporal_scaling: Whether each block should contain downscaling.
        layers_per_block: Number of layers per block.
        downsample_type: Downsampling pattern per block.
        patch_size: Size of spatial patches.
        patch_size_t: Size of temporal patches.
        resnet_norm_eps: Epsilon for ResNet normalization.
        is_causal: Whether to use causal behavior.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (256, 512, 1024, 2048),
        down_block_types: Tuple[str, ...] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, True),
        layers_per_block: Tuple[int, ...] = (4, 6, 6, 2, 2),
        downsample_type: Tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2 * patch_size_t
        self.is_causal = is_causal

        output_channel = out_channels

        self.conv_in = LTX2VideoCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # down blocks
        num_block_out_channels = len(block_out_channels)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if down_block_types[i] == "LTX2VideoDownBlock3D":
                down_block = LTX2VideoDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    downsample_type=downsample_type[i],
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_types[i]}")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = LTX2VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )

        # out
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = LTX2VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: Optional[bool] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        causal = causal or self.is_causal
        if feat_cache is not None and feat_idx is None:
            feat_idx = [0]

        hidden_states = hidden_states.reshape(
            batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(down_block, hidden_states, None, None, causal)

            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states, None, None, causal)
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

            hidden_states = self.mid_block(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        return hidden_states


# =============================================================================
# Decoder
# =============================================================================

class LTX2VideoDecoder3d(nn.Module):
    """Decoder layer for decoding latent representation to video.

    Args:
        in_channels: Number of latent channels.
        out_channels: Number of output channels.
        block_out_channels: Number of output channels for each block.
        spatio_temporal_scaling: Whether each block should contain upscaling.
        layers_per_block: Number of layers per block.
        patch_size: Size of spatial patches.
        patch_size_t: Size of temporal patches.
        resnet_norm_eps: Epsilon for ResNet normalization.
        is_causal: Whether to use causal behavior.
    """

    _UPSAMPLE_STRIDE_MAP = {
        "spatial": (1, 2, 2),
        "temporal": (2, 1, 1),
        "spatiotemporal": (2, 2, 2),
    }

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (256, 512, 1024),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True),
        layers_per_block: Tuple[int, ...] = (5, 5, 5, 5),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: Tuple[bool, ...] = (False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: Tuple[bool, ...] = (True, True, True),
        upsample_factor: Tuple[bool, ...] = (2, 2, 2),
        upsample_type: Tuple[str, ...] | None = None,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size**2
        self.is_causal = is_causal

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        if upsample_type is not None:
            upsample_type = tuple(reversed(upsample_type))
        output_channel = block_out_channels[0]

        self.conv_in = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.mid_block = LTX2VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            stride = (2, 2, 2)
            if upsample_type is not None:
                stride = self._UPSAMPLE_STRIDE_MAP[upsample_type[i]]

            up_block = LTX2VideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                spatial_padding_mode=spatial_padding_mode,
                upsample_stride=stride,
            )

            self.up_blocks.append(up_block)

        # out
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = LTX2VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # timestep embedding
        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        causal = causal or self.is_causal
        input_shape = _shape_of(hidden_states)
        
        if feat_cache is not None and feat_idx is None:
            feat_idx = [0]

        hidden_states = self.conv_in(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier

        if torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None:
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states, temb, None, causal)

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(up_block, hidden_states, temb, None, causal)
        else:
            hidden_states = self.mid_block(hidden_states, temb, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        hidden_states = self.norm_out(hidden_states)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal, feat_cache=feat_cache, feat_idx=feat_idx)

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return hidden_states



# =============================================================================
# Main VAE Model
# =============================================================================

class AutoencoderKLCausalLTX2Video(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin):
    """A VAE model with KL loss for encoding images into latents and decoding latent representations into images.
    
    Used in LTX-2. Supports causal encoding/decoding with streaming cache for memory-efficient
    processing of long videos.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        latent_channels: Number of latent channels.
        block_out_channels: Number of output channels for each encoder block.
        decoder_block_out_channels: Number of output channels for each decoder block.
        layers_per_block: Number of layers per encoder block.
        decoder_layers_per_block: Number of layers per decoder block.
        spatio_temporal_scaling: Whether each encoder block should downscale.
        decoder_spatio_temporal_scaling: Whether each decoder block should upscale.
        downsample_type: Downsampling pattern for encoder.
        upsample_residual: Whether to use residual in decoder upsampling.
        upsample_factor: Upsampling factor for each decoder block.
        timestep_conditioning: Whether to condition on timesteps.
        patch_size: Size of spatial patches.
        patch_size_t: Size of temporal patches.
        resnet_norm_eps: Epsilon for ResNet normalization.
        scaling_factor: Factor to scale latents.
        encoder_causal: Whether encoder should be causal.
        decoder_causal: Whether decoder should be causal.
        gradient_checkpointing: Whether to use gradient checkpointing.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (256, 512, 1024, 2048),
        down_block_types: Tuple[str, ...] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (256, 512, 1024),
        layers_per_block: Tuple[int, ...] = (4, 6, 6, 2, 2),
        decoder_layers_per_block: Tuple[int, ...] = (5, 5, 5, 5),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, True),
        decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True),
        decoder_inject_noise: Tuple[bool, ...] = (False, False, False, False),
        downsample_type: Tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        upsample_residual: Tuple[bool, ...] = (True, True, True),
        upsample_factor: Tuple[int, ...] = (2, 2, 2),
        decoder_upsample_type: Tuple[str, ...] | None = None,
        timestep_conditioning: bool = False,
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        scaling_factor: float = 1.0,
        encoder_causal: bool = True,
        decoder_causal: bool = True,
        gradient_checkpointing: bool = False,
        encoder_spatial_padding_mode: str = "zeros",
        decoder_spatial_padding_mode: str = "reflect",
        spatial_compression_ratio: int = None,
        temporal_compression_ratio: int = None,
    ) -> None:
        super().__init__()
        self.encoder = LTX2VideoEncoder3d(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=encoder_causal,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )
        self.decoder = LTX2VideoDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal,
            timestep_conditioning=timestep_conditioning,
            inject_noise=decoder_inject_noise,
            upsample_residual=upsample_residual,
            upsample_factor=upsample_factor,
            upsample_type=decoder_upsample_type,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

        self.spatial_compression_ratio = (
            patch_size * 2 ** sum(spatio_temporal_scaling)
            if spatial_compression_ratio is None
            else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_size_t * 2 ** sum(spatio_temporal_scaling)
            if temporal_compression_ratio is None
            else temporal_compression_ratio
        )

        # Memory optimization flags
        self.use_slicing = False
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # Tiling configuration
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

        # Cache managers for streaming inference
        self._decoder_cache = DecoderCacheManager()

        # Expose config-driven gradient checkpointing toggle for training
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        """Enable tiled VAE decoding for memory-efficient processing of large videos."""
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def _encode(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        """Internal encode method."""
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_framewise_decoding and num_frames > self.tile_sample_min_num_frames:
            return self._temporal_tiled_encode(x, causal=causal)

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x, causal=causal)

        enc = self.encoder(x, causal=causal)
        return enc

    # -------------------------------------------------------------------------
    # Cache Management (Legacy API for backward compatibility)
    # -------------------------------------------------------------------------
    def clear_decoder_cache(self) -> None:
        """Clear decoder cache (legacy API, delegates to cache manager)."""
        self._decoder_cache.clear()

    # -------------------------------------------------------------------------
    # Streaming Encode/Decode Methods
    # -------------------------------------------------------------------------
    def decode_with_cache(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        return_dict: bool = True,
        reset_cache: bool = False,
        prepend_prev_latent_frames: int = 1,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """Decode one latent chunk with persistent decoder feature cache.
        
        For decoder_causal=False model: decodes each chunk independently with:
        - Left context from previous chunk
        - Zero-appended right context placeholder
        
        Args:
            z: Input latent tensor [B, C, T, H, W]
            temb: Optional timestep embedding
            causal: Whether to use causal decoding
            return_dict: Whether to return dict or tuple
            reset_cache: Whether to reset cache before decoding
            prepend_prev_latent_frames: Number of previous frames to prepend as context
            
        Returns:
            DecoderOutput containing decoded video
        """
        if reset_cache:
            self._decoder_cache.clear()

        if self.use_slicing and z.shape[0] > 1:
            raise ValueError("decode_with_cache does not support batch slicing; pass batch size 1 or disable slicing.")
        if z.shape[2] <= 0:
            raise ValueError("Input latent chunk must contain at least 1 frame.")
        if prepend_prev_latent_frames < 0:
            raise ValueError(f"`prepend_prev_latent_frames` must be >= 0, got {prepend_prev_latent_frames}.")

        causal = self.decoder.is_causal if causal is None else causal
        self._decoder_cache.validate_mode(causal)

        ratio = self.temporal_compression_ratio
        z_chunk = z
        
        if not causal:
            z_chunk = self._decoder_cache.prepend_context(
                z, prepend_prev_latent_frames, ratio
            )

        decoded_full = self.decoder(
            z_chunk,
            temb,
            causal=causal,
            feat_cache=self._decoder_cache.feat_map,
            feat_idx=[0],
        )
        
        if not causal:
            decoded = self._decoder_cache.trim_output(
                decoded_full, prepend_prev_latent_frames, z.shape[2], ratio
            )
            self._decoder_cache.update_tail(z, prepend_prev_latent_frames)
        else:
            decoded = decoded_full

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    @apply_forward_hook
    def decode_per_frame_with_cache(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        reset_cache: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Stream-decode a latent video one latent frame at a time, yielding pixel chunks.

        Each iteration decodes a single latent frame and relies on the per-layer
        feature cache for temporal consistency. Yields the decoded pixel tensor
        for each latent frame so the caller can process/save/display each chunk
        immediately (low peak memory, low latency, no final ``torch.cat``).

        Streaming use: pass ``reset_cache=True`` on the first call of a new
        stream/session, then ``reset_cache=False`` on subsequent calls so the
        per-layer feature cache carries temporal state across calls.

        Args:
            z: Input latent tensor [B, C, T, H, W].
            temb: Optional timestep embedding.
            causal: Whether to use causal decoding.
            reset_cache: If True, clear the decoder feature cache before decoding.
                Set True only on the first chunk of a new stream.

        Yields:
            Decoded pixel tensor [B, C, T_out, H, W] for each single-latent-frame chunk.
        """
        if self.use_slicing and z.shape[0] > 1:
            raise ValueError(
                "decode_per_frame_with_cache does not support batch slicing; pass batch size 1 or disable slicing."
            )
        if z.shape[2] <= 0:
            raise ValueError("Input latent video must contain at least 1 frame.")

        causal = self.decoder.is_causal if causal is None else causal
        num_latent_frames = z.shape[2]

        if reset_cache:
            self._decoder_cache.clear()
        self._decoder_cache.validate_mode(causal)

        for t in range(num_latent_frames):
            z_chunk = z[:, :, t : t + 1, :, :]
            decoded_chunk = self.decoder(
                z_chunk,
                temb,
                causal=causal,
                feat_cache=self._decoder_cache.feat_map,
                feat_idx=[0],
            )
            yield decoded_chunk

    def encode(
        self, x: torch.Tensor, causal: Optional[bool] = None, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """Encode a batch of images into latents.

        Args:
            x: Input batch of images [B, C, T, H, W].
            causal: Whether to use causal encoding.
            return_dict: Whether to return a dict or tuple.

        Returns:
            The latent representations of the encoded videos.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice, causal=causal) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x, causal=causal)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Internal decode method."""
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio

        if self.use_framewise_decoding and num_frames > tile_latent_min_num_frames:
            return self._temporal_tiled_decode(z, temb, causal=causal, return_dict=return_dict)

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, temb, causal=causal, return_dict=return_dict)

        dec = self.decoder(z, temb, causal=causal)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Decode a batch of latents into images.

        Args:
            z: Input batch of latent vectors [B, C, T, H, W].
            temb: Optional timestep embedding.
            causal: Whether to use causal decoding.
            return_dict: Whether to return a dict or tuple.

        Returns:
            DecoderOutput containing decoded video.
        """
        if self.use_slicing and z.shape[0] > 1:
            if temb is not None:
                decoded_slices = [
                    self._decode(z_slice, t_slice, causal=causal).sample
                    for z_slice, t_slice in (z.split(1), temb.split(1))
                ]
            else:
                decoded_slices = [self._decode(z_slice, causal=causal).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, temb, causal=causal).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    # -------------------------------------------------------------------------
    # Tiling Methods
    # -------------------------------------------------------------------------

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend vertically between two tiles."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend horizontally between two tiles."""
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend temporally between two tiles."""
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    @staticmethod
    def _get_tile_positions(total_size: int, tile_size: int, overlap: int) -> list:
        """Compute tile start positions ensuring full coverage with given overlap.

        Returns a list of start positions such that every pixel in [0, total_size)
        is covered by at least one tile of size ``tile_size``.
        """
        if total_size <= tile_size:
            return [0]
        stride = tile_size - overlap
        positions = list(range(0, total_size - tile_size, stride))
        last = total_size - tile_size
        if not positions or positions[-1] < last:
            positions.append(last)
        return positions

    @staticmethod
    def _make_blend_mask(
        h: int,
        w: int,
        overlap_h: int,
        overlap_w: int,
        is_top: bool,
        is_bottom: bool,
        is_left: bool,
        is_right: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a 2D spatial blend mask with linear ramps in overlap regions.

        Interior pixels get weight 1.  Pixels in the overlap band of a non-edge
        side get linearly increasing/decreasing weights so that the weighted
        average of overlapping tiles transitions smoothly.

        Returns:
            Tensor of shape ``[h, w]`` with values in (0, 1].
        """
        mask_h = torch.ones(h, device=device, dtype=dtype)
        mask_w = torch.ones(w, device=device, dtype=dtype)

        if not is_top and overlap_h > 0:
            mask_h[:overlap_h] = torch.linspace(0, 1, overlap_h + 2, device=device, dtype=dtype)[1:-1]
        if not is_bottom and overlap_h > 0:
            mask_h[-overlap_h:] = torch.linspace(1, 0, overlap_h + 2, device=device, dtype=dtype)[1:-1]
        if not is_left and overlap_w > 0:
            mask_w[:overlap_w] = torch.linspace(0, 1, overlap_w + 2, device=device, dtype=dtype)[1:-1]
        if not is_right and overlap_w > 0:
            mask_w[-overlap_w:] = torch.linspace(1, 0, overlap_w + 2, device=device, dtype=dtype)[1:-1]

        return mask_h[:, None] * mask_w[None, :]

    def tiled_encode(
        self, x: torch.Tensor, causal: bool | None = None, tile_caches: dict | None = None,
    ) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x: Input batch of videos [B, C, T, H, W].
            causal: Whether to use causal encoding.
            tile_caches: Optional dict mapping ``(row_idx, col_idx)`` to a
                per-tile ``feat_cache`` list.  When provided each tile's
                encoder call receives its own persistent cache so temporal
                context is maintained across successive calls.

        Returns:
            The latent representation of the encoded videos.
        """
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i_idx, i in enumerate(range(0, height, self.tile_sample_stride_height)):
            row = []
            for j_idx, j in enumerate(range(0, width, self.tile_sample_stride_width)):
                tile = x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                if tile_caches is not None:
                    enc_tile = self.encoder(
                        tile, causal=causal,
                        feat_cache=tile_caches[(i_idx, j_idx)], feat_idx=[0],
                    )
                else:
                    enc_tile = self.encoder(tile, causal=causal)
                row.append(enc_tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self, z: torch.Tensor, temb: torch.Tensor | None, causal: bool | None = None,
        return_dict: bool = True, tile_caches: dict | None = None,
    ) -> DecoderOutput | torch.Tensor:
        r"""Decode a batch of images using a tiled decoder.

        Args:
            z: Input batch of latent vectors [B, C, T, H, W].
            temb: Optional timestep embedding.
            causal: Whether to use causal decoding.
            return_dict: Whether to return a DecoderOutput instead of a plain tuple.
            tile_caches: Optional dict mapping ``(row_idx, col_idx)`` to a
                per-tile ``feat_cache`` list.  When provided each tile's
                decoder call receives its own persistent cache so temporal
                context is maintained across successive calls.

        Returns:
            DecoderOutput or tuple containing decoded video.
        """

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i_idx, i in enumerate(range(0, height, tile_latent_stride_height)):
            row = []
            for j_idx, j in enumerate(range(0, width, tile_latent_stride_width)):
                tile = z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                if tile_caches is not None:
                    dec_tile = self.decoder(
                        tile, temb, causal=causal,
                        feat_cache=tile_caches[(i_idx, j_idx)], feat_idx=[0],
                    )
                else:
                    dec_tile = self.decoder(tile, temb, causal=causal)
                row.append(dec_tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        """Encode with temporal tiling for long videos."""
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (height > self.tile_sample_min_height or width > self.tile_sample_min_width):
                tile = self.tiled_encode(tile, causal=causal)
            else:
                tile = self.encoder(tile, causal=causal)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor], causal: Optional[bool] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Decode with temporal tiling for long videos."""
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (tile.shape[-1] > tile_latent_min_width or tile.shape[-2] > tile_latent_min_height):
                decoded = self.tiled_decode(tile, temb, causal=causal, return_dict=True).sample
            else:
                decoded = self.decoder(tile, temb, causal=causal)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                tile = tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                result_row.append(tile)
            else:
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :])

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        encoder_causal: Optional[bool] = None,
        decoder_causal: Optional[bool] = None,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.
        
        Args:
            sample: Input video tensor [B, C, T, H, W]
            temb: Optional timestep embedding
            sample_posterior: Whether to sample from posterior or use mode
            encoder_causal: Whether to use causal encoding
            decoder_causal: Whether to use causal decoding
            return_dict: Whether to return dict
            generator: Optional random generator
            
        Returns:
            Decoded video
        """
        x = sample
        posterior = self.encode(x, causal=encoder_causal).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, temb, causal=decoder_causal)
        if not return_dict:
            return (dec.sample,)
        return dec
