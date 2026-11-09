"""The packed Ref2VA sequence and its per-row timesteps (Diffusers MiniMax-H3 layout and timestep rules).

Sequence: [Qwen context (draft vision tokens, then text) | reference video rows (the draft latent) | audio rows (fixed
encoded silence) | target video rows].  At target noise sigma the target rows get t = 1 - sigma, the reference rows
t = max(1 - sigma, 0.999) (the native 0.001 conditioning noise; their input is 0.999 * draft + 0.001 * noise) and the
audio rows t = 1 (clean silence).  H3 predicts the data-ward velocity: x_next = x + (sigma - sigma_next) * v.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)

PATCH = (1, 2, 2)
REFERENCE_T = 0.999


@dataclass(frozen=True)
class PackedLayout:
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0

    def to(self, device):
        return PackedLayout(self.position_ids.to(device), self.token_tags.to(device), self.video_indices.to(device),
                            self.audio_indices.to(device), self.text_indices.to(device),
                            self.num_condition_video_rows, self.num_condition_audio_rows)


def canonical_sigma(sigma):
    """sigma as the float32 value the layout and the row checks use."""
    value = float(torch.as_tensor(sigma, dtype=torch.float32).item())
    assert 0 < value <= 1
    return value


def row_timesteps(layout, sigma):
    """(distinct timesteps, per-row index) for one transformer call at target noise `sigma`."""
    t = 1. - canonical_sigma(sigma)
    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices=layout.video_indices.cpu(), audio_indices=layout.audio_indices.cpu(),
        num_condition_video_rows=layout.num_condition_video_rows,
        num_condition_audio_rows=layout.num_condition_audio_rows, num_text_tokens=int(layout.text_indices.numel()),
        video_timestep=t, audio_timestep=1., condition_video_timestep=max(t, REFERENCE_T), condition_audio_timestep=1.)
    return unique.to(layout.position_ids.device), inverse.to(layout.position_ids.device)


def pack_audio_latents(audio_latents):
    """[B, 2, C, A] stereo latents -> channel-major rows [B, 2 * A, C]."""
    b, ch, c, a = audio_latents.shape
    assert ch == 2
    return audio_latents.permute(0, 1, 3, 2).reshape(b, ch * a, c).contiguous()


def patch_rows(latent):
    """[24, T, H, W] -> [1, T * H/2 * W/2, 96]."""
    return patchify_video_latents(latent[None], PATCH)[None]


def unpatch_rows(rows, shape):
    """Inverse of patch_rows for a latent of `shape` [C, T, H, W]."""
    c, t, h, w = shape
    return rows[0].reshape(t, h // 2, w // 2, c, 1, 2, 2).permute(3, 0, 4, 1, 5, 2, 6).reshape(c, t, h, w)


def pack(reference, target_shape, prompt, audio, device):
    """reference: draft latent [24, T, h, w]; target_shape [24, T, H, W]; prompt: encode_prompt() output;
    audio: silence latents [2, 32, A].  -> reference rows, audio rows, prompt embeds, layout, #reference rows."""
    tags = prompt['text_token_tags']
    assert (tags == 0).any(), 'missing the draft vision context'
    _, frames, height, width = target_shape
    assert reference.shape[1] == frames, 'draft and target must share latent frames'
    values = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        tags, [SimpleNamespace(kind='video', has_audio=False)], [reference[None]], [],
        frames, height, width, audio.shape[-1], PATCH, 2, 2, 0)
    layout = PackedLayout(*values).to(device)
    ref = patch_rows(reference).to(device)
    n = layout.num_condition_video_rows
    assert n == ref.shape[1] and layout.video_indices.numel() == n + frames * (height // 2) * (width // 2)
    assert layout.num_condition_audio_rows == 0
    return ref, pack_audio_latents(audio[None]).to(device), prompt['prompt_embeds'][None].to(device), layout, n
