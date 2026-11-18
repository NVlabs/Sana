"""Ulysses context parallelism over the packed Ref2VA sequence, and the transformer call for one step.

With cp > 1 Diffusers' context-parallel plan shards the packed rows over the cp ranks (the sequence is padded to a
multiple of cp; padding is kept out of the attention softmax); each attention all-to-alls q/k/v from sequence shards
to head shards (56 heads, so cp in 2, 4, 7, 8), runs full-sequence attention on its heads and all-to-alls back.  The
token refiner runs before the sequence is sharded.  cp = 1 is the plain single-GPU forward.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models._modeling_parallel import ContextParallelConfig
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor, _apply_rotary_emb

from .layout import REFERENCE_T, canonical_sigma, row_timesteps

HEADS = 56


def _seq_to_head(x, group):
    """[B, N_local, H, D] -> [B, N_full, H / cp, D]"""
    cp = group.size()
    b, n, h, d = x.shape
    x = x.view(b, n, cp, h // cp, d).permute(2, 0, 1, 3, 4).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    return y.permute(1, 0, 2, 3, 4).contiguous().view(b, cp * n, h // cp, d)


def _head_to_seq(x, group):
    """[B, N_full, H / cp, D] -> [B, N_local, H, D]"""
    cp = group.size()
    b, n, h, d = x.shape
    x = x.view(b, cp, n // cp, h, d).permute(1, 0, 2, 3, 4).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    return y.permute(1, 2, 0, 3, 4).contiguous().view(b, n // cp, cp * h, d)


class Ref2VACPProcessor(MiniMaxH3AttnProcessor):
    def __init__(self, group, backend):
        self.group = group
        self.valid_rows = None
        self._attention_backend = backend
        self._parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        assert attention_mask is None and self.valid_rows is not None
        q, k, v = [projection(hidden_states).unflatten(-1, (attn.heads, -1))
                   for projection in (attn.to_q, attn.to_k, attn.to_v)]
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        q, k, v = [_seq_to_head(x, self.group) for x in (q, k, v)]
        padded = q.shape[1]
        out = dispatch_attention_fn(q[:, :self.valid_rows].contiguous(), k[:, :self.valid_rows].contiguous(),
                                    v[:, :self.valid_rows].contiguous(), dropout_p=0., is_causal=False,
                                    backend=self._attention_backend, parallel_config=None)
        if padded > self.valid_rows:
            out = F.pad(out, (0, 0, 0, 0, 0, padded - self.valid_rows))
        out = _head_to_seq(out, self.group).flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](out))


def enable_cp(model, cp, backend):
    """Shard the packed sequence of `model` over all `cp` ranks of the default process group."""
    from torch.distributed.device_mesh import init_device_mesh
    assert cp > 1 and HEADS % cp == 0 and dist.get_world_size() == cp, cp
    mesh = init_device_mesh('cuda', (1, 1, cp), mesh_dim_names=('data', 'ring', 'ulysses'))
    model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=cp, ring_degree=1,
                                                          mesh=mesh['ring', 'ulysses']))
    for module in model.token_refiner.modules():
        processor = getattr(module, 'processor', None)
        if processor is not None and hasattr(processor, '_parallel_config'):
            processor._parallel_config = None
    group = mesh['ulysses'].get_group()
    for block in model.transformer_blocks:
        block.attn.set_processor(Ref2VACPProcessor(group, backend))


def forward(model, video, audio, prompt, layout, sigma, cp):
    """Velocity rows for the packed [reference | target] video rows `video` at target noise `sigma`."""
    sigma = canonical_sigma(sigma)
    times, indices = row_timesteps(layout, sigma)
    rows = times[indices]
    n = layout.num_condition_video_rows
    assert torch.equal(rows[layout.video_indices[:n]], torch.full_like(rows[layout.video_indices[:n]],
                                                                       max(1 - sigma, REFERENCE_T)))
    assert torch.equal(rows[layout.video_indices[n:]], torch.full_like(rows[layout.video_indices[n:]], 1 - sigma))
    length = layout.token_tags.numel()
    pad = (-length) % cp
    if cp > 1:
        for block in model.transformer_blocks:
            block.attn.processor.valid_rows = length
    return model(hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=prompt,
                 timestep=times, timestep_indices=F.pad(indices, (0, pad)),
                 token_tags=F.pad(layout.token_tags, (0, pad)), position_ids=F.pad(layout.position_ids, (0, 0, 0, pad)),
                 video_indices=layout.video_indices, audio_indices=layout.audio_indices,
                 text_indices=layout.text_indices, attention_kwargs={}, return_dict=False)[0]
