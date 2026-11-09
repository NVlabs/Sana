"""HyperFlow 8-step adapter (videorebirth/hyperflow, minimax_h3_hyperflow_8step_v1.0) merged into the Ref2VA base
weights, so the IC LoRA can sit on top of it.

Follows the official package (Video-Rebirth/hyperflow 1dd2f34, src/hyperflow_h3, Apache-2.0), which needs PEFT
and its own pipeline blocks; here only the model-side pieces are reproduced:
  * TwoTimeEmbedder: time embedding of the interval (t, r) = emb_t(t) + gate * (emb_r(r) - emb_t(t)); `endpoint`
    is a copy of `time_embedder` taken before any LoRA, with its own LoRA (`endpoint_time_embedder.*` keys);
  * LoRA keys `transformer.<module>.lora_{A,B}.weight`, delta = (lora_alpha / rank) * B @ A, merged in fp32;
  * the 9-point raw sigma grid of the file header, shifted with the video shift (12);
  * endpoint of step i is r = 1 - sigma_{i+1} for generated rows; conditioning / audio rows keep r = t.
"""
import copy
import json
import re

import torch
from safetensors import safe_open
from torch import nn

KEY = re.compile(r'^transformer\.(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$')


class TwoTimeEmbedder(nn.Module):
    def __init__(self, base, endpoint, time_proj, gate):
        super().__init__()
        self.base, self.endpoint, self.time_proj, self.gate = base, endpoint, time_proj, float(gate)
        self.endpoint_timesteps = None

    # The official transformer forward reads `time_embedder.linear_1.weight.dtype`.
    @property
    def linear_1(self):
        return self.base.linear_1

    @property
    def linear_2(self):
        return self.base.linear_2

    def forward(self, sample):
        t_emb = self.base(sample)
        if self.endpoint_timesteps is None:
            raise RuntimeError('HyperFlow endpoint timesteps not set for this forward')
        endpoint = self.endpoint_timesteps.to(sample.device)
        assert endpoint.shape[0] == sample.shape[0], (endpoint.shape, sample.shape)
        r_emb = self.endpoint(self.time_proj(endpoint).to(sample.dtype))
        return t_emb + self.gate * (r_emb - t_emb)


def _module_path(public):
    if public.startswith('endpoint_time_embedder.'):
        return 'time_embedder.endpoint.' + public[len('endpoint_time_embedder.'):]
    if public.startswith('time_embedder.'):
        return 'time_embedder.base.' + public[len('time_embedder.'):]
    return public


def install(model, path, device='cpu'):
    """Wrap `model.time_embedder` and merge every LoRA of the file into its target Linear (math on `device`,
    weights stay where they are).  Returns the header."""
    with safe_open(str(path), framework='pt', device='cpu') as f:
        meta = f.metadata()
        assert meta.get('hyperflow') == 'true', 'not a HyperFlow weights file'
        plan = {}
        for key in f.keys():
            m = KEY.match(key)
            assert m, key
            plan.setdefault(_module_path(m['module']), {})[m['matrix']] = key
        assert all(set(v) == {'A', 'B'} for v in plan.values())
        assert not isinstance(model.time_embedder, TwoTimeEmbedder)
        model.time_embedder = TwoTimeEmbedder(model.time_embedder, copy.deepcopy(model.time_embedder),
                                              copy.deepcopy(model.time_proj), float(meta['hyperflow_gate']))
        modules = dict(model.named_modules())
        scale = None
        with torch.no_grad():
            for path_, mats in plan.items():
                linear = modules.get(path_)
                assert isinstance(linear, nn.Linear), path_
                a, b = f.get_tensor(mats['A']).to(device).float(), f.get_tensor(mats['B']).to(device).float()
                rank = a.shape[0]
                assert rank == int(meta['lora_rank']) and b.shape == (linear.out_features, rank)
                scale = float(meta['lora_alpha']) / rank
                w = linear.weight
                w.copy_((w.to(device).float() + scale * (b @ a)).to(w.dtype).to(w.device))
    meta = dict(meta, merged_modules=len(plan), scale=scale)
    return meta


def sigmas(meta, shift):
    raw = torch.tensor(json.loads(meta['hyperflow_sigmas']), dtype=torch.float32)
    assert float(meta['hyperflow_video_shift']) == shift
    return [float(s) for s in shift * raw / (1. + (shift - 1.) * raw)]


def endpoints(times, sigma, next_sigma):
    """Per distinct timestep: generated rows (t == 1 - sigma) aim at 1 - next_sigma, all others keep r = t."""
    t = torch.as_tensor(times, dtype=torch.float32)
    gen = torch.isclose(t, torch.tensor(1. - sigma, dtype=torch.float32), atol=1e-6, rtol=0.)
    assert int(gen.sum()) == 1, (t.tolist(), sigma)
    r = t.clone()
    r[gen] = 1. - next_sigma
    return r
