"""H3 packed-sequence CP using SoL's differentiable Ulysses collectives.

Pad only the packed model layout for equal partition sizes; remove padding from
Q/K/V before attention, then restore it before the reverse all-to-all. Padding
therefore never participates in softmax, nor in text-token refinement.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor,_apply_rotary_emb
from .geometry import build_row_timestep_plan
from .collectives import seq_to_head,head_to_seq
import os
ATTENTION_BACKEND=os.environ.get('H3_ATTN_BACKEND','_flash_3_hub')

def attend(q,k,v,valid_rows,group,backend=None):
    backend=backend or ATTENTION_BACKEND
    q,k,v=(seq_to_head(x,group) for x in (q,k,v))
    padded_rows=q.shape[1]
    out=dispatch_attention_fn(q[:,:valid_rows].contiguous(),k[:,:valid_rows].contiguous(),v[:,:valid_rows].contiguous(),dropout_p=0.0,is_causal=False,backend=backend,parallel_config=None)
    if padded_rows>valid_rows:out=F.pad(out,(0,0,0,0,0,padded_rows-valid_rows))
    return head_to_seq(out,group)

class H3SoLCPProcessor(MiniMaxH3AttnProcessor):
    def __init__(self,group):
        self.group=group;self.valid_rows=None;self._attention_backend=ATTENTION_BACKEND;self._parallel_config=None
    def __call__(self,attn,hidden_states,rotary_emb=None,attention_mask=None):
        assert attention_mask is None and self.valid_rows is not None
        q,k,v=(projection(hidden_states).unflatten(-1,(attn.heads,-1)) for projection in (attn.to_q,attn.to_k,attn.to_v))
        q,k=attn.norm_q(q),attn.norm_k(k)
        if rotary_emb is not None:q,k=_apply_rotary_emb(q,*rotary_emb),_apply_rotary_emb(k,*rotary_emb)
        out=attend(q,k,v,self.valid_rows,self.group,self._attention_backend).flatten(2,3).type_as(q)
        return attn.to_out[1](attn.to_out[0](out))

def forward_padded(model,video,audio,prompt,layout,sigma,cp):
    times,indices=build_row_timestep_plan(layout,video_sigma=sigma,audio_sigma=0.,keyframe_noise_aug=1.)
    length=layout.token_tags.numel();pad=(-length)%cp
    for block in model.module.transformer_blocks:block.attn.processor.valid_rows=length
    tags=F.pad(layout.token_tags,(0,pad));positions=F.pad(layout.position_ids,(0,0,0,pad));indices=F.pad(indices,(0,pad))
    return model(hidden_states=video,audio_hidden_states=audio,encoder_hidden_states=prompt,timestep=times,timestep_indices=indices,token_tags=tags,position_ids=positions,video_indices=layout.video_indices,audio_indices=layout.audio_indices,text_indices=layout.text_indices,attention_kwargs={},return_dict=False)

def verify_collectives(group,device):
    """Compare CP output and all Q/K/V gradients with full-sequence FA3."""
    size=dist.get_world_size(group);rank=dist.get_rank(group);evidence=[]
    for rows in (64,67):
        torch.manual_seed(200+rows)
        values=[torch.randn(1,rows,56,128,device=device,dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        reference=dispatch_attention_fn(*values,dropout_p=0.,is_causal=False,backend=ATTENTION_BACKEND,parallel_config=None)
        reference.float().square().sum().backward()
        pad=(-rows)%size;padded=rows+pad;local=padded//size
        shards=[F.pad(x.detach(),(0,0,0,0,0,pad)).chunk(size,dim=1)[rank].contiguous().requires_grad_() for x in values]
        output=attend(*shards,rows,group)
        output.float().square().sum().backward()
        expected=F.pad(reference.detach(),(0,0,0,0,0,pad)).chunk(size,dim=1)[rank]
        torch.testing.assert_close(output,expected,rtol=.02,atol=.02)
        errors=[]
        for full,shard in zip(values,shards):
            target=F.pad(full.grad,(0,0,0,0,0,pad)).chunk(size,dim=1)[rank]
            torch.testing.assert_close(shard.grad,target,rtol=.04,atol=.04)
            errors.append(float((shard.grad.float()-target.float()).norm()/(target.float().norm()+1e-10)))
        error=torch.tensor(max(errors),device=device);dist.all_reduce(error,op=dist.ReduceOp.MAX,group=group)
        assert error.item()<.02
        evidence.append(dict(rows=rows,padded_rows=padded,relative_gradient_error=error.item()))
    return evidence
