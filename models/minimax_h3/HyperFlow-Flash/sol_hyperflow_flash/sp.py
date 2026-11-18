"""Minimal Ulysses sequence parallelism for the live DiT blocks (same wiring for every preset).

Parallelised: every live transformer block (AdaLN modulation, norms, the model's INT8 / low-rank linears, SwiGLU) on a contiguous
sequence shard of the packed [text | audio | video] rows (rank r owns rows [r*ceil(S/W), (r+1)*ceil(S/W))); attention is head-parallel
after an all-to-all (each rank: the FULL sequence for H/W heads) - so the attention back-end (dense cuDNN SDPA, Sol-Attn) sees exactly the full-sequence inputs of its heads; a second all-to-all returns the sequence shards. The last live block's
output is all-gathered. Replicated on every rank (not parallelised): text encoder, token refiner, input projections / time embedding,
norm_out / proj_out, the sampler update, video and audio VAE decode.

Collectives are injected (`a2a(recv_list, send_list)`, `allgather(out_tensor, in_tensor)`) so the data movement can be checked on CPU
by simulating all ranks (tests/test_cpu.py).
"""
from __future__ import annotations


class SP:
    def __init__(self, world: int, rank: int, a2a=None, allgather=None):
        self.W, self.r = int(world), int(rank)
        if a2a is None or allgather is None:
            import torch.distributed as dist

            a2a = a2a or dist.all_to_all
            allgather = allgather or dist.all_gather_into_tensor
        self.a2a, self.allgather = a2a, allgather
        self.S = None                     # full packed length of the current forward (set by the block pre-hook)

    def bounds(self, S: int):
        c = -(-S // self.W)
        return [(min(S, j * c), min(S, (j + 1) * c)) for j in range(self.W)]

    def sizes(self, S: int):
        return [hi - lo for lo, hi in self.bounds(S)]

    def shard(self, x, S: int, dim: int):
        lo, hi = self.bounds(S)[self.r]
        return x.narrow(dim, lo, hi - lo)

    # ---- sequence shard [1, S_r, H, E] <-> head shard [1, S, H/W, E] ------------------------------------------------------------
    def send_seq_to_head(self, x):
        hw = x.shape[2] // self.W
        return [x[0, :, j * hw:(j + 1) * hw].contiguous() for j in range(self.W)]

    def recv_seq_to_head(self, x, S: int):
        import torch

        hw = x.shape[2] // self.W
        return [torch.empty((n, hw, x.shape[3]), dtype=x.dtype, device=x.device) for n in self.sizes(S)]

    @staticmethod
    def assemble_seq_to_head(recv):
        import torch

        return torch.cat(recv, 0)[None]

    def send_head_to_seq(self, y, S: int):
        return [p.contiguous() for p in y[0].split(self.sizes(S), 0)]

    def recv_head_to_seq(self, y, S: int):
        import torch

        n = self.sizes(S)[self.r]
        return [torch.empty((n, y.shape[2], y.shape[3]), dtype=y.dtype, device=y.device) for _ in range(self.W)]

    @staticmethod
    def assemble_head_to_seq(recv):
        import torch

        return torch.cat(recv, 1)[None]

    def seq_to_head(self, x, S: int):
        if x.shape[2] % self.W:
            raise ValueError(f"heads {x.shape[2]} not divisible by world {self.W}")
        send, recv = self.send_seq_to_head(x), self.recv_seq_to_head(x, S)
        self.a2a(recv, send)
        return self.assemble_seq_to_head(recv)

    def head_to_seq(self, y, S: int):
        send, recv = self.send_head_to_seq(y, S), self.recv_head_to_seq(y, S)
        self.a2a(recv, send)
        return self.assemble_head_to_seq(recv)

    # ---- gather the sequence shards of [1, S_r, C] ------------------------------------------------------------------------------
    def gather(self, x, S: int):
        import torch
        import torch.nn.functional as F

        sz = self.sizes(S)
        c = max(sz)
        xp = x[0]
        if xp.shape[0] < c:
            xp = F.pad(xp, (0, 0, 0, c - xp.shape[0]))
        out = torch.empty((self.W * c, xp.shape[1]), dtype=x.dtype, device=x.device)
        self.allgather(out, xp.contiguous())
        out = out.view(self.W, c, -1)
        return torch.cat([out[j, :sz[j]] for j in range(self.W)], 0)[None]


def install_block_hooks(live_blocks, sp: SP):
    """Pre-hook on every live block: shard the hidden states once (first live block: they arrive full length), shard the per-row
    AdaLN indices and the rotary tables on every call; forward hook on the last live block: all-gather its output."""

    def pre(_mod, args):
        h, temb, ad, rot = args[0], args[1], args[2], args[3]
        S = int(ad.shape[0])
        sp.S = S
        if h.shape[1] == S:
            h = sp.shard(h, S, 1)
        rot_l = None if rot is None else (sp.shard(rot[0], S, 0), sp.shard(rot[1], S, 0))
        return (h, temb, sp.shard(ad, S, 0), rot_l) + tuple(args[4:])

    def post(_mod, _args, out):
        return sp.gather(out, sp.S)

    handles = [b.register_forward_pre_hook(pre) for b in live_blocks]
    handles.append(live_blocks[-1].register_forward_hook(post))
    return handles
