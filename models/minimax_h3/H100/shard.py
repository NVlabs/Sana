"""Per-layer weight sharding for GPUs that cannot hold MiniMax-H3 whole (inference-only, FSDP-style).

MiniMax-H3 carries a 33B DiT (66 GB bf16) and a Qwen3-VL-32B text encoder (67 GB); one 80 GB H100 holds
neither.  Each layer's parameters are flattened per dtype and split across the ranks, so a rank keeps 1/N
of every layer.  Right before a layer runs, an all-gather over NVLink rebuilds it into one of two
reusable buffers and the layer's parameters are pointed at views of that buffer; while layer i computes,
layer i+1 is gathered on a side stream and a separate process group, so the copy hides behind compute.
The math is untouched: every kernel reads the same weights it would read from a resident model.

    shards = LayerShards(blocks, group, device, exclude=lambda name: name.startswith("adaln_proj."))
    shards.before(i)            # call right before layer i runs (DiT: from the block loop;
                                # transformers decoders: install_pre_hooks)
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class LayerShards:
    def __init__(self, layers, group, device, exclude=lambda name: False):
        self.layers = list(layers)
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = torch.device(device)
        self.meta, self.shards = [], []
        sizes: dict = {}
        for layer in self.layers:
            by_dtype: dict = {}
            for name, p in layer.named_parameters():
                if exclude(name):
                    p.data = p.data.to(self.device)
                    continue
                by_dtype.setdefault(p.dtype, []).append(p)
            for b in layer.buffers():                               # small; keep resident
                b.data = b.data.to(self.device)
            meta, shard = {}, {}
            for dt, params in by_dtype.items():
                total = sum(p.numel() for p in params)
                padded = total + (-total % self.world)
                flat = torch.zeros(padded, dtype=dt)
                entries, off = [], 0
                for p in params:
                    n = p.numel()
                    flat[off:off + n].copy_(p.data.reshape(-1))
                    entries.append((p, off, n, tuple(p.shape)))
                    p.data = torch.empty(0, dtype=dt, device=self.device)
                    off += n
                per = padded // self.world
                shard[dt] = flat[self.rank * per:(self.rank + 1) * per].to(self.device)
                meta[dt] = entries
                sizes[dt] = max(sizes.get(dt, 0), padded)
                del flat
            self.meta.append(meta)
            self.shards.append(shard)
        self.bufs = [{dt: torch.empty(n, dtype=dt, device=self.device) for dt, n in sizes.items()} for _ in range(2)]
        self.stream = torch.cuda.Stream(device=self.device)
        self.ready = [None] * len(self.layers)

    @property
    def resident_bytes(self) -> int:
        shards = sum(t.numel() * t.element_size() for s in self.shards for t in s.values())
        bufs = sum(t.numel() * t.element_size() for b in self.bufs for t in b.values())
        return shards + bufs

    def _gather(self, i: int) -> None:
        buf = self.bufs[i % 2]
        with torch.cuda.stream(self.stream):
            for dt, shard in self.shards[i].items():
                full = buf[dt][:shard.numel() * self.world]
                dist.all_gather_into_tensor(full, shard, group=self.group)
                for p, off, n, shape in self.meta[i][dt]:
                    p.data = full[off:off + n].view(shape)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.ready[i] = ev

    def before(self, i: int) -> None:
        main = torch.cuda.current_stream(self.device)
        if self.ready[i] is None:                                   # first layer of a pass: not prefetched
            free = torch.cuda.Event()
            free.record(main)
            self.stream.wait_event(free)
            self._gather(i)
        main.wait_event(self.ready[i])
        self.ready[i] = None
        nxt = (i + 1) % len(self.layers)
        if self.ready[nxt] is None:
            # layer nxt reuses the buffer of layer i-1, whose compute is already queued on `main` before this point
            free = torch.cuda.Event()
            free.record(main)
            self.stream.wait_event(free)
            self._gather(nxt)

    def install_pre_hooks(self) -> list:
        return [layer.register_forward_pre_hook(lambda m, args, i=i: self.before(i))
                for i, layer in enumerate(self.layers)]


def largest_layer_list(module: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
    """The decoder stack of a transformers model: its largest ModuleList of identical layers."""
    best = ("", None, 0)
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 1 and len({type(x) for x in m}) == 1:
            n = sum(p.numel() for p in m.parameters())
            if n > best[2]:
                best = (name, m, n)
    if best[1] is None:
        raise RuntimeError("no layer stack found")
    return best[0], best[1]


def move_except(module: torch.nn.Module, skip: torch.nn.Module, device) -> None:
    """Move every parameter / buffer of `module` that is not inside `skip` to `device`."""
    skip_ids = {id(t) for t in list(skip.parameters()) + list(skip.buffers())}
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in skip_ids:
            t.data = t.data.to(device)
