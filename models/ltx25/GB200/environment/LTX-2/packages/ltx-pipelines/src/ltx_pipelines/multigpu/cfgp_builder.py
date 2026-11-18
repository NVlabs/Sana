"""CFG-parallel builder — the stage-1 replacement for SequenceParallelBuilder.

The vendor mgpu pipeline wires (ti2vid_two_stages_mgpu.py:103/117):
    stage_1._transformer_builder = SequenceParallelBuilder(...)   # All2All, every layer
    stage_2._transformer_builder = TiledDataParallelBuilder(...)  # one all_reduce/forward
Measured hot inference, 4x GB200: SP gives stage 1 **1.221 s/step vs 1.102 single-GPU**
-- i.e. four GPUs are SLOWER there, because All2All fires at every attention. CFG
parallelism instead splits the guidance batch `_guided_denoise` has already assembled:
**0.303 s/step, 3.6x**, with one all_gather of ~6.3 MB per step, independent of depth.

Stage 2 keeps TDP: it has a single forward per step (no guidance batch to split), and
TDP wins there (0.449 s/step vs 1.181 single-GPU).

Unlike SP this needs no module ops and no ltx-kernels -- it only wraps the finished
model, so it is a pure post-build decorator.
"""

from __future__ import annotations

from typing import Generic

import torch
import torch.distributed as dist

from ltx_core.batch_split import _merge_tensors, _split_perturbations
from ltx_core.loader.primitives import ModelBuilderProtocol
from ltx_core.model.model_protocol import LTXModelProtocol
from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder, InnerModelT


def _reindex(m, idx):
    """Permute a Modality along the batch dim."""
    import dataclasses
    if m is None:
        return None
    upd = {f.name: getattr(m, f.name).index_select(0, idx)
           for f in dataclasses.fields(m)
           if isinstance(getattr(m, f.name), torch.Tensor)
           and getattr(m, f.name).shape[:1] == idx.shape[:1]}
    return dataclasses.replace(m, **upd)


def _reindex_pert(p, perm):
    """_block_masks is (type, block, sample); permute the sample axis."""
    import copy as _copy
    if p is None:
        return None
    m = getattr(p, "_block_masks", None)
    if not isinstance(m, torch.Tensor) or m.shape[-1] != len(perm):
        return p
    new = _copy.copy(p)
    new._block_masks = m.index_select(-1, torch.tensor(perm, device=m.device))
    cpu = getattr(p, "_block_masks_cpu", None)
    if isinstance(cpu, torch.Tensor) and cpu.shape[-1] == len(perm):
        new._block_masks_cpu = cpu.index_select(-1, torch.tensor(perm))
    return new


def _even(batch: int, world: int) -> list[int]:
    b, r = divmod(batch, world)
    return [b + (1 if i < r else 0) for i in range(world)]


class CFGParallelWrapper(torch.nn.Module):
    def __init__(self, model, group=None):
        super().__init__()
        self._model = model
        self._group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)

    @property
    def num_blocks(self):
        return self._model.num_blocks

    @staticmethod
    def _order(batch):
        """Permutation applied before the contiguous split."""
        import os as _os
        spec = _os.environ.get("LTX_CFG_ORDER", "0,3,1,2")
        if not spec:
            return None
        try:
            perm = [int(x) for x in spec.split(",")]
        except ValueError:
            return None
        if sorted(perm) != list(range(batch)):
            return None          # wrong pass count for this spec -> leave as is
        return perm

    def forward(self, video, audio, perturbations):
        batch = (video or audio).latent.shape[0]
        # The torchrun version was only trustworthy because it counted; this one was
        # diagnosed twice without counting and both diagnoses were wrong.
        self._n_calls = getattr(self, "_n_calls", 0) + 1
        if self._n_calls <= 3 or self._n_calls % 20 == 0:
            # print() is useless here: mgpu runs its ranks as SPAWNED workers whose
            # stdout never reaches the parent -- the same reason the vendor SP run
            # produced no LTX_TIME at all. Write to the shared file instead.
            _msg = (f"[cfgb] rank{self.rank} call#{self._n_calls} batch={batch} "
                    f"world={self.world} -> {'SPLIT' if batch >= self.world else 'WHOLE'}")
            import os as _os
            _f = _os.environ.get("STAGE_TIME_FILE")
            if _f:
                with open(_f, "a") as _fh:
                    _fh.write(_msg + "\n")
            print(_msg, flush=True)
        if batch < self.world:
            self._n_whole = getattr(self, "_n_whole", 0) + 1
            return self._model(video=video, audio=audio, perturbations=perturbations)
        self._n_split = getattr(self, "_n_split", 0) + 1
        sizes = _even(batch, self.world)

        perm = self._order(batch)
        if perm is not None:
            idx = torch.tensor(perm, device=(video or audio).latent.device)
            video = _reindex(video, idx)
            audio = _reindex(audio, idx)
            perturbations = _reindex_pert(perturbations, perm)

        v = video.split(sizes) if video is not None else [None] * self.world
        a = audio.split(sizes) if audio is not None else [None] * self.world
        p = _split_perturbations(perturbations, sizes) if perturbations is not None else [None] * self.world

        _t0 = torch.cuda.Event(enable_timing=True)
        _t1 = torch.cuda.Event(enable_timing=True)
        _t0.record()
        ov, oa = self._model(video=v[self.rank], audio=a[self.rank], perturbations=p[self.rank])
        _t1.record()
        self._ev = getattr(self, "_ev", [])
        self._ev.append((_t0, _t1))

        def gather(local):
            if local is None:
                return None
            bufs = [torch.empty((s, *local.shape[1:]), dtype=local.dtype, device=local.device) for s in sizes]
            dist.all_gather(bufs, local.contiguous(), group=self._group)
            return _merge_tensors(bufs)

        gv, ga = gather(ov), gather(oa)
        if perm is not None:
            inv = [0] * batch
            for i, j in enumerate(perm):
                inv[j] = i
            iidx = torch.tensor(inv, device=(gv if gv is not None else ga).device)
            gv = gv.index_select(0, iidx) if gv is not None else None
            ga = ga.index_select(0, iidx) if ga is not None else None
        self._dump_balance()
        return gv, ga

    def _dump_balance(self, every=20):
        """Per-rank GPU time for the local passes. Balance is a measurement."""
        ev = getattr(self, "_ev", [])
        if len(ev) < every:
            return
        torch.cuda.synchronize()
        ms = sum(a.elapsed_time(b) for a, b in ev)
        self._ev = []
        import os as _os
        _f = _os.environ.get("STAGE_TIME_FILE")
        msg = (f"[cfgbal] rank{self.rank} order={_os.environ.get('LTX_CFG_ORDER','0,3,1,2')} "
               f"{every} calls local_gpu_ms={ms:.1f} mean={ms / every:.2f}")
        if _f:
            with open(_f, "a") as _fh:
                _fh.write(msg + "\n")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


class CFGParallelBuilder(DelegatingBuilder[InnerModelT], Generic[InnerModelT]):
    def __init__(
        self,
        inner: ModelBuilderProtocol[LTXModelProtocol],
        group=None,
        registry=None,
        tracker=None,
    ) -> None:
        # Same preparation SequenceParallelBuilder does. Without it the weights land
        # in fresh GPU storages on every build and the vendor refuses to capture a
        # CUDA graph over them.
        if registry is not None:
            cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            inner = inner.with_registry(registry).with_lora_load_device(cuda_device)
        self._inner = inner
        self._group = group
        self._tracker = tracker

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        # True only when a tracker is actually rebinding GPU-resident tensors across
        # builds. Claiming True without one would let CUDA graphs capture storages
        # that are later reallocated -- silent corruption instead of a clean abort.
        if self._tracker is not None:
            return True
        return getattr(self._inner, "keeps_gpu_resident_weights", False)

    def build(self, device=None, dtype=None, **kwargs):
        if self._tracker is not None:
            model = self._tracker.build(self._inner, device=device, dtype=dtype, **kwargs)
        else:
            model = self._inner.build(device=device, dtype=dtype, **kwargs)
        return CFGParallelWrapper(model, self._group)
