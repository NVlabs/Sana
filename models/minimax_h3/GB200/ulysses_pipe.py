"""Ulysses attention with the head exchange pipelined against attention, on the copy engine.

Adapted from SGLang's `ipc_a2a_multi.py` (Apache-2.0, sglang 0.5.22,
`IpcA2AMultiState.pipelined_attention`), cut down to what H3 needs and widened to uneven shards.

The sequential path (`ulysses_custom`) packs q/k/v, runs one NCCL all-to-all, attends over all local
heads, and runs a second all-to-all. Nothing overlaps, and NCCL's all-to-all is an SM kernel. Here
the local heads are split into `groups` head groups. Every rank writes each group's q/k/v rows
straight into the owner's receive slot with pitched `cuMemcpy2DAsync` copies (copy engine, no SMs)
on one copy stream, group by group, and publishes a counter per group with `cuStreamWriteValue32`.
Each group attends on its own stream as soon as its slot is complete (`cuStreamWaitValue32`), then
sends every peer its rows of the output and merges the blocks it receives, again on the copy engine.
So group g's exchange runs while group g-1 attends, and output copies overlap later groups.

Each head is attended independently, so splitting the heads changes no arithmetic: the result is
meant to be bit-identical to the sequential exchange (checked by the A/B harness, not assumed).

Shards are uneven (38247 rows over 4 ranks is [9562, 9562, 9562, 9561]): `counts` gives every rank's
rows, and every copy carries its own row count.

Buffer reuse needs no per-call parity (argument from SGLang's `_PipelinePool`): rank r sends a
group's output only after attending over its slot, and peer p returns from a call only after merging
all of r's blocks; p writes r's slots for the next call only after that, so after r read them. And p
writes its next block into r's output block only after attending over r's rows of that call, which r
sends only after its own previous call returned, so after r merged p's previous block.

Every rank must call `attention` the same number of times with the same shapes (the counters are
per call); the pool is allocated on first use, which is a collective.
"""

from __future__ import annotations

import ctypes
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_CU_MEMORYTYPE_DEVICE = 2


class _Memcpy2D(ctypes.Structure):  # CUDA_MEMCPY2D
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_int), ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_int), ("dstHost", ctypes.c_void_p),
        ("dstDevice", ctypes.c_uint64), ("dstArray", ctypes.c_void_p),
        ("dstPitch", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t),
    ]


class _MemOps:
    """cuStreamWriteValue32 / cuStreamWaitValue32 / cuMemcpy2DAsync through the driver API."""

    def __init__(self):
        cuda = ctypes.CDLL("libcuda.so.1")
        args = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self._write, self._wait = cuda.cuStreamWriteValue32_v2, cuda.cuStreamWaitValue32_v2
        for fn in (self._write, self._wait):
            fn.argtypes, fn.restype = args, ctypes.c_int
        self._copy2d = cuda.cuMemcpy2DAsync_v2
        self._copy2d.argtypes, self._copy2d.restype = [ctypes.POINTER(_Memcpy2D), ctypes.c_void_p], ctypes.c_int

    def write(self, stream, ptr: int, value: int) -> None:
        # default flags: the stream's earlier writes (the copies) land first
        rc = self._write(stream.cuda_stream, ptr, value & 0xFFFFFFFF, 0)
        if rc:
            raise RuntimeError(f"cuStreamWriteValue32 failed with CUresult {rc}")

    def wait(self, stream, ptr: int, value: int) -> None:
        # CU_STREAM_WAIT_VALUE_GEQ (0) compares cyclically, so counters may wrap
        rc = self._wait(stream.cuda_stream, ptr, value & 0xFFFFFFFF, 0)
        if rc:
            raise RuntimeError(f"cuStreamWaitValue32 failed with CUresult {rc}")

    def copy2d(self, stream, dst: int, dst_pitch: int, src: int, src_pitch: int, width: int,
               height: int) -> None:
        if height == 0:
            return
        desc = _Memcpy2D(srcMemoryType=_CU_MEMORYTYPE_DEVICE, srcDevice=src, srcPitch=src_pitch,
                         dstMemoryType=_CU_MEMORYTYPE_DEVICE, dstDevice=dst, dstPitch=dst_pitch,
                         WidthInBytes=width, Height=height)
        rc = self._copy2d(desc, stream.cuda_stream)
        if rc:
            raise RuntimeError(f"cuMemcpy2DAsync failed with CUresult {rc}")


@triton.jit
def _scatter_kernel(dst, q, k, v, rows, q_rs, k_rs, v_rs, head_base, dst_row0, rows_full,
                    HG: tl.constexpr, SM_GROUPS: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """Rows of heads `head_base + [0, SM_GROUPS * HG)` of q/k/v -> one member's receive slots
    `[group][part][rows_full][HG][D]`, rows `dst_row0 + [0, rows)`. Grid (row blocks, heads, 3)."""
    rb = tl.program_id(0)
    hl = tl.program_id(1)
    part = tl.program_id(2)
    g = hl // HG
    j = hl % HG
    row = rb * BR + tl.arange(0, BR)
    col = tl.arange(0, D)
    m = row[:, None] < rows
    h = head_base + hl
    if part == 0:
        x = tl.load(q + row[:, None].to(tl.int64) * q_rs + h * D + col[None, :], mask=m)
    elif part == 1:
        x = tl.load(k + row[:, None].to(tl.int64) * k_rs + h * D + col[None, :], mask=m)
    else:
        x = tl.load(v + row[:, None].to(tl.int64) * v_rs + h * D + col[None, :], mask=m)
    part_el = rows_full.to(tl.int64) * (HG * D)
    off = (g * 3 + part) * part_el + (dst_row0 + row[:, None]).to(tl.int64) * (HG * D) + j * D + col[None, :]
    tl.store(dst + off, x, mask=m)


def _shareable_zeros(n: int, dtype) -> torch.Tensor:
    """Kept out of expandable segments: freeing an IPC-exported expandable segment breaks the
    allocator on fabric-handle systems (GB200), per SGLang's `ipc_shareable_zeros`."""
    get = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if get is None or "expandable_segments:True" not in get():
        return torch.zeros(n, dtype=dtype, device="cuda")
    torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    try:
        return torch.zeros(n, dtype=dtype, device="cuda")
    finally:
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")


def _share(t: torch.Tensor, rank: int, world: int, group) -> list:
    """Every member's `t`, peers' opened in this rank's device context."""
    from torch.multiprocessing.reductions import reduce_tensor

    handles = [None] * world
    dist.all_gather_object(handles, reduce_tensor(t), group=group)
    dev = torch.cuda.current_device()
    out = []
    for member, (fn, args) in enumerate(handles):
        if member == rank:
            out.append(t)
            continue
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, torch.device):
                args[i] = torch.device(f"cuda:{dev}")
            elif isinstance(a, int) and i == 6:     # rebuild_cuda_tensor's device index
                args[i] = dev
        out.append(fn(*args))
    return out


class UlyssesPipe:
    """One Ulysses group's pipelined attention; `attention(q, k, v, attend)` per layer."""

    def __init__(self, world: int, rank: int, group, heads: int, head_dim: int, groups: int,
                 dtype=torch.bfloat16):
        if heads % (world * groups):
            raise ValueError(f"{heads} heads do not split into {world} ranks x {groups} groups")
        self.world, self.rank, self.group = world, rank, group
        self.heads, self.head_dim, self.groups = heads, head_dim, groups
        self.heads_local = heads // world
        self.hg = self.heads_local // groups
        self.dtype = dtype
        self.esz = torch.empty((), dtype=dtype).element_size()
        self.mem = _MemOps()
        self.cin = torch.cuda.Stream()
        self.gstreams = [torch.cuda.Stream() for _ in range(groups)]
        self.counts = None
        self.calls = 0
        self.groups_sent = 0
        # The first `sm_groups` head groups are written into every member's slot by one SM kernel
        # per member (NVLink stores) instead of pitched copy-engine copies (~110-300 GB/s for 512-B
        # rows): group 0's attention waited ~0.23 ms per layer for its slot. The later groups stay on
        # the copy engine, overlapped with attention. H3_PIPE_SM_GROUPS=0 restores copies only.
        self.sm_groups = min(int(os.environ.get("H3_PIPE_SM_GROUPS", "1")), groups)

    def _alloc(self, counts) -> None:
        """Receive slots, output blocks and counters for these shard sizes (a collective)."""
        world, hg, d = self.world, self.hg, self.head_dim
        if self.counts is not None:
            torch.cuda.synchronize()
            dist.barrier(group=self.group)      # no peer may still copy into the old mappings
        self.counts = list(counts)
        self.offs = [sum(counts[:p]) for p in range(world)]
        self.rows_full = sum(counts)
        self.max_local = max(counts)
        self.row_bytes = hg * d * self.esz                  # one token of one head group
        self.part_bytes = self.rows_full * self.row_bytes   # one part (q/k/v) of one group's slot
        self.group_bytes = 3 * self.part_bytes
        self.out_bytes = self.max_local * self.row_bytes    # one (source, group) output block
        in_numel = self.groups * self.group_bytes // self.esz
        out_numel = world * self.groups * self.out_bytes // self.esz
        self.inb = _share(_shareable_zeros(in_numel, self.dtype), self.rank, world, self.group)
        self.outb = _share(_shareable_zeros(out_numel, self.dtype), self.rank, world, self.group)
        # fin[p]: groups peer p has delivered into my slots; fout[g * world + p]: calls whose
        # group-g output peer p has delivered (separate per group: groups finish on separate streams)
        self.fin = _share(_shareable_zeros(world, torch.int32), self.rank, world, self.group)
        self.fout = _share(_shareable_zeros(self.groups * world, torch.int32), self.rank, world,
                           self.group)
        self.inb_ptr = [t.data_ptr() for t in self.inb]
        self.outb_ptr = [t.data_ptr() for t in self.outb]
        self.fin_ptr = [t.data_ptr() for t in self.fin]
        self.fout_ptr = [t.data_ptr() for t in self.fout]
        self.calls = 0
        self.groups_sent = 0
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def _slot_ptr(self, member: int, g: int, part: int, row: int) -> int:
        return self.inb_ptr[member] + g * self.group_bytes + part * self.part_bytes + row * self.row_bytes

    def _slot(self, g: int, part: int) -> torch.Tensor:
        start = (g * self.group_bytes + part * self.part_bytes) // self.esz
        n = self.part_bytes // self.esz
        return self.inb[self.rank].narrow(0, start, n).view(self.rows_full, self.hg, self.head_dim)

    def _out_block_ptr(self, member: int, source: int, g: int) -> int:
        return self.outb_ptr[member] + (source * self.groups + g) * self.out_bytes

    def attention(self, q, k, v, attend, counts) -> torch.Tensor:
        """q, k, v: `(rows_local, heads, head_dim)` with stride(-1) == 1 and stride(-2) == head_dim.
        `attend(q, k, v)` on `(rows_full, hg, head_dim)` returns the same shape. Returns this rank's
        `(rows_local, heads, head_dim)` output, contiguous."""
        if self.counts != list(counts):
            self._alloc(counts)
        world, r, G, hg, d = self.world, self.rank, self.groups, self.hg, self.head_dim
        rows_local = q.shape[0]
        assert rows_local == self.counts[r]
        for t in (q, k, v):
            assert t.stride(-1) == 1 and t.stride(-2) == d and t.shape[1] == self.heads
        mem, cin, esz = self.mem, self.cin, self.esz
        head_bytes = d * esz
        rb = self.row_bytes
        main = torch.cuda.current_stream()
        base = self.groups_sent
        self.groups_sent += G
        self.calls += 1
        call = self.calls
        peers = [(r + s) % world for s in range(1, world)]
        src = [(t.data_ptr(), t.stride(0) * esz) for t in (q, k, v)]
        my_off = self.offs[r]
        head0 = lambda p, g: p * self.heads_local + g * hg   # first head of rank p's group g

        own_done = []
        K = self.sm_groups
        if K:
            for p in (*peers, r):
                _scatter_kernel[(triton.cdiv(rows_local, 64), K * hg, 3)](
                    self.inb[p], q, k, v, rows_local, q.stride(0), k.stride(0), v.stride(0),
                    head0(p, 0), my_off, self.rows_full, HG=hg, SM_GROUPS=K, D=d, BR=64, num_warps=4)
            # default-flag stream writes order after the kernels' (peer) stores
            for p in peers:
                mem.write(main, self.fin_ptr[p] + 4 * r, base + K)
            own_done += [main.record_event()] * K
        ready = main.record_event()     # after the flag writes: cin's later flags must land after
        cin.wait_event(ready)
        with torch.cuda.stream(cin):
            for g in range(K, G):
                for p in (*peers, r):           # peers first: their copies gate remote attention
                    for part in range(3):
                        ptr, pitch = src[part]
                        mem.copy2d(cin, self._slot_ptr(p, g, part, my_off), rb,
                                   ptr + head0(p, g) * head_bytes, pitch, rb, rows_local)
                    if p == r:
                        own_done.append(cin.record_event())
                for p in peers:
                    mem.write(cin, self.fin_ptr[p] + 4 * r, base + g + 1)

        merged = torch.empty(rows_local, self.heads, d, dtype=q.dtype, device=q.device)
        m_ptr = merged.data_ptr()
        m_pitch = self.heads * head_bytes
        done = []
        for g in range(G):
            stream = self.gstreams[g]
            stream.wait_event(own_done[g])
            with torch.cuda.stream(stream):
                for p in peers:
                    mem.wait(stream, self.fin_ptr[r] + 4 * p, base + g + 1)
                out = attend(self._slot(g, 0), self._slot(g, 1), self._slot(g, 2)).contiguous()
                o_ptr = out.data_ptr()
                for p in peers:     # peer p's rows of my heads -> my block in p's output area
                    n = self.counts[p] * rb
                    mem.copy2d(stream, self._out_block_ptr(p, r, g), n,
                               o_ptr + self.offs[p] * rb, n, n, 1)
                for p in peers:
                    mem.write(stream, self.fout_ptr[p] + 4 * (g * world + r), call)
                mem.copy2d(stream, m_ptr + head0(r, g) * head_bytes, m_pitch,
                           o_ptr + my_off * rb, rb, rb, rows_local)
                for p in peers:
                    mem.wait(stream, self.fout_ptr[r] + 4 * (g * world + p), call)
                    mem.copy2d(stream, m_ptr + head0(p, g) * head_bytes, m_pitch,
                               self._out_block_ptr(r, p, g), rb, rb, rows_local)
                done.append(stream.record_event())
        for ev in done:
            main.wait_event(ev)
        # the next call's copies rewrite slots the caller's q/k/v were read into; the caller may also
        # free q/k/v, which the copy stream reads
        main.wait_stream(cin)
        return merged
