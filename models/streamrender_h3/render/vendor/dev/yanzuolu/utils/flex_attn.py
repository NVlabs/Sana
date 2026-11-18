"""Packed-sequence visibility rules and flex-attention backend dispatch.

A packed sequence is described by segment lengths, per-segment attention modes
(``causal``, ``full``, or ``noise``), a sink count, and an optional window size.
This module exposes two equivalent representations of that visibility contract:

* ``_prepare_flex_attention_mask`` returns variable-length query/key rectangles,
  attention types, and workload accounting for ``flex_flash_attn_func``. The
  optional ``kv_split_lens`` argument allows query and key segment layouts to
  differ.
* ``create_sparse_mask`` builds a torch ``BlockMask`` for
  ``torch.nn.attention.flex_attention`` while applying the compiler settings
  required to avoid materializing a dense quadratic mask.

``FlexAttention`` selects the available backend and consumes the corresponding
representation. Keeping mask construction and dispatch together ensures both
backends enforce the same visibility semantics.
"""

from __future__ import annotations

import bisect
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

from dev.yanzuolu.utils.flash_attn import FLASH_ATTN_4_AVAILABLE


torch._dynamo.config.cache_size_limit = int(os.environ.get("TORCH_DYNAMO_CACHE_SIZE_LIMIT", 2048))
torch._dynamo.config.accumulated_cache_size_limit = int(os.environ.get("TORCH_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT", 16384))


def _set_default_magi_kernel_backend() -> None:
    backend_env_vars = (
        "MAGI_ATTENTION_KERNEL_BACKEND",
        "MAGI_ATTENTION_FA4_BACKEND",
        "MAGI_ATTENTION_SDPA_BACKEND",
    )
    if (
        FLASH_ATTN_4_AVAILABLE
        and os.environ.get("FLEX_FLASH_ATTN_AVAILABLE") != "0"
        and not any(name in os.environ for name in backend_env_vars)
    ):
        os.environ["MAGI_ATTENTION_KERNEL_BACKEND"] = "fa4"


_set_default_magi_kernel_backend()


_magi_kernel_backend = None
_MagiAttentionKernelBackend = None
try:
    from magi_attention.api import flex_flash_attn_func
    FLEX_FLASH_ATTN_AVAILABLE = bool(int(os.environ.get("FLEX_FLASH_ATTN_AVAILABLE", "1")))
    try:
        from magi_attention.common.enum import MagiAttentionKernelBackend as _MagiAttentionKernelBackend
        from magi_attention.env.general import kernel_backend as _magi_kernel_backend
    except (ImportError, AttributeError):
        # Hopper's pinned pre-v1.1 MagiAttention does not expose backend selection.
        pass
except ModuleNotFoundError:
    FLEX_FLASH_ATTN_AVAILABLE = False


def _magi_auto_range_merge_enabled() -> bool:
    if _magi_kernel_backend is None or _MagiAttentionKernelBackend is None:
        return True
    return _magi_kernel_backend() != _MagiAttentionKernelBackend.FA4


# Eager flex_attention runs the HOP through ``math_attention``: it materializes
    # the full [1, H, Q, KV] fp32 score matrix and vmaps ``mask_mod`` over the whole
    # grid. At production sequence lengths that is tens of GB and instantly OOMs, so
    # the compiled kernel is not an optimization, it is the only usable form.
    # Default automatic dynamic-shape behavior is intentional here.
    #
    # No ``dynamic=`` argument, i.e. the default automatic dynamic: dynamo marks
    # the sequence length dynamic on the second distinct Q_LEN and from then on
    # reuses one kernel for every length. Measured on H100 over 28 consecutive
    # sequence lengths: ``dynamic=False`` compiled 28 times, automatic dynamic
    # compiled once.
    #
    # A symbolic length is not free per kernel, and the mechanism is worth naming
    # because it is not what it looks like. The ``IS_DIVISIBLE`` template flag is
    # decided by ``seq_len_q % 128 != 0`` at
    # ``torch/nn/attention/flex_attention.py:230`` in the forward (a structural
    # sympy comparison) and by ``statically_known_true`` at ``:606-611`` in the
    # backward. Neither can prove anything about a SymInt, so as soon as the length
    # is symbolic ``IS_DIVISIBLE`` is False for EVERY shape, aligned or not, and
    # every KV block -- not just the ragged last one -- runs the
    # ``CHECK_BLOCK_BOUNDARY=True`` branch (an extra modulo plus a ``tl.where`` per
    # block). That, not the ragged tail, is where the measured 2-10% pure-forward
    # penalty comes from. It does not survive into training: fwd+bwd is 0.99x/1.00x
    # of static at 8192/16384 tokens (10 heads x 256 head_dim, bf16, packed
    # sample-isolation + causal mask), and training runs fwd+bwd. So there is
    # nothing to buy by pinning the shape.
    #
    # Note this says nothing about whether any particular caller pads:
flex_attention = torch.compile(flex_attention)



def _prepare_flex_attention_mask(
    split_lens: list[int],
    attn_modes: list[str],
    sink: int = 0,
    window_size: int | None = None,
    device: str | torch.device = "cpu",
    kv_split_lens: list[int] | None = None,
) -> tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor, int]:
    """Enumerate the visible (query-block, key-block) rectangles of a packed sequence.

    Returns ``(q_ranges, k_ranges, attn_type_map, attn_workloads)``: two
    ``[N, 2]`` int32 tensors of half-open token spans, an ``[N]`` int32 map
    marking each rectangle full (0) or causal (1), and the total attended
    element count used for MFU accounting.

    Per segment ``i`` it emits the diagonal block -- causal or full per
    ``attn_modes[i]`` -- then one full block per earlier segment ``j < i`` that
    survives the sink/window filter. ``noise`` segments are never attended to
    from outside themselves, so they are skipped as keys and do not advance the
    non-noise index that sink and window are measured in.

    ``kv_split_lens`` lets the key side use a different segment layout from the
    query side, including bidirectional cross-attention between two modalities.
    """
    if kv_split_lens is None:
        kv_split_lens = split_lens
    assert len(split_lens) == len(attn_modes) == len(kv_split_lens), (
        "split_lens, attn_modes and kv_split_lens should have the same length."
    )

    q_ranges_list: list[list[int]] = []
    k_ranges_list: list[list[int]] = []
    attn_type_list: list[int] = []

    q_split_starts: list[int] = []
    q_split_ends: list[int] = []
    csum = 0
    for length in split_lens:
        q_split_starts.append(csum)
        csum += length
        q_split_ends.append(csum)

    kv_split_starts: list[int] = []
    kv_split_ends: list[int] = []
    csum = 0
    for length in kv_split_lens:
        kv_split_starts.append(csum)
        csum += length
        kv_split_ends.append(csum)

    type_full = 0
    type_causal = 1

    non_noise_indices: list[int] = []
    split_to_non_noise_idx: dict[int, int] = {}
    for idx, mode in enumerate(attn_modes):
        if mode != "noise":
            split_to_non_noise_idx[idx] = len(non_noise_indices)
            non_noise_indices.append(idx)

    for i, (q_len, q_mode) in enumerate(zip(split_lens, attn_modes)):
        if q_len == 0:
            continue
        q_start = q_split_starts[i]
        q_end = q_split_ends[i]
        k_start = kv_split_starts[i]
        k_end = kv_split_ends[i]
        if k_start == k_end:
            raise ValueError(f"Query split {i} has non-zero q_len but zero kv_len.")

        q_ranges_list.append([q_start, q_end])
        k_ranges_list.append([k_start, k_end])
        attn_type_list.append(type_causal if q_mode == "causal" else type_full)

        if i in split_to_non_noise_idx:
            q_logic_idx = split_to_non_noise_idx[i]
        else:
            q_logic_idx = bisect.bisect_right(non_noise_indices, i)

        for j in range(i):
            k_len = kv_split_lens[j]
            k_mode = attn_modes[j]
            if k_len == 0 or k_mode == "noise":
                continue

            j_non_noise_idx = split_to_non_noise_idx[j]
            is_in_sink = j_non_noise_idx < sink
            is_in_window = True
            if window_size is not None:
                is_in_window = q_logic_idx - j_non_noise_idx <= window_size
            if not (is_in_sink or is_in_window):
                continue

            q_ranges_list.append([q_start, q_end])
            k_ranges_list.append([kv_split_starts[j], kv_split_ends[j]])
            attn_type_list.append(type_full)

    if q_ranges_list:
        q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
        k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
        attn_type_map = torch.tensor(attn_type_list, dtype=torch.int32, device=device)
    else:
        q_ranges = torch.zeros((0, 2), dtype=torch.int32, device=device)
        k_ranges = torch.zeros((0, 2), dtype=torch.int32, device=device)
        attn_type_map = torch.zeros((0,), dtype=torch.int32, device=device)

    q_lens = q_ranges[:, 1] - q_ranges[:, 0]
    k_lens = k_ranges[:, 1] - k_ranges[:, 0]
    full_counts = q_lens * k_lens
    causal_counts = (q_lens * (q_lens + 1)) // 2
    block_counts = torch.where(attn_type_map == type_causal, causal_counts, full_counts)
    attn_workloads = block_counts.sum().item()

    return q_ranges, k_ranges, attn_type_map, int(attn_workloads)


# Pre-packed reference implementation of ``create_sparse_mask``, superseded by the
# packed implementation below. Kept commented out as a comparison reference.
#
# def create_sparse_mask(document_lens, split_lens, attn_modes, device="cpu", *, sink, window_size):
#     """Port of the old sparse flex-attention mask builder.
#
#     Returns a flex-attention mask predicate implementing causal/full/noise segment
#     visibility with per-sample, sink, and window constraints.
#     """
#     assert len(set(sink)) == 1, "Currently only support same sink for all samples."
#     sink = sink[0]
#     assert len(set(window_size)) == 1, "Currently only support same window_size for all samples."
#     window_size: Optional[int] = window_size[0]
#
#     def causal_mask(b, h, q_idx, kv_idx):
#         """Standard lower triangle."""
#         return q_idx >= kv_idx
#
#     def full_and_noise_mask(b, h, q_idx, kv_idx):
#         """Same full/noise segment, excluding text/causal segments."""
#         return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)
#
#     def remove_noise_mask(b, h, q_idx, kv_idx):
#         """Disallow attending to other noise segments."""
#         return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))
#
#     def sample_mask(b, h, q_idx, kv_idx):
#         """Restrict attention to tokens from the same sample."""
#         return document_id[q_idx] == document_id[kv_idx]
#
#     def sink_window_mask(b, h, q_idx, kv_idx):
#         q_non_noise_idx = non_noise_idx_map[q_idx]
#         kv_non_noise_idx = non_noise_idx_map[kv_idx]
#
#         same_split = split_id[q_idx] == split_id[kv_idx]
#         in_sink = kv_non_noise_idx < sink
#
#         if window_size is None:
#             # 0-dim constant: a shape-(1,) tensor here leaks an extra dim through
#             # create_block_mask's vmap (5D mask -> unpack error). Latent upstream
#             # bug, never hit there because magi flex-flash always short-circuits
#             # this torch-flex fallback path.
#             in_window = torch.ones((), dtype=torch.bool, device=device)
#         else:
#             distance = q_non_noise_idx - kv_non_noise_idx
#             in_window = distance <= window_size
#
#         return same_split | in_sink | in_window
#
#     full_and_noise_tmp = []
#     noise_tmp = []
#     for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
#         value = i if mode in ["full", "noise"] else -1
#         full_and_noise_tmp.extend([value] * length)
#         value_noise = i if mode == "noise" else -1
#         noise_tmp.extend([value_noise] * length)
#
#     full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
#     noise_seq_id = torch.Tensor(noise_tmp).to(device)
#     document_id = torch.cat([torch.full((length,), i) for i, length in enumerate(document_lens, start=1)]).to(device)
#
#     split_id_tmp = []
#     non_noise_idx_tmp = []
#     non_noise_counter = 0
#     non_noise_indices = []
#     for mode in attn_modes:
#         if mode != "noise":
#             non_noise_indices.append(non_noise_counter)
#             non_noise_counter += 1
#         else:
#             non_noise_indices.append(non_noise_counter)
#
#     for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
#         split_id_tmp.extend([i] * length)
#         non_noise_idx_tmp.extend([non_noise_indices[i]] * length)
#
#     split_id = torch.Tensor(split_id_tmp).to(device)
#     non_noise_idx_map = torch.Tensor(non_noise_idx_tmp).to(device)
#
#     return and_masks(
#         or_masks(causal_mask, full_and_noise_mask),
#         remove_noise_mask,
#         sample_mask,
#         sink_window_mask,
#     )


# Bit layout of the packed per-token word ``_packed_sparse_mask`` builds. Every field
# is already non-negative, and 20 bits leaves ~1e6 of headroom on quantities that are
# at most a few thousand in the largest pack we can build. Total 62 bits, so the
# int64 sign bit is never touched.
_SPLIT_BITS, _ID_BITS = 20, 20
_FAN_SHIFT = _SPLIT_BITS               # 20
_NOISE_SHIFT = _FAN_SHIFT + 1          # 21
_DOC_SHIFT = _NOISE_SHIFT + 1          # 22
_NN_SHIFT = _DOC_SHIFT + _ID_BITS      # 42
_MASK_SPLIT = (1 << _SPLIT_BITS) - 1
_MASK_ID = (1 << _ID_BITS) - 1


def _packed_sparse_mask(document_lens, split_lens, attn_modes, device="cpu", *, sink, window_size):
    """Exact drop-in for the reference implementation above, with one lookup table instead of five.

    Same signature, same two preconditions, same visibility rule. Verified
    elementwise-identical to the reference implementation above over 16,777,216 mask
    elements with 0 mismatches, and BlockMask-identical, by
    ``tests/test_packed_sparse_mask.py``. Do not redesign the bit layout
    or the boolean algebra below without re-running that probe.

    WHY it exists: the reference predicate issues ten indirect loads (five
    per-token tables x q and kv sides), and above a load-count cliff inductor stops
    fusing the mask build and REALIZES the full ``[1, 1, N, N]`` bool -- 4030 MiB of
    transient GPU memory every training step at the production packed length 64960,
    and it does so on the compiled path too, not just eager.

    HOW it collapses to one: the five tables are algebraically redundant. ``split_id``
    is the segment index, and the two other per-segment tables are functions of it
    plus that segment's mode --

        fan[t]   = i  if mode[i] in {full, noise} else -1
        noise[t] = i  if mode[i] == noise        else -1

    -- so ``fan[q] == fan[kv] and fan[q] >= 0`` is exactly ``same_segment and
    is_fan[q]``: equal non-negative fan ids mean the same segment, and the same
    segment means the same mode. Only four fields per token survive (split_id, two
    mode bits, doc_id, non-noise index), which is 62 bits, one int64, ONE load per
    side.
    """
    assert len(set(sink)) == 1, "Currently only support same sink for all samples."
    sink = sink[0]
    assert len(set(window_size)) == 1, "Currently only support same window_size for all samples."
    window_size = window_size[0]

    # The non-noise index is DOCUMENT-RELATIVE: the counter restarts at every document
    # boundary, matching ``_prepare_flex_attention_mask``, which is called once per
    # sample and so numbers from 0 there too. A pack-global counter would make
    # ``nn_k < sink`` satisfiable only inside the first document, and ``same_doc`` then
    # hides those tokens from every other one, so documents 2..B would get no sink.
    # The window test is a DIFFERENCE of two indices in the same document, so the
    # per-document offset cancels and only the sink semantics move.
    doc_ends, csum = [], 0
    for length in document_lens:
        csum += length
        doc_ends.append(csum)

    non_noise_indices, counter, pos, doc = [], 0, 0, 0
    for length, mode in zip(split_lens, attn_modes):
        while doc < len(doc_ends) and pos == doc_ends[doc]:
            doc, counter = doc + 1, 0
        non_noise_indices.append(counter)
        if mode != "noise":
            counter += 1
        pos += length
        assert doc >= len(doc_ends) or pos <= doc_ends[doc], (
            f"split of length {length} straddles the end of document {doc}"
        )

    values = []
    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        word = i
        if mode in ("full", "noise"):
            word |= 1 << _FAN_SHIFT
        if mode == "noise":
            word |= 1 << _NOISE_SHIFT
        word |= non_noise_indices[i] << _NN_SHIFT
        values.extend([word] * length)

    # doc_id is per-SAMPLE, not per-segment, so it is OR-ed in over the sample spans
    # rather than the segment spans. Production numbers samples from 1.
    doc_values = []
    for d, length in enumerate(document_lens, start=1):
        doc_values.extend([d << _DOC_SHIFT] * length)
    assert len(values) == len(doc_values), (len(values), len(doc_values))

    assert max(len(split_lens), len(document_lens)) <= _MASK_ID, "field width exceeded"
    table = torch.tensor([v | d for v, d in zip(values, doc_values)],
                         dtype=torch.int64, device=device)

    def packed_mask(b, h, q_idx, kv_idx):
        pq = table[q_idx]
        pk = table[kv_idx]
        same_seg = (pq & _MASK_SPLIT) == (pk & _MASK_SPLIT)
        is_fan_q = ((pq >> _FAN_SHIFT) & 1) != 0
        is_noise_q = ((pq >> _NOISE_SHIFT) & 1) != 0
        is_noise_k = ((pk >> _NOISE_SHIFT) & 1) != 0
        same_doc = ((pq >> _DOC_SHIFT) & _MASK_ID) == ((pk >> _DOC_SHIFT) & _MASK_ID)
        nn_q = pq >> _NN_SHIFT
        nn_k = pk >> _NN_SHIFT

        visible = (q_idx >= kv_idx) | (same_seg & is_fan_q)
        visible = visible & ~(is_noise_k & ~(is_noise_q & same_seg))
        visible = visible & same_doc
        sink_window = same_seg | (nn_k < sink)
        if window_size is None:
            # 0-dim constant: a shape-(1,) tensor here leaks an extra dim through
            # create_block_mask's vmap (5D mask -> unpack error). Latent upstream
            # bug, never hit there because magi flex-flash always short-circuits
            # this torch-flex fallback path.
            sink_window = sink_window | torch.ones((), dtype=torch.bool, device=device)
        else:
            sink_window = sink_window | ((nn_q - nn_k) <= window_size)
        return visible & sink_window

    return packed_mask


# Applied SCOPED around ``create_block_mask`` only (see ``create_sparse_mask`` below).
# Together with ``_packed_sparse_mask`` this takes the transient peak of the mask build at the
# production packed length 64960 from 4030 MiB to 5.9 MiB (679x), at build time
# inside run-to-run noise, with the BlockMask bit-identical and flex_attention's
# forward and dq/dk/dv all bitwise equal.
#
# WHY ONE KNOB, AND WHY THIS ONE. Inductor has two independent realize decisions,
# each an OR (torch 2.9.1):
#
#   A  has_exceeded_max_reads()        torch/_inductor/ir.py:8345, during lowering
#      Pointwise and ( num_reads() > realize_acc_reads_threshold          # 8
#                      or has_large_inner_fn()                            # opcount > 30
#                      or (realize_acc_reads_size_threshold is not None   # default None
#                          and accumulated read bytes > it) )
#
#   B  should_realize_on_reuse(users)  torch/_inductor/ir.py:8357, on a reused buffer
#      users > 1 and ( num_reads() > realize_reads_threshold              # 4
#                      or has_large_inner_fn() )                          # opcount > 30
#
# ``has_large_inner_fn()`` is SHARED by both, so raising ``realize_opcount_threshold``
# alone closes the only branch our predicate trips.
#
# The read-count gates never fire, and the reason is easy to get wrong: for a
# Pointwise, ``num_reads()`` is ``len(inner_fn_opcount().read_buffers)``
# (torch/_inductor/ir.py:1045) -- the number of DISTINCT BUFFERS read, not the number
# of load instructions. ``table[q_idx]`` and ``table[kv_idx]`` are two loads but one
# buffer. The packed predicate reads exactly ONE, versus five for the five-table
# reference implementation it replaced, so both 4 and 8 are far away.
#
# That is measured, not argued -- a torch 2.9.1 benchmark sweep covered all
# 16 subsets of four candidate knobs and split 8:8 on the presence of this one alone,
# and a repro confirmed the three others are no-ops (an apparent 52-vs-2 MiB difference
# was allocator carryover from the preceding measurement; it vanished once the sweep
# order was rotated). ``tests/test_packed_sparse_mask.py`` does not reproduce that
# sweep -- it is the standing regression gate, measuring predicate x knob at production
# lengths.
#
# Consequence for anyone editing the packed predicate: the headroom is in buffers,
# not bits. Adding a new per-token field by opening a SEPARATE tensor walks toward the
# gates (four more such tensors re-arms gate B, eight re-arms gate A); packing it into
# the spare bits of the existing word costs nothing. Two bits are free today, and
# ``_SPLIT_BITS``/``_ID_BITS`` are 20 each against real values in the low thousands,
# so there is room to narrow them. Re-run the probes either way.
# Same if torch changes -- this reads two private thresholds out of ``_inductor.config``
# and depends on the OR structure above. RE-MEASURE ON ANY TORCH UPGRADE; the gate is
# ``tests/test_packed_sparse_mask.py``, which fails loudly if the peak goes
# back to O(N^2).
#
# RE-MEASURED on torch 2.11.0+cu128 (H200) by that gate, and the pairing holds. peak
# over N^2 at N=65536: (reference, no knob) 1.0005, (reference, knob) 1.0005,
# (packed, no knob) 1.0151, (packed, knob) 0.0016 -- 4098 MiB down to 6.52 MiB, 629x,
# consistent with the 679x result on 2.9.1. NEITHER HALF WORKS ALONE, and
# packed-without-knob is in fact slightly WORSE than the reference (316 vs 256 MiB
# at N=16384): the saving is
# in buffer count, so when fusion does not happen the int64 table is just one more
# intermediate. ``realize_opcount_threshold`` still exists on 2.11, default still 30.
# The (packed, knob) cell measures ``create_sparse_mask`` whole, so unlike the other
# three it carries its own int64 table (512 KiB at N=65536) inside the measured window
# -- that is the entire difference between 629x here and the 681x an earlier shape
# reported, not a regression.
_FUSION_KNOBS = {"realize_opcount_threshold": 4096}


def create_sparse_mask(document_lens, split_lens, attn_modes, device="cpu", *, sink, window_size,
                       block_size):
    """Build the flex-attention ``BlockMask`` for a packed sequence.

    Owns the ``create_block_mask`` call rather than just returning the predicate,
    because the predicate and ``_FUSION_KNOBS`` are only useful together and the knob
    has to be live *during* the build: the predicate is an unevaluated closure, and
    ``_compile=True`` means inductor traces it inside ``create_block_mask``, which is
    where the realize decision the knob controls is taken. A caller that patched the
    config around building the predicate alone would have restored it before it
    mattered -- measured, that is the ``(packed, no knob)`` row of the matrix above,
    i.e. no saving at all.

    ``H=None`` broadcasts the head-independent mask. ``_compile=True`` is kept because
    eager ``create_block_mask`` goes through ``create_mask`` and is much slower; it is
    a SPEED property, not a memory one.
    """
    # ``_packed_sparse_mask`` numbers the non-noise index PER DOCUMENT. Do not
    # re-globalise it: a pack-global counter silently drops the sink on every document
    # but the first, which is what ``_case_sink_vs_rectangles`` in
    # ``tests/test_packed_sparse_mask.py`` measures against the rectangle backend.
    seqlen = sum(document_lens)
    sparse_mask = _packed_sparse_mask(document_lens, split_lens, attn_modes, device,
                                      sink=sink, window_size=window_size)
    with torch._inductor.config.patch(_FUSION_KNOBS):
        return create_block_mask(
            sparse_mask, B=1, H=None, Q_LEN=seqlen, KV_LEN=seqlen,
            device=device, BLOCK_SIZE=block_size, _compile=True,
        )


class _FlexAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def tflops(self, args, kwargs, output) -> float:
        attn_workloads = sum(kwargs["attn_workloads"]) / 1e12
        if FLEX_FLASH_ATTN_AVAILABLE:
            _, h, d = output.shape
        else:
            _, h, _, d = output.shape
        return h * (4 * d * attn_workloads)

    def forward(self, *args, attn_workloads, **kwargs):
        # attn_workloads is MFU accounting only (tflops); callers that do not
        # provide it contribute 0 rather than risk an undefined backend default.
        if attn_workloads is None:
            attn_workloads = 0
        # magi's flex-flash only supports head_dim <= 128 (its sanity_check
        # asserts ``head_dim <= 128``); larger head dims (e.g. 256) fall back to
        # the compiled flex_attention path.
        if FLEX_FLASH_ATTN_AVAILABLE and args[0].shape[-1] <= 128:
            return flex_flash_attn_func(*args, **kwargs)[0]
        else:
            return flex_attention(*args, **kwargs)


class FlexAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flex_attn_impl = _FlexAttention()

    def forward(
        self,
        packed_query_states: torch.Tensor,
        packed_key_states: torch.Tensor,
        packed_value_states: torch.Tensor,
        attention_mask: BlockMask,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor,
        attn_workloads: List[int] | int | None = None,
        sample_lens: List[int] | torch.Tensor | None = None,  # DEPRECATED: ignored (padding support removed)
        dtype=torch.bfloat16,
    ):
        allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        assert dtype in allowed_dtypes, f"FlexAttention dtype must be one of {allowed_dtypes}, got {dtype}"
        half_dtypes = (torch.float16, torch.bfloat16)
        def half(x):
            # dtype=float32 requests fp32 attention: promote anything below fp32.
            # The half path keeps the original semantics -- bf16/fp16 inputs pass
            # through untouched, while other inputs downcast to the requested dtype.
            if dtype == torch.float32:
                return x if x.dtype == torch.float32 else x.to(torch.float32)
            return x if x.dtype in half_dtypes else x.to(dtype)
        q_dtype = packed_query_states.dtype

        packed_query_states = half(packed_query_states)
        packed_key_states = half(packed_key_states)
        packed_value_states = half(packed_value_states)

        if FLEX_FLASH_ATTN_AVAILABLE and packed_query_states.shape[-1] <= 128:
            packed_attn_output = self._flex_attn_impl(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                auto_range_merge=_magi_auto_range_merge_enabled(),
                attn_workloads=attn_workloads,
            )
        else:
            # torch flex takes the packed [N, h, d] as [1, h, N, d] views --
            # permute/unsqueeze copy nothing and preserves the existing packed call
            # shape. The removed padding machinery required a per-call GPU sync plus
            # three unconditional full copies (~13% of the training step). The pack
            # must already span the mask's
            # Q_LEN/KV_LEN, and a caller that violates that fails loudly on
            # flex_attention's shape check against the BlockMask.
            packed_attn_output = self._flex_attn_impl(
                packed_query_states.permute(1, 0, 2).unsqueeze(0),  # [1, num_head, L, head_dim]
                packed_key_states.permute(1, 0, 2).unsqueeze(0),
                packed_value_states.permute(1, 0, 2).unsqueeze(0),
                block_mask=attention_mask,
                attn_workloads=attn_workloads,
            )[0].transpose(0, 1)  # [L, num_head, head_dim]

        return packed_attn_output.to(q_dtype)
