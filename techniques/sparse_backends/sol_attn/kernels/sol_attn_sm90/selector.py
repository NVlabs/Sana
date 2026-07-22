"""CTA-local exact-block selector helpers for the SOL Attention CuteDSL port.

The production SOL Attention kernel must not materialize selected indices in HBM.  This
module keeps the representation as four scalar 32-bit mask words per route
group.  The initial lowering target is a fixed GROUP_SIZE scan:

    for off in 0..GROUP_SIZE-1:
        if mask[off]:
            load/compute exact block off

That removes the cuTile `ct.min(exact_offsets)` reduction tree while preserving
the same increasing-offset exact order as the Triton SOL Attention implementation.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def sol_attn_bfind_b32(x: Int32, *, loc=None, ip=None) -> Int32:
    """Return PTX bfind.u32 result: 0-based high-set-bit index."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "bfind.u32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def sol_attn_popc_b32(x: Int32, *, loc=None, ip=None) -> Int32:
    """Return the number of set bits using one PTX popc.b32."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def sol_attn_mask_word(mask0: Int32, mask1: Int32, mask2: Int32, mask3: Int32, word: Int32) -> Int32:
    res = mask0
    if word == Int32(1):
        res = mask1
    if word == Int32(2):
        res = mask2
    if word == Int32(3):
        res = mask3
    return res


@cute.jit
def sol_attn_test_exact_bit(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    offset: Int32,
) -> cutlass.Boolean:
    word = offset // Int32(32)
    bit = offset - word * Int32(32)
    m = sol_attn_mask_word(mask0, mask1, mask2, mask3, word)
    return (m & (Int32(1) << bit)) != Int32(0)


@cute.jit
def sol_attn_test_exact_bit_limited_words(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    offset: Int32,
    group_words: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    """Bit test specialized to the number of mask words used by a route group."""

    bit = offset & Int32(31)
    if const_expr(group_words == 1):
        return (mask0 & (Int32(1) << bit)) != Int32(0)
    if const_expr(group_words == 2):
        m = mask0
        if offset >= Int32(32):
            m = mask1
        return (m & (Int32(1) << bit)) != Int32(0)
    if const_expr(group_words == 3):
        word = offset // Int32(32)
        m = mask0
        if word == Int32(1):
            m = mask1
        if word == Int32(2):
            m = mask2
        return (m & (Int32(1) << bit)) != Int32(0)
    return sol_attn_test_exact_bit(mask0, mask1, mask2, mask3, offset)


@cute.jit
def sol_attn_set_exact_bit(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    offset: Int32,
):
    """Set one dynamic exact-block bit and return updated mask words."""

    word = offset // Int32(32)
    bit = offset - word * Int32(32)
    bit_value = Int32(1) << bit
    if word == Int32(0):
        mask0 = mask0 | bit_value
    if word == Int32(1):
        mask1 = mask1 | bit_value
    if word == Int32(2):
        mask2 = mask2 | bit_value
    if word == Int32(3):
        mask3 = mask3 | bit_value
    return mask0, mask1, mask2, mask3


@cute.jit
def sol_attn_abs_i32(x: Int32) -> Int32:
    res = x
    if x < Int32(0):
        res = Int32(0) - x
    return res


@cute.jit
def sol_attn_route_is_exact(
    q_block_idx: Int32,
    kv_block_idx: Int32,
    col_mean: Float32,
    thresh: Float32,
    valid: cutlass.Boolean,
) -> cutlass.Boolean:
    """SOL Attention exact predicate: threshold-selected or forced local block."""

    local = sol_attn_abs_i32(q_block_idx - kv_block_idx) <= Int32(1)
    return ((col_mean > thresh) or local) and valid


@cute.jit
def sol_attn_count_exact_bits_scan(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    group_size: cutlass.Constexpr[int],
) -> Int32:
    count = Int32(0)
    for off in cutlass.range_constexpr(group_size):
        if sol_attn_test_exact_bit(mask0, mask1, mask2, mask3, Int32(off)):
            count += Int32(1)
    return count


@cute.jit
def sol_attn_selected_offset_by_rank_scan(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    rank: Int32,
    group_size: cutlass.Constexpr[int],
) -> Int32:
    """Return the increasing-offset selected block at `rank`, or -1."""

    count = Int32(0)
    selected = Int32(-1)
    done = False
    for off in cutlass.range_constexpr(group_size):
        if sol_attn_test_exact_bit(mask0, mask1, mask2, mask3, Int32(off)) and not done:
            if count == rank:
                selected = Int32(off)
                done = True
            count += Int32(1)
    return selected


@cute.jit
def sol_attn_mask_word_constexpr(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    word: cutlass.Constexpr[int],
) -> Int32:
    if const_expr(word == 0):
        return mask0
    if const_expr(word == 1):
        return mask1
    if const_expr(word == 2):
        return mask2
    return mask3


@cute.jit
def sol_attn_selected_offset_by_rank_ffs(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    rank: Int32,
) -> Int32:
    """Return selected offset by popping mask bits with PTX ffs.b32."""

    count = Int32(0)
    selected = Int32(-1)
    done = False
    for word in cutlass.range_constexpr(4):
        m = sol_attn_mask_word_constexpr(mask0, mask1, mask2, mask3, word)
        while m != Int32(0):
            lowbit = m & (Int32(0) - m)
            bit = sol_attn_bfind_b32(lowbit)
            if count == rank and not done:
                selected = Int32(word * 32) + bit
                done = True
            m = m & (m - Int32(1))
            count += Int32(1)
    return selected


@cute.jit
def sol_attn_selected_offset_by_rank_bfind_desc(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    rank: Int32,
) -> Int32:
    """Return the descending-offset selected block at ``rank``.

    The SM100 SOL Attention pipeline historically scans route pairs from high to low.
    Popping the high bit of words 3..0 preserves that exact transaction order
    while avoiding work for zero mask bits.
    """

    word = Int32(3)
    m = mask3
    remaining = rank
    found_word = False
    count = sol_attn_popc_b32(mask3)
    if remaining < count:
        found_word = True
    else:
        remaining -= count
        word = Int32(2)
        m = mask2
    count = sol_attn_popc_b32(mask2)
    if not found_word:
        if remaining < count:
            found_word = True
        else:
            remaining -= count
            word = Int32(1)
            m = mask1
    count = sol_attn_popc_b32(mask1)
    if not found_word:
        if remaining < count:
            found_word = True
        else:
            remaining -= count
            word = Int32(0)
            m = mask0
    while remaining > Int32(0):
        bit = sol_attn_bfind_b32(m)
        m = m ^ (Int32(1) << bit)
        remaining -= Int32(1)
    return word * Int32(32) + sol_attn_bfind_b32(m)


@cute.jit
def sol_attn_count_exact_bits_popc(mask0: Int32, mask1: Int32, mask2: Int32, mask3: Int32) -> Int32:
    return (
        sol_attn_popc_b32(mask0)
        + sol_attn_popc_b32(mask1)
        + sol_attn_popc_b32(mask2)
        + sol_attn_popc_b32(mask3)
    )


@cute.jit
def sol_attn_pop_exact_offset_bfind_desc(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
):
    """Pop and return the highest selected offset from four mask words."""

    word = Int32(0)
    m = mask0
    if mask1 != Int32(0):
        word = Int32(1)
        m = mask1
    if mask2 != Int32(0):
        word = Int32(2)
        m = mask2
    if mask3 != Int32(0):
        word = Int32(3)
        m = mask3
    bit = sol_attn_bfind_b32(m)
    bit_value = Int32(1) << bit
    if word == Int32(0):
        mask0 = mask0 ^ bit_value
    if word == Int32(1):
        mask1 = mask1 ^ bit_value
    if word == Int32(2):
        mask2 = mask2 ^ bit_value
    if word == Int32(3):
        mask3 = mask3 ^ bit_value
    return word * Int32(32) + bit, mask0, mask1, mask2, mask3


@cute.jit
def sol_attn_count_exact_bits_ffs(mask0: Int32, mask1: Int32, mask2: Int32, mask3: Int32) -> Int32:
    count = Int32(0)
    for word in cutlass.range_constexpr(4):
        m = sol_attn_mask_word_constexpr(mask0, mask1, mask2, mask3, word)
        while m != Int32(0):
            m = m & (m - Int32(1))
            count += Int32(1)
    return count


__all__ = [
    "sol_attn_count_exact_bits_scan",
    "sol_attn_count_exact_bits_ffs",
    "sol_attn_count_exact_bits_popc",
    "sol_attn_bfind_b32",
    "sol_attn_popc_b32",
    "sol_attn_pop_exact_offset_bfind_desc",
    "sol_attn_route_is_exact",
    "sol_attn_selected_offset_by_rank_bfind_desc",
    "sol_attn_selected_offset_by_rank_ffs",
    "sol_attn_selected_offset_by_rank_scan",
    "sol_attn_set_exact_bit",
    "sol_attn_test_exact_bit",
    "sol_attn_test_exact_bit_limited_words",
]
