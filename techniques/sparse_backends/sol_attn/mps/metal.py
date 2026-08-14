"""Simdgroup-matrix Metal forward for 64/128-dimensional half-precision heads."""

from __future__ import annotations

import torch

from .preprocess import _routing_thresholds


_SOURCE = r"""
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
#include <ATen/native/mps/kernels/PrefillAttention.h>

template <typename T, int BD>
[[kernel, max_total_threads_per_threadgroup(128)]] void sol_reduce_summaries(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device float* QC [[buffer(3)]],
    device T* KC [[buffer(4)]],
    device T* VC [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& blocks [[buffer(8)]],
    constant ulong& sq_b [[buffer(9)]],
    constant ulong& sq_t [[buffer(10)]],
    constant ulong& sq_h [[buffer(11)]],
    constant ulong& sk_b [[buffer(12)]],
    constant ulong& sk_t [[buffer(13)]],
    constant ulong& sk_h [[buffer(14)]],
    constant ulong& sv_b [[buffer(15)]],
    constant ulong& sv_t [[buffer(16)]],
    constant ulong& sv_h [[buffer(17)]],
    uint dim [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint block = group % blocks;
  const uint batch_head = group / blocks;
  const uint head = batch_head % heads;
  const uint batch = batch_head / heads;
  const uint token_start = block * 64;
  const uint block_length = min(64u, tokens - token_start);

  if (dim >= BD) {
    return;
  }

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  float v_sum = 0.0f;
  for (uint offset = 0; offset < block_length; ++offset) {
    const uint token = token_start + offset;
    q_sum += float(Q[batch * sq_b + token * sq_t + head * sq_h + dim]);
    k_sum += float(K[batch * sk_b + token * sk_t + head * sk_h + dim]);
    v_sum += float(V[batch * sv_b + token * sv_t + head * sv_h + dim]);
  }

  const ulong summary_offset =
      ((ulong(batch) * heads + head) * blocks + block) * BD + dim;
  QC[summary_offset] = q_sum / float(block_length);
  KC[summary_offset] = T(k_sum / float(block_length));
  VC[summary_offset] = T(v_sum);
}

template <bool WEIGHT_DENOMINATOR, typename T, typename AccumType,
      typename STile, typename OTile, int LDV>
METAL_FUNC void sol_accumulate_tile(
    thread STile& scores,
    thread OTile& output,
    threadgroup T* values,
    short values_offset,
    thread AccumType* maximum,
  thread AccumType* denominator,
  uint key_start,
  short key_column,
  uint blocks,
  uint tokens) {
  constexpr short rows_per_thread = STile::kRowsPerThread;
  constexpr short value_tiles = OTile::kTileCols;
  constexpr short key_tiles = STile::kTileCols;
  using MMAFrag = typename STile::MMAFrag_t;
  MMATile<AccumType, 1, 1, MMAFrag> value_tile;

  AccumType next_maximum[rows_per_thread];
  AccumType correction[rows_per_thread];
  PREFILL_PRAGMA_UNROLL
  for (short row = 0; row < rows_per_thread; ++row) {
    next_maximum[row] = maximum[row];
  }
  scores.template row_reduce<MaxOp>(next_maximum);
  scores.template row_bin_op<ExpSubOp>(next_maximum);
  PREFILL_PRAGMA_UNROLL
  for (short row = 0; row < rows_per_thread; ++row) {
    correction[row] = next_maximum[row] == -INFINITY
        ? AccumType(1)
        : fast::exp2(maximum[row] - next_maximum[row]);
    maximum[row] = next_maximum[row];
  }

  output.template row_bin_op<MulOp>(correction);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  PREFILL_PRAGMA_UNROLL
  for (short value_tile_index = 0; value_tile_index < value_tiles; ++value_tile_index) {
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
      const short key_offset = key_tile_index * 8;
      const short dim_offset = value_tile_index * 8;
      value_tile.template load<T, 1, 1, LDV, 1>(
          &values[values_offset + key_offset * LDV + dim_offset]);
      simdgroup_barrier(mem_flags::mem_none);
      MMAFrag::mma(
          output.frag_at(0, value_tile_index),
          scores.frag_at(0, key_tile_index),
          value_tile.frag_at(0, 0),
          output.frag_at(0, value_tile_index));
    }
  }

  if constexpr (WEIGHT_DENOMINATOR) {
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        const uint block = key_start + key_column + key_tile_index * 8 + element;
        const uint block_length = block < blocks
            ? min(64u, tokens - block * 64)
            : 0u;
        scores.frag_at(0, key_tile_index)[element] *= AccumType(block_length);
      }
    }
  }

  AccumType tile_sum[rows_per_thread] = {0};
  scores.template row_reduce<SumOp>(tile_sum);
  PREFILL_PRAGMA_UNROLL
  for (short row = 0; row < rows_per_thread; ++row) {
    denominator[row] = denominator[row] * correction[row] + tile_sum[row];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename T, int BD, int LDK>
METAL_FUNC void sol_route_tile(
  const device float* route_query,
    threadgroup T* keys,
    threadgroup uchar* route_flags,
    uint summary_start,
    uint blocks,
    uint route_query_block,
    float route_threshold,
    float score_scale,
    uint sink_start,
    uint sink_end,
    uint sink_q_start,
    uint sink_q_end,
    uint lane,
    uint simdgroup) {
  constexpr short TD = BD / 8;
  using AccumType = c10::metal::accum_t<T>;
  using MMAFrag = BaseMMAFrag<AccumType, 8, 8>;
  MMATile<AccumType, 1, 1, MMAFrag> route_query_fragment;
  MMATile<AccumType, 1, 2, MMAFrag> route_key_fragment;
  MMATile<AccumType, 1, 2, MMAFrag> route_score_tile;
  route_score_tile.clear();

  const short2 coordinate = MMAFrag::get_coord(lane);
  const short row = coordinate.y;
  const short column = coordinate.x;
  const short key_offset = row * LDK + column;
  if (simdgroup < 4) {
    PREFILL_PRAGMA_UNROLL
    for (short dim_tile = 0; dim_tile < TD; ++dim_tile) {
        MMAFrag::load(
          route_query_fragment.frag_at(0, 0),
          &route_query[column + dim_tile * 8], Int<0>{}, Int<1>{});
      route_key_fragment.template load<T, 1, 1, LDK, 1>(
          &keys[key_offset + simdgroup * 16 + dim_tile * 8 * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(
          route_score_tile, route_query_fragment, route_key_fragment,
          route_score_tile);
    }
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < 2; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        route_score_tile.frag_at(0, key_tile_index)[element] *= score_scale;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool query_in_sink =
      route_query_block >= sink_q_start && route_query_block < sink_q_end;
  if (simdgroup < 4 && row == 0) {
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < 2; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        const uint local_block =
            simdgroup * 16 + column + key_tile_index * 8 + element;
        const uint block = summary_start + local_block;
        const bool neighbor =
            abs(int(route_query_block) - int(block)) <= 1;
        const bool sink = block >= sink_start && block < sink_end;
        route_flags[local_block] = block < blocks && (
            query_in_sink || neighbor || sink ||
            route_score_tile.frag_at(0, key_tile_index)[element] > route_threshold
        );
      }
    }
  }
}

template <typename T, int BD>
[[kernel, max_total_threads_per_threadgroup(128)]] void sol_route_mask_debug(
    const device float* QC [[buffer(0)]],
    const device T* KC [[buffer(1)]],
    const device float* thresholds [[buffer(2)]],
    device uchar* routes [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& blocks [[buffer(6)]],
    constant uint& sink_start [[buffer(7)]],
    constant uint& sink_end [[buffer(8)]],
    constant uint& sink_q_start [[buffer(9)]],
    constant uint& sink_q_end [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  constexpr short BK = 64;
  constexpr short LDK = BK + 8;
  using KLoader = BlockLoaderT<T, BK, BD, 1, LDK, 0, 128>;

  const uint route_query_block = group % blocks;
  const uint batch_head = group / blocks;
  const uint head = batch_head % heads;
  const uint batch = batch_head / heads;
  QC += ((ulong(batch) * heads + head) * blocks + route_query_block) * BD;
  KC += (ulong(batch) * heads + head) * blocks * BD;
  thresholds += (ulong(batch) * heads + head) * blocks + route_query_block;
  routes += ((ulong(batch) * heads + head) * blocks + route_query_block) * blocks;

  threadgroup T keys[BK * LDK];
  threadgroup uchar route_flags[BK];
  const uint thread_index = simdgroup * 32 + lane;

  for (uint summary_start = 0; summary_start < blocks; summary_start += BK) {
    const uint summary_count = min(uint(BK), blocks - summary_start);
    KLoader key_loader(KC + ulong(summary_start) * BD, BD, keys, simdgroup, lane);
    key_loader.load_safe(short2(BD, summary_count));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sol_route_tile<T, BD, LDK>(
      QC, keys, route_flags, summary_start, blocks,
        route_query_block, thresholds[0], scale * 1.44269504089f,
        sink_start, sink_end, sink_q_start, sink_q_end, lane, simdgroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < summary_count) {
      routes[summary_start + thread_index] = route_flags[thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int BD, int BQ, int TGP_SIZE>
[[kernel, max_total_threads_per_threadgroup(256)]] void sol_attn_tiled(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
  const device float* QC [[buffer(3)]],
  const device T* KC [[buffer(4)]],
  const device T* VC [[buffer(5)]],
  const device float* thresholds [[buffer(6)]],
  device T* O [[buffer(7)]],
  constant float& scale [[buffer(8)]],
  constant uint& tokens [[buffer(9)]],
  constant uint& heads [[buffer(10)]],
  constant uint& blocks [[buffer(11)]],
  constant uint& sink_start [[buffer(12)]],
  constant uint& sink_end [[buffer(13)]],
  constant uint& sink_q_start [[buffer(14)]],
  constant uint& sink_q_end [[buffer(15)]],
  constant ulong& sq_b [[buffer(16)]],
  constant ulong& sq_t [[buffer(17)]],
  constant ulong& sq_h [[buffer(18)]],
  constant ulong& sk_b [[buffer(19)]],
  constant ulong& sk_t [[buffer(20)]],
  constant ulong& sk_h [[buffer(21)]],
  constant ulong& sv_b [[buffer(22)]],
  constant ulong& sv_t [[buffer(23)]],
  constant ulong& sv_h [[buffer(24)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  constexpr short BK = 64;
  constexpr short LDQ = BQ == 64 ? BD : BD + 8;
  constexpr short LDK = BQ == 64 ? BK : BK + 8;
  constexpr short LDV = BQ == 64 ? BD : BD + 8;
    constexpr short shared_count = LDK * BD > BK * LDV
      ? LDK * BD
      : BK * LDV;
  constexpr short TD = BD / 8;
  constexpr short TK = BK / 8;
  using AccumType = c10::metal::accum_t<T>;
  using MMAFrag = BaseMMAFrag<AccumType, 8, 8>;
  using QLoader = BlockLoaderT<T, BQ, BD, LDQ, 1, 1, TGP_SIZE>;
  using KLoader = BlockLoaderT<T, BK, BD, 1, LDK, 0, TGP_SIZE>;
  using VLoader = BlockLoaderT<T, BK, BD, LDV, 1, 0, TGP_SIZE>;

  const uint query_tiles = (tokens + BQ - 1) / BQ;
  const uint query_tile = group % query_tiles;
  const uint batch_head = group / query_tiles;
  const uint head = batch_head % heads;
  const uint batch = batch_head / heads;
  const uint query_start = query_tile * BQ;
  const uint query_size = min(uint(BQ), tokens - query_start);
  const uint route_query_block = query_start / 64;

  Q += batch * sq_b + query_start * sq_t + head * sq_h;
  K += batch * sk_b + head * sk_h;
  V += batch * sv_b + head * sv_h;
  QC += ((ulong(batch) * heads + head) * blocks + route_query_block) * BD;
  KC += (ulong(batch) * heads + head) * blocks * BD;
  VC += (ulong(batch) * heads + head) * blocks * BD;
  thresholds += (ulong(batch) * heads + head) * blocks + route_query_block;
  O += ((ulong(batch) * tokens + query_start) * heads + head) * BD;

  threadgroup T query_shared[BQ * LDQ];
  threadgroup T key_value_shared[shared_count];
  threadgroup T* keys = key_value_shared;
  threadgroup T* values = key_value_shared;
  threadgroup uchar* route_flags =
      reinterpret_cast<threadgroup uchar*>(key_value_shared);

  QLoader query_loader(Q, int(sq_t), query_shared, simdgroup, lane);
  if (query_size < BQ) {
    query_loader.load_safe(short2(BD, query_size));
  } else {
    query_loader.load_unsafe();
  }

  MMATile<AccumType, 1, 1, MMAFrag> query_tile_fragment;
  MMATile<AccumType, 1, TK, MMAFrag> key_tile_fragment;
  MMATile<AccumType, 1, TK, MMAFrag> score_tile;
  MMATile<AccumType, 1, TD, MMAFrag> output_tile;
  output_tile.clear();

  const short2 coordinate = MMAFrag::get_coord(lane);
  const short row = coordinate.y;
  const short column = coordinate.x;
  const short query_row = 8 * simdgroup;
  const short query_offset = (query_row + row) * LDQ + column;
  const short key_offset = row * LDK + column;
  const short value_offset = row * LDV + column;
  constexpr short rows_per_thread = decltype(score_tile)::kRowsPerThread;
  const AccumType score_scale = AccumType(scale * 1.44269504089f);
  const uint thread_index = simdgroup * 32 + lane;
  const float route_threshold = thresholds[0];
  AccumType maximum[rows_per_thread];
  AccumType denominator[rows_per_thread] = {0};
  PREFILL_PRAGMA_UNROLL
  for (short index = 0; index < rows_per_thread; ++index) {
    maximum[index] = -INFINITY;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Unrouted blocks use K centroids and V sums, matching the Triton forward.
  for (uint summary_start = 0; summary_start < blocks; summary_start += BK) {
    const uint summary_count = min(uint(BK), blocks - summary_start);
    KLoader key_loader(KC + ulong(summary_start) * BD, BD, keys, simdgroup, lane);
    VLoader value_loader(VC + ulong(summary_start) * BD, BD, values, simdgroup, lane);
    key_loader.load_safe(short2(BD, summary_count));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    score_tile.clear();
    PREFILL_PRAGMA_UNROLL
    for (short dim_tile = 0; dim_tile < TD; ++dim_tile) {
      query_tile_fragment.template load<T, 1, 1, LDQ, 1>(
          &query_shared[query_offset + dim_tile * 8]);
      key_tile_fragment.template load<T, 1, 1, LDK, 1>(
          &keys[key_offset + dim_tile * 8 * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(score_tile, query_tile_fragment, key_tile_fragment, score_tile);
    }
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemsPerFrag; ++element) {
        score_tile.frag_at(0, key_tile_index)[element] *= score_scale;
      }
    }

    // Route scores and summary QK both consume KC. Reuse KC storage for flags
    // only after both calculations have finished, then retain flags in a
    // register mask before V/exact loads overwrite the shared buffer.
    sol_route_tile<T, BD, LDK>(
      QC, keys, route_flags, summary_start, blocks,
        route_query_block, route_threshold, score_scale,
        sink_start, sink_end, sink_q_start, sink_q_end, lane, simdgroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong route_mask = 0;
    for (uint local_block = 0; local_block < summary_count; ++local_block) {
      route_mask |= ulong(route_flags[local_block] != 0) << local_block;
    }

    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
      const uint block_column = summary_start + column + key_tile_index * 8;
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        const uint block = block_column + element;
        if (block >= blocks || ((route_mask >> (block - summary_start)) & 1ul)) {
          score_tile.frag_at(0, key_tile_index)[element] = -INFINITY;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    value_loader.load_safe(short2(BD, summary_count));
    sol_accumulate_tile<true, T, AccumType,
              decltype(score_tile), decltype(output_tile), LDV>(
      score_tile, output_tile, values, value_offset, maximum, denominator,
      summary_start, column, blocks, tokens);

    // Consume exact blocks while this summary group's route mask is in registers.
    for (uint local_key_block = 0; local_key_block < summary_count; ++local_key_block) {
      if (((route_mask >> local_key_block) & 1ul) == 0) {
        continue;
      }
      const uint key_start = (summary_start + local_key_block) * BK;
      const uint key_count = min(uint(BK), tokens - key_start);
      KLoader exact_key_loader(
          K + ulong(key_start) * sk_t, int(sk_t), keys, simdgroup, lane);
      VLoader exact_value_loader(
          V + ulong(key_start) * sv_t, int(sv_t), values, simdgroup, lane);
      if (key_count < BK) {
        exact_key_loader.load_safe(short2(BD, key_count));
      } else {
        exact_key_loader.load_unsafe();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      score_tile.clear();
      PREFILL_PRAGMA_UNROLL
      for (short dim_tile = 0; dim_tile < TD; ++dim_tile) {
        query_tile_fragment.template load<T, 1, 1, LDQ, 1>(
            &query_shared[query_offset + dim_tile * 8]);
        key_tile_fragment.template load<T, 1, 1, LDK, 1>(
            &keys[key_offset + dim_tile * 8 * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(score_tile, query_tile_fragment, key_tile_fragment, score_tile);
      }
      PREFILL_PRAGMA_UNROLL
      for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
        PREFILL_PRAGMA_UNROLL
        for (short element = 0; element < MMAFrag::kElemCols; ++element) {
          score_tile.frag_at(0, key_tile_index)[element] *= score_scale;
        }
      }
      if (key_count < BK) {
        PREFILL_PRAGMA_UNROLL
        for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
          const short key_column = column + key_tile_index * 8;
          PREFILL_PRAGMA_UNROLL
          for (short element = 0; element < MMAFrag::kElemCols; ++element) {
            if (key_column + element >= key_count) {
              score_tile.frag_at(0, key_tile_index)[element] = -INFINITY;
            }
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (key_count < BK) {
        exact_value_loader.load_safe(short2(BD, key_count));
      } else {
        exact_value_loader.load_unsafe();
      }
      sol_accumulate_tile<false, T, AccumType,
                          decltype(score_tile), decltype(output_tile), LDV>(
          score_tile, output_tile, values, value_offset, maximum, denominator,
          0, 0, blocks, tokens);
    }
  }

  PREFILL_PRAGMA_UNROLL
  for (short index = 0; index < rows_per_thread; ++index) {
    if (maximum[index] == -INFINITY) {
      denominator[index] = AccumType(1);
    }
  }
  output_tile.template row_bin_op<DivOp>(denominator);
  device T* output_ptr = O + ulong(query_row + row) * heads * BD + column;
  if (query_row + row < query_size) {
    output_tile.template store_safe<T, 1, 1>(
        output_ptr, heads * BD, short2(BD - column, query_size - (query_row + row)));
  }
}

instantiate_kernel("sol_attn_tiled_f16_d64_bq32", sol_attn_tiled, half, 64, 32, 128)
instantiate_kernel("sol_attn_tiled_f16_d128_bq32", sol_attn_tiled, half, 128, 32, 128)
instantiate_kernel("sol_attn_tiled_bf16_d64_bq32", sol_attn_tiled, bfloat, 64, 32, 128)
instantiate_kernel("sol_attn_tiled_bf16_d128_bq32", sol_attn_tiled, bfloat, 128, 32, 128)
instantiate_kernel("sol_attn_tiled_f16_d64_bq64", sol_attn_tiled, half, 64, 64, 256)
instantiate_kernel("sol_attn_tiled_f16_d128_bq64", sol_attn_tiled, half, 128, 64, 256)
instantiate_kernel("sol_attn_tiled_bf16_d64_bq64", sol_attn_tiled, bfloat, 64, 64, 256)
instantiate_kernel("sol_attn_tiled_bf16_d128_bq64", sol_attn_tiled, bfloat, 128, 64, 256)
instantiate_kernel("sol_reduce_summaries_f16_d64", sol_reduce_summaries, half, 64)
instantiate_kernel("sol_reduce_summaries_f16_d128", sol_reduce_summaries, half, 128)
instantiate_kernel("sol_reduce_summaries_bf16_d64", sol_reduce_summaries, bfloat, 64)
instantiate_kernel("sol_reduce_summaries_bf16_d128", sol_reduce_summaries, bfloat, 128)
instantiate_kernel("sol_route_mask_debug_f16_d64", sol_route_mask_debug, half, 64)
instantiate_kernel("sol_route_mask_debug_f16_d128", sol_route_mask_debug, half, 128)
instantiate_kernel("sol_route_mask_debug_bf16_d64", sol_route_mask_debug, bfloat, 64)
instantiate_kernel("sol_route_mask_debug_bf16_d128", sol_route_mask_debug, bfloat, 128)
"""


_library = None


def _get_library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


def sol_attn_tiled_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
    query_block_size: int = 64,
) -> torch.Tensor:
    if q.dtype not in (torch.float16, torch.bfloat16) or q.shape[-1] not in (64, 128):
        raise ValueError("tiled Metal forward requires float16/bfloat16 heads of dimension 64 or 128")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same BTHD shape")
    if q.shape[1] == 0:
        return torch.empty_like(q)
    if query_block_size not in (32, 64):
      raise ValueError("query_block_size must be 32 or 64")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, tokens, heads, head_dim = q.shape
    blocks = (tokens + 63) // 64
    summary_shape = (batch, heads, blocks, head_dim)
    q_centroids = torch.empty(summary_shape, device=q.device, dtype=torch.float32)
    k_centroids = torch.empty(summary_shape, device=q.device, dtype=q.dtype)
    v_sums = torch.empty(summary_shape, device=q.device, dtype=v.dtype)
    dtype_name = "f16" if q.dtype == torch.float16 else "bf16"
    summary_kernel = getattr(
      _get_library(), f"sol_reduce_summaries_{dtype_name}_d{head_dim}"
    )
    summary_kernel(
      q, k, v, q_centroids, k_centroids, v_sums,
      tokens, heads, blocks,
      q.stride(0), q.stride(1), q.stride(2),
      k.stride(0), k.stride(1), k.stride(2),
      v.stride(0), v.stride(1), v.stride(2),
      threads=batch * heads * blocks * 128,
      group_size=128,
      arg_casts={6: "int32", 7: "int32", 8: "int32"},
    )
    thresholds = _routing_thresholds(
      q_centroids, k_centroids.float(), scale, float(tau), thresh_type
    ).contiguous()
    output = torch.empty_like(q)
    query_tiles = (tokens + query_block_size - 1) // query_block_size
    group_size = query_block_size * 4
    kernel = getattr(
      _get_library(),
      f"sol_attn_tiled_{dtype_name}_d{head_dim}_bq{query_block_size}",
    )
    kernel(
      q, k, v, q_centroids, k_centroids, v_sums, thresholds, output,
      scale, tokens, heads, k_centroids.shape[2],
      int(sink_blocks[0]), int(sink_blocks[1]),
      int(sink_q[0]), int(sink_q[1]),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        threads=batch * heads * query_tiles * group_size,
        group_size=group_size,
      arg_casts={
        9: "int32", 10: "int32", 11: "int32",
        12: "int32", 13: "int32", 14: "int32", 15: "int32",
      },
    )
    return output


def _routing_debug_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
):
    """Return production Metal routing inputs and mask for parity tests only."""
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, tokens, heads, head_dim = q.shape
    blocks = (tokens + 63) // 64
    summary_shape = (batch, heads, blocks, head_dim)
    q_centroids = torch.empty(summary_shape, device=q.device, dtype=torch.float32)
    k_centroids = torch.empty(summary_shape, device=q.device, dtype=q.dtype)
    unused_v_sums = torch.empty(summary_shape, device=q.device, dtype=k.dtype)
    dtype_name = "f16" if q.dtype == torch.float16 else "bf16"
    summary_kernel = getattr(
      _get_library(), f"sol_reduce_summaries_{dtype_name}_d{head_dim}"
    )
    summary_kernel(
      q, k, k, q_centroids, k_centroids, unused_v_sums,
      tokens, heads, blocks,
      q.stride(0), q.stride(1), q.stride(2),
      k.stride(0), k.stride(1), k.stride(2),
      k.stride(0), k.stride(1), k.stride(2),
      threads=batch * heads * blocks * 128,
      group_size=128,
      arg_casts={6: "int32", 7: "int32", 8: "int32"},
    )
    thresholds = _routing_thresholds(
      q_centroids, k_centroids.float(), scale, float(tau), thresh_type
    ).contiguous()
    routes = torch.empty(
      (batch, heads, blocks, blocks), device=q.device, dtype=torch.uint8
    )
    route_kernel = getattr(
      _get_library(), f"sol_route_mask_debug_{dtype_name}_d{head_dim}"
    )
    route_kernel(
      q_centroids, k_centroids, thresholds, routes,
      scale, heads, blocks,
      int(sink_blocks[0]), int(sink_blocks[1]),
      int(sink_q[0]), int(sink_q[1]),
      threads=batch * heads * blocks * 128,
      group_size=128,
      arg_casts={
        5: "int32", 6: "int32", 7: "int32", 8: "int32",
        9: "int32", 10: "int32",
      },
    )
    return routes.bool(), q_centroids, k_centroids, thresholds


__all__ = ["sol_attn_tiled_mps"]