"""CUDA candidate for chunkwise BiGDN forward (RTX 5090 / sm_120).

Staged port. `run_cuda(inp, mode)` lets us swap each phase Triton<->CUDA:
  mode="c"   : Triton A + Triton B + CUDA Phase-C-fused(+divide)   [Stage 1]
  mode="bc"  : Triton A + CUDA B + CUDA C                          [Stage 2]
  mode="all" : full CUDA                                           [Stage 3]
Correctness is validated phase-by-phase against Triton in the harness.
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load_inline

from diffusion.model.ops.fused_gdn_chunkwise import phase_a, phase_b_triton, phase_c

_EXT = None

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

namespace {

constexpr int D = 128;
constexpr int WM = 16;                 // wmma tile
constexpr int KT = D / WM;             // 8 k-tiles
constexpr int NCT = D / WM;            // 8 column tiles (output dim)
constexpr int ROWS = 64;               // s-rows per CTA
constexpr int RT = ROWS / WM;          // 4 row tiles
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;

// ───────────────────────── Phase C + fused divide ─────────────────────────
// Grid (BH*F, S_TILES): one CTA per (bh,f,s-tile of ROWS rows). M kept OUT of
// smem (it is bf16 in HBM, ~14MB total -> L2-resident); wmma b-fragments load
// directly from global so occupancy is set by the small Q working set, not by
// the 32KB M. den[r] = sum_d Q*z; out = num/(den+eps) bf16 to [B,N,H,D].
__global__ void phase_c_fused_kernel(
    const bf16* __restrict__ qkv, // [B,N,3,H,D]
    long sb, long sn, long s3, long sh, long sd,
    const float* __restrict__ q_inv_rms, // [B,N]
    const float* __restrict__ q_norm_w,  // [H*D]
    const float* __restrict__ rope_cos,  // [N,D]
    const float* __restrict__ rope_sin,  // [N,D]
    const bf16*  __restrict__ M,          // [BH,F,D,D] bf16
    const float* __restrict__ z,          // [BH,F,D]
    bf16* __restrict__ out,               // [B,N,H,D]
    int B, int H, int F, int S, int S_TILES, float eps) {
  const int bhf = blockIdx.x;           // 0..BH*F-1
  const int s_tile = blockIdx.y;        // 0..S_TILES-1
  const int bh = bhf / F;
  const int f  = bhf % F;
  const int b  = bh / H;
  const int h  = bh % H;
  const int N  = F * S;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int s0 = s_tile * ROWS;

  extern __shared__ char smem_raw[];
  float* z_s    = reinterpret_cast<float*>(smem_raw);          // D floats
  float* qnw_s  = z_s + D;                                     // D floats
  bf16*  Qrot_s = reinterpret_cast<bf16*>(qnw_s + D);          // ROWS*D bf16
  float* den_s  = reinterpret_cast<float*>(Qrot_s + ROWS * D); // ROWS floats

  const bf16*  M_f = M + (long)bhf * D * D;
  const float* z_f = z + (long)bhf * D;

  for (int i = tid; i < D; i += THREADS) { z_s[i] = z_f[i]; qnw_s[i] = q_norm_w[h * D + i]; }
  __syncthreads();

  const bf16* qkv_q = qkv + (long)b * sb + 0 * s3 + (long)h * sh;
  const int n_base = f * S;

  // ── Qrot = relu(Qn)*cos + relu(Qn[d^1])*sin, all in fp32 (match Triton) ──
  for (int idx = tid; idx < ROWS * D; idx += THREADS) {
    const int r = idx >> 7, d = idx & 127, s = s0 + r;
    float qrot = 0.f;
    if (s < S) {
      const int n = n_base + s;
      const float invrms = q_inv_rms[b * N + n];
      const long base = (long)n * sn;
      float qd  = __bfloat162float(qkv_q[base + (long)d * sd]) * invrms * qnw_s[d];
      float qd1 = __bfloat162float(qkv_q[base + (long)(d ^ 1) * sd]) * invrms * qnw_s[d ^ 1];
      qd  = qd  > 0.f ? qd  : 0.f;
      qd1 = qd1 > 0.f ? qd1 : 0.f;
      qrot = qd * rope_cos[(long)n * D + d] + qd1 * rope_sin[(long)n * D + d];
    }
    Qrot_s[idx] = __float2bfloat16(qrot);
  }
  // den uses fp32 Q (match Triton) — recompute qv from global (qkv is L1/L2-hot).
  for (int r = warp; r < ROWS; r += WARPS) {
    const int s = s0 + r;
    float acc = 0.f;
    if (s < S) {
      const int n = n_base + s;
      const float invrms = q_inv_rms[b * N + n];
      for (int d = lane; d < D; d += 32) {
        float qn = __bfloat162float(qkv_q[(long)n * sn + (long)d * sd]) * invrms * qnw_s[d];
        float qv = qn > 0.f ? qn : 0.f;
        acc += qv * z_s[d];
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    }
    if (lane == 0) den_s[r] = acc;
  }
  __syncthreads();

  // ── num = Qrot @ M (b-frag from global/L2). warp w owns col-tile w. ──
  const int ct = warp;                  // 0..7
  __shared__ float tile_smem[WARPS][WM * WM];
  wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[RT];
  #pragma unroll
  for (int rt = 0; rt < RT; ++rt) wmma::fill_fragment(acc[rt], 0.f);
  #pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::row_major> b_frag;
    wmma::load_matrix_sync(b_frag, M_f + (kt * WM) * D + ct * WM, D);
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, Qrot_s + (rt * WM) * D + kt * WM, D);
      wmma::mma_sync(acc[rt], a_frag, b_frag, acc[rt]);
    }
  }
  #pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    wmma::store_matrix_sync(tile_smem[warp], acc[rt], WM, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < WM * WM; i += 32) {
      const int rr = i >> 4, cc = i & 15;
      const int s = s0 + rt * WM + rr;
      if (s < S) {
        const int n = n_base + s;
        const int d = ct * WM + cc;
        out[(((long)b * N + n) * H + h) * D + d] =
            __float2bfloat16(tile_smem[warp][i] / (den_s[rt * WM + rr] + eps));
      }
    }
    __syncwarp();
  }
}

// ───────────────────────── Phase A — KV stream ─────────────────────────
// Per (bh,f): I_P_kv[D,D]=I-K_rot^T diag(b) K_rot, A[D,D]=K_rot^T diag(b) V.
// grid (BH*F,), 16 warps (512 thr). warp w owns (output=w>>3, row-tile=w&7) and
// its 8 col-tiles -> 8 acc frags/warp (avoids register spills). Krot prep shared.
constexpr int AW_KV = 16;          // warp w owns one output's row-tile (8 frags)
constexpr int ATH_KV = AW_KV * 32; // 512 threads
constexpr int ACHUNK = 32;            // S-rows staged per sync cycle
constexpr int ASUB = ACHUNK / WM;     // wmma k-subtiles per chunk
__global__ __launch_bounds__(ATH_KV) void phase_a_kv_kernel(
    const bf16* __restrict__ qkv, long sb, long sn, long s3, long sh, long sd,
    const float* __restrict__ beta,       // [BH, F*S]
    const float* __restrict__ k_inv_rms,  // [B, N]
    const float* __restrict__ k_norm_w,   // [H*D]
    const float* __restrict__ rope_cos,   // [N,D]
    const float* __restrict__ rope_sin,   // [N,D]
    bf16* __restrict__ I_P_kv,            // [BH,F,D,D]
    bf16* __restrict__ A,                 // [BH,F,D,D]
    int B, int H, int F, int S, float k_scale) {
  const int bhf = blockIdx.x, bh = bhf / F, f = bhf % F, b = bh / H, h = bh % H;
  const int N = F * S, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int out_sel = warp >> 3;   // 0=P_kv, 1=A
  const int rt = warp & 7;         // row-tile (d in [16*rt, 16*rt+16))

  // Assumes contiguous packed qkv: sd==1, so K/V rows are contiguous in d
  // (16-byte cp.async friendly). sn is the per-token stride.
  extern __shared__ char sm[];
  bf16*  bufK = reinterpret_cast<bf16*>(sm);            // [2][ACHUNK,D] raw K
  bf16*  bufV = bufK + 2 * ACHUNK * D;                  // [2][ACHUNK,D] raw V
  float* knw  = reinterpret_cast<float*>(bufV + 2 * ACHUNK * D); // [D]
  bf16*  Krot = reinterpret_cast<bf16*>(knw + D);       // [ACHUNK,D]
  bf16*  bKrot= Krot + ACHUNK * D;                      // [ACHUNK,D]
  bf16*  bV   = bKrot + ACHUNK * D;                     // [ACHUNK,D]
  float* irms_s = reinterpret_cast<float*>(bV + ACHUNK * D); // [ACHUNK]
  float* beta_s = irms_s + ACHUNK;                      // [ACHUNK]

  for (int i = tid; i < D; i += ATH_KV) knw[i] = k_norm_w[h * D + i];

  wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[NCT];
  #pragma unroll
  for (int ct = 0; ct < NCT; ++ct) wmma::fill_fragment(acc[ct], 0.f);

  const bf16* qkv_k = qkv + (long)b * sb + 1 * s3 + (long)h * sh;
  const bf16* qkv_v = qkv + (long)b * sb + 2 * s3 + (long)h * sh;
  const float* beta_bhf = beta + (long)bh * N + (long)f * S;
  const int n_base = f * S;
  const int NCH = (S + ACHUNK - 1) / ACHUNK;
  const int VEC = 8;                       // 8 bf16 = 16 bytes per cp.async
  const int COPIES = (ACHUNK * D) / VEC;   // 16-byte copies per chunk per tensor

  // cp.async stage raw K,V for chunk c into double-buffer slot c%2.
  #define STAGE(c) {                                                            \
    const int _s0 = (c) * ACHUNK; const int _slot = ((c) & 1) * ACHUNK * D;     \
    for (int t = tid; t < COPIES; t += ATH_KV) {                               \
      const int _sl = t / (D / VEC), _d = (t % (D / VEC)) * VEC, _s = _s0 + _sl; \
      if (_s < S) {                                                            \
        const long _ko = (long)(n_base + _s) * sn + _d;                        \
        __pipeline_memcpy_async(&bufK[_slot + _sl * D + _d], &qkv_k[_ko], 16);  \
        __pipeline_memcpy_async(&bufV[_slot + _sl * D + _d], &qkv_v[_ko], 16);  \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                       \
  }

  STAGE(0);
  __syncthreads();
  for (int c = 0; c < NCH; ++c) {
    if (c + 1 < NCH) STAGE(c + 1);
    __pipeline_wait_prior(c + 1 < NCH ? 1 : 0);
    const int s0 = c * ACHUNK;
    // hoist per-row constants (invrms, beta) once -> avoid 128x redundant loads.
    for (int i = tid; i < ACHUNK; i += ATH_KV) {
      const int s = s0 + i;
      irms_s[i] = (s < S) ? k_inv_rms[b * N + n_base + s] : 0.f;
      beta_s[i] = (s < S) ? beta_bhf[s] : 0.f;
    }
    __syncthreads();
    const bf16* Kb = bufK + (c & 1) * ACHUNK * D;
    const bf16* Vb = bufV + (c & 1) * ACHUNK * D;
    // vectorized prep: each thread does a 4-wide d-block (d4..d4+3). RoPE pair
    // d^1 stays within the block, so one float4 cos/sin/knw load covers all 4.
    for (int t = tid; t < ACHUNK * (D / 4); t += ATH_KV) {
      const int sl = t / (D / 4), d4 = (t % (D / 4)) * 4, s = s0 + sl;
      float kr[4] = {0, 0, 0, 0}, bvv[4] = {0, 0, 0, 0};
      float be = 0.f;
      if (s < S) {
        const int n = n_base + s;
        const float invrms = irms_s[sl];
        be = beta_s[sl];
        const float4 c4 = *reinterpret_cast<const float4*>(&rope_cos[(long)n * D + d4]);
        const float4 s4 = *reinterpret_cast<const float4*>(&rope_sin[(long)n * D + d4]);
        const float4 w4 = *reinterpret_cast<const float4*>(&knw[d4]);
        const float wv[4] = {w4.x, w4.y, w4.z, w4.w};
        const float cv[4] = {c4.x, c4.y, c4.z, c4.w};
        const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
        float kn[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          float v = __bfloat162float(Kb[sl * D + d4 + j]) * invrms * wv[j];
          kn[j] = (v > 0.f ? v : 0.f) * k_scale;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          kr[j] = kn[j] * cv[j] + kn[j ^ 1] * sv[j];      // d^1 within block
          bvv[j] = be * __bfloat162float(Vb[sl * D + d4 + j]);
        }
      }
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        Krot[sl * D + d4 + j]  = __float2bfloat16(kr[j]);
        bKrot[sl * D + d4 + j] = __float2bfloat16(be * kr[j]);
        bV[sl * D + d4 + j]    = __float2bfloat16(bvv[j]);
      }
    }
    __syncthreads();
    const bf16* bsrc = out_sel ? bV : bKrot;
    #pragma unroll
    for (int ks = 0; ks < ASUB; ++ks) {
      wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, Krot + ks * WM * D + rt * WM, D);
      #pragma unroll
      for (int ct = 0; ct < NCT; ++ct) {
        wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::row_major> bfr;
        wmma::load_matrix_sync(bfr, bsrc + ks * WM * D + ct * WM, D);
        wmma::mma_sync(acc[ct], a_frag, bfr, acc[ct]);
      }
    }
    __syncthreads();
  }
  #undef STAGE

  bf16* dst = (out_sel ? A : I_P_kv) + (long)bhf * D * D;
  __shared__ float st[AW_KV][WM * WM];
  #pragma unroll
  for (int ct = 0; ct < NCT; ++ct) {
    wmma::store_matrix_sync(st[warp], acc[ct], WM, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < WM * WM; i += 32) {
      const int d = rt * WM + (i >> 4), dp = ct * WM + (i & 15);
      float v = st[warp][i];
      if (out_sel == 0) v = (d == dp ? 1.f : 0.f) - v;   // I - P_kv
      dst[(long)d * D + dp] = __float2bfloat16(v);
    }
    __syncwarp();
  }
}

// ───────────────────────── Phase A — Z stream ─────────────────────────
// Per (bh,f): I_P_z[D,D]=I-K^T diag(b) K (no RoPE), B[D]=sum_s b_s K_s.
__global__ void phase_a_z_kernel(
    const bf16* __restrict__ qkv, long sb, long sn, long s3, long sh, long sd,
    const float* __restrict__ beta,
    const float* __restrict__ k_inv_rms,
    const float* __restrict__ k_norm_w,
    bf16* __restrict__ I_P_z,             // [BH,F,D,D]
    float* __restrict__ Bz,               // [BH,F,D]
    int B, int H, int F, int S, float k_scale) {
  const int bhf = blockIdx.x, bh = bhf / F, f = bhf % F, b = bh / H, h = bh % H;
  const int N = F * S, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  extern __shared__ char sm[];
  bf16*  K_s  = reinterpret_cast<bf16*>(sm);            // [ACHUNK,D] bf16 (K)
  bf16*  bK   = K_s + ACHUNK * D;                        // [ACHUNK,D] bf16 (beta*K)
  float* knw  = reinterpret_cast<float*>(bK + ACHUNK * D); // [D]
  float* Bacc = knw + D;                                 // [D]

  for (int i = tid; i < D; i += THREADS) { knw[i] = k_norm_w[h * D + i]; Bacc[i] = 0.f; }

  wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc_p[NCT];
  #pragma unroll
  for (int ct = 0; ct < NCT; ++ct) wmma::fill_fragment(acc_p[ct], 0.f);

  const bf16* qkv_k = qkv + (long)b * sb + 1 * s3 + (long)h * sh;
  const float* beta_bhf = beta + (long)bh * N + (long)f * S;
  const int n_base = f * S;
  __syncthreads();

  for (int s0 = 0; s0 < S; s0 += ACHUNK) {
    for (int idx = tid; idx < ACHUNK * D; idx += THREADS) {
      const int sl = idx >> 7, d = idx & 127, s = s0 + sl;
      float kvv = 0.f, be = 0.f;
      if (s < S) {
        const int n = n_base + s;
        float kn = __bfloat162float(qkv_k[(long)n * sn + (long)d * sd]) * k_inv_rms[b * N + n] * knw[d];
        kvv = (kn > 0.f ? kn : 0.f) * k_scale;
        be = beta_bhf[s];
      }
      K_s[idx] = __float2bfloat16(kvv);
      const float bk = be * kvv;
      bK[idx] = __float2bfloat16(bk);
      if (s < S) atomicAdd(&Bacc[d], bk);
    }
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < ASUB; ++ks) {
      wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, K_s + ks * WM * D + warp * WM, D);
      #pragma unroll
      for (int ct = 0; ct < NCT; ++ct) {
        wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::row_major> bp;
        wmma::load_matrix_sync(bp, bK + ks * WM * D + ct * WM, D);
        wmma::mma_sync(acc_p[ct], a_frag, bp, acc_p[ct]);
      }
    }
    __syncthreads();
  }

  bf16* IPz = I_P_z + (long)bhf * D * D;
  __shared__ float st[WARPS][WM * WM];
  #pragma unroll
  for (int ct = 0; ct < NCT; ++ct) {
    wmma::store_matrix_sync(st[warp], acc_p[ct], WM, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < WM * WM; i += 32) {
      const int d = warp * WM + (i >> 4), dp = ct * WM + (i & 15);
      IPz[(long)d * D + dp] = __float2bfloat16((d == dp ? 1.f : 0.f) - st[warp][i]);
    }
    __syncwarp();
  }
  for (int i = tid; i < D; i += THREADS) Bz[(long)bhf * D + i] = Bacc[i];
}

// ═══════════════════ CAM path (live model) — no prep, [B,H,D,N] ═══════════════════
// cam_scan_bidi_chunkwise: identity norm/RoPE, skip_relu, skip_z, num_only,
// output transposed to [B,H,D,N]. K_rot==K==raw k. Reads q/k/v directly (no
// packing), writes fp32 out directly (no permute). 16 warps; warp w owns
// (out=w>>3 {P_kv,A}, row-tile=w&7).
__global__ __launch_bounds__(ATH_KV) void cam_phase_a_kv_kernel(
    const float* __restrict__ k,   // [B,H,D,N]
    const float* __restrict__ v,   // [B,H,D,N]
    const float* __restrict__ beta, // [BH,F,S]
    bf16* __restrict__ I_P_kv, bf16* __restrict__ A,
    int B, int H, int F, int S) {
  const int bhf = blockIdx.x, bh = bhf / F, f = bhf % F;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int out_sel = warp >> 3, rt = warp & 7;
  const long Nd = (long)F * S;             // tokens per (b,h)
  const long kbase = (long)bh * D * Nd;    // k/v base for this bh

  extern __shared__ char sm[];
  bf16* Ksm  = reinterpret_cast<bf16*>(sm);   // [ACHUNK,D]
  bf16* bKsm = Ksm + ACHUNK * D;              // [ACHUNK,D]
  bf16* bVsm = bKsm + ACHUNK * D;             // [ACHUNK,D]

  wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[NCT];
  #pragma unroll
  for (int ct = 0; ct < NCT; ++ct) wmma::fill_fragment(acc[ct], 0.f);

  const float* beta_bhf = beta + (long)bhf * S;
  const int n_base = f * S;

  for (int s0 = 0; s0 < S; s0 += ACHUNK) {
    // stage K/V D-MAJOR [d][sl]: read along N (contiguous, coalesced — k/v are
    // [B,H,D,N]) and write contiguously. a_frag = K is then row_major; the wmma
    // transpose moves to the b-operand (col_major). Avoids the uncoalesced
    // strided global loads that were the L1TEX bottleneck (ncu).
    for (int idx = tid; idx < D * ACHUNK; idx += ATH_KV) {
      const int d = idx / ACHUNK, sl = idx - d * ACHUNK, s = s0 + sl;
      float kk = 0.f, bv = 0.f, bk = 0.f;
      if (s < S) {
        const long off = kbase + (long)d * Nd + (n_base + s);
        const float be = beta_bhf[s];
        kk = k[off]; bk = be * kk; bv = be * v[off];
      }
      Ksm[idx]  = __float2bfloat16(kk);   // Ksm[d*ACHUNK + sl]
      bKsm[idx] = __float2bfloat16(bk);
      bVsm[idx] = __float2bfloat16(bv);
    }
    __syncthreads();
    const bf16* bsrc = out_sel ? bVsm : bKsm;
    #pragma unroll
    for (int ks = 0; ks < ASUB; ++ks) {
      // a = K[rt-rows, ks-cols] row_major (d-major smem, ldm=ACHUNK)
      wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, Ksm + rt * WM * ACHUNK + ks * WM, ACHUNK);
      #pragma unroll
      for (int ct = 0; ct < NCT; ++ct) {
        // b = (betaK)^T : col_major from d-major betaK, ldm=ACHUNK
        wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::col_major> bfr;
        wmma::load_matrix_sync(bfr, bsrc + ct * WM * ACHUNK + ks * WM, ACHUNK);
        wmma::mma_sync(acc[ct], a_frag, bfr, acc[ct]);
      }
    }
    __syncthreads();
  }

  bf16* dst = (out_sel ? A : I_P_kv) + (long)bhf * D * D;
  __shared__ float st[AW_KV][WM * WM];
  #pragma unroll
  for (int ct = 0; ct < NCT; ++ct) {
    wmma::store_matrix_sync(st[warp], acc[ct], WM, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < WM * WM; i += 32) {
      const int d = rt * WM + (i >> 4), dp = ct * WM + (i & 15);
      float val = st[warp][i];
      if (out_sel == 0) val = (d == dp ? 1.f : 0.f) - val;
      dst[(long)d * D + dp] = __float2bfloat16(val);
    }
    __syncwarp();
  }
}

// num = Q @ M_hist, write transposed fp32 to [B,H,D,N]. grid (BH*F, S_TILES).
__global__ void cam_phase_c_kernel(
    const float* __restrict__ q,   // [B,H,D,N]
    const bf16*  __restrict__ M,   // [BH,F,D,D] bf16
    float* __restrict__ out,       // [B,H,D,N]
    int B, int H, int F, int S, int S_TILES) {
  const int bhf = blockIdx.x, s_tile = blockIdx.y, f = bhf % F;
  const int bh = bhf / F;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int s0 = s_tile * ROWS;
  const long Nd = (long)F * S;
  const long qbase = (long)bh * D * Nd;
  const int n_base = f * S;

  extern __shared__ char smem_raw[];
  bf16* Qsm = reinterpret_cast<bf16*>(smem_raw);   // [D, ROWS] d-major
  const bf16* M_f = M + (long)bhf * D * D;

  // stage Q D-MAJOR [i][r]: read q[i, n] along N (coalesced; q is [B,H,D,N]).
  // a_frag = Q is then col_major (a[s,i]=Q[i,s]); b_frag = M stays row_major.
  for (int idx = tid; idx < D * ROWS; idx += THREADS) {
    const int i = idx / ROWS, r = idx - i * ROWS, s = s0 + r;
    float qv = 0.f;
    if (s < S) qv = q[qbase + (long)i * Nd + (n_base + s)];
    Qsm[idx] = __float2bfloat16(qv);             // Qsm[i*ROWS + r]
  }
  __syncthreads();

  const int ct = warp;
  __shared__ float tile_smem[WARPS][WM * WM];
  wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[ROWS / WM];
  #pragma unroll
  for (int rt = 0; rt < ROWS / WM; ++rt) wmma::fill_fragment(acc[rt], 0.f);
  #pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::row_major> b_frag;
    wmma::load_matrix_sync(b_frag, M_f + (kt * WM) * D + ct * WM, D);
    #pragma unroll
    for (int rt = 0; rt < ROWS / WM; ++rt) {
      wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, Qsm + (kt * WM) * ROWS + rt * WM, ROWS);
      wmma::mma_sync(acc[rt], a_frag, b_frag, acc[rt]);
    }
  }
  #pragma unroll
  for (int rt = 0; rt < ROWS / WM; ++rt) {
    // col-major store -> tile_smem[j*16 + s]; then consecutive lanes write
    // consecutive n (=s) for a fixed j -> coalesced output (out is [B,H,D,N]).
    wmma::store_matrix_sync(tile_smem[warp], acc[rt], WM, wmma::mem_col_major);
    __syncwarp();
    for (int i = lane; i < WM * WM; i += 32) {
      const int s = s0 + rt * WM + (i & 15);
      if (s < S) {
        const int j = ct * WM + (i >> 4);
        out[qbase + (long)j * Nd + (n_base + s)] = tile_smem[warp][i];
      }
    }
    __syncwarp();
  }
}

// cam Phase B: serial F scan, fwd+rev combined -> bf16 M_hist directly (no fp32
// roundtrip). skip_z. M state kept bf16 in smem. grid (BH,), 8 warps, warp w
// owns output row-tile w. M = g*(I-P_kv)@M + A.
__global__ void cam_phase_b_kernel(
    const bf16* __restrict__ I_P_kv,  // [BH,F,D,D]
    const bf16* __restrict__ A,       // [BH,F,D,D]
    const float* __restrict__ decay,  // [BH,F]
    bf16* __restrict__ M_hist,        // [BH,F,D,D]
    int BH, int F) {
  const int bh = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rt = warp;                      // row-tile (8 warps -> 8 row-tiles)
  extern __shared__ char smem_raw[];
  bf16* Mbf = reinterpret_cast<bf16*>(smem_raw);  // [D,D] running state
  __shared__ float st[WARPS][WM * WM];

  const bf16* IP_bh = I_P_kv + (long)bh * F * D * D;
  const bf16* A_bh  = A + (long)bh * F * D * D;
  const float* g_bh = decay + (long)bh * F;

  // helper macro replaced by inline: compute M_new for frame f into st/Mbf
  #define CAM_B_FRAME(f, WRITE_ADD, dstframe)                                   \
  {                                                                             \
    const bf16* IP_f = IP_bh + (long)(f) * D * D;                               \
    const bf16* A_f  = A_bh + (long)(f) * D * D;                                \
    const float g = g_bh[(f)];                                                  \
    wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[NCT];              \
    _Pragma("unroll") for (int ct = 0; ct < NCT; ++ct) wmma::fill_fragment(acc[ct], 0.f); \
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {                         \
      wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::row_major> a_frag;  \
      wmma::load_matrix_sync(a_frag, IP_f + (long)(rt * WM) * D + kt * WM, D);   \
      _Pragma("unroll") for (int ct = 0; ct < NCT; ++ct) {                       \
        wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::row_major> b_frag; \
        wmma::load_matrix_sync(b_frag, Mbf + (long)(kt * WM) * D + ct * WM, D);   \
        wmma::mma_sync(acc[ct], a_frag, b_frag, acc[ct]);                         \
      }                                                                          \
    }                                                                            \
    __syncthreads(); /* all warps done reading Mbf before overwrite */          \
    _Pragma("unroll") for (int ct = 0; ct < NCT; ++ct) {                         \
      wmma::store_matrix_sync(st[warp], acc[ct], WM, wmma::mem_row_major);        \
      __syncwarp();                                                              \
      for (int i = lane; i < WM * WM; i += 32) {                                 \
        const int d = rt * WM + (i >> 4), dp = ct * WM + (i & 15);               \
        float mnew = g * st[warp][i] + __bfloat162float(A_f[(long)d * D + dp]);   \
        Mbf[(long)d * D + dp] = __float2bfloat16(mnew);                          \
        bf16* hp = M_hist + ((long)bh * F + (dstframe)) * D * D + (long)d * D + dp; \
        if (WRITE_ADD) *hp = __float2bfloat16(__bfloat162float(*hp) + mnew);      \
        else *hp = __float2bfloat16(mnew);                                       \
      }                                                                          \
      __syncwarp();                                                              \
    }                                                                            \
    __syncthreads();                                                             \
  }

  // forward
  for (int i = tid; i < D * D; i += THREADS) Mbf[i] = __float2bfloat16(0.f);
  __syncthreads();
  for (int f = 0; f < F; ++f) CAM_B_FRAME(f, false, f);
  // reverse (combined): for f_src=F-1..1, add into M_hist[f_src-1]
  for (int i = tid; i < D * D; i += THREADS) Mbf[i] = __float2bfloat16(0.f);
  __syncthreads();
  for (int f = F - 1; f >= 1; --f) CAM_B_FRAME(f, true, f - 1);
  #undef CAM_B_FRAME
}

}  // namespace

void cam_phase_b(torch::Tensor I_P_kv, torch::Tensor A, torch::Tensor decay,
                 torch::Tensor M_hist, int F) {
  const int BH = I_P_kv.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t smem = sizeof(bf16) * D * D;
  cam_phase_b_kernel<<<BH, THREADS, smem, stream>>>(
      reinterpret_cast<const bf16*>(I_P_kv.data_ptr()),
      reinterpret_cast<const bf16*>(A.data_ptr()),
      decay.data_ptr<float>(), reinterpret_cast<bf16*>(M_hist.data_ptr()), BH, F);
}

void cam_phase_a_kv(torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor I_P_kv, torch::Tensor A, int F, int S) {
  const int B = k.size(0), H = k.size(1), BH = B * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t smem = sizeof(bf16) * (3 * ACHUNK * D);
  cam_phase_a_kv_kernel<<<BH * F, ATH_KV, smem, stream>>>(
      k.data_ptr<float>(), v.data_ptr<float>(), beta.data_ptr<float>(),
      reinterpret_cast<bf16*>(I_P_kv.data_ptr()), reinterpret_cast<bf16*>(A.data_ptr()),
      B, H, F, S);
}

void cam_phase_c(torch::Tensor q, torch::Tensor M, torch::Tensor out, int F, int S) {
  const int B = q.size(0), H = q.size(1), BH = B * H;
  const int S_TILES = (S + ROWS - 1) / ROWS;
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t smem = sizeof(bf16) * (ROWS * D);
  dim3 grid(BH * F, S_TILES);
  cam_phase_c_kernel<<<grid, THREADS, smem, stream>>>(
      q.data_ptr<float>(), reinterpret_cast<const bf16*>(M.data_ptr()),
      out.data_ptr<float>(), B, H, F, S, S_TILES);
}

void phase_a_kv(torch::Tensor qkv, torch::Tensor beta, torch::Tensor k_inv_rms,
                torch::Tensor k_norm_w, torch::Tensor rope_cos, torch::Tensor rope_sin,
                torch::Tensor I_P_kv, torch::Tensor A, int F, int S, double k_scale) {
  const int B = qkv.size(0), H = qkv.size(3), BH = I_P_kv.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t smem = sizeof(float) * (D + 2 * ACHUNK) + sizeof(bf16) * (7 * ACHUNK * D);
  static bool s_kv = false;
  if (!s_kv) { cudaFuncSetAttribute(phase_a_kv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); s_kv = true; }
  phase_a_kv_kernel<<<BH * F, ATH_KV, smem, stream>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()),
      qkv.stride(0), qkv.stride(1), qkv.stride(2), qkv.stride(3), qkv.stride(4),
      beta.data_ptr<float>(), k_inv_rms.data_ptr<float>(), k_norm_w.data_ptr<float>(),
      rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
      reinterpret_cast<bf16*>(I_P_kv.data_ptr()), reinterpret_cast<bf16*>(A.data_ptr()),
      B, H, F, S, (float)k_scale);
}

void phase_a_z(torch::Tensor qkv, torch::Tensor beta, torch::Tensor k_inv_rms,
               torch::Tensor k_norm_w, torch::Tensor I_P_z, torch::Tensor Bz,
               int F, int S, double k_scale) {
  const int B = qkv.size(0), H = qkv.size(3), BH = I_P_z.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t smem = sizeof(bf16) * (2 * ACHUNK * D) + sizeof(float) * (2 * D);
  phase_a_z_kernel<<<BH * F, THREADS, smem, stream>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()),
      qkv.stride(0), qkv.stride(1), qkv.stride(2), qkv.stride(3), qkv.stride(4),
      beta.data_ptr<float>(), k_inv_rms.data_ptr<float>(), k_norm_w.data_ptr<float>(),
      reinterpret_cast<bf16*>(I_P_z.data_ptr()), Bz.data_ptr<float>(),
      B, H, F, S, (float)k_scale);
}

void phase_c_fused(torch::Tensor qkv, torch::Tensor q_inv_rms, torch::Tensor q_norm_w,
                   torch::Tensor rope_cos, torch::Tensor rope_sin,
                   torch::Tensor M, torch::Tensor z, torch::Tensor out,
                   int F, int S, double eps) {
  const int B = qkv.size(0);
  const int H = qkv.size(3);
  const int BH = M.size(0);
  const int S_TILES = (S + ROWS - 1) / ROWS;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(BH * F, S_TILES);
  size_t smem = sizeof(float) * (D + D + ROWS) + sizeof(bf16) * (ROWS * D);
  phase_c_fused_kernel<<<grid, THREADS, smem, stream>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()),
      qkv.stride(0), qkv.stride(1), qkv.stride(2), qkv.stride(3), qkv.stride(4),
      q_inv_rms.data_ptr<float>(), q_norm_w.data_ptr<float>(),
      rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
      reinterpret_cast<const bf16*>(M.data_ptr()), z.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr()),
      B, H, F, S, S_TILES, (float)eps);
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void phase_c_fused(torch::Tensor qkv, torch::Tensor q_inv_rms, torch::Tensor q_norm_w,
                   torch::Tensor rope_cos, torch::Tensor rope_sin,
                   torch::Tensor M, torch::Tensor z, torch::Tensor out,
                   int F, int S, double eps);
void phase_a_kv(torch::Tensor qkv, torch::Tensor beta, torch::Tensor k_inv_rms,
                torch::Tensor k_norm_w, torch::Tensor rope_cos, torch::Tensor rope_sin,
                torch::Tensor I_P_kv, torch::Tensor A, int F, int S, double k_scale);
void phase_a_z(torch::Tensor qkv, torch::Tensor beta, torch::Tensor k_inv_rms,
               torch::Tensor k_norm_w, torch::Tensor I_P_z, torch::Tensor Bz,
               int F, int S, double k_scale);
void cam_phase_a_kv(torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor I_P_kv, torch::Tensor A, int F, int S);
void cam_phase_b(torch::Tensor I_P_kv, torch::Tensor A, torch::Tensor decay,
                 torch::Tensor M_hist, int F);
void cam_phase_c(torch::Tensor q, torch::Tensor M, torch::Tensor out, int F, int S);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("phase_c_fused", &phase_c_fused);
  m.def("phase_a_kv", &phase_a_kv);
  m.def("phase_a_z", &phase_a_z);
  m.def("cam_phase_a_kv", &cam_phase_a_kv);
  m.def("cam_phase_b", &cam_phase_b);
  m.def("cam_phase_c", &cam_phase_c);
}
"""


def _system_cuda_home():
    """Locate a CUDA toolkit with *usable* headers (cuda_runtime.h, nv/target,
    crt/host_config.h) and an nvcc, returning ``(home, include_dir)``.

    Some conda nvcc installs (and the original svideo box) ship nvcc without
    co-located headers; a complete toolkit may instead live under
    ``/usr/local/cuda-*`` (``include/`` layout) or in a conda env
    (``targets/<arch>/include`` layout). We probe both, honoring an explicit
    ``$CUDA_HOME``/``$CONDA_PREFIX`` first so the same kernel builds on any
    cluster (CW H100, GB200, RTX 5090)."""
    import sys as _sys

    env = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), os.environ.get("CONDA_PREFIX"), _sys.prefix]
    cands = [c for c in env if c] + ["/usr/local/cuda-12.9", "/usr/local/cuda", "/usr/local/cuda-12"]
    for c in cands:
        if not c or not os.path.isfile(os.path.join(c, "bin", "nvcc")):
            continue
        # Accept either the classic include/ layout or conda's targets/<arch>/include.
        from glob import glob

        inc_dirs = [os.path.join(c, "include")] + glob(os.path.join(c, "targets", "*", "include"))
        for inc in inc_dirs:
            if os.path.isfile(os.path.join(inc, "cuda_runtime.h")) and os.path.isfile(
                os.path.join(inc, "nv", "target")
            ):
                return c, inc
    return None, None


def build(name="cw_cuda_ext_v1", extra_cuda=None):
    global _EXT
    if _EXT is not None:
        return _EXT
    # Compile for the GPU actually present (sm_90 H100, sm_100 GB200, sm_120
    # 5090) so the default-on path produces a runnable cubin everywhere; an
    # explicit TORCH_CUDA_ARCH_LIST still wins. The kernels are arch-portable
    # (wmma + cp.async, sm_80+; no clusters/TMA).
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            _maj, _min = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_maj}.{_min}"
        except Exception:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    import torch.utils.cpp_extension as C

    home, inc = _system_cuda_home()
    incs = []
    if home:
        os.environ["CUDA_HOME"] = home
        os.environ["PATH"] = os.path.join(home, "bin") + os.pathsep + os.environ.get("PATH", "")
        C.CUDA_HOME = home  # override torch's import-time cached value
        incs = [inc]
    _EXT = load_inline(
        name=name,
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"] + (extra_cuda or []),
        extra_include_paths=incs,
        with_cuda=True,
        verbose=True,
    )
    return _EXT


def cuda_phase_c_fused(inp, M_hist, z_hist, eps=1e-6):
    ext = build()
    qkv = inp["qkv"]
    B, N, _, H, Dd = qkv.shape
    out = torch.empty(B, N, H, Dd, device=qkv.device, dtype=qkv.dtype)
    M_bf = M_hist.to(torch.bfloat16).contiguous()
    ext.phase_c_fused(
        qkv.contiguous(),
        inp["q_inv_rms"].contiguous(),
        inp["q_norm_w"].contiguous(),
        inp["rope_cos"].contiguous(),
        inp["rope_sin"].contiguous(),
        M_bf,
        z_hist.contiguous(),
        out,
        inp["F"],
        inp["S"],
        eps,
    )
    return out


def cuda_phase_a(inp, k_scale=1.0):
    """CUDA Phase A -> (I_P_kv, A, I_P_z, B_z), shapes/dtypes matching Triton."""
    ext = build()
    qkv = inp["qkv"].contiguous()
    B, N, _, H, Dd = qkv.shape
    BH, F, S = B * H, inp["F"], inp["S"]
    dev = qkv.device
    BD = 1 << (Dd - 1).bit_length()  # next pow2 (==128 for Dd<=128)
    I_P_kv = torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16)
    A = torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16)
    I_P_z = torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16)
    B_z = torch.empty(BH, F, BD, device=dev, dtype=torch.float32)
    beta = inp["beta"].contiguous().float()
    kir = inp["k_inv_rms"].contiguous().float()
    knw = inp["k_norm_w"].contiguous().float()
    rc, rs = inp["rope_cos"].contiguous(), inp["rope_sin"].contiguous()
    ext.phase_a_kv(qkv, beta, kir, knw, rc, rs, I_P_kv, A, F, S, k_scale)
    ext.phase_a_z(qkv, beta, kir, knw, I_P_z, B_z, F, S, k_scale)
    return I_P_kv, A, I_P_z, B_z


def cam_scan_bidi_chunkwise_cuda(q, k, v, beta, decay, *, dot_precision=None):
    """Drop-in for fused_gdn_chunkwise.cam_scan_bidi_chunkwise.

    CUDA fast path when head_dim==128 & fp32 contiguous inputs (the deployed
    sana_wm config: attention_head_dim=128). Falls back to the Triton entry
    otherwise so this is always safe to substitute.
    """
    B, H, D, N = q.shape
    F = beta.shape[2]
    if (
        not _CUDA_CAM_DISABLED
        and D == 128
        and q.dtype == torch.float32
        and N % F == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    ):
        try:
            return run_cuda_cam(q, k, v, beta, decay)
        except Exception as e:  # missing toolkit / unexpected shape -> Triton
            _disable_cuda_cam(e)
    from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise

    return cam_scan_bidi_chunkwise(q, k, v, beta, decay, dot_precision=dot_precision)


_CAM_BUFS = {}
# Triton phase_b (fp32 state) is fast (0.04ms) & accurate. The CUDA cam_phase_b
# kept bf16 M-state across frames -> drift (max_rel 0.14) and was slow (serial,
# BH CTAs) -> disabled by default. fp32-state CUDA phase B is future work.
_CAM_PHASE_B = os.environ.get("CAM_PHASE_B", "triton")  # "triton" (default) or "cuda"

# Once the CUDA path fails to build/run (e.g. no nvcc / no CUDA toolkit on this
# box), latch it off so the drop-ins fall back to Triton silently for the rest
# of the process instead of retrying the (failing, slow) compile every call.
_CUDA_CAM_DISABLED = False


def _disable_cuda_cam(err):
    global _CUDA_CAM_DISABLED
    if not _CUDA_CAM_DISABLED:
        _CUDA_CAM_DISABLED = True
        print(f"[SANA_GDN_CUDA] CUDA cam kernels unavailable ({err}); using Triton")


def cam_scan_chunkwise_cuda(
    q, k, v, beta, decay, *, reverse=False, init_state=None, save_final_state=False, dot_precision=None
):
    """Drop-in for fused_gdn_chunkwise.cam_scan_chunkwise (the STREAMING/cached
    causal cam scan). CUDA cam_phase_a + cam_phase_c (read q/k/v [B,H,D,N] direct,
    write transposed fp32 direct), Triton phase_b for the state-cached single
    direction. bf16 GEMM. Falls back to Triton for unsupported shapes/dtype."""
    B, H, D, N = q.shape
    F = beta.shape[2]
    S = N // F
    use_cuda = (
        not _CUDA_CAM_DISABLED
        and D == 128
        and q.dtype == torch.float32
        and N % F == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    )
    if use_cuda:
        try:
            ext = build()
            BH, BD, dev = B * H, 128, q.device
            I_P_kv = torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16)
            A = torch.empty_like(I_P_kv)
            ext.cam_phase_a_kv(k.contiguous(), v.contiguous(), beta.contiguous().float(), I_P_kv, A, F, S)
            dummy = torch.empty(1, device=dev, dtype=torch.float32)
            init_kv = init_state.to(torch.float32).contiguous() if init_state is not None else None
            init_z = torch.zeros(BH, BD, device=dev, dtype=torch.float32) if init_state is not None else None
            direction = 2 if reverse else 1
            pb = phase_b_triton(
                I_P_kv,
                A,
                dummy,
                dummy,
                decay.contiguous(),
                F=F,
                dot_precision=0,
                direction=direction,
                init_state_kv=init_kv,
                init_state_z=init_z,
                return_final_state=save_final_state,
                skip_z=True,
            )
            if save_final_state:
                M_fwd, _zf, M_rev, _zr, final_kv, _fz = pb
            else:
                M_fwd, _zf, M_rev, _zr = pb
            M_use = M_rev if reverse else M_fwd
            M_bf = M_use.to(torch.bfloat16).contiguous()
            out = torch.empty(B, H, D, N, device=dev, dtype=torch.float32)
            ext.cam_phase_c(q.contiguous(), M_bf, out, F, S)
            return (out, final_kv) if save_final_state else out
        except Exception as e:  # missing toolkit / unexpected shape -> Triton
            _disable_cuda_cam(e)
    from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_chunkwise

    return cam_scan_chunkwise(
        q,
        k,
        v,
        beta,
        decay,
        reverse=reverse,
        init_state=init_state,
        save_final_state=save_final_state,
        dot_precision=dot_precision,
    )


def run_cuda_cam(q, k, v, beta, decay, reuse_buffers=True):
    """CUDA cam path: reads q/k/v [B,H,D,N] directly, writes fp32 [B,H,D,N]
    directly (no packing, no transpose). Phase B via Triton (skip_z).

    reuse_buffers=True caches the I_P_kv/A/M_bf/out scratch by shape — a
    streaming deployment reuses these every step, so per-call torch.empty/.to
    (which otherwise hits cudaMalloc sync under memory pressure) is amortised."""
    ext = build()
    B, H, D, N = q.shape
    BH, F = B * H, beta.shape[2]
    S = N // F
    dev = q.device
    BD = 1 << (D - 1).bit_length()
    k = k.contiguous()
    v = v.contiguous()
    q = q.contiguous()
    beta_f = beta.contiguous().float()
    key = (B, H, D, N, F, dev)
    buf = _CAM_BUFS.get(key) if reuse_buffers else None
    if buf is None:
        buf = dict(
            I_P_kv=torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16),
            A=torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16),
            M_bf=torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16),
            out=torch.empty(B, H, D, N, device=dev, dtype=torch.float32),
        )
        if reuse_buffers:
            _CAM_BUFS[key] = buf
    I_P_kv, A, M_bf, out = buf["I_P_kv"], buf["A"], buf["M_bf"], buf["out"]
    ext.cam_phase_a_kv(k, v, beta_f, I_P_kv, A, F, S)
    if _CAM_PHASE_B == "cuda":
        ext.cam_phase_b(I_P_kv, A, decay.contiguous().float(), M_bf, F)
    else:
        dummy = torch.empty(1, device=dev, dtype=torch.float32)
        M_hist, _, _, _ = phase_b_triton(
            I_P_kv,
            A,
            dummy,
            dummy,
            decay.contiguous(),
            F=F,
            dot_precision=0,
            direction=0,
            combined_history=True,
            skip_z=True,
        )
        M_bf.copy_(M_hist)
    ext.cam_phase_c(q, M_bf, out, F, S)
    return out


def run_cuda(inp, mode="ac", dot_precision=0, eps=1e-6):
    qkv = inp["qkv"]
    F, S = inp["F"], inp["S"]
    if mode in ("ac", "abc_a"):
        I_P_kv, A, I_P_z, B_z = cuda_phase_a(inp)
    else:
        I_P_kv, A, I_P_z, B_z = phase_a(
            qkv,
            inp["beta"],
            inp["q_inv_rms"],
            inp["k_inv_rms"],
            inp["q_norm_w"],
            inp["k_norm_w"],
            inp["rope_cos"],
            inp["rope_sin"],
            F=F,
            S=S,
            dot_precision=dot_precision,
        )
    M_hist, z_hist, _, _ = phase_b_triton(
        I_P_kv, A, I_P_z, B_z, inp["decay"], F=F, dot_precision=dot_precision, direction=0, combined_history=True
    )
    return cuda_phase_c_fused(inp, M_hist, z_hist, eps=eps)
