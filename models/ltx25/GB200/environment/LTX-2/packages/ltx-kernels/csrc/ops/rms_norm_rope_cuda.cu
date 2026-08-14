/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// #pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include "static_switch.h"


template<int kNThreads_, int kLogN_, typename input_t_, typename output_t_, bool norm_affine_>
struct norm_rope_kernel_traits {
    using input_t = input_t_;
    using output_t = output_t_;

    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static constexpr int OutkNBytes = sizeof(output_t);
    
    static constexpr bool norm_affine = norm_affine_;
    
    static_assert(kNBytes == 1 || kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : kNBytes == 2 ? 8 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using vec_t_out = typename BytesToType<OutkNBytes * kNElts>::Type;

    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};


template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void norm_rope_cvt_kernel(NormRopeHadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    constexpr bool norm_affine = Ktraits::norm_affine;

    using input_t = typename Ktraits::input_t;
    using output_t = typename Ktraits::output_t;
    using vec_t = typename Ktraits::vec_t;
    using out_vec_t = typename Ktraits::vec_t_out;
    using weights_t = typename Ktraits::input_t;
    using freqs_t = typename Ktraits::input_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    // Shared memory.
    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;
    weights_t *weights = norm_affine ? reinterpret_cast<weights_t*>(params.weights_ptr) : nullptr;

    float x_vals[kNChunks][kNElts];
    float weights_vals[kNChunks][kNElts];

    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

    //RMS Norm START
    float thread_squared_sum = 0.0f;
    #pragma unroll
    for (size_t c = 0; c < kNChunks; c++)
    {
        #pragma unroll
        for (size_t i = 0; i < kNElts; i++)
        {
            thread_squared_sum += x_vals[c][i] * x_vals[c][i]; 
        }
        
    }
    SumOp<float> sum_op;
    float warp_sum = Allreduce<32>::run(thread_squared_sum, sum_op);
    float *smem_sum = reinterpret_cast<float*>(smem_);
    if(threadIdx.x % 32 == 0){
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    float norm = 0.0f;
     #pragma unroll  
    for (size_t i = 0; i < kNWarps; i++)
    {
        norm += smem_sum[i];
    }
    
    norm *= 1.0f/params.dim;
    norm = rsqrtf(norm);
    if constexpr (norm_affine){
        load_input<kNChunks, kNElts, weights_t>(weights, weights_vals, params.dim);
    }
    #pragma unroll
    for (size_t c = 0; c < kNChunks; c++)
    {
        #pragma unroll
        for (size_t i = 0; i < kNElts; i++)
        {
            if constexpr (norm_affine){
                x_vals[c][i] *= (norm * weights_vals[c][i]);
            } else {
                x_vals[c][i] *= norm;
            }
        }
        
    }
    //RMS NORM END
    
    //ROPE START
    float sin_freqs_vals[kNChunks][kNElts];
    float cos_freqs_vals[kNChunks][kNElts];

    freqs_t *cos_freqs = reinterpret_cast<freqs_t*>(params.cos_freq_ptr) + batch_id * params.cos_freq_batch_stride;
    freqs_t *sin_freqs = reinterpret_cast<freqs_t*>(params.sin_freq_ptr) + batch_id * params.sin_freq_batch_stride;

    load_input<kNChunks, kNElts, freqs_t>(cos_freqs, cos_freqs_vals, params.dim);
    load_input<kNChunks, kNElts, freqs_t>(sin_freqs, sin_freqs_vals, params.dim);

    #pragma unroll
    for (size_t c = 0; c < kNChunks; c++)
    {
        #pragma unroll
        for (size_t i = 0; i < kNElts; i+=2)
        {
            float x_1 = x_vals[c][i];
            float x_2 = x_vals[c][i+1];
            x_vals[c][i] = -x_2*sin_freqs_vals[c][i] + x_1*cos_freqs_vals[c][i];
            x_vals[c][i+1] = x_1*sin_freqs_vals[c][i+1] + x_2*cos_freqs_vals[c][i+1];
        }
    }
    //ROPE END
    
    store_output<kNChunks, kNElts, output_t, false>(out, x_vals, params.dim, params.scale);
}

template<int kNThreads, int kLogN, typename input_t, typename output_t, bool norm_affine>
void norm_rope_cvt_launch(NormRopeHadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = norm_rope_kernel_traits<kNThreads, kLogN, input_t, output_t, norm_affine>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &norm_rope_cvt_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename output_t, bool norm_affine>
void rms_norm_rope_cuda(NormRopeHadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 3) {
        norm_rope_cvt_launch<1, 3, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 4) {
        norm_rope_cvt_launch<2, 4, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 5) {
        norm_rope_cvt_launch<4, 5, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 6) {
        norm_rope_cvt_launch<8, 6, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 7) {
        norm_rope_cvt_launch<16, 7, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 8) {
        norm_rope_cvt_launch<32, 8, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 9) {
        norm_rope_cvt_launch<32, 9, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 10) {
        norm_rope_cvt_launch<128, 10, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 11) {
        norm_rope_cvt_launch<256, 11, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 12) {
        norm_rope_cvt_launch<256, 12, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 13) {
        norm_rope_cvt_launch<256, 13, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 14) {
        norm_rope_cvt_launch<256, 14, input_t, output_t, norm_affine>(params, stream);
    } else if (params.log_N == 15) {
        norm_rope_cvt_launch<256, 15, input_t, output_t, norm_affine>(params, stream);
    }
}


template void rms_norm_rope_cuda<at::BFloat16, at::Float8_e4m3fn, false>(NormRopeHadamardParamsBase &params, cudaStream_t stream);
template void rms_norm_rope_cuda<at::BFloat16, at::Float8_e4m3fn, true>(NormRopeHadamardParamsBase &params, cudaStream_t stream);

template void rms_norm_rope_cuda<at::BFloat16, at::BFloat16, false>(NormRopeHadamardParamsBase &params, cudaStream_t stream);
template void rms_norm_rope_cuda<at::BFloat16, at::BFloat16, true>(NormRopeHadamardParamsBase &params, cudaStream_t stream);

// template void fast_hadamard_transform_cuda<at::BFloat16, at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);

// template void fast_hadamard_transform_cuda<at::Float8_e4m3fn, at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);
// template void fast_hadamard_transform_cuda<at::Float8_e4m3fn, at::Float8_e4m3fn>(HadamardParamsBase &params, cudaStream_t stream);
