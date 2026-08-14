#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#include "kernels/geforce/static_switch.h"

namespace sm89 {
template<bool use_fast_accum>
void fp8_bias_gemm_cuda(void* Aptr, void* SFA, void* Bptr, void* SFB, void* bias_ptr, void* out, int M, int N, int K, cudaStream_t stream);
}

namespace blockwise {
static void sm89_fp8_gemm_1d2d_bias(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const torch::Tensor& bias,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const bool use_fast_accum) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (use_fast_accum) {
        sm89::fp8_bias_gemm_cuda<true>(
                a.data_ptr(), sfa.data_ptr(),
                b.data_ptr(), sfb.data_ptr(),
                bias.data_ptr(), d.data_ptr(),
                m, n, k, stream);
    } else {
        sm89::fp8_bias_gemm_cuda<false>(
                a.data_ptr(), sfa.data_ptr(),
                b.data_ptr(), sfb.data_ptr(),
                bias.data_ptr(), d.data_ptr(),
                m, n, k, stream);
    }
}
}