/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Host entry point for the fused RMS-norm + RoPE kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "fast_hadamard_transform.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

template<typename input_t, typename output_t, bool norm_affine>
void rms_norm_rope_cuda(NormRopeHadamardParamsBase &params, cudaStream_t stream);

void set_norm_rope_hadamard_params(NormRopeHadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor cos_freqs,
                         const at::Tensor sin_freqs,
                         const at::Tensor weights,
                         const at::Tensor out,

                         bool norm_affine,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.cos_freq_ptr = cos_freqs.data_ptr();
    params.sin_freq_ptr = sin_freqs.data_ptr();
    if (norm_affine){
        params.weights_ptr = weights.data_ptr();
    } else {
        params.weights_ptr = nullptr;
    }
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.cos_freq_batch_stride = cos_freqs.stride(0);
    params.sin_freq_batch_stride = sin_freqs.stride(0);

    params.scale = scale;

}

at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs, bool out_16bit) {
    auto input_type = x.scalar_type();
    float scale = 1.0f; // :D
    TORCH_CHECK(input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    cos_freqs = cos_freqs.reshape({-1, dim_og});
    sin_freqs = sin_freqs.reshape({-1, dim_og});
    at::Tensor weights;
    bool norm_affine = false;
    if(weights_.has_value()){
        weights = weights_.value();
        norm_affine = true;
    }
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    at::Tensor out;
    if (out_16bit){
        out = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    } else {
        out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn));
    }
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    NormRopeHadamardParamsBase params;
    set_norm_rope_hadamard_params(params, batch_size, dim, 1, x, cos_freqs, sin_freqs, weights, out, norm_affine, scale);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    if (norm_affine){
        if (out_16bit){
            rms_norm_rope_cuda<at::BFloat16, at::BFloat16, true>(params, stream);
        } else {
            rms_norm_rope_cuda<at::BFloat16, at::Float8_e4m3fn, true>(params, stream);
        }

    } else {
        if (out_16bit){
            rms_norm_rope_cuda<at::BFloat16, at::BFloat16, false>(params, stream);
        } else {
            rms_norm_rope_cuda<at::BFloat16, at::Float8_e4m3fn, false>(params, stream);
        }
    }
    return out.reshape(shapes_og);
}
