#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

// Forward declaration of CUDA kernel template
template<typename out_t>
void rms_norm_split_rope_cuda(
    void* x,
    void* sin_freqs,
    void* cos_freqs,
    void* weights,
    int b,
    int s,
    int n,
    int h,
    long cos_sb, long cos_sn, long cos_ss,
    long sin_sb, long sin_sn, long sin_ss,
    void* out,
    cudaStream_t stream
);

at::Tensor rms_norm_split_rope(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8
) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "Input must be BFloat16");
    TORCH_CHECK(sin_freqs.scalar_type() == at::ScalarType::BFloat16, "sin_freqs must be BFloat16");
    TORCH_CHECK(cos_freqs.scalar_type() == at::ScalarType::BFloat16, "cos_freqs must be BFloat16");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(sin_freqs.is_cuda(), "sin_freqs must be on CUDA");
    TORCH_CHECK(cos_freqs.is_cuda(), "cos_freqs must be on CUDA");

    // Get dimensions
    // x: [b, s, h]
    // cos, sin: [b, n, s, d] where n*d = h/2
    int b = x.size(0);
    int s = x.size(1);
    int h = x.size(2);

    TORCH_CHECK(cos_freqs.dim() == 4, "cos_freqs must be 4D");
    TORCH_CHECK(sin_freqs.dim() == 4, "sin_freqs must be 4D");

    int n = cos_freqs.size(1);
    int d = h / n;

    
    // Require a contiguous innermost (d/2) dim for the vectorized int4 freq load,
    // but keep the outer (b, n, s) strides: apply_split_rotary_emb hands us a
    // swapaxes view (logical [b, n, s, d/2], physical [b, s, n, d/2]) whose inner
    // stride is already 1, so this never copies it. The strides are forwarded to
    // the kernel so the read is correct regardless of the physical layout.
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    if (cos_freqs.stride(-1) != 1) { cos_freqs = cos_freqs.contiguous(); }
    if (sin_freqs.stride(-1) != 1) { sin_freqs = sin_freqs.contiguous(); }

    long cos_sb = cos_freqs.stride(0), cos_sn = cos_freqs.stride(1), cos_ss = cos_freqs.stride(2);
    long sin_sb = sin_freqs.stride(0), sin_sn = sin_freqs.stride(1), sin_ss = sin_freqs.stride(2);

    // Create output tensor
    at::Tensor out;
    if (out_fp8) {
        out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn));
    } else {
        out = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    }
    
    // Setup CUDA
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    if (out_fp8) {
        rms_norm_split_rope_cuda<at::Float8_e4m3fn>(
            x.data_ptr(),
            sin_freqs.data_ptr(),
            cos_freqs.data_ptr(),
            weights.data_ptr(),  // weights (optional, not used yet)
            b,
            s,
            n,
            h,
            cos_sb, cos_sn, cos_ss,
            sin_sb, sin_sn, sin_ss,
            (void*)out.data_ptr(),
            stream
        );
    } else {
        rms_norm_split_rope_cuda<at::BFloat16>(
            x.data_ptr(),
            sin_freqs.data_ptr(),
            cos_freqs.data_ptr(),
            weights.data_ptr(),  // weights (optional, not used yet)
            b,
            s,
            n,
            h,
            cos_sb, cos_sn, cos_ss,
            sin_sb, sin_sn, sin_ss,
            (void*)out.data_ptr(),
            stream
        );
    }

    return out;
}
