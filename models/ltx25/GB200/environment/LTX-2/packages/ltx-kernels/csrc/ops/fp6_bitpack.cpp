#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

void fp6_pack_cuda(
    at::Tensor& x,
    at::Tensor& out,
    cudaStream_t stream
);

void fp6_unpack_cuda(
    at::Tensor& x,
    at::Tensor& out,
    cudaStream_t stream
);

at::Tensor fp6_pack(at::Tensor &x) {
    // TORCH_CHECK(x.dtype() == torch::kUInt8, "Input tensor must be uint8");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D [m, n]");

    int64_t m = x.size(0);
    int64_t n = x.size(1);

    TORCH_CHECK(n % 8 == 0, "n must be divisible by 8, got ", n);

    // Output shape: [m, n*3/4] since 4 elements of 8-bit = 32 bits, 4 elements of 6-bit = 24 bits = 3 bytes
    int64_t n_packed = n * 3 / 4;

    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(x.device());

    at::Tensor out = torch::empty({m, n_packed}, options);

    at::cuda::CUDAGuard device_guard{x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    fp6_pack_cuda(x, out, stream);

    return out;
}

at::Tensor fp6_unpack(at::Tensor &x, int64_t original_n) {
    TORCH_CHECK(x.dtype() == torch::kUInt8, "Input tensor must be uint8");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D [m, n_packed]");
    TORCH_CHECK(original_n % 8 == 0, "original_n must be divisible by 8, got ", original_n);

    int64_t m = x.size(0);
    int64_t n_packed = x.size(1);

    TORCH_CHECK(n_packed == original_n * 3 / 4,
                "Packed size mismatch: expected ", original_n * 3 / 4, " got ", n_packed);

    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(x.device());

    at::Tensor out = torch::empty({m, original_n}, options);

    at::cuda::CUDAGuard device_guard{x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    fp6_unpack_cuda(x, out, stream);

    return out;
}
