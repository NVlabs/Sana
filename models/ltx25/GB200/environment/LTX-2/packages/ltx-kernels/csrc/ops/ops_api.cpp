
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>


at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs,  bool out_16bit);

at::Tensor fp6_pack(at::Tensor &x);
at::Tensor fp6_unpack(at::Tensor &x, int64_t original_n);

at::Tensor rms_norm_split_rope(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_rope", &rms_norm_rope,
          "fused  norm + rope + cvt");
    m.def("fp6_pack", &fp6_pack,
          "Pack 8-bit to 6-bit by dropping e_1 and e_2 bits");
    m.def("fp6_unpack", &fp6_unpack,
          "Unpack 6-bit to 8-bit (with e_1 and e_2 set to 0)");
    m.def("rms_norm_split_rope", &rms_norm_split_rope,
          "RMS norm + split RoPE with optional FP8 output");
}
