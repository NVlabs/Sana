#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#ifdef __SM90__
#include "sm90_fp8_gemm_1d2d_bias.hpp"
#endif

#include "sm89_fp8_gemm_1d2d.hpp"

namespace blockwise{
template <int N>
static auto get_shape(const torch::Tensor& t) {
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

#ifdef __SM90__
static void fp8_gemm_nt_sm90(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d,
                        const std::optional<torch::Tensor>& bias,
                        const std::optional<torch::Tensor>& c, const int num_sms) {

    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    // The SM90 kernel always adds bias; synthesize a zero bias when the layer is
    // bias-less (e.g. the no-bias video FFN of v3 checkpoints), mirroring SM89 below.
    torch::Tensor bias_tensor = bias.has_value()
        ? bias.value()
        : torch::zeros({n}, d.options().dtype(torch::kFloat32));
    sm90_fp8_gemm_1d2d_bias(a.first, a.second, b.first, b.second, bias_tensor, c, d, m, n, k, num_sms);
}
#endif

static void fp8_gemm_nt_sm89(const std::pair<torch::Tensor, torch::Tensor>& a,
                             const std::pair<torch::Tensor, torch::Tensor>& b,
                             const torch::Tensor& d,
                             const std::optional<torch::Tensor>& bias,
                             const bool use_fast_accum = true) {

    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [n, k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);

    // The SM89 kernel always adds bias; synthesize a zero bias when the layer is
    // bias-less so we add 0 rather than uninitialized memory (mirrors SM90 above).
    torch::Tensor bias_tensor = bias.has_value()
        ? bias.value()
        : torch::zeros({n}, d.options().dtype(torch::kFloat32));

    blockwise::sm89_fp8_gemm_1d2d_bias(
        a.first, a.second,    // a data, sfa scales
        b.first, b.second,    // b data, sfb scales
        bias_tensor,          // bias (or empty tensor)
        d,                    // output
        m, n, k,
        use_fast_accum);      // pass through accumulation mode
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
#ifdef __SM90__
    m.def("fp8_gemm_nt_sm90", &fp8_gemm_nt_sm90,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("bias") = std::nullopt,
          py::arg("c") = std::nullopt,
          py::arg("num_sms") = 132
    );
#endif
    m.def("fp8_gemm_nt_sm89", &fp8_gemm_nt_sm89,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("bias") = std::nullopt,
          py::arg("use_fast_accum") = true
    );
}
};






