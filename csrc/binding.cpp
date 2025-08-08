#include <torch/extension.h>

void dual_gemm_silu_hadamard_launcher(
    at::Tensor x, at::Tensor w1, at::Tensor w2, at::Tensor out);

torch::Tensor dual_gemm_forward(torch::Tensor x,
                                torch::Tensor w1,
                                torch::Tensor w2) {
  auto out = at::empty({x.size(0), w1.size(1)}, x.options());
  dual_gemm_silu_hadamard_launcher(x.contiguous(), w1.contiguous(), w2.contiguous(), out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemm_forward", &dual_gemm_forward, "Dual GEMM SiLU Hadamard (CUDA)");
}

