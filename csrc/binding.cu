#include "torch_api/forward.h"

using namespace dual_gemm::torchapi;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemm_forward", &dual_gemm_forward,
        "Fused Dual-GEMM + SiLU*Mul (x[M,K], w1[K,N], w2[K,N], storeGEMM1?, "
        "storeGEMM2?, dtype=optional)");
  m.def("dual_gemm_forward_infer", &dual_gemm_forward_infer,
        "Fused Dual-GEMM (dtype inferred from x)");

  m.def("dual_gemm_batched_forward", &dual_gemm_batched_forward,
        "Batched fused Dual-GEMM (x[B,M,K], w1[B,K,N], w2[B,K,N], storeGEMM1?, "
        "storeGEMM2?,"
        "dtype=optional)");
  m.def("dual_gemm_batched_forward_infer", &dual_gemm_batched_forward_infer,
        "Batched fused Dual-GEMM (dtype inferred from x)");

  m.def("dual_gemm_broadcast_forward", &dual_gemm_broadcast_forward,
        "Batched fused Dual-GEMM with B1 broadcast (x[B,M,K], w1[B,K,N], "
        "storeGEMM1?, storeGEMM2?,"
        "w2[K,N], dtype=optional)");
  m.def("dual_gemm_broadcast_forward_infer", &dual_gemm_broadcast_forward_infer,
        "Batched fused Dual-GEMM with B1 broadcast (dtype inferred from x)");
}
