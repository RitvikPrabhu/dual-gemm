#include "torch_api/forward.h"

using namespace dual_gemm::torchapi;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemm_forward", &dual_gemm_forward,
        "Fused Dual-GEMM + SiLU*Mul (x[M,K], w1[K,N], w2[K,N], storeGEMM1?, "
        "storeGEMM2? (dtype inferred from x)");

  m.def("dual_gemm_batched_forward", &dual_gemm_batched_forward,
        "Batched fused Dual-GEMM (x[B,M,K], w1[B,K,N], w2[B,K,N], storeGEMM1?, "
        "storeGEMM2? (dtype inferred from x)");

  m.def("dual_gemm_broadcast_forward", &dual_gemm_broadcast_forward,
        "Batched fused Dual-GEMM with B1 broadcast (x[B,M,K], w1[B,K,N], "
        "w2[B,K,N], storeGEMM1?, storeGEMM2? (dtype inferred from x)");
}
