#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "../device/dual_gemm.h"
#include "../dual_gemm_common.h"
#include "../thread/left_silu_and_mul.h"

namespace dual_gemm {
namespace torchapi {

namespace py = pybind11;

constexpr int kStages = 3;
constexpr bool kSplitKSerial = false;

using ArchTag = cutlass::arch::Sm80;
using RM = cutlass::layout::RowMajor;

template <typename T> struct AtenScalar;
template <> struct AtenScalar<cutlass::half_t> {
  using type = at::Half;
};
template <> struct AtenScalar<cutlass::bfloat16_t> {
  using type = at::BFloat16;
};
template <> struct AtenScalar<float> {
  using type = float;
};

// TF32 instruction tile for float, TensorOp tile for FP16/BF16
template <typename Element>
using InstrShapeFor =
    typename std::conditional<std::is_same<Element, float>::value,
                              cutlass::gemm::GemmShape<16, 8, 8>, // TF32 path
                              cutlass::gemm::GemmShape<16, 8, 16> // FP16/BF16
                                                                  // path
                              >::type;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;

inline void check_shapes_2d(const at::Tensor &x, const at::Tensor &w1,
                            const at::Tensor &w2) {
  TORCH_CHECK(x.is_cuda() && w1.is_cuda() && w2.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(x.dim() == 2 && w1.dim() == 2 && w2.dim() == 2,
              "expected 2D tensors");
  TORCH_CHECK(x.size(1) == w1.size(0) && x.size(1) == w2.size(0),
              "shape mismatch: x[B,K], w*[K,N]");
  TORCH_CHECK(w1.size(1) == w2.size(1), "w1 and w2 must have same N");
}

} // namespace torchapi
} // namespace dual_gemm
