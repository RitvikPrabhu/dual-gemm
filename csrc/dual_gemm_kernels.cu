// csrc/dual_gemm_kernels.cu
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

// Your project headers — keep your relative paths as in your repo
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"
#include "dual_gemm_run.h"
#include "test_run.h"

// ----------------------------
// Globals (copied from snippet)
// ----------------------------
cutlass::gemm::GemmCoord problem_size(4096, 4096, 8192);
cutlass::gemm::GemmCoord batch_problem_size(321, 256, 512);

constexpr int kStages = 3;
constexpr bool kSplitKSerial = false;
constexpr bool kUseBias = true;
constexpr int kBatchCount = 37;

using ElementOperandA = cutlass::half_t;
using ElementOperandB = cutlass::half_t;
using ElementOutput   = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute     = cutlass::half_t;

constexpr auto kScaleType = kUseBias
  ? cutlass::epilogue::thread::ScaleType::NoBetaScaling
  : (kSplitKSerial
      ? cutlass::epilogue::thread::ScaleType::Default
      : cutlass::epilogue::thread::ScaleType::Nothing);

using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
  ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementAccumulator, ElementCompute, kScaleType>;

using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
  ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementAccumulator, ElementCompute, kScaleType>;

using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementOutput, ElementCompute>;

const ElementCompute alpha0 = ElementCompute(1);
const ElementCompute beta0  = ElementCompute(kUseBias ? 1 : 0);
const ElementCompute alpha1 = ElementCompute(1);
const ElementCompute beta1  = ElementCompute(kUseBias ? 1 : 0);

// If your DualFusedGemmRun needs this functor in TU:
template <typename T>
struct LeftSiLUAndMul {
  struct Params{};
  CUTLASS_HOST_DEVICE LeftSiLUAndMul(Params) {}
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
  CUTLASS_HOST_DEVICE T operator()(T const& lhs, T const& rhs) const {
    cutlass::epilogue::thread::SiLu<T> silu;
    cutlass::multiplies<T> mul;
    return mul(silu(lhs), rhs);
  }
  template <int kCount>
  CUTLASS_HOST_DEVICE cutlass::Array<T, kCount> operator()(
      cutlass::Array<T, kCount> const& lhs,
      cutlass::Array<T, kCount> const& rhs) const {
    cutlass::epilogue::thread::SiLu<T> silu;
    cutlass::multiplies<T> mul;
    return mul(silu(lhs), rhs);
  }
};

// ----------------------------
// The three runners (unchanged)
// ----------------------------
bool run_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA, cutlass::layout::RowMajor,
    ElementOperandB, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
    ElementOutput,   cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp0, EpilogueOutputOp1, EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages, kStoreD0, kStoreD1, kSplitKSerial>;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Fused FP16 TN GEMMs + Epilogue2...\n";
  bool passed = fusedGemm.run(problem_size, alpha0, beta0, alpha1, beta1);
  std::cout << (passed ? "Pass\n" : "Fail\n");
  return passed;
}

bool run_batched_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA, cutlass::layout::RowMajor,
    ElementOperandB, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
    ElementOutput,   cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp0, EpilogueOutputOp1, EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages, kStoreD0, kStoreD1, kSplitKSerial>;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Batched Fused FP16 TN GEMMs + Epilogue2...\n";
  bool passed = fusedGemm.run(
      batch_problem_size, alpha0, beta0, alpha1, beta1,
      /*batch_count*/  kBatchCount,
      /*broadcast_b1*/ false,
      /*is_profiling*/ false);
  std::cout << (passed ? "Pass\n" : "Fail\n");
  return passed;
}

bool run_broadcast_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA, cutlass::layout::RowMajor,
    ElementOperandB, /*B0*/ cutlass::layout::RowMajor, /*B1*/ cutlass::layout::ColumnMajor,
    ElementOutput,   cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp0, EpilogueOutputOp1, EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages, kStoreD0, kStoreD1, kSplitKSerial>;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Broadcast Fused FP16 TN GEMMs + Epilogue2...\n";
  bool passed = fusedGemm.run(
      problem_size, alpha0, beta0, alpha1, beta1,
      /*batch_count*/ 1,
      /*broadcast_b1*/ true,
      /*is_profiling*/ true);
  std::cout << (passed ? "Pass\n" : "Fail\n");
  return passed;
}

