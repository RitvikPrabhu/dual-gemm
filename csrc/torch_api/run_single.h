#pragma once
#include "common.h"

namespace dual_gemm {
namespace torchapi {

template <typename Element, bool StoreD0, bool StoreD1>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
run_dual_gemm_typed_impl(const at::Tensor &x_in, const at::Tensor &w1_in,
                         const at::Tensor &w2_in) {
  using ElementOperandA = Element;
  using ElementOperandB = Element;
  using ElementOutput = Element;
  using ElementAccumulator = float; // accumulate in fp32
  using ElementCompute = float;

  static constexpr int kVec = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, kVec, ElementAccumulator, ElementCompute,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using EpilogueOutputOp1 = EpilogueOutputOp0;
  using EpilogueOutputOp2 =
      cutlass::epilogue::thread::LeftSiLUAndMul<ElementOutput, kVec,
                                                ElementOutput, ElementCompute>;

  using InstrShape = InstrShapeFor<Element>;

  using DualGemmT = cutlass::gemm::device::DualGemm<
      ElementOperandA, RM, ElementOperandB, RM, RM, ElementOutput, RM,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, ArchTag,
      ThreadblockShape, WarpShape, InstrShape, EpilogueOutputOp0,
      EpilogueOutputOp1, EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, kStages,
      StoreD0, StoreD1, kSplitKSerial>;

  check_shapes_2d(x_in, w1_in, w2_in);

  using AT = typename AtenScalar<Element>::type;

  at::Tensor x = x_in.contiguous();
  at::Tensor b0 = w1_in.contiguous();
  at::Tensor b1 = w2_in.contiguous();

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = b0.size(1);

  at::Tensor d0 =
      StoreD0 ? at::empty({M, N}, x.options()) : at::empty({0}, x.options());
  at::Tensor d1 =
      StoreD1 ? at::empty({M, N}, x.options()) : at::empty({0}, x.options());
  at::Tensor d2 = at::empty({M, N}, x.options());
  at::Tensor c0 = at::empty({M, N}, x.options());
  at::Tensor c1 = at::empty({M, N}, x.options());

  int lda = int(K);
  int ldb0 = int(N);
  int ldb1 = int(N);
  int ldc = int(N);
  int ldd = int(N);

  auto A_ptr = reinterpret_cast<Element const *>(x.data_ptr<AT>());
  auto B0_ptr = reinterpret_cast<Element const *>(b0.data_ptr<AT>());
  auto B1_ptr = reinterpret_cast<Element const *>(b1.data_ptr<AT>());
  auto C0_ptr = reinterpret_cast<Element const *>(c0.data_ptr<AT>());
  auto C1_ptr = reinterpret_cast<Element const *>(c1.data_ptr<AT>());
  auto D0_ptr =
      StoreD0 ? reinterpret_cast<Element *>(d0.data_ptr<AT>()) : nullptr;
  auto D1_ptr =
      StoreD1 ? reinterpret_cast<Element *>(d1.data_ptr<AT>()) : nullptr;
  auto D2_ptr = reinterpret_cast<Element *>(d2.data_ptr<AT>());

  cutlass::TensorRef<Element const, RM> refA(A_ptr, RM::Stride(lda));
  cutlass::TensorRef<Element const, RM> refB0(B0_ptr, RM::Stride(ldb0));
  cutlass::TensorRef<Element const, RM> refB1(B1_ptr, RM::Stride(ldb1));
  cutlass::TensorRef<Element const, RM> refC0(C0_ptr, RM::Stride(ldc));
  cutlass::TensorRef<Element const, RM> refC1(C1_ptr, RM::Stride(ldc));
  cutlass::TensorRef<Element, RM> refD0(D0_ptr, RM::Stride(ldd));
  cutlass::TensorRef<Element, RM> refD1(D1_ptr, RM::Stride(ldd));
  cutlass::TensorRef<Element, RM> refD2(D2_ptr, RM::Stride(ldd));

  cutlass::gemm::GemmCoord problem_size{int(M), int(N), int(K)};

  using Params0 = typename EpilogueOutputOp0::Params;
  using Params1 = typename EpilogueOutputOp1::Params;
  using Params2 = typename EpilogueOutputOp2::Params;

  Params0 ep0(ElementCompute(1), ElementCompute(0));
  Params1 ep1(ElementCompute(1), ElementCompute(0));
  Params2 ep2;

  using ArgumentsT = typename DualGemmT::Arguments;

  ArgumentsT args(cutlass::gemm::DualGemmMode::kGemm, problem_size, refA, refB0,
                  refC0, refD0, refB1, refC1, refD1, refD2, ep0, ep1, ep2, 1, 1,
                  0, 0, 0, 0, 0);

  DualGemmT op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
              "DualGemm: unsupported shapes/dtypes");

  size_t ws_bytes = DualGemmT::get_workspace_size(args);
  at::Tensor wbuf =
      (ws_bytes ? at::empty({(long)ws_bytes}, x.options().dtype(at::kByte))
                : at::Tensor());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto st = op(args, ws_bytes ? wbuf.data_ptr() : nullptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "DualGemm run failed");

  return {d0, d1, d2};
}

template <typename Element>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
run_dual_gemm_typed(const at::Tensor &x_in, const at::Tensor &w1_in,
                    const at::Tensor &w2_in, bool storeD0, bool storeD1) {
  if (storeD0 && storeD1) {
    return run_dual_gemm_typed_impl<Element, true, true>(x_in, w1_in, w2_in);
  } else if (storeD0 && !storeD1) {
    return run_dual_gemm_typed_impl<Element, true, false>(x_in, w1_in, w2_in);
  } else if (!storeD0 && storeD1) {
    return run_dual_gemm_typed_impl<Element, false, true>(x_in, w1_in, w2_in);
  } else {
    return run_dual_gemm_typed_impl<Element, false, false>(x_in, w1_in, w2_in);
  }
}

} // namespace torchapi
} // namespace dual_gemm
