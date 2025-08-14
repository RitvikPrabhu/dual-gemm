#pragma once
#include "common.h"

namespace dual_gemm {
namespace torchapi {

template <typename Element, bool StoreD0, bool StoreD1>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
run_dual_gemm_batched_typed_impl(const at::Tensor &x_in,
                                 const at::Tensor &w1_in,
                                 const at::Tensor &w2_in) {
  using ElementOperandA = Element;
  using ElementOperandB = Element;
  using ElementOutput = Element;
  using ElementAccumulator = float;
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

  TORCH_CHECK(x_in.dim() == 3 && w1_in.dim() == 3 && w2_in.dim() == 3,
              "expected x[B,M,K], w1[B,K,N], w2[B,K,N]");

  using AT = typename AtenScalar<Element>::type;

  at::Tensor x = x_in.contiguous();
  at::Tensor b0 = w1_in.contiguous();
  at::Tensor b1 = w2_in.contiguous();

  const int64_t B = x.size(0);
  const int64_t M = x.size(1);
  const int64_t K = x.size(2);
  const int64_t N = b0.size(2);

  at::Tensor d0 =
      StoreD0 ? at::empty({B, M, N}, x.options()) : at::empty({0}, x.options());
  at::Tensor d1 =
      StoreD1 ? at::empty({B, M, N}, x.options()) : at::empty({0}, x.options());
  at::Tensor d2 = at::empty({B, M, N}, x.options());
  at::Tensor c0 = at::empty({B, M, N}, x.options());
  at::Tensor c1 = at::empty({B, M, N}, x.options());

  int lda = int(K);
  int ldb0 = int(N);
  int ldb1 = int(N);
  int ldc = int(N);
  int ldd = int(N);

  int64_t bsA = M * K;
  int64_t bsB0 = K * N;
  int64_t bsB1 = K * N;
  int64_t bsC = M * N;
  int64_t bsD = M * N;

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

  typename EpilogueOutputOp0::Params ep0(ElementCompute(1), ElementCompute(0));
  typename EpilogueOutputOp1::Params ep1(ElementCompute(1), ElementCompute(0));
  typename EpilogueOutputOp2::Params ep2;

  typename DualGemmT::Arguments args(cutlass::gemm::DualGemmMode::kBatched,
                                     problem_size, refA, refB0, refC0, refD0,
                                     refB1, refC1, refD1, refD2, ep0, ep1, ep2,
                                     1, int(B), bsA, bsB0, bsB1, bsC, bsD);

  DualGemmT op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
              "DualGemm(batched): unsupported configuration");

  size_t ws_bytes = DualGemmT::get_workspace_size(args);
  at::Tensor wbuf =
      (ws_bytes ? at::empty({(long)ws_bytes}, x.options().dtype(at::kByte))
                : at::Tensor());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto st = op(args, ws_bytes ? wbuf.data_ptr() : nullptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "DualGemm(batched) run failed");

  return {d0, d1, d2};
}

template <typename Element>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
run_dual_gemm_batched_typed(const at::Tensor &x_in, const at::Tensor &w1_in,
                            const at::Tensor &w2_in, bool storeD0,
                            bool storeD1) {
  if (storeD0 && storeD1) {
    return run_dual_gemm_batched_typed_impl<Element, true, true>(x_in, w1_in,
                                                                 w2_in);
  } else if (storeD0 && !storeD1) {
    return run_dual_gemm_batched_typed_impl<Element, true, false>(x_in, w1_in,
                                                                  w2_in);
  } else if (!storeD0 && storeD1) {
    return run_dual_gemm_batched_typed_impl<Element, false, true>(x_in, w1_in,
                                                                  w2_in);
  } else {
    return run_dual_gemm_batched_typed_impl<Element, false, false>(x_in, w1_in,
                                                                   w2_in);
  }
}

} // namespace torchapi
} // namespace dual_gemm
