#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "device/dual_gemm.h"          
#include "thread/left_silu_and_mul.h"  
#include "dual_gemm_common.h"

using ElementOperandA    = cutlass::half_t;   
using ElementOperandB    = cutlass::half_t;   
using ElementOutput      = cutlass::half_t;   
using ElementAccumulator = float;             
using ElementCompute     = float;

constexpr int  kStages       = 3;
constexpr bool kSplitKSerial = false;
constexpr bool kStoreD0      = true;
constexpr bool kStoreD1      = true;

using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Nothing  
>;

using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Nothing
>;

using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementOutput,
    ElementCompute
>;

using ArchTag          = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape        = cutlass::gemm::GemmShape< 64, 32, 32>;
using InstrShape       = cutlass::gemm::GemmShape< 16,  8, 16>;

using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA, cutlass::layout::RowMajor,      
    ElementOperandB, cutlass::layout::RowMajor,      
                     cutlass::layout::RowMajor,      
    ElementOutput,   cutlass::layout::RowMajor,      
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp, ArchTag,
    ThreadblockShape, WarpShape, InstrShape,
    EpilogueOutputOp0, EpilogueOutputOp1, EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages, kStoreD0, kStoreD1, kSplitKSerial
>;


static inline void check_inputs(const at::Tensor& x,
                                const at::Tensor& w1,
                                const at::Tensor& w2) {
  TORCH_CHECK(x.is_cuda() && w1.is_cuda() && w2.is_cuda(), "All tensors must be CUDA");
  TORCH_CHECK(x.scalar_type()==at::kHalf && w1.scalar_type()==at::kHalf && w2.scalar_type()==at::kHalf,
              "x, w1, w2 must be float16");
  TORCH_CHECK(x.dim()==2 && w1.dim()==2 && w2.dim()==2, "expected 2D tensors");
  TORCH_CHECK(x.size(1)==w1.size(0) && x.size(1)==w2.size(0), "shape mismatch: x[B,K], w*[K,N]");
  TORCH_CHECK(w1.size(1)==w2.size(1), "w1 and w2 must have same N");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_forward(const at::Tensor& x_in,
                  const at::Tensor& w1_in,
                  const at::Tensor& w2_in) {
  check_inputs(x_in, w1_in, w2_in);

  at::Tensor x  = x_in.contiguous();    
  at::Tensor b0 = w1_in.contiguous();   
  at::Tensor b1 = w2_in.contiguous();   

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = w1_in.size(1);

  at::Tensor d0 = at::empty({M, N}, x.options());
  at::Tensor d1 = at::empty({M, N}, x.options());
  at::Tensor d2 = at::empty({M, N}, x.options());

  at::Tensor c0 = at::empty({M, N}, x.options());
  at::Tensor c1 = at::empty({M, N}, x.options());

  int lda = static_cast<int>(x.stride(0));
  int ldb = static_cast<int>(b0.stride(0));
  int ldc = static_cast<int>(d0.stride(0));
  int ldd = ldc;
  
  using RM = cutlass::layout::RowMajor;

  auto A_ptr  = reinterpret_cast<ElementOperandA const*>(x.data_ptr<at::Half>());
  auto B0_ptr = reinterpret_cast<ElementOperandB const*>(b0.data_ptr<at::Half>());
  auto B1_ptr = reinterpret_cast<ElementOperandB const*>(b1.data_ptr<at::Half>());
  auto C0_ptr = reinterpret_cast<ElementOutput   const*>(c0.data_ptr<at::Half>());
  auto C1_ptr = reinterpret_cast<ElementOutput   const*>(c1.data_ptr<at::Half>());
  auto D0_ptr = reinterpret_cast<ElementOutput         *>(d0.data_ptr<at::Half>());
  auto D1_ptr = reinterpret_cast<ElementOutput         *>(d1.data_ptr<at::Half>());
  auto D2_ptr = reinterpret_cast<ElementOutput         *>(d2.data_ptr<at::Half>());

  cutlass::TensorRef<ElementOperandA const, RM> refA (A_ptr,  RM::Stride(lda));
  cutlass::TensorRef<ElementOperandB const, RM> refB0(B0_ptr, RM::Stride(ldb));
  cutlass::TensorRef<ElementOperandB const, RM> refB1(B1_ptr, RM::Stride(ldb));
  cutlass::TensorRef<ElementOutput   const, RM> refC0(C0_ptr, RM::Stride(ldc));
  cutlass::TensorRef<ElementOutput   const, RM> refC1(C1_ptr, RM::Stride(ldc));
  cutlass::TensorRef<ElementOutput         , RM> refD0(D0_ptr, RM::Stride(ldd));
  cutlass::TensorRef<ElementOutput         , RM> refD1(D1_ptr, RM::Stride(ldd));
  cutlass::TensorRef<ElementOutput         , RM> refD2(D2_ptr, RM::Stride(ldd));

  cutlass::gemm::GemmCoord problem_size(
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K));

  EpilogueOutputOp0::Params ep0(ElementCompute(1), ElementCompute(0));
  EpilogueOutputOp1::Params ep1(ElementCompute(1), ElementCompute(0));
  EpilogueOutputOp2::Params ep2;  // default

  DualGemm::Arguments args(
      cutlass::gemm::DualGemmMode::kGemm,
      problem_size,
      refA,
      refB0,
      refC0,
      refD0,
      refB1,
      refC1,
      refD1,
      refD2,
      ep0,
      ep1,
      ep2,
      1,
      1,
      0,
      0,
      0,
      0,
      0);

  DualGemm op;

  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
              "DualGemm: configuration not supported for these shapes/dtypes");

  size_t ws_bytes = DualGemm::get_workspace_size(args);
  at::Tensor wbuf = (ws_bytes ? at::empty({(long)ws_bytes}, x.options().dtype(at::kByte))
                              : at::Tensor());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto st = op(args, ws_bytes ? wbuf.data_ptr() : nullptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "DualGemm run failed");

  return std::make_tuple(d0, d1, d2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemm_forward", &dual_gemm_forward,
        "Fused Dual-GEMM + SiLU*Mul (fp16 I/O, fp32 accumulate)");
}

