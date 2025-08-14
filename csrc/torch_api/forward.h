#pragma once
#include "common.h"
#include "run_batched.h"
#include "run_broadcast.h"
#include "run_single.h"

namespace dual_gemm {
namespace torchapi {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_forward_expl(const at::Tensor &x_in, const at::Tensor &w1_in,
                       const at::Tensor &w2_in, bool kStoreD0, bool kStoreD1,
                       c10::optional<at::ScalarType> dtype_opt) {
  TORCH_CHECK(x_in.is_cuda() && w1_in.is_cuda() && w2_in.is_cuda(),
              "All tensors must be CUDA");

  at::ScalarType want = dtype_opt.has_value() ? *dtype_opt : x_in.scalar_type();
  TORCH_CHECK(want == at::kHalf || want == at::kBFloat16 || want == at::kFloat,
              "dtype must be one of {float16, bfloat16, float32}");

  at::Tensor x = (x_in.scalar_type() == want) ? x_in : x_in.to(want);
  at::Tensor w1 = (w1_in.scalar_type() == want) ? w1_in : w1_in.to(want);
  at::Tensor w2 = (w2_in.scalar_type() == want) ? w2_in : w2_in.to(want);

  x = x.contiguous();
  w1 = w1.contiguous();
  w2 = w2.contiguous();

  if (want == at::kHalf) {
    return run_dual_gemm_typed<cutlass::half_t>(x, w1, w2, kStoreD0, kStoreD1);
  } else if (want == at::kBFloat16) {
    return run_dual_gemm_typed<cutlass::bfloat16_t>(x, w1, w2, kStoreD0,
                                                    kStoreD1);
  } else {
    return run_dual_gemm_typed<float>(x, w1, w2, kStoreD0, kStoreD1);
  }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_forward(const at::Tensor &x_in, const at::Tensor &w1_in,
                  const at::Tensor &w2_in, bool kStoreD0, bool kStoreD1) {
  return dual_gemm_forward_expl(x_in, w1_in, w2_in, kStoreD0, kStoreD1,
                                c10::nullopt);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_batched_forward_expl(const at::Tensor &x_in, const at::Tensor &w1_in,
                               const at::Tensor &w2_in, bool kStoreD0,
                               bool kStoreD1,
                               c10::optional<at::ScalarType> dtype_opt) {
  TORCH_CHECK(x_in.is_cuda() && w1_in.is_cuda() && w2_in.is_cuda(),
              "All tensors must be CUDA");
  at::ScalarType want = dtype_opt.has_value() ? *dtype_opt : x_in.scalar_type();
  TORCH_CHECK(want == at::kHalf || want == at::kBFloat16 || want == at::kFloat,
              "dtype must be one of {float16, bfloat16, float32}");

  at::Tensor x = (x_in.scalar_type() == want) ? x_in : x_in.to(want);
  at::Tensor w1 = (w1_in.scalar_type() == want) ? w1_in : w1_in.to(want);
  at::Tensor w2 = (w2_in.scalar_type() == want) ? w2_in : w2_in.to(want);

  TORCH_CHECK(x.dim() == 3 && w1.dim() == 3 && w2.dim() == 3,
              "expected x[B,M,K], w1[B,K,N], w2[B,K,N]");

  if (want == at::kHalf) {
    return run_dual_gemm_batched_typed<cutlass::half_t>(x, w1, w2, kStoreD0,
                                                        kStoreD1);
  } else if (want == at::kBFloat16) {
    return run_dual_gemm_batched_typed<cutlass::bfloat16_t>(x, w1, w2, kStoreD0,
                                                            kStoreD1);
  } else {
    return run_dual_gemm_batched_typed<float>(x, w1, w2, kStoreD0, kStoreD1);
  }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_batched_forward(const at::Tensor &x_in, const at::Tensor &w1_in,
                          const at::Tensor &w2_in, bool kStoreD0,
                          bool kStoreD1) {
  return dual_gemm_batched_forward_expl(x_in, w1_in, w2_in, kStoreD0, kStoreD1,
                                        c10::nullopt);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_broadcast_forward_expl(const at::Tensor &x_in,
                                 const at::Tensor &w1_in,
                                 const at::Tensor &w2_in, bool kStoreD0,
                                 bool kStoreD1,
                                 c10::optional<at::ScalarType> dtype_opt) {
  TORCH_CHECK(x_in.is_cuda() && w1_in.is_cuda() && w2_in.is_cuda(),
              "All tensors must be CUDA");
  at::ScalarType want = dtype_opt.has_value() ? *dtype_opt : x_in.scalar_type();
  TORCH_CHECK(want == at::kHalf || want == at::kBFloat16 || want == at::kFloat,
              "dtype must be one of {float16, bfloat16, float32}");

  at::Tensor x = (x_in.scalar_type() == want) ? x_in : x_in.to(want);
  at::Tensor w1 = (w1_in.scalar_type() == want) ? w1_in : w1_in.to(want);
  at::Tensor w2 = (w2_in.scalar_type() == want) ? w2_in : w2_in.to(want);

  TORCH_CHECK(x.dim() == 3 && w1.dim() == 3 && w2.dim() == 2,
              "expected x[B,M,K], w1[B,K,N], w2[K,N]");

  if (want == at::kHalf) {
    return run_dual_gemm_broadcast_typed<cutlass::half_t>(x, w1, w2, kStoreD0,
                                                          kStoreD1);
  } else if (want == at::kBFloat16) {
    return run_dual_gemm_broadcast_typed<cutlass::bfloat16_t>(
        x, w1, w2, kStoreD0, kStoreD1);
  } else {
    return run_dual_gemm_broadcast_typed<float>(x, w1, w2, kStoreD0, kStoreD1);
  }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_broadcast_forward(const at::Tensor &x_in, const at::Tensor &w1_in,
                            const at::Tensor &w2_in, bool kStoreD0,
                            bool kStoreD1) {
  return dual_gemm_broadcast_forward_expl(x_in, w1_in, w2_in, kStoreD0,
                                          kStoreD1, c10::nullopt);
}

} // namespace torchapi
} // namespace dual_gemm
