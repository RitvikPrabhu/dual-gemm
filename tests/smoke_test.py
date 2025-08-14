import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import fused_swiglu_ampere as _C


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp16", "half", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float", "float32"):
        return torch.float32
    if s in ("fp8"):
        return torch.float8_e4m3fnuz
    raise ValueError(f"Unrecognized dtype: {s}")


@torch.no_grad()
def run_case_single(B: int, D: int, H: int, dtype: torch.dtype):
    device = "cuda"

    x = torch.randn(B, D, device=device, dtype=dtype)
    w1 = torch.randn(D, H, device=device, dtype=dtype)
    w2 = torch.randn(D, H, device=device, dtype=dtype)

    _ = _C.dual_gemm_forward(x, w1, w2, False, False)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        d0, d1, d2 = _C.dual_gemm_forward(x, w1, w2, True, True)
        torch.cuda.synchronize()

    for name, t in (("d0", d0), ("d1", d1), ("d2", d2)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
        if t.dtype != dtype:
            raise RuntimeError(f"{name} dtype {t.dtype} != expected {dtype}")

    xf = x.float()
    d0_ref = (xf @ w1.float()).to(dtype)
    d1_ref = (xf @ w2.float()).to(dtype)

    d2_ref = (F.silu(d0.float()) * d1.float()).to(dtype)

    max_d0 = (d0.float() - d0_ref.float()).abs().max().item()
    max_d1 = (d1.float() - d1_ref.float()).abs().max().item()
    max_d2 = (d2.float() - d2_ref.float()).abs().max().item()

    eps = 1e-8
    rel_d0 = (
        d0.float() - d0_ref.float()
    ).abs().max() / d0_ref.float().abs().max().clamp_min(eps)
    rel_d1 = (
        d1.float() - d1_ref.float()
    ).abs().max() / d1_ref.float().abs().max().clamp_min(eps)
    rel_d2 = (
        d2.float() - d2_ref.float()
    ).abs().max() / d2_ref.float().abs().max().clamp_min(eps)
    rel_d0 = rel_d0.item()
    rel_d1 = rel_d1.item()
    rel_d2 = rel_d2.item()

    dtype_label = {
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
        torch.float32: "float",
    }[dtype]

    print(f"max|d0 - (x@w1).{dtype_label}()|: {max_d0}")
    print(f"max|d1 - (x@w2).{dtype_label}()|: {max_d1}")
    print(f"max|d2 - SiLU(d0)*d1         |: {max_d2}")
    print(f"rel_max d0: {rel_d0}")
    print(f"rel_max d1: {rel_d1}")
    print(f"rel_max d2: {rel_d2}")

    print("\nCUDA profile:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


@torch.no_grad()
def run_case_batched(B: int, D: int, H: int, dtype: torch.dtype):
    device = "cuda"

    x = torch.randn(B, B, D, device=device, dtype=dtype)
    w1 = torch.randn(B, D, H, device=device, dtype=dtype)
    w2 = torch.randn(B, D, H, device=device, dtype=dtype)

    _ = _C.dual_gemm_batched_forward(x, w1, w2, False, False)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        d0, d1, d2 = _C.dual_gemm_batched_forward(x, w1, w2, True, True)
        torch.cuda.synchronize()

    for name, t in (("d0", d0), ("d1", d1), ("d2", d2)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
        if t.dtype != dtype:
            raise RuntimeError(f"{name} dtype {t.dtype} != expected {dtype}")

    xf = x.float()
    d0_ref = torch.matmul(xf, w1.float()).to(dtype)
    d1_ref = torch.matmul(xf, w2.float()).to(dtype)
    d2_ref = (F.silu(d0.float()) * d1.float()).to(dtype)

    max_d0 = (d0.float() - d0_ref.float()).abs().max().item()
    max_d1 = (d1.float() - d1_ref.float()).abs().max().item()
    max_d2 = (d2.float() - d2_ref.float()).abs().max().item()

    eps = 1e-8
    rel_d0 = (
        d0.float() - d0_ref.float()
    ).abs().max() / d0_ref.float().abs().max().clamp_min(eps)
    rel_d1 = (
        d1.float() - d1_ref.float()
    ).abs().max() / d1_ref.float().abs().max().clamp_min(eps)
    rel_d2 = (
        d2.float() - d2_ref.float()
    ).abs().max() / d2_ref.float().abs().max().clamp_min(eps)
    rel_d0, rel_d1, rel_d2 = rel_d0.item(), rel_d1.item(), rel_d2.item()

    dtype_label = {
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
        torch.float32: "float",
    }.get(dtype, str(dtype))

    print(f"max|d0 - (x@w1).{dtype_label}()|: {max_d0}")
    print(f"max|d1 - (x@w2).{dtype_label}()|: {max_d1}")
    print(f"max|d2 - SiLU(d0)*d1         |: {max_d2}")
    print(f"rel_max d0: {rel_d0}")
    print(f"rel_max d1: {rel_d1}")
    print(f"rel_max d2: {rel_d2}")

    print("\nCUDA profile:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


@torch.no_grad()
def run_case_broadcast(B: int, D: int, H: int, dtype: torch.dtype):
    device = "cuda"

    x = torch.randn(B, B, D, device=device, dtype=dtype)
    w1 = torch.randn(B, D, H, device=device, dtype=dtype)
    w2 = torch.randn(D, H, device=device, dtype=dtype)

    _ = _C.dual_gemm_broadcast_forward(x, w1, w2, False, False)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        d0, d1, d2 = _C.dual_gemm_broadcast_forward(x, w1, w2, True, True)
        torch.cuda.synchronize()

    for name, t in (("d0", d0), ("d1", d1), ("d2", d2)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
        if t.dtype != dtype:
            raise RuntimeError(f"{name} dtype {t.dtype} != expected {dtype}")

    xf = x.float()
    d0_ref = torch.matmul(xf, w1.float()).to(dtype)
    d1_ref = torch.matmul(xf, w2.float()).to(dtype)
    d2_ref = (F.silu(d0.float()) * d1.float()).to(dtype)

    max_d0 = (d0.float() - d0_ref.float()).abs().max().item()
    max_d1 = (d1.float() - d1_ref.float()).abs().max().item()
    max_d2 = (d2.float() - d2_ref.float()).abs().max().item()

    eps = 1e-8
    rel_d0 = (
        d0.float() - d0_ref.float()
    ).abs().max() / d0_ref.float().abs().max().clamp_min(eps)
    rel_d1 = (
        d1.float() - d1_ref.float()
    ).abs().max() / d1_ref.float().abs().max().clamp_min(eps)
    rel_d2 = (
        d2.float() - d2_ref.float()
    ).abs().max() / d2_ref.float().abs().max().clamp_min(eps)
    rel_d0, rel_d1, rel_d2 = rel_d0.item(), rel_d1.item(), rel_d2.item()

    dtype_label = {
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
        torch.float32: "float",
    }.get(dtype, str(dtype))

    print(f"max|d0 - (x@w1).{dtype_label}()|: {max_d0}")
    print(f"max|d1 - (x@w2).{dtype_label}()|: {max_d1}")
    print(f"max|d2 - SiLU(d0)*d1         |: {max_d2}")
    print(f"rel_max d0: {rel_d0}")
    print(f"rel_max d1: {rel_d1}")
    print(f"rel_max d2: {rel_d2}")

    print("\nCUDA profile:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--H", type=int, default=11008)
    ap.add_argument("--dtype", type=str, default="fp16", help="fp16 | bf16 | fp32")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = parse_dtype(args.dtype)
    print(f"\n=== B={args.B} D={args.D} H={args.H} dtype={dtype} Mode: Single ===")
    run_case_single(args.B, args.D, args.H, dtype)

    print(f"\n=== B={args.B} D={args.D} H={args.H} dtype={dtype} Mode: Batched ===")
    run_case_batched(args.B, args.D, args.H, dtype)

    print(f"\n=== B={args.B} D={args.D} H={args.H} dtype={dtype} Mode: Broadcast ===")
    run_case_broadcast(args.B, args.D, args.H, dtype)


if __name__ == "__main__":
    main()
