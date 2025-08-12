import argparse
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

import dual_gemm as _C  

@torch.no_grad()
def run_case(B: int, D: int, H: int):
    device = "cuda"
    dtype  = torch.float16

    torch.manual_seed(0)

    x  = torch.randn(B, D, device=device, dtype=dtype)
    w1 = torch.randn(D, H, device=device, dtype=dtype)
    w2 = torch.randn(D, H, device=device, dtype=dtype)

    _ = _C.dual_gemm_forward(x, w1, w2)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        d0, d1, d2 = _C.dual_gemm_forward(x, w1, w2)
        torch.cuda.synchronize()

    for name, t in (("d0", d0), ("d1", d1), ("d2", d2)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"{name} contains NaN/Inf")

    # 1) GEMM refs (fp32 accum, then store fp16)
    xf  = x.float()
    d0_ref = (xf @ w1.float()).to(torch.float16)
    d1_ref = (xf @ w2.float()).to(torch.float16)

    # 2) Fused epilogue ref built FROM THE RETURNED d0/d1 (matches kernel)
    d2_ref = (F.silu(d0.float()) * d1.float()).to(torch.float16)

    # Errors (measure in fp32 for readability)
    max_d0 = (d0.float() - d0_ref.float()).abs().max().item()
    max_d1 = (d1.float() - d1_ref.float()).abs().max().item()
    max_d2 = (d2.float() - d2_ref.float()).abs().max().item()

    print(f"max|d0 - (x@w1).half()|: {max_d0}")
    print(f"max|d1 - (x@w2).half()|: {max_d1}")
    print(f"max|d2 - SiLU(d0)*d1  |: {max_d2}")

    print("\nCUDA profile:")
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=100))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--H", type=int, default=11008)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print(f"\n=== B={args.B} D={args.D} H={args.H} ===")
    run_case(args.B, args.D, args.H)

if __name__ == "__main__":
    main()

