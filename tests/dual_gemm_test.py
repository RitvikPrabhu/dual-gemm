import os, sys
import pytest
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
_C = pytest.importorskip("dual_gemm")  


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-3, atol=1e-3)
    return dict(rtol=3e-2, atol=3e-2)


def _make_inputs(mode, B, M, K, N, dtype, device):
    if mode == "single":
        x  = torch.randn(M, K, device=device, dtype=dtype)
        w1 = torch.randn(K, N, device=device, dtype=dtype)
        w2 = torch.randn(K, N, device=device, dtype=dtype)
    elif mode == "batched":
        x  = torch.randn(B, M, K, device=device, dtype=dtype)
        w1 = torch.randn(B, K, N, device=device, dtype=dtype)
        w2 = torch.randn(B, K, N, device=device, dtype=dtype)
    elif mode == "broadcast":
        x  = torch.randn(B, M, K, device=device, dtype=dtype)
        w1 = torch.randn(B, K, N, device=device, dtype=dtype)
        w2 = torch.randn(K, N, device=device, dtype=dtype)
    else:
        raise ValueError(mode)
    return x, w1, w2


def _call(mode, x, w1, w2, storeD0, storeD1):
    if mode == "single":
        return _C.dual_gemm_forward(x, w1, w2, storeD0, storeD1)
    if mode == "batched":
        return _C.dual_gemm_batched_forward(x, w1, w2, storeD0, storeD1)
    if mode == "broadcast":
        return _C.dual_gemm_broadcast_forward(x, w1, w2, storeD0, storeD1)
    raise ValueError(mode)


def _refs(mode, x, w1, w2, out_dtype):
    xf  = x.float()
    w1f = w1.float()
    w2f = w2.float()

    if mode == "single":
        d0f = xf @ w1f
        d1f = xf @ w2f
    elif mode == "batched":
        d0f = torch.matmul(xf, w1f)  
        d1f = torch.matmul(xf, w2f)
    elif mode == "broadcast":
        d0f = torch.matmul(xf, w1f)  
        d1f = torch.matmul(xf, w2f)  
    else:
        raise ValueError(mode)

    d0_ref = d0f.to(out_dtype)
    d1_ref = d1f.to(out_dtype)
    d2_ref = (F.silu(d0f) * d1f).to(out_dtype)
    return d0_ref, d1_ref, d2_ref


@pytest.mark.parametrize("mode", ["single", "batched", "broadcast"])
@pytest.mark.parametrize("storeD0,storeD1", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dual_gemm_all_combinations(mode, storeD0, storeD1, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for dual_gemm tests")

    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported on this device")

    device = "cuda"
    torch.manual_seed(0)

    B, M, K, N = 3, 8, 64, 96

    x, w1, w2 = _make_inputs(mode, B, M, K, N, dtype, device)
    d0, d1, d2 = _call(mode, x, w1, w2, storeD0, storeD1)

    assert d0.device.type == d1.device.type == d2.device.type == "cuda"
    assert d0.dtype == d1.dtype == d2.dtype == dtype

    if mode == "single":
        exp = (M, N)
    else:
        exp = (B, M, N)
    assert tuple(d0.shape) == exp and tuple(d1.shape) == exp and tuple(d2.shape) == exp

    d0_ref, d1_ref, d2_ref = _refs(mode, x, w1, w2, dtype)

    tol = _tol(dtype)

    torch.testing.assert_close(d2, d2_ref, **tol)

    if storeD0:
        torch.testing.assert_close(d0, d0_ref, **tol)
    if storeD1:
        torch.testing.assert_close(d1, d1_ref, **tol)

    if not storeD0:
        assert d0.is_cuda and d0.dtype == dtype
    if not storeD1:
        assert d1.is_cuda and d1.dtype == dtype

