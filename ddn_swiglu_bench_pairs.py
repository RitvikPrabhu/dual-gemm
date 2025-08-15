import os, time, importlib, warnings
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 220,
})

warnings.filterwarnings("ignore", message="invalid value encountered")

OUTDIR   = "bench_out"
PAIRS    = [(2114, 5637),(4096, 14336), (6144, 16384), (8192, 28672), (12288, 32768), (16384, 53248), (256000, 682666)]
DTYPES   = ["fp16", "bf16", "fp32"]
SEQLENS  = [2048]
BATCHES  = [2, 4, 8]
IMPLS    = ["eager", "fused_single", "fused_batched", "fused_broadcast"]
TF32     = True
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 123
WARMUP   = 3
ITERS    = 10
SPEEDUP_THR = 1.00
PAIRWISE_THR = 1.00
SINGLE_B_AGNOSTIC = True

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

try:
    fused = importlib.import_module("fused_swiglu_ampere")
    HAVE_FUSED = True
except Exception as e:
    print("[warn] couldn't import fused_swiglu_ampere:", e)
    HAVE_FUSED = False
    IMPLS = [i for i in IMPLS if not i.startswith("fused")] or ["eager"]

def nv_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def new_events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def time_ms(fn):
    times = []
    if torch.cuda.is_available():
        s,e = new_events()
        for _ in range(max(1, WARMUP)):
            fn()
        torch.cuda.synchronize()
        for _ in range(max(1, ITERS)):
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(max(1, WARMUP)):
            fn()
        for _ in range(max(1, ITERS)):
            t0=time.perf_counter(); fn(); t1=time.perf_counter()
            times.append((t1-t0)*1e3)
    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr)>1 else 0.0)

def randn_like(shape, dtype, device, g):
    return torch.randn(*shape, generator=g, device=device, dtype=dtype).contiguous()

def make_tensors(H, Hf, L, B, dtype, device, seed, broadcast):
    g = torch.Generator(device=device).manual_seed(seed)
    if B == 1:
        x = randn_like((L, H), dtype, device, g)
        W1 = randn_like((H, Hf), dtype, device, g)
        W3 = randn_like((H, Hf), dtype, device, g)
        Wout = randn_like((Hf, H), dtype, device, g)
    else:
        x = randn_like((B, L, H), dtype, device, g)
        W1 = randn_like((B, H, Hf), dtype, device, g)
        W3 = randn_like((H, Hf), dtype, device, g) if broadcast else randn_like((B, H, Hf), dtype, device, g)
        Wout = randn_like((Hf, H), dtype, device, g) if broadcast else randn_like((B, Hf, H), dtype, device, g)
    return x, W1, W3, Wout

def eager_swiglu_ffn(x, W1, W3, Wout):
    if x.dim()==2:
        a = x @ W1
        b = x @ W3
        y = torch.nn.functional.silu(a) * b
        out = y @ Wout
    else:
        a = torch.matmul(x, W1)
        b = torch.matmul(x, W3)
        y = torch.nn.functional.silu(a) * b
        out = torch.matmul(y, Wout)
    return y, out

def fused_single_once(x_2d, W1_2d, W3_2d):
    d0,d1,d2 = fused.dual_gemm_forward(x_2d, W1_2d, W3_2d, False, False)
    return d2

def fused_single_ffn(x, W1, W3, Wout):
    if x.dim()==2:
        d2 = fused_single_once(x.contiguous(), W1.contiguous(), W3.contiguous())
        out = d2 @ Wout.contiguous()
        return d2, out
    B = x.shape[0]
    d2_list = []
    for b in range(B):
        xb = x[b].contiguous()
        W1_b = (W1 if W1.dim()==2 else W1[b]).contiguous()
        W3_b = (W3 if W3.dim()==2 else W3[b]).contiguous()
        d2_b = fused_single_once(xb, W1_b, W3_b)
        d2_list.append(d2_b)
    d2 = torch.stack(d2_list, dim=0)
    out = torch.matmul(d2, Wout.contiguous())
    return d2, out

def fused_batched_ffn(x, W1, W3, Wout):
    d0,d1,d2 = fused.dual_gemm_batched_forward(x, W1, W3, False, False)
    out = torch.matmul(d2, Wout)
    return d2, out

def fused_broadcast_ffn(x, W1, W3, Wout):
    d0,d1,d2 = fused.dual_gemm_broadcast_forward(x, W1, W3, False, False)
    out = torch.matmul(d2, Wout)
    return d2, out

def fp32_ref(x, W1, W3, Wout):
    return eager_swiglu_ffn(x.float(), W1.float(), W3.float(), Wout.float())

def err_metrics(test, ref):
    t, r = test.float(), ref.float()
    d = t - r
    denom = torch.linalg.norm(r).item() + 1e-12
    rel_l2 = float(torch.linalg.norm(d).item()/denom)
    max_ref = float(r.abs().max().item()) + 1e-12
    rel_max = float(d.abs().max().item() / max_ref)
    max_rel_elem = float((d.abs()/(r.abs()+1e-12)).max().item())
    return {
        "rel_l2": rel_l2 if np.isfinite(rel_l2) else float('inf'),
        "rel_max_norm": rel_max if np.isfinite(rel_max) else float('inf'),
        "max_rel_elem": max_rel_elem if np.isfinite(max_rel_elem) else float('inf'),
    }

def flops_ffn(L,H,Hf,B=1):
    return 6.0*B*L*H*Hf

def bench_case(impl, dtype_name, H, Hf, L, B, device):
    dtype = DTYPE_MAP[dtype_name]
    torch.backends.cuda.matmul.allow_tf32 = bool(TF32)
    torch.backends.cudnn.allow_tf32 = bool(TF32)

    B_eff = 1 if (impl == "fused_single" and SINGLE_B_AGNOSTIC) else B
    broadcast = (impl == "fused_broadcast")
    x,W1,W3,Wout = make_tensors(H,Hf,L,B_eff,dtype,device,SEED,broadcast)
    y_ref,out_ref = fp32_ref(x,W1,W3,Wout)

    def run_once():
        if impl == "eager":
            return eager_swiglu_ffn(x,W1,W3,Wout)
        if impl == "fused_single":
            if not HAVE_FUSED: raise RuntimeError("no fused module")
            return fused_single_ffn(x,W1,W3,Wout)
        if impl == "fused_batched":
            if not HAVE_FUSED: raise RuntimeError("no fused module")
            return fused_batched_ffn(x,W1,W3,Wout)
        if impl == "fused_broadcast":
            if not HAVE_FUSED: raise RuntimeError("no fused module")
            return fused_broadcast_ffn(x,W1,W3,Wout)
        raise ValueError(impl)

    total_ms, total_std = time_ms(lambda: run_once())

    if impl == "eager":
        s1_ms,_ = time_ms(lambda: eager_swiglu_ffn(x,W1,W3,Wout)[0])
        y,_ = eager_swiglu_ffn(x,W1,W3,Wout)
        s2_ms,_ = time_ms(lambda: (y @ Wout) if x.dim()==2 else torch.matmul(y,Wout))
    elif impl == "fused_single":
        if x.dim()==2:
            s1_ms,_ = time_ms(lambda: fused_single_once(x,W1,W3))
        else:
            def stage1_loop():
                for b in range(x.shape[0]):
                    fused_single_once(x[b], W1 if W1.dim()==2 else W1[b], W3 if W3.dim()==2 else W3[b])
            s1_ms,_ = time_ms(stage1_loop)
        d2,_ = run_once()
        s2_ms,_ = time_ms(lambda: (d2 @ Wout) if x.dim()==2 else torch.matmul(d2,Wout))
    else:
        if impl == "fused_batched":
            s1_ms,_ = time_ms(lambda: fused.dual_gemm_batched_forward(x,W1,W3,False,False)[2])
        else:
            s1_ms,_ = time_ms(lambda: fused.dual_gemm_broadcast_forward(x,W1,W3,False,False)[2])
        d2,_ = run_once()
        s2_ms,_ = time_ms(lambda: (d2 @ Wout) if x.dim()==2 else torch.matmul(d2,Wout))

    y, out = run_once(); nv_sync()
    errs = err_metrics(out, out_ref)
    tflops = flops_ffn(L,H,Hf,B_eff)/(total_ms*1e-3)/1e12

    return {
        "impl":impl,"dtype":dtype_name,"B":B,"H":H,"Hff":Hf,"L":L,
        "total_ms":total_ms,"total_std_ms":total_std,
        "stage1_ms":s1_ms,"stage2_ms":s2_ms,"tflops":tflops,
        "err_out_rel_l2":errs["rel_l2"],
        "err_rel_max_norm":errs["rel_max_norm"],
        "err_max_rel_elem":errs["max_rel_elem"],
    }

def heat_time_methods(df, title, outfile):
    impl_cols = [c for c in ["eager","fused_batched","fused_broadcast","fused_single"] if c in df['impl'].unique()]
    pivot = (df.pivot_table(index="H", columns="impl", values="total_ms", aggfunc="mean")
               .reindex(columns=impl_cols))
    if pivot.empty:
        return
    plt.figure(figsize=(max(6, 1.0*pivot.shape[1]), max(4, 0.7*pivot.shape[0])))
    plt.imshow(pivot.values, aspect='auto', interpolation='nearest')
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha='right')
    plt.yticks(range(pivot.shape[0]), pivot.index)
    cb = plt.colorbar(); cb.set_label('ms')
    plt.title(title)
    plt.xlabel("method (impl)")
    plt.ylabel("Hidden size H")
    plt.tight_layout(); plt.savefig(outfile); plt.close()

def error_plot(merged, outfile):
    d = merged[merged['impl']!='eager'].copy()
    if d.empty:
        return
    agg = (d.groupby(['dtype','impl','B','H'], as_index=False)
             .agg(err=('err_rel_max_norm','median')))

    impl_order = [i for i in ['fused_batched','fused_broadcast','fused_single'] if i in agg['impl'].unique()]
    dtype_order = [i for i in ['fp16','bf16','fp32'] if i in agg['dtype'].unique()]
    size_map = {1:60, 2:70, 4:90, 8:110, 16:130, 32:160}
    color_map = {
        'fused_batched':   '#1f77b4',
        'fused_broadcast': '#ff7f0e',
        'fused_single':    '#2ca02c',
    }
    marker_map = {'fp16':'o','bf16':'s','fp32':'^'}

    fig, ax = plt.subplots(figsize=(10,6))

    for impl in impl_order:
        for dtype in dtype_order:
            for B in sorted(agg['B'].unique()):
                sub = agg[(agg['impl']==impl)&(agg['dtype']==dtype)&(agg['B']==B)]
                if sub.empty:
                    continue
                x = sub['H'].to_numpy()
                y = sub['err'].to_numpy()
                ax.scatter(x, y,
                           s=size_map.get(int(B), 80),
                           c=color_map.get(impl, '#444444'),
                           marker=marker_map.get(dtype, 'o'),
                           alpha=0.85,
                           edgecolor='none')

    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Hidden size H')
    ax.set_ylabel('Normalized max-diff error (median over L)')
    ax.set_title('Error vs H — all dtypes, batches, and fused implementations')
    ax.grid(True, which='both', ls=':', lw=0.6, alpha=0.5)

    impl_handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor=color_map[i], markersize=8, label=i)
                    for i in impl_order]
    dtype_handles = [Line2D([0],[0], marker=marker_map[d], color='#333333', linestyle='None', markersize=8, label=d)
                     for d in dtype_order]
    bs_vals = sorted({int(b) for b in agg['B'].unique()})
    size_handles = [plt.scatter([], [], s=size_map.get(b,80), c='#777777', label=f'B={b}') for b in bs_vals]

    leg1 = ax.legend(handles=impl_handles, title='Implementation', loc='upper right')
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=dtype_handles, title='Dtype', loc='lower left')
    ax.add_artist(leg2)
    ax.legend(handles=size_handles, title='Batch size', loc='lower right', scatterpoints=1)

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)

def geomean(x):
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v) & (v>0)]
    return float(np.exp(np.log(v).mean())) if v.size else float('nan')

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "tables"), exist_ok=True)
    device = torch.device(DEVICE)
    rows = []

    if TF32:
        print("[info] TF32 enabled for fp32 matmuls")

    for dtype in DTYPES:
        for (H,Hf) in PAIRS:
            for L in SEQLENS:
                for B in BATCHES:
                    for impl in IMPLS:
                        if impl.startswith("fused") and not HAVE_FUSED:
                            continue
                        if impl=="fused_single" and B<1:
                            continue
                        if impl in ("fused_batched","fused_broadcast") and B==1:
                            continue
                        try:
                            r = bench_case(impl, dtype, H, Hf, L, B, str(device))
                            rows.append(r)
                            print(f"[ok] {impl} {dtype} H={H} Hff={Hf} L={L} B={B} -> {r['total_ms']:.3f} ms, {r['tflops']:.2f} TF/s")
                        except Exception as e:
                            print(f"[skip] {impl} {dtype} H={H} Hff={Hf} L={L} B={B}: {e}")

    if not rows:
        print("No results."); return

    df = pd.DataFrame(rows)
    csv = os.path.join(OUTDIR, "results.csv"); df.to_csv(csv, index=False); print("Wrote", csv)

    base = df[df['impl']=="eager"][['dtype','B','H','Hff','L','total_ms']].rename(columns={'total_ms':'eager_ms'})
    merged = pd.merge(df, base, on=['dtype','B','H','Hff','L'], how='left')
    merged['speedup'] = merged['eager_ms'] / merged['total_ms']

    for dtype in merged['dtype'].unique():
        for B in BATCHES:
            for L in SEQLENS:
                sub = merged[(merged['dtype']==dtype)&(merged['B']==B)&(merged['L']==L)]
                if len(sub)==0:
                    continue
                out_time = os.path.join(OUTDIR, f"heat_time_methods_dtype-{dtype}_B{B}_L{L}.png")
                heat_time_methods(sub, f"Runtime (ms) — dtype={dtype}, B={B}, L={L}", out_time)
                print("Wrote", out_time)

    err_out = os.path.join(OUTDIR, "error_plot.png")
    error_plot(merged, err_out)
    print("Wrote", err_out)

if __name__ == "__main__":
    main()

