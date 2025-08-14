# FFN SwiGLU: Practitioner Guidance

- **bf16** best throughput: `eager` at B=1, H=2048, Hff=8192, L=2048 → 585.76 TF/s (0.352 ms).
- **fp16** best throughput: `eager` at B=1, H=2048, Hff=8192, L=2048 → 578.23 TF/s (0.357 ms).
- **fp32** best throughput: `fused_single` at B=1, H=2048, Hff=8192, L=2048 → 100.99 TF/s (2.041 ms).
- **bf16** error vs FP32-eager (rel L2, OUT): median=3.32e-03, 95p= 3.32e-03, max=3.32e-03.
- **fp16** error vs FP32-eager: no finite values recorded.
- **fp32** error vs FP32-eager (rel L2, OUT): median=2.08e-04, 95p= 4.17e-04, max=4.18e-04.

## Recommendations

- **bf16**: Prefer eager, or validate fused numerics per shape.
- **fp16**: Prefer eager, or validate fused numerics per shape.
- **fp32**: Prefer eager, or validate fused numerics per shape.

### Batched vs Broadcast
- Use `fused_batched` when each sample has unique W1/W3 and B is moderate/large.
- Use `fused_broadcast` when one branch (W3) is shared across batch; memory traffic drops.
- For B=1, `fused_single` is the fused option.
