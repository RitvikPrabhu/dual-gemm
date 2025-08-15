# FFN SwiGLU: Practitioner Guidance (Simple)

- **bf16** best throughput: `eager` at B=4, H=2048, Hff=8192, L=2048 → 574.04 TF/s (1.437 ms).
- **fp16** best throughput: `eager` at B=4, H=2048, Hff=8192, L=2048 → 573.82 TF/s (1.437 ms).
- **fp32** best throughput: `eager` at B=4, H=2048, Hff=8192, L=2048 → 276.98 TF/s (2.977 ms).

## Batched vs Broadcast vs Single
- `fused_single` uses the single-matrix fused kernel; for B>1 we loop per-sample (for comparison only).
- `fused_batched` expects distinct weights per sample; fastest when B is moderate/large.
- `fused_broadcast` shares one branch across the batch; best when W3 is shared.
