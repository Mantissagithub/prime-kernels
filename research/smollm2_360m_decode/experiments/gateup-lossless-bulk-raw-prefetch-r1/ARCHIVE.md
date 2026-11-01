# Experiment 218 research evidence

Measured directory is `outputs/megakernel-20261025/gateup-lossless-bulk-raw-prefetch-r1` in tiny-prl. The candidate replaces CSR-dependent per-row raw-escape hints with cooperative aligned cache-line hints over each layer's complete raw table. Plane/header hints, SIMD decode, projection math and row mapping are unchanged. Original SIMD control and all CSR allocations remain intact.

Reject this candidate. SIMD control median is 75.456001 us per layer versus 83.616003 us with bulk hints, 10.8143% slower. Five alternating placements and five replays each yield 800 timed layer calls per arm across 32 distinct layers. Full decoded weight and FP32/BF16 output comparisons pass, as do NaN-poisoned graph writes. Dense executes eagerly only for validation; this is not an RL or full serving result.

Both kernels use 80 registers and 7,680 bytes shared memory with no spills or stack. Both guards return zero with no stop reason, peak GPU memory is 1,019 MiB and swap is zero. Neighboring layer cache-line overlap is expected. The rounded-up exclusive end is not issued as an address.

The source-equivalence proof checks the frozen control and exact prefetch-policy change. Reports are deterministic gzip with hashes in `raw-output-manifest.json`. No weights or compiled libraries are included. Source scripts retain measured absolute paths and require the bounded pinned-offline-uv operator workflow.
