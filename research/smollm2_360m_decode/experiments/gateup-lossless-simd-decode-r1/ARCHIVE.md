# Experiment 217 research evidence

Measured directory is `outputs/megakernel-20261025/gateup-lossless-simd-decode-r1` in tiny-prl. PRMT and packed integer arithmetic reconstruct pairs of BF16 words while preserving original raw-escape hints, row mapping, FMA order and reduction. Original compressed control remains unchanged. Source equivalence and CPU arithmetic proofs are included.

This is a compressed-decoder component improvement, not a serving or RL win. The two-arm first run gives 86.304 to 75.424 us per layer. A balanced three-arm run in the same session gives dense 43.136001, original compressed 86.458500, SIMD 75.467002 us per layer. SIMD latency is 12.7130% lower than compressed control but 74.9513% higher than dense, so serving adoption is rejected. Each arm has six placements, five replays per placement and 32 distinct layer calls per replay, totaling 960 timed calls per arm. The schedule is dense/control/SIMD/SIMD/control/dense repeated three times.

The candidate GPU decoder reconstructs all 157,286,400 original BF16 words exactly. Dense, compressed and SIMD projection outputs match in FP32 and BF16, with 327,680 elements checked per arm and poisoned replay writes verified for all three graphs. Raw event ranges do not overlap between arms in the balanced run. The CPU arithmetic proof checks all 65,536 byte pairs, 16,075 valid delta/base cases including decoded exponent 255, 64,300 sign/mantissa corners and four PRMT selector mappings. Raw escape behavior is reviewed in source and exercised by the full-weight GPU test.

SIMD uses 80 registers versus compressed control 56 and dense 64, all with 7,680 bytes shared memory, no spills and no stack. Both runs pass the guards at 1,019 MiB sampled GPU peak with zero swap. These measurements exclude SwiGLU, full inference and RL. Scripts retain measured absolute paths. Use `harness_3arm.py --skip-build --output NEW_PATH` through the bounded pinned-offline-uv operator workflow to reproduce the three-arm measurement.

Reports and SASS are deterministic gzip with original and compressed hashes in `raw-output-manifest.json`. No weights or compiled libraries are included.
