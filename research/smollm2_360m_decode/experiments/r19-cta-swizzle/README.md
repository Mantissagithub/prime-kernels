# R19 B2 CTA mapping swizzle

R19 is a private copy of frozen R16 with one mapping change: B1 keeps `blockIdx.x * kWarps + warp`; B2 uses `warp * kCTAs + blockIdx.x`. All `gw`-based task loops retain the same 192-global-warp stride, and each warp retains its original arithmetic order. The source-derived CPU checker confirms bijection and exact-once coverage for QKV, O/down, and gate/up projection tiles. Those task sets spread across all 24 CTAs under R19, versus 13, 8, and 20 active CTAs under R16. This establishes work coverage, not GPU speed or correctness.

R19 completes the full 46-case check: 34/34 original and initial fixtures pass, and the wider set passes 6/12. The six wide failures exactly match R16's pass/fail set, with no added pass or regression.

Two independent direct decoder-only ABBA sessions used three warmups and three blocks each. In both sessions, B2 medians improved by 0.39% at ctx32 and 0.34% at ctx128; R19 beat R16 in all three pooled blocks for each B2 shape in both runs. B1 differences stayed within about 0.03%, consistent with a neutral control. All four timing cases in both runs had exact 0-ULP final hidden and all-layer KV parity against native and across candidates. This is a small, measured decoder-component result; it does not support a broad performance claim or a serving/RL gain.

`result.json` records both sessions, validation comparisons, guard data, and source/build hashes. `check_mapping.py` and `mapping-check.json` contain the CPU-only coverage check. The compiled binary is omitted. Build with CUDA 13.0 for SM89 under the host resource guard.
