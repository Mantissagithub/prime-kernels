# R13 B1 projection weight lookahead

This folder archives the R13 source, include, and SM89 build command. R13 changes only the B1 `dots` and `dots16` loops: it issues four successive BF16 weight loads before applying the same FP32 RN multiply/add operations in the original order. It does not change B2 WMMA, reduction order, dispatch, dtype boundaries, barriers, or acceptance gates. The compiled library and capture tensors are intentionally not archived. Source and library hashes, plus the compact measurements, are in `result.json`.

## Direct decoder timing

The separate CUDA-event ABBA run measured the direct 32-layer `decode360` call through final hidden state. The LM head is excluded, as are prefill, KV prefix restore, native reference work, and telemetry. It used three warmups, three alternating ABBA/BAAB blocks, and 120 samples per candidate per case. The six-arm summary uses the recorded per-arm upper medians (`sorted20[10]`), so its median can differ slightly from the standard median of all 120 event samples.

| Case | Median of 120 samples, R12 → R13 | Median of six reported arm medians, R12 → R13 | Ratio from six arm medians |
|---|---:|---:|---:|
| B1, context 32 | 16.8314 → 5.2817 ms | 16.8320 → 5.2817 ms | 3.187× |
| B1, context 128 | 16.9216 → 5.3729 ms | 16.9216 → 5.3729 ms | 3.149× |
| B2, context 32 | 8.4675 → 8.4675 ms | 8.4685 → 8.4673 ms | 1.000× |
| B2, context 128 | 8.7081 → 8.7132 ms | 8.7085 → 8.7132 ms | 0.999× |

All four native comparisons and R12/R13 comparisons had exact final hidden state and all 32 layers of KV (0 ULP). These measurements support a B1 decoder-kernel speed result on these two contexts. They do not establish vLLM, end-to-end serving, long-context, multi-step, or RL throughput gains.

## Numerical validation scope

R13 passes the original 30 held-out cases and four initial controls (34/34 total). It passes 6/12 wider-context cases and fails 6/12, matching R12's pass/fail cases and first post-block/KV divergence layers. The complete 46-case acceptance set is therefore not passed. The exact operator reports and guards are in sibling `prime-rl/outputs/megakernel-20261025/validation-operator-r1/reports/` under `r13-heldout30`, `r13-initial4`, and `r13-wide12`. The direct timing result and guard are in `prime-rl/outputs/megakernel-20261025/r13-direct-bench-r1/`.

Line-mapped SM89 disassembly ties the four-load sequence to both changed B1 loops: `dots` (kernel include line 80) and `dots16` (lines 570 and 614). In each case the four source weights load before dependent projection arithmetic; the R12 control has a per-iteration load followed by its arithmetic. This confirms the intended machine-code change. It does not by itself explain the speedup or quantify memory-stall reduction. See `result.json` for source-line locations and dump hashes.

## Build

From this directory, run `./build.sh` with CUDA 13.0 and SM89 support. The script emits `build/libdecode360_r13.so` and nvcc/ptxas logs.
