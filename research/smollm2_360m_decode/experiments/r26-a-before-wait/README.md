# R26 A-fragment preload before B wait

R26 starts from the validated R25 PV64 base. It changes only when A fragments
are loaded in `wmma_project_tile`: preload A(k=0) while the first B tile is in
flight, then load A(k+1) before waiting for prefetched B(k+1). The original
increasing-K K16 MMA order, B buffers, async waits, warp barriers, BF16
conversion, and zero fill are preserved. The R25 include is unchanged.

This archive contains source, build and direct-benchmark scripts, plus compact
preflight and result summaries. The `.so` and tensor `.pt` files are not
archived. R26's built library hash is recorded in `direct-bench-summary-r1.json`
and `validation-summary-r1.json`.

## Full decoder validation

R26 passed all unchanged 46-case HF gates: initial 4/4, held-out 30/30, and wide
12/12. All 472 recorded gate values are true; the R25 case-ID set is unchanged.
This establishes the reported fixture validation only, not general model
quality or serving behavior. Detailed validation, per-case summaries, guard
hashes, and raw comparison hashes are in `validation-summary-r1.json`.

PTXAS reports B1 at 147 registers/thread and B2 at 156, with zero spills for
both. B2 uses four more registers than R25's 152. Static shared memory remains
6,944 bytes for B1 and 13,344 bytes for B2. This is compile metadata, not a
runtime occupancy measurement.

## Direct decoder timing

Two bounded seed-17 direct runs used three alternating ABBA/BAAB blocks per
fixture, three warmups and twenty CUDA-event samples per arm placement. Each
run's reported case statistic is the median of six placement medians. In each
block, both R26 placements must beat both R25 placements for a strict win.
The Oct 31 repeat reused the frozen harness and candidate libraries.

| Fixture | Run | R25 (ms) | R26 (ms) | Change | Strict both-beat-both blocks |
|---|---|---:|---:|---:|---:|
| B1, ctx32 | Oct 28 | 3.657728 | 3.657808 | +0.0022% | 0/3 |
| B1, ctx32 | Oct 31 | 3.644928 | 3.643392 | −0.0421% | 2/3 |
| B1, ctx128 | Oct 28 | 3.749376 | 3.748864 | −0.0137% | 1/3 |
| B1, ctx128 | Oct 31 | 3.734544 | 3.735040 | +0.0133% | 1/3 |
| B2, ctx32 | Oct 28 | 7.921232 | 7.837776 | −1.0536% | 3/3 |
| B2, ctx32 | Oct 31 | 7.902720 | 7.821712 | −1.0251% | 3/3 |
| B2, ctx128 | Oct 28 | 8.165968 | 8.084992 | −0.9916% | 3/3 |
| B2, ctx128 | Oct 31 | 8.146432 | 8.066880 | −0.9765% | 3/3 |

Across the two bounded runs, B2 reductions reproduce at 0.98–1.05% and meet
the strict both-beat-both criterion in all six blocks per shape. B1 remains a
neutral control: neither shape meets that strict criterion in both runs. Each
run's four fixtures passed the native and cross-candidate hidden/KV checks
bitwise at 0 ULP (65 values per candidate-output set). The Oct 31 run's first
B2 ctx32 block crossed a telemetry-reported clock state from 2250/7001 to
2625/8001; later blocks remained at the latter state. The measurement is not
clock-controlled. These are two bounded decoder-component runs, not serving,
full-RL, or broad performance evidence. Raw event records and compact
summaries are in `direct-bench-r1.json`, `direct-bench-summary-r1.json`, and
`direct-bench-r2-summary.json`; the Oct 31 raw output remains under Prime-RL
`outputs/megakernel-20261025/decode-360-cuda-r26-a-before-wait/results-r2/`.

## Archived artifact hashes

| Artifact | SHA-256 |
|---|---|
| `block360.cu` | `fb94a427319c8da1694c6457bebadb8949600e3a9d10fabc685bba2376931f68` |
| `block360_kernel.inc` | `48423f3f430d886e18aa3300a322ba74c9659c3efa9c5dd1f896c85f33f3150a` |
| `build.sh` | `7e4e9bee34fe95b0337c007adfb98e7d8681069b41b32e4aabe56fcf1a5418c0` |
| `direct_bench.py` | `423c1922320ae7d5ccd1353c6d218c2a1206443d3c72a6127df4c0e295d67381` |
| `source-preflight-r1.json` | `f129ed94632c22309f3fd3b15d2b524e128453a056a4c45975227a36f9e552be` |
| `library-preflight-r1.json` | `e821b60f8a57260b90c1d8a149dc274c736b3177fcf639b21bb61bd87ed44529` |
| `validation-summary-r1.json` | `1d07da4306aa625489a7249bb1cd3cd49876a997240c200a3767301113ec3dca` |
| `direct-bench-r1.json` | `cf48a2143b822839a94f2c86491b07820bc1c84f0f5dd329aca06b87b1d688e7` |
| `direct-bench-summary-r1.json` | `42f9b5ebe90e05dbcb24ad795b117f4f8c6b1e7a65d8a8672180d21bf4f40a78` |
| `direct-bench-r2-summary.json` | `564484448232e9968ff4bed6ad73356a1c5bb33f0a3620815fafa527b7cadaaa` |

Original guarded outputs remain under
`prime-rl/outputs/megakernel-20261025/decode-360-cuda-r26-a-before-wait/`.
