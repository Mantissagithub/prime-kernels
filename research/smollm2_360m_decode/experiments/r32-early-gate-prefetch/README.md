# R32: early gate/up weight prefetch

R32 is a private source candidate based on the tested R30 FAST projection + split-KV implementation. It moves the existing gate/up phase-ahead prefetch block from after O projection/residual to the point after merge shared synchronization and debug capture, immediately before O projection. The block and its row assignments are unchanged; it is moved, not duplicated.

The testable hypothesis is that gate/up weight requests can progress while O projection consumes its independently prefetched O weights. This might hide some weight-fetch latency before gate/up computation begins. It could also contend for cache/request resources, and the cache behavior has not been measured. This is a scheduling experiment, not a speed estimate.

Arithmetic order, all grid and shared barriers, decoder ABI, row mapping, and the other three R30 hint sites remain unchanged. The six-case direct harness compares R30 and R32 for frozen contexts 32/128 at B1/B2 and seeded synthetic context 1024 at B1/B2, with 3 warmups, 3 alternating A-B-B-A blocks, and 120 CUDA-event samples per arm/case. Final hidden state and every KV layer are checked bitwise outside timing. `run_case_cap3g.sh` runs a single whitelisted case through the existing guarded runner with a 3072 MiB GPU cap.

`source_preflight.py` verifies that host source is byte-identical to R30 and that the kernel differs only by relocating the exact existing gate/up prefetch block.

## Build and bounded direct result

The private DSO SHA-256 is `5789a5ed94d7304da00d48ae9e7fbe8399ceeb9b2f7774b639894781112dbc19`. Ptxas reports B1 at 241 registers/36,644 bytes shared memory and B2 at 245 registers/43,056 bytes; both have zero spills and zero stack. Build used `candidate/build.sh` under the canonical lock with CUDA hidden, 2 GiB memory, zero swap, and a 120-second deadline.

All six cases ran separately with the 3072 MiB cap, exact final-hidden/all-KV parity, and 120 raw CUDA-event samples per arm. The table reports the median of all samples; “strict blocks” counts blocks where both R32 placements beat both R30 placements.

| Case | R30 ms | R32 ms | Change | Strict blocks |
|---|---:|---:|---:|---:|
| B1, context 32 | 2.889408 | 2.915328 | +0.897% | 0/3 |
| B1, context 128 | 2.940928 | 2.968576 | +0.940% | 0/3 |
| B2, context 32 | 3.129344 | 3.114928 | -0.461% | 2/3 |
| B2, context 128 | 3.213312 | 3.195008 | -0.570% | 3/3 |
| B1, synthetic context 1024 | 3.241984 | 3.275776 | +1.042% | 0/3 |
| B2, synthetic context 1024 | 3.665920 | 3.649920 | -0.436% | 3/3 |

The relocation regresses B1 at all three contexts, exceeding 1% at context 1024, while B2 improves modestly. Do not adopt or promote this shape-dependent result. Raw events, parity, and cap telemetry are in `results/` and `result_summary.json`.
