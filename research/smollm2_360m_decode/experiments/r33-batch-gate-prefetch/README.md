# R33: batch-specialized gate/up prefetch

R33 starts from tested private R30. It keeps R30's existing post-O gate/up prefetch schedule for B1, where R32 regressed, and uses R32's early gate/up prefetch schedule for B2, where R32 improved. Both sites use `if constexpr`: each compiled batch specialization contains exactly one hint block, and there is no runtime branch. The hint body and row assignment are identical.

All arithmetic, barriers, ABI, other prefetch sites, and projection mappings remain R30-identical. The bounded direct harness compares R30/R33 across contexts 32/128 at B1/B2 and synthetic 1024 at B1/B2, with 3 warmups, three alternating A-B-B-A blocks, and 120 CUDA-event samples per arm/case. It checks final hidden and every layer's KV bitwise outside timing. The per-case runner uses the existing guarded runner with a 3072 MiB GPU cap.

`source_preflight.py` verifies byte-identical host source and exact compile-time specialization of the single R30 hint block.

## Build and bounded direct result

The private DSO SHA-256 is `a8e95aebd452b91c50c8fa78ceb3c8b451d28cbe99d6bd2f77e472408d41c439`. Ptxas reports B1 at 241 registers/36,644 bytes shared memory and B2 at 245 registers/43,056 bytes; both have zero spills and zero stack. Build ran under the canonical GPU lock with CUDA hidden, 2 GiB memory, zero swap, and a 120-second deadline.

All six cases ran separately with the 3072 MiB GPU cap and exact final-hidden/all-KV parity. The table uses the median over all 120 raw event samples per arm. “Strict blocks” counts blocks where both R33 placements beat both R30 placements.

| Case | R30 ms | R33 ms | Change | Strict blocks |
|---|---:|---:|---:|---:|
| B1, context 32 | 2.886656 | 2.886656 | +0.000% | 0/3 |
| B1, context 128 | 2.939904 | 2.941136 | +0.042% | 0/3 |
| B2, context 32 | 3.130352 | 3.116032 | -0.457% | 2/3 |
| B2, context 128 | 3.213168 | 3.197952 | -0.474% | 2/3 |
| B1, synthetic context 1024 | 3.243008 | 3.241984 | -0.032% | 1/3 |
| B2, synthetic context 1024 | 3.666944 | 3.651584 | -0.419% | 2/3 |

The B1 short cases are neutral, and B2 shows small median gains without strict wins in all three blocks. This is not robust evidence for adoption; keep R33 private. Raw samples, parity, and cap telemetry are in `results/` and `result_summary.json`.
