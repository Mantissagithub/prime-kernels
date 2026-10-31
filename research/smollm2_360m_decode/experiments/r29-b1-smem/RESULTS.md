# R29 B1 shared-memory extent check

The candidate compiled successfully. `ptxas` reports B1 at 249 registers and 21,784 bytes static shared memory, with no spills; B2 remains at 236 registers and 43,056 bytes, with no spills. The shared-memory reduction is confirmed, but the interleaved timings do not establish a speedup.

Six separate guarded processes ran the existing R28-control/R29-candidate harness at contexts 32, 128, and 1024 for B1 and B2. Each arm had 120 CUDA-event samples per case (three warmups and three A-B-B-A/B-A-A-B blocks, 20 repeats per placement). Pooled placement medians:

| Batch/context | R28 ms | R29 ms | Change | Blocks with both R29 placements faster than both R28 placements |
|---|---:|---:|---:|---:|
| B1 / 32 | 2.98547 | 2.98594 | +0.02% | 0/3 |
| B1 / 128 | 3.03514 | 3.03821 | +0.10% | 0/3 |
| B1 / 1024* | 3.34285 | 3.34131 | −0.05% | 2/3 |
| B2 / 32 | 3.15333 | 3.15194 | −0.04% | 2/3 |
| B2 / 128 | 3.23891 | 3.24046 | +0.05% | 0/3 |
| B2 / 1024* | 3.68947 | 3.68845 | −0.03% | 1/3 |

*Synthetic seeded 1024-token inputs; contexts 32/128 use frozen fixtures.

Exact R28/R29 parity passed for all six cases: final hidden states and all 32 layers of K and V were bitwise equal; maximum K/V absolute difference was zero. Every run returned 0, with zero swap, and stayed below the 3,072 MiB cap. Guard-reported GPU peaks ranged from 1,315 to 1,869 MiB. These results confirm the extent-only source change preserves the observed outputs; they do not justify calling R29 faster or promoting it.

Build library SHA256: `0e37fd1f377d6cf03dbca9068637a261c09d326eb14de4a1263ce6f4efd1b908`. Per-case raw events and guard telemetry are in `results/` and `guard-*/`.
