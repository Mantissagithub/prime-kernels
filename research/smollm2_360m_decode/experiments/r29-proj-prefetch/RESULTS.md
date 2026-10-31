# R29 projection prefetch comparison

The candidate compiled with no spills. `ptxas` reports B1 at 242 registers, 36,644 bytes static shared memory, and a 48-byte stack frame; B2 at 254 registers, 43,056 bytes shared memory, and a 48-byte stack frame. B2 is 18 registers above FAST R28 and near the per-thread register limit.

Six separate guarded direct-decoder runs compared FAST R28 (A) with the prefetch candidate (B). Each arm has 120 CUDA-event samples per case. Table values are the median across all raw event samples, matching the experiment ledger definition.

| Batch/context | R28 ms | R29 ms | Change | Blocks where both R29 placements beat both R28 placements |
|---|---:|---:|---:|---:|
| B1 / 32 | 2.98803 | 3.01773 | +0.99% | 0/3 |
| B1 / 128 | 3.03821 | 3.07187 | +1.11% | 0/3 |
| B1 / 1024* | 3.34336 | 3.36794 | +0.74% | 0/3 |
| B2 / 32 | 3.15597 | 3.16006 | +0.13% | 0/3 |
| B2 / 128 | 3.24198 | 3.24506 | +0.09% | 0/3 |
| B2 / 1024* | 3.69050 | 3.68640 | −0.11% | 3/3 |

*Seeded synthetic context; contexts 32/128 use frozen fixtures.

Exact R28/R29 parity passed for all six cases: final hidden states and all 32 layers of K and V were bitwise equal, with zero maximum K/V difference. All guards returned 0 with zero swap; GPU peaks ranged from 1,317 to 1,911 MiB, below the 3,072 MiB cap. B2/context 1024 is the sole timing win across all three blocks, but its pooled raw-event improvement is only 0.11%. The prefetch candidate slows the other five shapes, including B1/context 1024. Do not promote this general candidate; keep the B2 long-context result as a narrow follow-up hypothesis only.

Candidate library SHA256: `74c7121409dba08aab1d4a06ecc84ebd165b43cc10a09296c6e3ef621d3f731c`. Raw events and guard telemetry are in `results/` and `guard-*/`.
