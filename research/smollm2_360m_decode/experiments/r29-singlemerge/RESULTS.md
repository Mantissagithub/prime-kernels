# R29 adaptive single-CTA merge comparison

The candidate compiled under the shared GPU lock with CUDA hidden, a 2 GiB systemd memory limit, zero swap, and a 120-second runtime limit. `ptxas` reports B1 at 249 registers and 36,644 bytes static shared memory, no spills; B2 at 245 registers and 43,056 bytes, no spills. Relative to FAST R28, B2 uses nine more registers.

Six separate guarded direct-decoder runs compared FAST R28 (A) with the candidate (B), under the canonical GPU runner and a 3,072 MiB cap monitor. Each arm has 120 CUDA-event samples per case. The table reports the canonical median over all 120 raw CUDA-event samples per arm; the paired-block comparison remains based on placement medians:

| Batch/context | R28 ms | R29 ms | Change | Blocks where both R29 placements beat both R28 placements |
|---|---:|---:|---:|---:|
| B1 / 32 | 2.985984 | 2.983936 | −0.0686% | 3/3 |
| B1 / 128 | 3.035136 | 3.038208 | +0.1012% | 0/3 |
| B1 / 1024* | 3.341280 | 3.352576 | +0.3380% | 0/3 |
| B2 / 32 | 3.154944 | 3.155968 | +0.0324% | 0/3 |
| B2 / 128 | 3.241648 | 3.243008 | +0.0419% | 0/3 |
| B2 / 1024* | 3.686400 | 3.708928 | +0.6112% | 0/3 |

*Seeded synthetic context; its sequence length is 1025, so it exercises the single-merge path.

All six cases passed exact R28/R29 hidden-state and all-layer K/V parity: hidden state and all 32 key/value layers were bitwise equal, with zero maximum K/V difference. All guards returned 0 with zero swap. Guard GPU peaks ranged from 1,315 to 1,869 MiB; the independent cap monitor also stayed below 3,072 MiB. The candidate is slower in both long-context cases, so the single-merge hypothesis is not supported by this run. No promotion is recommended.

Candidate library SHA256: `4e3b300f90b8ab1d9899806ab8916ac9a53b3b33dc0cd68f1c7a2880d84a142e`. Raw event data and guard telemetry are in `results/` and `guard-*/`.
