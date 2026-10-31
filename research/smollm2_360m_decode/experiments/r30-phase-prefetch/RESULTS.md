# R30 phase-ahead weight prefetch comparison

R30 compiled under the shared GPU lock with CUDA hidden, a 2 GiB systemd
memory cap, zero swap, 100% CPU quota and a 120-second runtime cap. `ptxas`
reported B1 at 241 registers and 36,644 bytes static shared memory; B2 at 245
registers and 43,056 bytes. Both have zero spills and zero stack bytes.

Six one-case processes compared FAST R28 (A) with R30 (B), each through the
canonical GPU runner and an external 3,072 MiB cap monitor. Each arm has 120
CUDA-event samples per case. Table values are the median of those 120 raw
samples; the final column is the count of blocks where both R30 placement
medians beat both R28 placement medians.

| Batch/context | R28 ms | R30 ms | Change | Both R30 placements beat both R28 placements |
|---|---:|---:|---:|---:|
| B1 / 32 | 2.988032 | 2.888704 | −3.3242% | 3/3 |
| B1 / 128 | 3.038208 | 2.945024 | −3.0671% | 3/3 |
| B1 / 1024* | 3.341312 | 3.239936 | −3.0340% | 3/3 |
| B2 / 32 | 3.155104 | 3.130192 | −0.7896% | 3/3 |
| B2 / 128 | 3.240960 | 3.209216 | −0.9795% | 3/3 |
| B2 / 1024* | 3.690496 | 3.665024 | −0.6902% | 2/3 |

*Seeded synthetic context; contexts 32/128 use frozen fixtures.

Exact R28/R30 parity passed in all six cases: final hidden state and all 32
layers of K/V were bitwise equal, with maximum K/V difference zero. Guards
returned 0 with zero swap. Guard GPU peaks were 1,315–1,869 MiB; the independent
cap monitor observed 1,317–1,991 MiB, below 3,072 MiB. Minimum available host
memory was 8,421,687,296 bytes or higher.

This is a repeatable direct decoder-step result on the six measured shapes; it
excludes LM-head and end-to-end serving work. The next serving comparison is
being prepared separately. This does not resolve FAST R28's existing
per-case numerical gate, so it does not authorize package promotion.

Candidate source SHA-256: `5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638`.
Candidate include SHA-256: `60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83`.
Candidate DSO SHA-256: `625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2`.
Raw samples and per-case guard artifacts are in `results/` and `guard-*/`.
