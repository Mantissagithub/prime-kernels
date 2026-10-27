# R23 B2 projection A-tile zero-fill

R23 is an isolated R22 copy. Its only kernel change is in `wmma_project_tile`: initialize A-tile rows 2–15 to BF16 zero once per warp/helper invocation, then write only rows 0–1 on each K16 step. The conversion, K16 MMA accumulation order, weight load order, barriers, ABI, and acceptance gates are unchanged. The include is byte-identical to R22.

## Validation

R23 passes 45/46 full decoder cases with the exact R22 pass set: initial 4/4, held-out 30/30, wide 11/12. The only failure remains seed777 B2/context8000. No additional failure was introduced.

Two guarded direct decoder-only runs used the same four seed-17 B1/B2 context-32/128 fixtures, with three alternating ABBA/BAAB blocks, three warmups, and 20 CUDA-event samples per arm entry. Timing excludes LM head, prefill, KV restore, and parity. The reported arm statistic is the median of its six stored upper-medians, not a pooled raw-sample median.

| Fixture | R1 R22 → R23 | R1 delta | R1 strict block wins | R2 R22 → R23 | R2 delta | R2 strict block wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1, ctx32 | 3.648512 → 3.649536 ms | +0.028066% | 0/3 | 3.651072 → 3.649456 ms | −0.044261% | 1/3 |
| B1, ctx128 | 3.740672 → 3.740688 ms | +0.000427% | 0/3 | 3.740672 → 3.741696 ms | +0.027369% | 0/3 |
| B2, ctx32 | 7.958096 → 7.912960 ms | −0.567171% | 3/3 | 7.956880 → 7.909888 ms | −0.590584% | 3/3 |
| B2, ctx128 | 8.202240 → 8.161792 ms | −0.493136% | 3/3 | 8.198768 → 8.157184 ms | −0.507204% | 3/3 |

B1 is effectively flat. B2 shows a small consistent direct-decoder reduction, with all six blocks per B2 shape across the two runs satisfying the strict both-beat-both criterion. For each of the four shapes in each run, 195 hidden/KV comparisons across native, cross-candidate, and R22 references were bitwise exact at 0 ULP. This is a decoder-component result only; it does not establish a serving or RL wall-time improvement.

Validation and performance details are in `r23-direct-bench-two-run-summary.md` and `r22-r23-passset-comparison-r1.json`. The summary records raw source, DSO, direct-run, and pass-set artifact hashes. Raw direct outputs remain under the Prime-RL ignored output tree. No DSO or PTX is included in this archive.

## Archived source hashes

| Artifact | SHA-256 |
| --- | --- |
| `block360.cu` | `cff9f0f0ac3d2dcdcd8fe9026346cf01313c1247588ba8f213d4005a33b15298` |
| `block360_kernel.inc` | `4c1a684303d0329238ab5f3058c7f4058ae0fe36c9b13a44adc3dbb649d8be55` |
| `build.sh` | `d294ef662186cf3c5815c97f3be83af23ebaac83adec516b8922cd778e8cd9c7` |
| `direct_bench.py` | `d12d64ab8b7469d93cafd2708310a12b763eb773be5365b24b1c353e80e7c6cf` |
