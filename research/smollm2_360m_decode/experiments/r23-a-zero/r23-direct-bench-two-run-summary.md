# R23 A-zero validation and direct timing summary

## Validation pass set

The R23 GPU validation queue completed under the canonical guarded runner. The operator derived each case result as `all(case.gates.values())` and compared R22/R23 pass sets from the three per-group comparison files.

| Group | Cases | R22 passes | R23 passes | Pass set |
|---|---:|---:|---:|---|
| Initial | 4 | 4 | 4 | Same |
| Held out | 30 | 30 | 30 | Same |
| Wide | 12 | 11 | 11 | Same |
| Total | 46 | 45 | 45 | Same |

The sole failure is `seed777-b2-ctx8000`; it remains in the R22 and R23 failure sets. No new R23 validation failure appeared.

## Direct decoder timing

Two separately guarded runs used the same frozen direct benchmark: four seed-17 B1/B2 context-32/128 fixtures, three ABBA/BAAB blocks, three warmups per library, and 20 CUDA-event samples per arm entry. Prefill, KV restoration, parity checks, and telemetry are outside the event intervals. LM head is excluded.

For each arm, the six stored values are upper medians of its six 20-event samples, two per block across three blocks. The reported arm statistic is the median of those six upper medians, not a pooled median of raw events. A block is an R23 win only if both R23 arm medians are below both R22 arm medians (`max(R23) < min(R22)`).

| Fixture | Run 1 median of six (R22 → R23) | R23 delta | R23 block wins | Run 2 median of six (R22 → R23) | R23 delta | R23 block wins |
|---|---:|---:|---:|---:|---:|---:|
| B1, ctx32 | 3.648512 → 3.649536 ms | +0.028066% | 0/3 | 3.651072 → 3.649456 ms | -0.044261% | 1/3 |
| B1, ctx128 | 3.740672 → 3.740688 ms | +0.000427% | 0/3 | 3.740672 → 3.741696 ms | +0.027369% | 0/3 |
| B2, ctx32 | 7.958096 → 7.912960 ms | -0.567171% | 3/3 | 7.956880 → 7.909888 ms | -0.590584% | 3/3 |
| B2, ctx128 | 8.202240 → 8.161792 ms | -0.493136% | 3/3 | 8.198768 → 8.157184 ms | -0.507204% | 3/3 |

Both B1 fixtures remain effectively flat across the two runs. Both B2 fixtures show a small, consistent R23 reduction, with all six B2 blocks meeting the strict both-beat-both win criterion. This is evidence for a small direct-decoder improvement on these fixtures; it does not establish production serving impact or a hardware cause.

For each fixture in each run, parity was exact across 195 comparisons: 65 final-hidden/KV tensors for R22 versus native, 65 for R23 versus native, and 65 for R22 versus R23. All checks were bitwise equal, with zero differing elements and maximum BF16 ULP 0.

## Provenance and guards

| Artifact | SHA-256 |
|---|---|
| R23 `block360.cu` | `cff9f0f0ac3d2dcdcd8fe9026346cf01313c1247588ba8f213d4005a33b15298` |
| R23 `block360_kernel.inc` | `4c1a684303d0329238ab5f3058c7f4058ae0fe36c9b13a44adc3dbb649d8be55` |
| R23 shared library | `dfe9fe54a3cbf6c1043709b9eeb9fead1722f075f6fc524cdaa006fd9621a3c7` |
| R22 `block360.cu` | `180be7870c87b5993f02529d89546b8f771b261398af59608d35ab3fc12003f8` |
| R22 shared library | `6c614aafb4907dea94b3c599c9bf97936e60027e447330838e9ba5a094fcfcec` |
| Direct benchmark | `d12d64ab8b7469d93cafd2708310a12b763eb773be5365b24b1c353e80e7c6cf` |
| R22/R23 pass-set comparison | `1bb6cbe16f454d8167fcbaac401dcf580226509aeb681beb88992d4820fc16f0` |
| Run 1 direct result | `140e223a4e7e20c60f666706389b389776f38acee33380d33d03239e7171021e` |
| Run 2 direct result | `8d260d120c9be43f75838f2210639eff0bc6e935e55d23be3f6cc8cd6f141fea` |

The validation guards completed the initial and held-out runs with return code 0; the wide group returned code 1 solely for its matched 12th failure. Direct timing runs 1 and 2 both returned code 0. All five guard records report zero swap and no guard stop. Peak GPU memory was 1357/1331/4323 MiB for initial/held-out/wide validation, then 1401 MiB for each direct run. Direct run peak cgroup memory was 1,915,920,384 bytes and 1,757,470,720 bytes, respectively.

Raw reports are under `reports/r23-{initial4,heldout30,wide12}-r1/`; direct results are `decode-360-cuda-r23-a-zero/direct-bench-r1.json` and `direct-bench-r2.json`.
