# R2 formal-parameter map and chunk-64 derivation

## Captured parameter evidence

`decode360-pv-launch-args-r2/results-r1/launch-args.jsonl` reports `cuKernelGetParamInfo` success for formal argument 0 at device offset 0, size 152 bytes, then `CUDA_ERROR_INVALID_VALUE` (1) at the next index. It used no function-API fallback. The captured launch is grid `[2,1,30]`, block `[32,8,1]`, dynamic shared bytes 2176. The matching saved eager PV replay is exact at all 1920 BFloat16 outputs.

The raw 152-byte parameter argument independently aligns with the extracted SASS constant-bank fields if the formal argument begins at `c[0][0x160]`:

| SASS c[0] offset | Raw formal byte offset | Captured uint32 | Interpretation/evidence |
|---|---:|---:|---|
| `0x178` | `0x18` | 8001 | SASS K/stride use; matches native key length |
| `0x1a0` | `0x40` | 64 | output width |
| `0x1a4` | `0x44` | 8001 | key length |
| `0x1a8` | `0x48` | 64 | value K-row element stride |
| `0x1ec` | `0x8c` | 64 | per-Y work span/inner tile parameter |
| `0x1f0` | `0x90` | 1 | scalar mode/stride parameter used by address math |

The first captured 8 bytes are a pointer pair and line up with SASS use of `c[0][0x160]`/`[0x164]`; subsequent captured pointer/stride fields line up at `[0x168]`, `[0x170]`, `[0x178]`, etc. The complete 152-byte size is 0x98, which contains the highest SASS field `[0x1f0]` (four bytes) when based at 0x160 and ends at 0x1f8. Together these provide an independent, internally consistent base mapping. The base remains an inference from launch metadata, pointer-field layout, and shape constants; the cubin formal-parameter base has not been separately read from `.nv.info`.

## Native schedule evidence and remaining inference

The SASS loads `c[0][0x1ec]` into R19 at `0x4610`, then advances the loop tile register with `R28 += R19 * c[0][4]` at `0x4630`. The observed block Y dimension is 8; the same `c[0][4]` parameter controls the Y fold width (R5 at `0x4780`), which runs 8->4->2->1 (`0x47a0`-`0x4830`). Therefore the loop tile advances by `64 * 8 = 512` K positions. The chunk-end calculation also multiplies block Y by `c[0][0x1ec]` (`0x0530`).

One FP32 accumulator chain is zero-seeded at `0x04d0`, persists over the loop backedge `0x46a0 -> 0x0510`, and is stored as one partial per Y lane at `0x4730`. The unrolled body walks BF16 value addresses in K-row stride increments and applies FFMA to the running chain. Its final Y-lane combination is the halves tree, matching R21's `((p0+p4)+(p2+p6))+((p1+p5)+(p3+p7))` form.

The targeted candidate models each Y lane as a single FMA accumulator over a 64-token chunk, with Y lanes starting at consecutive 64-token segments per 512-token tile, and then the halves tree. This segment start and the exact shared probability load permutation are not fully reconstructed from SASS. Treat the candidate as a targeted, native-parameter-derived hypothesis to test against the saved corpus, not as a proven per-K traversal.
