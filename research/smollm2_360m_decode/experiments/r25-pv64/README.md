# R25 B2 long-context PV chunk-64 schedule

R25 starts from R23 A-zero. `block360.cu` is byte-identical to R23; the only kernel edit is the B2 PV dispatch in `block360_kernel.inc`. For `max(seq_s) > 2048`, except the existing homogeneous B2 K=512 branch, R25 selects eight Y groups, chunk 64, one persistent FP32 `__fmaf_rn` accumulator per Y lane, and the halves fold. The B1 path, B2 K=256 path, remaining B2 paths, QK, softmax, rounding boundaries, workspace, task indexing, and ABI remain unchanged.

## Full decoder validation

R25 passes all 46/46 cases with the existing gates unchanged: initial 4/4, held-out 30/30, wide 12/12. Relative to R23, it adds `seed777-b2-ctx8000` and removes no passes. At that formerly failing case, final hidden and logits are 0 ULP. The wide attention internal debug capture has a maximum 10 ULP diagnostic delta, while the acceptance-stage gates pass; internal debug bitwise identity is not an acceptance criterion.

The 2048 context case (effective sequence length 2049) also passes through the new schedule. Validation summary SHA-256 is `8c4875dba13e08deae3c9eb5b99375a70d55f3a8f8388efb445b436cb86e0019`. The summary records raw group comparison and guard hashes. The R25 DSO was `2c307e4a21eb080e1d1bd9999611e7f7eb32829d5375b0af2bf3dd9634c6efe6`; no shared library is archived. PTXAS reported B2 at 152 registers/thread, zero spills, 16-byte stack and 13,344 shared bytes; B1 at 147 registers/thread, zero spills, 8-byte stack and 6,944 shared bytes. Workspace remains 536,320 bytes and the decoder ABI remains 28 arguments.

## Native PV schedule evidence and limits

The CUPTI R2 capture observed exactly one matching GEMVX launch. `cuKernelGetParamInfo` returned one formal argument of 152 bytes; the launch grid was `[2,1,30]`, block `[32,8,1]`, with 2,176 dynamic shared bytes. The saved native PV result contains 1,920 BF16 outputs and is bitwise exact. The chunk-64 candidate replay matched all 1,920 values for all 32 saved layers; fresh Torch also matched all layers. The older fixed R21 schedule matched 21/32 layers in the same probe.

The captured SASS advances the K tile by 64 times the eight Y groups (512 positions). The candidate's chunk-64 schedule is supported by the captured launch parameters and replay, but exact native per-K traversal and the complete SASS parameter mapping remain inferred rather than proven. The targeted corpus replay is component evidence; full decoder acceptance is established by the independent 46-case validation above.

The archive includes `native-pv-chunk64-summary-r1.json`, a compact decoded-parameter summary with no raw formal-argument bytes or pointer addresses; `parameter-map.md`; and the CPU/replay source scripts. `chunk64-replay-r1.json` contains per-layer mismatch counts and checksums, not saved tensors. Raw CUPTI argument bytes, `.so`, `.pt`, and raw model tensors are not archived.

## Direct decoder timing

The direct R23/R25 benchmark used the median of six placement statistics, where each placement statistic is the upper median of 20 CUDA-event samples. Three ABBA/BAAB blocks were collected per case. The full decoder step was timed; LM head, prefill, cache restore, parity, and telemetry were outside timing.

| Fixture | R23 control | R25 | Change | Strict both-beat-both blocks | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| B1, ctx32 | 3.649568 ms | 3.648000 ms | −0.042963% | 1/3 | Neutral unaffected-path control |
| B2, ctx2048 | 12.654096 ms | 11.650048 ms | −7.934568% | 3/3 | Correctness-valid decoder-component reduction |
| B2, ctx8000 | 26.344016 ms | 22.657537 ms | −13.993613% | 3/3 | Diagnostic only: R23 fails this case's numerical gate |

R25 native hidden/KV checks are 0 ULP in all three benchmark fixtures. At context 2048, both R23 and R25 pass their native hidden/KV gates. At context 8000, R25 is correct under the full 46-case validator but R23 is not, so the timing delta cannot be treated as a quality-equivalent comparison. These are decoder-only results; they do not establish serving or main-RL wall-time gains. Compact benchmark metadata is `direct-bench-summary-r2.json` (SHA-256 `57f5b065656effbc981df047cc772fcfcaf9ef4cedbfb5724f9be615dde32572`); raw events remain in the ignored Prime-RL outputs with SHA-256 `67e6aacb2c76895517259dc95b3ccb0c5248304e2282e18053603d140d85fe6a`.

## Archived source hashes

| Artifact | SHA-256 |
| --- | --- |
| `block360.cu` | `cff9f0f0ac3d2dcdcd8fe9026346cf01313c1247588ba8f213d4005a33b15298` |
| `block360_kernel.inc` | `48423f3f430d886e18aa3300a322ba74c9659c3efa9c5dd1f896c85f33f3150a` |
| `build.sh` | `bb5b6ac42a8eebdb9662740c1ac0ce693b38e93da7022e488f5c87b96edec450` |
| `source_preflight.py` | `35eb063008140244ab0b3569ebea2346ea3db50a9bf88afd092e4bf2e89db8d1` |
| `validation-summary-r1.json` | `8c4875dba13e08deae3c9eb5b99375a70d55f3a8f8388efb445b436cb86e0019` |
| `targeted_replay.py` | `9e2c83b33f4c176baafaf43d6db197604759fd705834464b48411c774762ffdc` |
