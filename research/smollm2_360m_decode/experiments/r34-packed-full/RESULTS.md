# Full packed B2 candidate rejected

The first six-case direct decoder comparison passes exact final-hidden and all-layer KV parity against R30, but fails the speed gate on every B2 shape. R34 is not adopted. No serving or RL improvement is established.

| Batch/context | R30 pooled event median | R34 | Change | Strict winning blocks |
|---|---:|---:|---:|---:|
| 1/32 | 2.886288 ms | 2.889728 ms | +0.119% | 0/3 |
| 1/128 | 2.941952 ms | 2.939104 ms | -0.097% | 2/3 |
| 2/32 | 3.128320 ms | 3.152832 ms | +0.784% | 0/3 |
| 2/128 | 3.212224 ms | 3.243008 ms | +0.958% | 0/3 |
| 1/1024 | 3.247104 ms | 3.239936 ms | -0.221% | 3/3 |
| 2/1024 | 3.669024 ms | 3.689472 ms | +0.557% | 0/3 |

Each case runs ABBA/BAAB/ABBA blocks with 20 events per placement and 120 events per arm. Strict winning blocks require both candidate placement medians to beat both control medians. The displayed medians pool raw events rather than placement medians. LM head is excluded. Frozen fixtures cover contexts 32/128; context 1024 uses a seeded synthetic prompt with real model weights.

B1 compiles to 241 registers and 36,644 bytes shared memory, unchanged from R30. B2 uses 239 registers and 32,816 bytes versus R30's 245 and 43,056. Both have zero stack/spills. The inner guard returns zero with no stop reason, 1,955 MiB sampled GPU peak, zero sampled swap and minimum host available RAM of 8,625,176,576 bytes.

This archive preserves measured source, harness, raw event samples, guard results and compiler resource output. Absolute paths refer to the original measured working directory under `prime-rl/outputs/megakernel-20261025/r34-packed-full`. No compiled libraries or model weights are included. The README describes the source-only handoff before this measurement; this file records the subsequent root-run result.

The isolated projection signal did not survive full integration. Selective packing of QKV/O while retaining FP32 gate/up and down is the next hypothesis. The cause of R34's regression has not been isolated by a full-kernel stage profile.
