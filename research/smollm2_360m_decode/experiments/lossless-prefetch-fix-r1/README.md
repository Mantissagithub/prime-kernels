# Experiment 225: repair the integrated lossless decoder regression

The fixed opt-in 360M decoder is 9.18–10.72% faster than the original lossless decoder at batch two. It also beats R30 by 0.93–1.25% in the packaged-build comparison. These are full 32-layer decoder-step CUDA-event measurements, excluding the LM head, prefill, KV restoration, cache refresh and RL orchestration. Batch one is effectively unchanged. No verified RL wall-time speedup is claimed.

The root agent implemented this fix directly. An independent read-only review found no correctness blocker.

## Matched measurements

Each process uses three alternating blocks, six placements per arm/case and 20 events per placement: 120 events per arm/case. The values below are medians of placement medians. Every B2 block passes `max(candidate placement medians) < min(control placement medians)`. B1 does not claim an improvement.

| B2 context | Original lossless | Fixed lossless | Latency change |
|---:|---:|---:|---:|
| 32 | 3.459072 ms | 3.088336 ms | −10.72% |
| 128 | 3.539312 ms | 3.173376 ms | −10.34% |
| 1024 | 3.993600 ms | 3.627088 ms | −9.18% |

The table above uses `reports/old-new-r1.json.gz`, with a frozen original library and the actual rebuilt package. The following table is a separate matched process against R30. Do not combine values across the two timing blocks.

| B2 context | R30 | Fixed package | Latency change |
|---:|---:|---:|---:|
| 32 | 3.124880 ms | 3.085824 ms | −1.25% |
| 128 | 3.211696 ms | 3.173792 ms | −1.18% |
| 1024 | 3.663248 ms | 3.629072 ms | −0.93% |

Two preceding private-build processes independently measured 0.93–1.26% lower B2 latency than R30, passing every B2 strict block at contexts 32, 128 and 1024. All six cases in each process preserve R30 final hidden states and every layer's K/V bits. The packaged test also replays a captured graph with encoded buffers deliberately poisoned and mode flags disabled; it must still match R30 through the dense fallback. GU-only in-place mutation and refresh must change graph output and match eager execution.

## Changes and proof

- Stage warp-owned CSR metadata and first-row hints before the existing grid barrier; synchronize same-warp readers with `__syncwarp`.
- Keep the rounded SiLU intermediate and up projection in registers instead of a global workspace store/load.
- Use sign-extending PRMT selectors and a mask to form the same BF16 sign/mantissa pairs. CPU proof exhausts all 65,536 byte pairs for both selectors. The PTX selector semantics are documented in the [NVIDIA instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt).
- Share the B2 projection body between encoded and native fallback loads. Retain the original B1 path, accumulator ordering, reduction and rounding boundaries.
- Load the always-valid plane alongside the header. Only raw escape loads remain predicated; escape results still select the original raw words. This removes the header-to-plane address dependency at the cost of reading the plane for escaped chunks too.

The exact candidate helper was extracted into `scripts/decode_words.cu` for an independent bit test. It checks all 65,536 BF16 patterns over 4,915,200 words per pass, three passes including in-place refresh and restoration. Cache creation occurs in inference mode and refresh outside it. All passes are bit-exact and the descriptor pointer stays stable. The packaged B2 kernel uses 246 registers, 44,848 bytes shared memory and zero local/stack spill; B1 uses 241 registers and 36,644 bytes shared memory.

`decision.tsv` records all measured trials. Whole-layer hinting, deeper lookahead, L1 hinting, memory-clobber removal and paired load/reconstruct splitting did not beat the retained schedule and were discarded. Source preparers and compressed reports preserve those outcomes. Private phase profiles record scheduled prefetch/barrier, normalization and GU/barrier intervals. They are diagnostic cycle intervals, not pure DRAM costs or wall-time shares; performance claims use uninstrumented comparisons. The first profile invocation failed its local-package import before timing and is preserved in guard records.

## Full-parameter RL validation

The actual packaged fix completes two updates in the unchanged fork-server/stream64 recipe, with the route active and three cache refreshes through generations 1–4. All 32 layers remain encoded. Loss, gradient norm and mismatch KL are finite at both updates. The run records 392 rollouts, 196 tasks and 43,007 tokens in 80.872738 seconds; this wrapper interval includes post-run validation. It is a functional validation, not a matched wall-time comparison. Guarded elapsed is 81.155801 seconds, sampled GPU peak 7,222 MiB, peak job memory 7,413,661,696 bytes, minimum host-available memory 3,882,037,248 bytes and job swap zero. The training optimization/reduction dtypes remain FP32.

## Reproduction and artifact identities

Measurements originate in `prime-rl/outputs/megakernel-20261025/lossless-prefetch-fix-r1`. The original source corresponds to prime-kernels commit `1a23274`; its unchanged kernel sources are included in `frozen-source`. `accepted/block360_kernel.inc` is the measured final include used in the package. Source preparers retain the original workspace paths and must run against that frozen source state, in their recorded order; they are research snapshots rather than portable installers. The packaged benchmark, old/new benchmark and RL wrapper snapshots retain their original dependency paths. The inference recipe remains explicitly opt-in.

Reports include source/include/library hashes. `manifest.json` records original and archived hashes for each file; gzip timestamps are fixed to zero. All compressed JSON and manifest entries were checked after archiving. No weights or compiled libraries are included. The model snapshot and frozen native validators are the same as Experiment 224.
