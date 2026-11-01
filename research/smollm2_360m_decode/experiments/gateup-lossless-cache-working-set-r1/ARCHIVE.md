# Experiment 221 research evidence

Measured directory is `outputs/megakernel-20261025/gateup-lossless-cache-working-set-r1` in tiny-prl. No decoder, arithmetic, hint policy or weight format is changed. All 32 encoded layers remain on the GPU; only graph call-to-layer mapping changes. Each graph contains 32 calls with exact outputs checked against its mapped dense references.

R1 compares 1/2/4/32 distinct layer sets. Medians are 36.016/36.030/38.190/61.209 us per call. Distinct encoded byte totals including metadata are 8,000,548 / 16,000,232 / 31,990,512 / 256,225,056. Each case has six placements, five replays per placement and 960 timed calls. The one-layer case uses layer zero, so R2 tests that possible content/address bias.

R2 has 33 cases: 32 individually anchored single-layer graphs and one all-layer graph. Warm-anchor medians range 31.3920–31.4855 us, with median 31.4410 us. All-layer traversal is 53.6480 us in the same block. All cases use six placements, one untimed 32-call prewarm before each placement and five timed replays, giving 960 timed calls, 192 prewarm calls and 32 correctness replay calls per case. Prewarm is outside CUDA events. R1 and R2 are separate blocks with different setup; comparing their absolute medians is not an optimization result.

Both runs verify all 157,286,400 decoded BF16 words. Every case verifies 327,680 FP32 and BF16 outputs and poisoned replay writes. Guards pass with zero swap, GPU peaks 1,017 MiB in R1 and 1,065 MiB in R2. R2 tensor peak is 935,925,760 bytes, below the 3 GiB cap.

The tight anchor spread supports a within-block reuse/working-set effect rather than layer-zero content being the main explanation. This does not identify a DRAM-only cause, isolate a causal cache percentage or establish a full-model/serving/RL speedup. A rolling row-prefetch overlap trial is being prepared separately; it has no performance result here.

Reports are deterministic gzip with original and compressed hashes in `raw-output-manifest.json`. Scripts bind source, DSO and encoder hashes to the earlier predicated experiment. No weights or compiled library are included.
