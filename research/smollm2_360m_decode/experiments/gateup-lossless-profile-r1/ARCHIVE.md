# Experiment 215 research evidence

Original measured directory is `outputs/megakernel-20261025/gateup-lossless-profile-r1` in tiny-prl. This profiling clone preserves the rejected compressed projection candidate and records local warp intervals across 32 layers. It is a diagnostic, not a speed improvement or serving change.

Profile/control outputs match exactly, all decoded weight words match, all recorded cycle cells are positive and the GPU escape count matches 1,227,498 CPU-encoded chunks. Interleaved profile/control timings yield 4.5698% median instrumentation overhead, within the absolute 5% gate, but raw event ranges are broad and overlap. Do not interpret this run as a new latency win.

Median warp-local cycle sums across 32 layers are 6,624,619 for the whole interval, 3,036,549 for prefetch setup, 2,918,431.5 for decode, 280,017.5 for activation/FMA and 286,034.5 for reduction. These are scheduled source-region intervals, not pure arithmetic or memory latency. Independent load completion can occur in later intervals. Their medians need not add, and they are not device wall-time shares. They motivate an isolated test of raw-escape prefetch setup, without establishing its causal cost.

The instrumented kernel uses 76 registers versus the control's 56, both with 7,680 bytes shared memory and no spills or stack. Guards pass, peak GPU memory is 1,019 MiB and swap is zero. First build failed only because the local output directory was missing; its log is preserved. After creating that directory, the bounded second build succeeded before the GPU run. The source generator checks exact stripping of instrumentation back to the original decoder and projection math.

Reports are deterministic gzip with hashes in `raw-output-manifest.json`. No weights or compiled libraries are included. Scripts retain their measured absolute paths and require the bounded operator workflow with pinned offline uv.
