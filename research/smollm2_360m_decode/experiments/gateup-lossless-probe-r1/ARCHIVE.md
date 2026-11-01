# Experiment 214 research evidence

Original measured directory is `outputs/megakernel-20261025/gateup-lossless-probe-r1` in tiny-prl. This is an isolated B2 raw gate/up projection test across all 32 SmolLM2-360M layers. It excludes SwiGLU, serving and RL.

The candidate is rejected. Median latency is 43.125 us dense versus 86.425 us lossless, 100.41% higher. Five alternating placements per arm contain five graph replays each, for 800 timed layer calls per arm. Every one of 157,286,400 decoded BF16 weight words matches, as do 327,680 FP32 and BF16 projection outputs per arm. NaN poisoning after capture verifies replay writes. Counted encoded storage, including CSR row offsets, is 256,225,056 versus 314,572,800 bytes, an 18.55% reduction.

The bounded build compiled source `b0218ea40e2f9012c802dde20adfb428d0361ad952761c07a698a7e60e7cf09c` into DSO `ead4d09da929822502fb5f32f5fbb57a349da9c2da4de36e8da4cfc538e3c192` before the guarded run. The report matches source, harness and DSO hashes. Compiler diagnostics show no spills or stack use. Dense and compressed kernels use 64 and 56 registers respectively, both with 7,680 bytes shared memory. Both guards exit zero, peak GPU memory is 1,019 MiB, and cgroup swap is zero.

An independent Luna review checked the final source and run artifacts. Its first source review missed two defects in an earlier draft; root caught them before any build or GPU execution, and the corrected revision was re-reviewed. That invalid draft produced no timing evidence. No serving change is adopted. Decode profiling is the next investigation, not an established explanation of the regression.

Reports and SASS are deterministic gzip files with original and compressed hashes in `raw-output-manifest.json`. No weights or compiled libraries are included. Source scripts preserve their measured absolute working paths; run them only through the separately guarded operator workflow, using the pinned offline uv environment.
