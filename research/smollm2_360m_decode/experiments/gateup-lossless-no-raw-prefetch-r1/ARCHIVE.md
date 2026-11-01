# Experiment 216 research evidence

Measured directory is `outputs/megakernel-20261025/gateup-lossless-no-raw-prefetch-r1` in tiny-prl. The candidate removes only two CSR raw-offset reads and the conditional raw-escape L2 prefetch call. Plane/header hints, row mapping, decode and projection math are unchanged. The original compressed control is preserved. `source_deletion_proof.py` checks the exact source difference.

Reject the candidate. Compressed control median is 86.522996 us per layer versus 103.832997 us without raw prefetch, 20.0062% slower. Five alternating placements per arm with five graph replays each give 800 timed layer calls per arm across 32 distinct layers. Dense projection runs eagerly for validation, not timed as an arm. All decoded weight words and both compressed outputs match dense FP32 and BF16 results. Poisoned replay output checks pass. All CSR memory remains allocated and counted, isolating hint removal.

Both compressed kernels use 56 registers and 7,680 bytes shared memory, with no spills or stack. Both guards return zero, GPU peak is 1,019 MiB and swap is zero. This raw projection result excludes SwiGLU, full serving and RL. The raw prefetch hints are retained; no serving change is adopted. Decode instruction work is the next investigation, not an established speedup.

Reports are deterministic gzip with original and compressed hashes in `raw-output-manifest.json`. No weights or compiled libraries are included. Source scripts retain their measured absolute paths and require the pinned offline uv environment inside the bounded operator workflow.
