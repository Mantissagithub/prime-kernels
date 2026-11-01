# Experiment 220 research evidence

Measured directory is `outputs/megakernel-20261025/gateup-lossless-packed-ranges-r1` in tiny-prl. One uint32 record holds a 24-bit raw start and 8-bit raw count for each row. It replaces two CSR offset reads in prefetch setup while preserving the same per-row hint intervals, decoder and projection math. The actual packer checks field bounds and verifies unpacked starts/counts against CSR spans before GPU transfer. The fixed row has at most 120 eight-weight chunks, fitting the count field.

No measurable latency improvement. Predicated control median is 56.736000 us versus packed ranges 56.704000 us per layer, a nominal 0.0564% difference with overlapping raw event ranges. Do not adopt the extra metadata table from this run. Five alternating placements and five replays over 32 layers give 800 timed calls per arm. All decoded weights and FP32/BF16 projection outputs match exactly, and poisoned replay writes pass.

The experiment retains CSR for the control and allocates an additional 655,360-byte packed table; all metadata totals 256,880,416 encoded bytes. Both kernels use 62 registers and 7,680 bytes shared memory with no spills or stack. Both guards pass with 1,019 MiB sampled GPU memory and zero swap. This raw projection test excludes serving and RL. A cache-working-set diagnostic is being prepared to distinguish streaming/cache pressure from instruction work; it has no result here.

Source and CPU packing proofs are included. Reports are deterministic gzip with hashes in `raw-output-manifest.json`. No weights or compiled library are included. Scripts retain their original measured paths and require the bounded pinned-offline-uv operator workflow.
