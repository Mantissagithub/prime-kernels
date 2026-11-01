# Gate/up lossless decode profile r1

This experiment profiles the existing lossless gate/up projection for the
actual B2 input and all 32 SmolLM2-360M layers. It keeps the projection math,
row mapping, barriers, and original `run_gateup_lossless` ABI unchanged. The
profile export is a generated clone beside that original kernel.

`prepare_profile.py` accepts only the measured probe source with SHA-256
`b0218ea40e2f9012c802dde20adfb428d0361ad952761c07a698a7e60e7cf09c`. It
creates `projection_probe.cu` and strips the clock and escape-count statements
from the cloned kernel to check byte-for-byte equality with the baseline
kernel. It does not compile or access a GPU.

The profile buffer has shape `[32 layers, 24 CTAs, 8 warps, 6 uint64 values]`.
The first five values are whole local interval, row-offset checks and prefetch
issue, decode pair, activation/FMA, and reduction cycles. The last value
counts escaped chunks handled by that warp. Phase intervals accumulate per
lane across the rows assigned to the warp; one warp maximum is taken per phase
after projection work finishes. Cycle sums across layers are sums of local
worker intervals, not device wall time. These `clock64()` intervals measure
scheduled local work, not isolated arithmetic cost or memory latency. A load
can complete later than its issue interval when there is no dependency boundary
to expose its completion. The escape flag uses the header already loaded by the
profile decoder.

The harness retains the probe's exact BF16 weight round-trip and FP32/BF16
output comparisons. It records uninstrumented and profiled event timings in a
separate overhead field. The report keeps measurements when instrumentation
overhead exceeds 5 percent and marks the phase diagnosis as materially
perturbed. The existing 3 GiB GPU
allocation guard remains in force. This harness covers only raw gate/up
projection, not SwiGLU or the surrounding MLP.

From this directory, regenerate the source and run the parent-owned CUDA
measurement with:

```sh
python3 prepare_profile.py
python3 harness.py --ptxas-verbose --output results-r1.json
```

The harness does not overwrite an existing result or shared library. The
coordinator owns the build and GPU run. Do not use `--build-only` as a profile
result; it does not validate or measure the kernel.
