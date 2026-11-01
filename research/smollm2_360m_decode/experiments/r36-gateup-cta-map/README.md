# R36 gate/up CTA row assignment

R36 tests one scheduling change against the accepted R30 phase-prefetch candidate. For B=2, the gate/up projection starts each warp at `blockIdx.x * kWarps + warp`; B=1 retains R30's `gw`. Each warp continues with stride `kGlobalWarps`.

The change applies only to the gate/up projection loop. It leaves the paired gate/up rows, row math, FMA order, stores, barriers, prefetch schedule, other projections, outputs, and ABI unchanged. It adds no arrays or workspace and changes no dtype.

## Why test this

The R30 B2 profile shows gate/up is dominant: about 3.56M of 3.64M cycles at context 32 and 128. Isolated QKV and O packing did not improve the real projection phase. The current B2 `gw` ordering assigns a warp-strided sequence of CTA indices to adjacent warp rows. R36 gives each CTA eight contiguous starting rows. This may improve memory locality or warp arrival distribution. It may also worsen tail balance. The hypothesis does not predict a gain.

## Frozen comparison

`bench_r30_r36.py` compares the R30 baseline (A) with R36 (B) using the same four frozen B1/B2 fixtures at contexts 32 and 128, plus seeded synthetic B1/B2 cases at context 1024. It runs three interleaved ABBA blocks per case, 20 CUDA-event samples per placement, and three warmups per arm. It checks final hidden state and every layer's K/V cache bitwise outside timing. There are six cases and 120 samples per arm per case.

Build the candidate separately with `candidate/build.sh`. Run `source_preflight.py` before compiling. The harness reads R30's existing library as A and the R36 library under `r36build` as B. Results belong under `results/`. This preparation did not invoke nvcc, a GPU, or model imports.

## Profiling and interpretation

The R30 B2 profile is the reason for choosing this loop. The measured phase split makes gate/up worth isolating. It does not establish whether memory locality, CTA arrival, or the tail is the limiter, and therefore cannot establish an expected speedup. Judge R36 only from the paired run and preserve all six cases, ABBA ordering, exact parity checks, and event count.
