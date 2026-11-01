# R37 B2 early down-weight prefetch

R37 tests one placement change against the accepted R30 phase-prefetch candidate. For B=2, it issues the existing down-weight L2 hints after norm2/debug output and immediately before gate/up. The loop, rows, addresses, number of hints, and stride are unchanged. B=1 keeps the original hint position after gate/up. Gate/up and down projection math, row mapping, barriers, workspace, outputs, and ABI are unchanged.

The hypothesis is that B2 down-weight hint latency can overlap gate/up work. The hints may contend with gate/up for cache or DRAM bandwidth and erase or reverse a gain; no speedup is assumed.

`source_preflight.py` reconstructs the expected R30 include with only this conditional relocation and verifies the `.cu` is byte-identical to R30. `candidate/build.sh` builds separately under `r37build`. The initial build compiled successfully but exited at the final hash step because that line named the R30 library; the script now hashes the R37 library. `bench_r30_r37.py` compares R30 as A with R37 as B in a six-case direct decoder-step harness. It uses four frozen B1/B2 fixtures at contexts 32 and 128 plus seeded synthetic B1/B2 at context 1024; each case gets three warmups per arm and three interleaved ABBA blocks with 20 CUDA-event samples per placement. Final hidden state and every layer's K/V are checked bitwise outside timing.

`run_all_six_cap3g.sh` runs all six cases under the existing offline runner and stops the user systemd unit if GPU memory exceeds 3072 MiB. GPU benchmark and model loading have not been performed.
