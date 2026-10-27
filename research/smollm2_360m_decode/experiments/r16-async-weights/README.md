# R16 asynchronous B2 weight copies

R16 builds on the R13 four-load B1 experiment. Its B2 WMMA projection uses two warp-private shared-memory weight tiles and direct SM89 `cp.async.ca.shared.global` PTX, overlapping the next tile copy with the current A-tile/MMA work. K16 accumulation order and BF16 boundaries are preserved. B1 is unchanged.

Three interleaved CUDA-event ABBA/BAAB blocks measured B2 decoder latency reductions of 5.86% at context32 and 5.63% at context128. Both candidate arm medians beat both controls in every B2 block. B1 controls are neutral. All four timing fixtures have exact final hidden and all32 layers of K/V against native and directly between R13/R16. The original34 fixtures pass; wider validation remains6/12 with the same pass set as R13. This is a decoder-component speed result; serving and RL gains remain unestablished.

The exact source/include/build command and results are archived here. Binaries and captures are omitted. `result.json` records hashes, resources, timings, and guard data. Run `./build.sh` with CUDA13.0/SM89 under the host guard. Raw evidence resides in sibling `prime-rl/outputs/megakernel-20261025/r16-direct-bench-r1/results-r1.json`; its guard and summary are in `validation-operator-r1/reports/r16-direct-bench-r1/`. Numerical reports are in `validation-operator-r1/reports/r16-{heldout30,initial4,wide12}-r1/`.
