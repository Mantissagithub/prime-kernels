# R15b attention accuracy experiment

R15b builds on the R12 arithmetic branch with context-dependent PV schedules and complete B1 task iteration. QK uses warp-private BF16 WMMA for B1/K256 and homogeneous B2/K512, rounding raw QK to BF16 before scaling. This source has the original projection loops; the separate R13 lookahead experiment is archived alongside it.

The unchanged 46-case suite passes 43 cases: original heldout30 30/30, initial controls 4/4, and wider seed-777 cases 9/12. All B1 cases pass. B2 contexts511/2048/8000 still fail strict per-layer checks at post-block layers8/11/22, despite matching greedy IDs. This remains an experimental accuracy branch. It establishes no speed, serving, probability, reward, or RL acceptance result.

`result.json` records source/library hashes, every case verdict, and guard records. The canonical source snapshots are included; binaries and capture tensors are omitted. Run `./build.sh` with CUDA13.0/SM89 under the documented host guard to build the library. Numerical evidence resides in sibling `prime-rl/outputs/megakernel-20261025/validation-operator-r1/reports/r15b-{heldout30,initial4,wide12}-r2/`.
