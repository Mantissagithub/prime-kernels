# R21 B2 PV reduction order

R21 is a private copy of R20 with one change: the B2 general-SIMT PV schedule uses eight y groups, eight-token chunks, four round-robin accumulators, ordered FP32 RN local folds, and the halves y-reduction tree. It applies when the batch maximum sequence exceeds 2048 or both B2 sequence lengths are 512. All other kernel paths, including the R20 QK, projections, softmax, B1 PV, cp.async buffers, warp mapping, ABI, and acceptance gates, are preserved.

R21 passes 45/46 fixtures: all 34 original and initial fixtures and 11/12 wide fixtures. It adds the B2 ctx511 and ctx2048 cases relative to R20 without regressions. The remaining B2 ctx8000 case has matching greedy IDs; its first post-block mismatch is layer 10, where KV is still exact, followed by first KV mismatch at layer 11. The layer-15 capture is later diagnostic evidence and does not identify the cause of the layer-10 difference.

The R3 primitive probe matched saved native PV on all six selected captures for this schedule. That is limited to those captures. No R21 timing, serving, or RL speed claim is recorded; R20 component measurements do not establish R21 performance.

The compact operator validation report is `r21-validation-summary-r1.json`. Source, include, build, and library hashes are recorded there. The SM89 build reported zero spills, 148 registers/thread for B1 and 142 for B2, and 6,944/13,344 bytes static shared memory. The compiled binary is omitted.
