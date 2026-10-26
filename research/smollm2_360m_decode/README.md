# SmolLM2-360M full-decoder research snapshot

This directory archives the exact R8 source and include used by the recorded
full-decoder checks. The kernel performs one single-launch decode step across
all 32 layers, including embedding lookup, paged KV reads/writes, and final
RMSNorm. It remains research code and is not accepted for serving.

## R8 changes

R8 keeps R7's source, including its four-lane QK and observed RMSNorm
reduction. The include changes only PV schedule dispatch: B1 uses WMMA through
K=16 and then a y4/chunk32 FP32 accumulation path; B2 uses y2/chunk16 through
K=32 and then y2/chunk32 FP32 accumulation. Products use separate RN
multiply/add operations, partition partials fold serially, and the result is
rounded once to BF16. Other decoder arithmetic is unchanged.

## Validation

All four initial B1/B2 context-8/32 fixtures pass bitwise: every per-layer
block output and KV tensor, final hidden state, and final logits are 0 BF16
ULP from native; greedy IDs match.

On the frozen 30-case held-out suite (seeds 17/101/2027, batches 1/2, contexts
1/8/16/32/128), R8 passes 29 and fails one numerical gate. All 30 greedy IDs
match; final logits are bitwise equal to native in the 29 passing cases. The
sole failure is `seed17-b1-ctx16`: the first post-block output diverges at
layer 9 while its K/V is still exact, then K/V first diverges at layer 10.
This case fails the unchanged numerical gates. R8 improves on R7's 24/30 but
does not pass the full held-out suite. The exact gates are per-layer output
and K/V <=1 ULP, final hidden <=1 ULP, final logits <=2 ULP, and exact greedy
IDs. A causal-layer diagnostic for the failed case found rotated Q, raw/scaled
QK, and probabilities bitwise exact at layer 9; PV differed in one output by
one BF16 ULP. This single-case diagnostic does not replace the held-out run.

R7 remains summarized in `validation-summary.json`: its R7 source passed the
four initial fixtures and 24/30 held-out cases. R6 passed 22/30 and R4 passed
20/30. Their source and result files remain in the Prime-RL experiment archive.

## Build

With CUDA 13.0 and SM89 support, build the shared library with:

```sh
mkdir -p build
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build/libdecode360_r8.so
```

The archived R8 source/include/library hashes and validation provenance are in
`validation-summary.json` and `prime-rl/outputs/megakernel-20261025/decode-360-cuda-r8/provenance.json`.

## Evidence

- Initial four: `prime-rl/outputs/megakernel-20261025/decode-360-validation-r10/results-r3-r8/comparison.json` and `decode-360-validation-r10/guard-r8/result.json`.
- Full held-out 30: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r14/results-r1/comparison.json` and `block-360-heldout-r14/guard/result.json`.
- Causal-layer diagnostic: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r15/results-r1/comparison.json` and `block-360-heldout-r15/guard/result.json`.
- RMSNorm primitive replay: `prime-rl/outputs/megakernel-20261025/norm360-probe-r3/results-r3/probe-result.json`.

These are numerical checks only. There is no speed, reward, quality, or
production claim; one held-out case still fails the original gates.
