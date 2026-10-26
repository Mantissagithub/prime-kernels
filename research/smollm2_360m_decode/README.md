# SmolLM2-360M full-decoder research snapshot

This directory contains the exact R7 source and include used by the recorded full-decoder checks. The kernel performs one single-launch decode step across all 32 layers, including embedding lookup, paged KV reads/writes, and final RMSNorm. It remains research code and is not accepted for serving.

## R7 change

R7 keeps R6's four-lane QK reduction and changes only the RMSNorm reduction sequence. It uses four ordered FP32 input accumulators, a 128-lane shared reduction tree, descending warp shuffles, mean-factor multiplication, a separate epsilon addition, and approximate reciprocal square root. The normalized value is rounded to BF16 before multiplication by the BF16 weight. PV and other decoder arithmetic are unchanged. The build uses no fast-math flag.

## Validation

The four initial B1/B2 context-8/32 fixtures pass bitwise: all per-layer block outputs and KV, final hidden state, and logits are 0 BF16 ULP from native; greedy IDs match.

On the frozen 30-case held-out suite (seeds 17/101/2027, batches 1/2, contexts 1/8/16/32/128), R7 passes 24 and fails six numerical gates. Greedy IDs match in all 30. It repairs both R6 norm-stage failures without a pass-to-fail case. The unchanged gates are per-layer output/KV <=1 ULP, final hidden <=1 ULP, logits <=2 ULP, and exact greedy IDs. The run completed without a resource stop; its guard returned 1 for the numerical failures. R7 is not full-suite accepted.

The earlier R6 candidate passed four initial fixtures and 22/30 held-out cases; R4 passed four initial fixtures and 20/30 held-out cases. These prior results and their hashes are retained compactly in `validation-summary.json`.

## Build

With CUDA 13.0 and SM89 support, build the shared library with:

```sh
mkdir -p build
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build/libdecode360_r7.so
```

The archived R7 source/include hashes, compiled-library hash, and validation provenance are in `validation-summary.json` and `prime-rl/outputs/megakernel-20261025/decode-360-cuda-r7/provenance.json`.

## Evidence

- Initial four: `prime-rl/outputs/megakernel-20261025/decode-360-validation-r10/results-r2-r7/comparison.json` and `decode-360-validation-r10/guard-r7/result.json`.
- Full held-out 30: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r13/results-r1/comparison.json` and `block-360-heldout-r13/guard/result.json`.
- RMSNorm primitive replay: `prime-rl/outputs/megakernel-20261025/norm360-probe-r3/results-r3/probe-result.json`.

No speed, reward, quality, or production claim follows from these numerical checks. R13 improves held-out exactness but still fails six strict numerical gates.
