# SmolLM2-360M full-decoder research snapshot

This directory contains the R6 source used by the recorded full-decoder checks. It performs one single-launch decode step across all 32 layers, including embedding lookup, paged KV reads/writes, and final RMSNorm. The snapshot is research code and is not accepted for serving.

## R6 change

R6 keeps the R4 decoder and changes only the QK product: four SIMT lanes accumulate strided K dimensions with FP32 round-to-nearest multiply/add, then reduce at shuffle offsets 2 and 1 before BF16 rounding. PV and RMSNorm are unchanged. The selected reduction is a generic matching candidate; the primitive sweep did not uniquely identify native cuBLAS's arithmetic order. The candidate uses no fast-math flag.

## Validation

R6 passes all four initial B1/B2 context-8/32 fixtures bitwise: per-layer block outputs, KV, final hidden state, and logits are 0 BF16 ULP from native, and greedy IDs match. The unchanged gates are per-layer output/KV <=1 ULP, final hidden <=1 ULP, logits <=2 ULP, and exact greedy IDs.

On the frozen 30-case held-out set (seeds 17/101/2027, batches 1/2, contexts 1/8/16/32/128), R6 passes 22 and fails 8 numerical gates. Greedy IDs match on all 30. It fixes two R4 failures, with no prior pass becoming a failure. The run completed without a resource stop; its guard returned 1 for the numerical failures. This is not full-suite acceptance.

R12 recaptured the eight failed cases at their first post-block divergence layers. At those targets, rotated Q, raw QK, scaled scores, and softmax probabilities match native in all eight. Six cases first differ at PV output by one BF16 element; two have exact norm2 input and weight but differ at norm2 output. This diagnostic does not itself change the acceptance result.

The earlier R4 candidate passed 4/4 initial fixtures and 20/30 held-out fixtures, with all 30 greedy IDs matching. Its source and validation history are summarized in `validation-summary.json`; R4 remains a historical baseline, not the source in this directory.

## Build

With CUDA 13.0 and SM89 support, build the shared library with:

```sh
mkdir -p build
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -Xcompiler=-fPIC --shared \
  block360.cu -o build/libdecode360_r6.so
```

The archived R6 source and include hashes are recorded in `validation-summary.json`. Build and library provenance is under `prime-rl/outputs/megakernel-20261025/decode-360-cuda-r6/`.

## Evidence

- Initial four: `prime-rl/outputs/megakernel-20261025/decode-360-validation-r10/results-r1/comparison.json` and `decode-360-validation-r10/guard/result.json`.
- Full held-out 30: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r11/results-r1/comparison.json` and `block-360-heldout-r11/guard/result.json`.
- Targeted eight-case diagnostic: `prime-rl/outputs/megakernel-20261025/block-360-heldout-r12/results-r1/comparison.json` and `block-360-heldout-r12/guard/result.json`.
- Source, build, and compiled-library hashes: `prime-rl/outputs/megakernel-20261025/decode-360-cuda-r6/provenance.json`.

No speed, reward, quality, or production claim follows from these numerical checks. See `validation-summary.json` for exact hashes, case IDs, and guard details.
