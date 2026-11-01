# Lossless gate/up predicated decode trial

This source-only trial compares the frozen SIMD decoder with a candidate that uses PTX-predicated read-only global loads for compressed-plane and raw-escape chunks. The candidate keeps the SIMD pair arithmetic, FMA order, reduction order, and original raw-prefetch hints. Its projection kernel is otherwise the frozen SIMD control kernel.

The candidate uses an 8-byte plane load only for non-escape chunks and a 16-byte raw load only for escape chunks. A PTX `selp` chooses raw index zero for non-escape headers before address calculation. `decode_weights_predicated` validates every decoded BF16 word with the same candidate helper used by the projection kernel.

`source_equivalence_proof.py` checks the frozen SIMD source hash, unchanged control kernel, candidate kernel equivalence, and candidate decode wiring. `cpu_arithmetic_proof.py` checks the packed SIMD arithmetic and raw-word selection for zero, infinity, NaN, subnormal, and finite BF16 patterns. These checks use no GPU, model checkpoint, or PyTorch weights.

The harness checks all 157,286,400 BF16 weight words, exact FP32 and BF16 projection outputs, and NaN-poisoned output writes. It alternates five placements per arm with five replays per placement and enforces the 3 GiB allocated-memory cap. Build and run only after separate review:

```bash
uv run python harness.py --build-only
uv run python harness.py --skip-build --output results-r1.json
```
