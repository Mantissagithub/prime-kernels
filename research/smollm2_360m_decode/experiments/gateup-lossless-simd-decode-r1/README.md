# Lossless gate/up SIMD decode trial

This source only trial compares the frozen lossless gate/up projection against an otherwise identical kernel using packed integer/SIMD decode logic. The original compressed control kernel, CSR row offsets, raw escape hints, FMA order, and reduction order are retained. Only the candidate's two decode helper calls differ.

`decode_weights_simd` runs the same SIMD helper over the full encoded weight set for exact BF16 word comparison. The harness keeps the dense validation copy and the compressed control and SIMD candidate in the same run. It retains the existing 3 GiB allocated-memory guard.

`source_equivalence_proof.py` verifies the frozen control hash and proves the candidate kernel differs only by its two helper calls. `cpu_arithmetic_proof.py` checks all byte pairs, paired deltas, every legal base including representable exponent 255, sign/mantissa corners, and the PRMT selectors against both plane words. Its frozen counts are in `proof-result.json`. The proof imports no PyTorch and loads no weights.

No build, model load, GPU run, or weight write is part of preparing this trial. Build and execute only after separate review:

```bash
uv run python harness.py --build-only
uv run python harness.py --skip-build --output results-r1.json
```
