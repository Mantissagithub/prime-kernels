# R20 composed 360M decoder candidate

R20 composes R18's R15b attention/PV behavior and B2 cp.async path with R17's eight-load B1 `dots`/`dots16` and R19's B2 warp-to-CTA mapping. The source and include are archived exactly as built. R20 passes 43/46 cases: all 34 original and initial fixtures, plus 9/12 wide fixtures. The wide pass set is identical to R18, with no additions or regressions. The remaining failures are the same three B2 wide contexts.

A four-fixture direct decoder ABBA compared R20 with R15b. Across three blocks, both R20 placements beat both R15b placements for B1 and B2 at contexts 32 and 128. Using the median of the six reported arm statistics, B1 latency changed from 16.876 to 4.566 ms (3.696×) at ctx32 and 16.962 to 4.657 ms (3.642×) at ctx128; B2 changed from 8.463 to 7.957 ms (−5.99%) and 8.705 to 8.200 ms (−5.80%). The operator summary also reports standard medians of all 120 event samples; `result.json` records both aggregation methods because they differ by up to about 0.001 ms in these cases. All four fixtures have exact 0-ULP final hidden and all-layer KV parity against native and between candidates.

This measures the combined decoder directly against R15b. It does not claim a vLLM, serving, or RL wall-time gain. The compiled binary is omitted; source, include, and build script hashes plus raw-evidence hashes are in `result.json`.
