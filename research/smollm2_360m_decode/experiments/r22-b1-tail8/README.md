# R22 B1 projection tail loads

R22 is a frozen R21 copy with one B1 projection change. At K=960, the `dots` path loads and converts the six remaining BF16 weights after its eight-weight groups, and the `dots16` path does the same for its four remaining weights. Each tail then uses the existing sequential FP32 RN multiply/add steps. K=2560 has no tail. Accumulation order, other projection sizes, and all other kernel paths are unchanged.

Full validation remains 45/46 with R21's exact pass set: the initial four and held-out 30 pass, as do 11 of 12 wide cases. Only seed777 B2/context8000 fails numerical hidden/KV/logit gates; its greedy IDs still match. Acceptance gates and dtype boundaries are unchanged. The compact full-validation source/library hashes and guard results are in `r22-validation-summary-r1.json`.

Three interleaved direct decoder-only ABBA/BAAB blocks (ABBA, BAAB, ABBA) use 20 samples per arm per block, excluding LM head. Prefill, KV restore, parity, and reference work are outside timed intervals. The median of six reported arm upper-medians shows B1/context32 at 4.568703889846802 -> 3.651520013809204 ms (-20.07536%) and B1/context128 at 4.6600799560546875 -> 3.7437119483947754 ms (-19.66421%). R22 beats both R21 placements in every B1 block. B2 is neutral (+0.011% and +0.056% at contexts 32 and 128). Across four shapes, 195 hidden/KV checks per shape are bitwise exact at 0 ULP. These are decoder-component measurements only; they do not establish a serving or RL wall-time improvement. See `r22-direct-summary-r1.json` for compact timing metadata and the original run at `outputs/megakernel-20261025/r22-direct-bench-r1/results-r1.json` for full events.

`direct-bench.py` is the harness used for the direct measurement. `build.sh` builds the archived source. Compiled libraries and tensor captures are omitted.

| Artifact | SHA-256 |
| --- | --- |
| `block360.cu` | `180be7870c87b5993f02529d89546b8f771b261398af59608d35ab3fc12003f8` |
| `block360_kernel.inc` | `4c1a684303d0329238ab5f3058c7f4058ae0fe36c9b13a44adc3dbb649d8be55` |
| `build.sh` | `573fce41ec2ba3ad656495232d34a6ca9668ead170dae6181cd24cb00d4c813d` |
| `build/libdecode360_r22_tail8.so` (not archived) | `6c614aafb4907dea94b3c599c9bf97936e60027e447330838e9ba5a094fcfcec` |
