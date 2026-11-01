# R35-R2 independent measurement audit (R2 and R3)

## Verdict

Do not adopt R35-R2 on the claim that it is faster. The two fresh-process measurements show small decoder-only differences, but the unchanged B1 kernel reports gains similar to B2. The placebo effect makes the measured B2 gains inconclusive. In R3, B2 at context 128 also misses the strict per-block criterion in one of three blocks.

## Scope and identity

Both `all-six-r2.json` and `all-six-r3.json` identify the same control and candidate source, include, and shared-library SHA256s:

| Artifact | SHA256 |
|---|---|
| R30 `candidate/block360.cu` | `5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638` |
| R30 `candidate/block360_kernel.inc` | `60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83` |
| R30 `build/libdecode360_r30_phase_prefetch.so` | `625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2` |
| R35-R2 `candidate/block360.cu` | `844431b41692356d1c5936b45b5cf9104a0efa958feb501b1a827d13c7884b0f` |
| R35-R2 `candidate/block360_kernel.inc` | `e4f9ef22300d6bedd71e5a3ffdfc879e5a2b39251fd4cc8605b661a95b992264` |
| R35-R2 `r35build-r2/libdecode360_r35_selective_packed.so` | `21960588872d6728e5814cb2cad094222a31ebf2d35b0865050700dd0068e204` |

R2 and R3 each ran six cases with 3 warmups per arm, three ABBA/BAAB blocks, six placements per arm, and 20 CUDA-event samples per placement (120 samples per arm per case). The pooled median below is computed over the 120 raw event values; ranges are the minimum and maximum raw events. Positive percentage means R35-R2's pooled median was faster: `(R30 median / R35 median - 1) × 100`.

| Case | R2 R30 → R35-R2 median ms | R2 speed | R2 strict blocks | R3 R30 → R35-R2 median ms | R3 speed | R3 strict blocks |
|---|---:|---:|---:|---:|---:|---:|
| B1, context 32 | 2.886656 → 2.887840 | -0.041% | 0/3 | 2.887792 → 2.887680 | +0.004% | 0/3 |
| B1, context 128 | 2.942976 → 2.939904 | +0.105% | 3/3 | 2.942976 → 2.940656 | +0.079% | 3/3 |
| B2, context 32 | 3.129344 → 3.121152 | +0.263% | 3/3 | 3.128320 → 3.120640 | +0.246% | 3/3 |
| B2, context 128 | 3.213312 → 3.210240 | +0.096% | 3/3 | 3.212288 → 3.210240 | +0.064% | 2/3 |
| Synthetic B1, context 1024 | 3.247888 → 3.239936 | +0.245% | 3/3 | 3.248144 → 3.240960 | +0.222% | 3/3 |
| Synthetic B2, context 1024 | 3.667968 → 3.658752 | +0.252% | 3/3 | 3.668992 → 3.658752 | +0.280% | 3/3 |

Raw event ranges (R30 / R35-R2, milliseconds) were:

- B1 context 32: R2 `2.878464–3.026848 / 2.878464–2.979680`; R3 `2.876416–2.976704 / 2.877440–2.991200`.
- B1 context 128: R2 `2.931712–2.994496 / 2.928672–2.991168`; R3 `2.932736–2.984992 / 2.931712–2.990080`.
- B2 context 32: R2 `3.120192–3.188736 / 3.113984–3.176448`; R3 `3.120128–3.191776 / 3.113920–3.163136`.
- B2 context 128: R2 `3.204096–3.274752 / 3.202048–3.263488`; R3 `3.204256–3.260416 / 3.203072–3.255968`.
- Synthetic B1 context 1024: R2 `3.233792–3.265536 / 3.227648–3.258368`; R3 `3.238912–3.263488 / 3.229696–3.254272`.
- Synthetic B2 context 1024: R2 `3.658528–3.683232 / 3.649536–3.671040`; R3 `3.658752–3.689472 / 3.649536–3.669312`.

Strict blocks means both R35 placement medians were below both R30 placement medians within each block. It is a direction-consistency check, not a statistical significance test. R2 passes this check in 5/6 cases; R3 passes it in 4/6. Raw event ranges overlap between arms in every case and run. R3's B2 context-128 comparison fails strict ordering in the first block.

The unchanged-B1 comparisons are the important placebo control: their source and B1 machine code are unchanged, yet B1 context 128 and synthetic B1 show 0.08–0.22% pooled-median gains in both runs, matching the scale of most B2 gains. R2's B1 context-32 result is slightly slower; R3 is effectively tied. The timing data therefore do not isolate a reliable B2 packing effect from measurement/run variation.

## CPU-only SASS evidence

`reproduce_sass.py` uses only Python's standard library and `cuobjdump`; it does not import PyTorch, initialize CUDA, or run GPU work. It runs:

```sh
cuobjdump --dump-sass --gpu-architecture sm_89 <library.so>
```

The script parses each named function section through the next `Function :` boundary, then hashes the normalized instruction addresses and operation text. Results from the current R30 and R35-R2 DSOs:

| Function | R30 instructions | R35-R2 instructions | normalized SASS SHA256 | encoded bytes per arm | encoded little-endian SHA256 |
|---|---:|---:|---|---:|---|
| B1 `_Z9mk_kernelILi1EEv6Params` | 7,360 | 7,360 | same: `1672c31bd8f1d51d65c6a6bdac734526a71e9882f2435ced7cd9b9674b878d5a` | 117,760 | same: `912d14f46ffe8b591a79f7fa26a2e13d22cbb516000c4015874b0c3343919858` |
| B2 `_Z9mk_kernelILi2EEv6Params` | 7,768 | 7,880 | differs | 124,288 / 126,080 | differs: R30 `498788e3dbccc93aa647fb9a23d0eeea70cc5f56ee9339e0bf37ae5b8a3758d7`; R35-R2 `14516e89f713f171cd0c127cea1ac93f6b2dc17ea3ad138fb57373e129d5afd1` |

For B1, first instruction is `0000 IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;`; last is `1cbf0 NOP;`. The parser requires every instruction to have exactly two 16-hex-digit machine words, serializes both words as little-endian bytes, and refuses empty or unsupported instruction sections. It checks decoded instruction text and encoded bytes. This establishes identical B1 instruction encodings in the embedded `sm_89` function; it does not claim that the complete DSOs or cubins are byte-identical. `cuobjdump --dump-resource-usage` also reports identical B1 resources in both libraries: 241 registers, 36,644 B shared memory, zero stack/local. B2 differs, so the identical B1 conclusion comes from parsing only the B1 sections of full dumps, not comparing whole-library dumps. An earlier exploratory `cuobjdump --function` call emitted a warning and was discarded; the evidence here comes from the unfiltered dumps and explicit function-boundary parser.

Reproduce from the `prime-rl` project root with:

```sh
uv run --no-sync python outputs/megakernel-20261025/r35-selective-packed/audit-r2-r3/reproduce_sass.py
```

## Correctness and execution limits

In all six cases of both runs, R30 and R35-R2 match bitwise for final hidden output and K/V values across all 32 layers. This establishes candidate-versus-control parity for these measured inputs. The harness's separate native 1-ULP diagnostic fails for both arms; it is explicitly labeled diagnostic-only, so these runs do not establish native parity acceptance.

The decoder-only CUDA event interval excludes LM head, prefill, KV prefix restoration, parity checks, and telemetry. No full decode/user latency or RL throughput claim follows. Both R2 and R3 inner and outer GPU guards returned code 0 with 1,955 MiB peak GPU memory, below the 3,072 MiB cap; neither reported a cap trip. The runs used an RTX 4060 Laptop GPU, Torch 2.13.0+cu130, and Transformers 5.6.2.
