# Lossless gate/up weight codec survey

This CPU-only survey measures whether the cached SmolLM2-360M BF16 gate/up matrices admit useful byte reduction without changing any weight bit. It encodes only `model.layers.{0..31}.mlp.{gate_proj,up_proj}.weight`, one matrix at a time, and checks each complete matrix by bitwise roundtrip before recording its statistics.

## Codec contract

Each group of eight BF16 weights is represented by an 8-byte plane containing the original sign and seven mantissa bits, plus a 32-bit header. The header stores the minimum exponent in its high byte and eight three-bit exponent deltas in its low 24 bits. A group escapes when its minimum exponent is 255 or any delta exceeds 7. Escaped groups store their original 16 bytes in a dense raw table; each escape header's low 24 bits gives that table's ordinal. Decode identifies escapes from the header high byte and indexes the raw table from the header itself. Escape data takes precedence, preserving every 16-bit pattern, including NaNs, infinities, and signed zero. The survey raises on any bitwise mismatch.

The reported encoded size is `weight_count + 4 * chunk_count + 16 * escaped_chunk_count` bytes. The report includes per-matrix and aggregate raw/encoded sizes, escape rates, nonfinite counts, checkpoint and script SHA-256 hashes, and roundtrip status. It writes only `results.json`; it does not save encoded weights or mutate the checkpoint.

## Memory and scope

Safetensors is opened on CPU. The script processes one 2560x960 matrix at a time, keeps its temporary encoding and decode buffers only for that matrix, then releases them before loading the next. It never loads the full model or uses a GPU. Run it only in the separately guarded operator pass:

```sh
python survey.py
```

`python survey.py --self-test` checks crafted CPU bit patterns, including two nonadjacent escaped chunks and signed-zero/NaN/infinity encodings, without opening the checkpoint.

This survey establishes 3-bit exponent-delta feasibility only. It does not implement or benchmark a GPU decoder, prove end-to-end speed, or establish a faster kernel. A GPU prototype is worth pursuing only if measured storage saving reaches at least 10%; integration would also need to account for extra GPU cache use and weight reload invalidations.
