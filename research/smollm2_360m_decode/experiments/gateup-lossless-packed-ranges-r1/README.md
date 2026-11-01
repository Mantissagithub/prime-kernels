# Packed raw row ranges

This experiment checks one prefetch-setup hypothesis: replace the two CSR offset loads per row with one packed-range load. For the fixed `[5120, 960]` gate/up weight shape, each row has at most `960 / 8 = 120` escape chunks. The new `uint32` table stores the row's first raw ordinal in bits 0–23 and its escape count in bits 24–31.

The original CSR offsets remain allocated, transferred, counted, and used by the predicated control. The candidate has an additional packed table with `5120 * 4 = 20,480` bytes per layer. The candidate reads that table only in its per-row L2 prefetch setup and reconstructs the same raw byte interval. Its decode helper, SIMD arithmetic, accumulation, reduction, and output path are copied unchanged from the frozen predicated control.

`cpu_packed_ranges_proof.py` exercises zero and maximum row counts, ordinals near the 24-bit limit, exact prefetch-interval mapping, and invalid-bound rejection without model weights. `source_equivalence_proof.py` checks the frozen source hash and proves that the added kernel is the control clone with only the packed prefetch lookup changed. The harness retains the dense eager validation arm, compares the frozen predicated control and packed candidate for exact FP32 and BF16 outputs, checks NaN-poisoned writes, and records all allocated weight and metadata bytes against the 3 GiB cap. Timing uses five alternating placements and five replays per placement, or 800 projection calls per compressed arm.

Measured results and resource-guard reports are included as compressed JSON; see `ARCHIVE.md` for the no-measurable-gain result. No weights or compiled library are included. Rebuild and benchmark only through the bounded pinned-offline-uv operator workflow; the harness arguments are:

```bash
uv run python harness.py --build-only
uv run python harness.py --skip-build --output results-r1.json
```
