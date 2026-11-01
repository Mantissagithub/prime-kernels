# Cache working-set diagnostic

The two measured harnesses reuse the frozen predicated decoder and change only call-to-layer mapping. Every graph has 32 projection calls. R1 touches 1, 2, 4 or 32 distinct encoded layer sets. R2 checks one repeated layer at every one of the 32 anchors, plus the all-layer traversal, with an explicit prewarm before each timed placement.

This is a diagnostic, not a production kernel optimization, serving result or RL speedup. Both runs are complete; measured reports and guard records are included as compressed JSON. See `ARCHIVE.md` for the results and interpretation limits.

Source, library and encoder hashes are checked before running. The library and weights are not included. Scripts retain the original measured sibling paths and must run through the bounded pinned-offline-uv operator workflow.
