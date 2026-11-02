# B1 scheduled-phase profile

Private instrumentation only. Hardware counters are admin-restricted (RmProfilingAdminOnly=1); no driver setting is changed. Thread0 of each CTA records four cycle intervals per layer: encoded metadata/first hints plus grid barrier; norm2; GU loop; down-weight hints plus grid barrier. GU is thread0 scheduled work, not maximum across all warps. Barrier intervals include waiting. These are not pure memory costs, additive wall shares or speedups.

Uninstrumented package is the control. Interleaved events measure profiler overhead; exact hidden/KV and captured fallback/refresh checks retained. Inspect added spills before interpretation. No production edit or RL speedup. Initial preparation failed an ambiguous source anchor, so the first build had no source; the corrected anchor includes the down-matrix pointer. Both build logs retained.

The first GPU launch failed before measurement because a library named profile.so shadowed the Python standard-library profile module. The binary was renamed gu_timing.so without rebuilding; fresh r2 guard paths are used.
