# Private B1 compiler ordering trial

Empty inline-assembly memory clobbers at the same five boundary locations as the profiling intervals, restricted to B1. No timestamps, global profile writes or source math changes. Tests whether ordering constraints retain the profiler-associated scheduling change. No driver or production change; B2 should remain byte-identical. Require exact parity and interleaved uninstrumented timing; a faster profiling build alone is not evidence.
