# Private B1 compact GU loop

Only the outer GU chunk-loop unroll directive changes: B1 requests one, B2 keeps four. No codec math or accumulator order change. Tests the instruction-footprint/loop-overhead tradeoff without profiling globals. Source-helper bits remain unchanged; exact hidden/KV and captured fallback/refresh are the gate. No production edit or speed claim before measurement.
