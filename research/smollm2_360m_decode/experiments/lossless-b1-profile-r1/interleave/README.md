# Private B1 CTA interleaving

One-line trial: use warp-major global warp numbering for B1 as already used for B2. It balances leftover GU rows across CTAs rather than concentrating them in the first eight CTAs. All per-row accumulation order and reduction math remain unchanged. Global mapping also affects other projection/attention work; exact hidden/KV, captured poisoned fallback and refresh checks must pass before interpreting timing. No profiling instrumentation in this candidate. No production edits or speed claim before measurement.
