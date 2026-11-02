# Private B1 lossless-GU trial

Hypothesis: the existing B2 decoder may also lower B1 latency when enabled for B1. Only the encoded-table eligibility and templated projection branch change; B2 source behavior, decoding, reductions and rounding remain unchanged. No production edits or performance claim.

Run the six-case R30 comparison under the canonical micro guard. Exact checks compare BF16 integer views, including signed zero. Captured poisoned fallback and GU-only refresh are checked for both B1/B2 at context32. Native HF comparison remains diagnostic; R30 exact parity is the acceptance gate. Timing is one32-layer decoder step excluding LMhead/prefill/cache refresh/RL orchestration. Three interleaved blocks, six placements per arm per case,20events per placement. Repeated process confirmation required before adopting a close result.
