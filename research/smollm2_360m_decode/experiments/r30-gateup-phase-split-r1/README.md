# R30 gate/up phase split

This source-only probe tests whether R30's existing combined gate/up interval is dominated by gate/up projection and SwiGLU math or by the following down-weight prefetch issue loop. The combined eleven-slot profile remains intact. Two private subslots measure only the gate/up math loop and the subsequent down-prefetch issue loop. Neither subslot is added to the eleven-slot totals or their fractions.

`prepare_sources.py` pins both R30 baseline source hashes, generates `r30-profile/` from unique source anchors, checks the subslot boundaries and decoder source ABI, and proves exact baseline recovery after removing instrumentation. Review the generated `block360_kernel.inc` diff manually before building. Instrumentation changes only the private profiling clone. It adds no barrier or arithmetic and preserves the kernel workspace ABI and existing projection loops.

`profile_bench.py` defaults to the B2 context 32 and 128 pilot cases. Each timed case compares the uninstrumented R30 library to this instrumented clone with CUDA-event ABBA samples. It checks exact profile-clone parity for final hidden state and every K/V cache tensor before writing measurements. The analyzer rejects cases whose median instrumentation overhead exceeds five percent.

Raw clock64 values are stored as `[32 layers, 13 slots, 24 CTAs, 2 fields]`. The first eleven slots preserve the previous work and existing grid-wait measurements. Slots twelve and thirteen report gate/up math and down-prefetch issue separately. Reports use same-CTA medians and distributions, never sums across SMs. The original combined interval remains the denominator basis, so the subslots are not double-counted.

Previously observed R30 and R35 SM clocks were 2250 MHz and 2340 MHz. These runs are not clock-matched, so cross-variant cycle comparisons are not causal. This R30 clone decomposes one R30 interval within the same kernel variant.

No optimization is proposed by this probe. No CUDA build, GPU call, model load, or production-source change has been performed. The clone is prepared for review and remains unmeasured until the guarded build and benchmark are run.

## Commands

Run source preparation and CPU checks from this directory:

```sh
UV_CACHE_DIR=/tmp/tiny-prl-profile-uv uv run --offline --no-project --python /home/pradheep/.venv/bin/python prepare_sources.py
```

After manual diff review and the root-owned build/GPU guard, build with `./build_profile.sh r30`. The benchmark defaults to `seed17-b2-ctx32,seed17-b2-ctx128` and writes `results/phase-split.json`. Summarize with `uv run --offline --no-project --python /home/pradheep/.venv/bin/python analyze_phase_split.py results/phase-split.json`.
