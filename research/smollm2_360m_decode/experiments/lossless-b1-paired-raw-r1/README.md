# Experiment 232: paired raw-load trial rejected

A B1-only inline PTX helper groups both conditional gate/up raw requests before reconstruction, with the original clamping, predicates and integer arithmetic. PtxAS still separates the actual raw requests (representative instructions at 0x17120 and 0x17430); the intended overlap was not retained. B1 has 8,616 instructions and 253 registers, no spills; B2 is byte-identical to the control (9,048 instructions, 246 registers).

The independent two-output proof checks all 65,536 BF16 patterns, four passes of 4,915,200 words per output, and 64 mixed exponent pairs at four pair positions over 248 bases. Descriptor stability and refresh pass. Proof r1 stopped in the reference calculation because CUDA roll does not support uint16. The reference was corrected to roll an int16 view; decoder source did not change. The failed script and guards are retained; proof r2 passes.

One full decoder process, three alternating blocks and 120 events per arm/case, passes exact hidden/KV bits for six fixtures plus captured fallback/refresh. B1 latency changes +0.196%, +0.473%, +0.175% at contexts 32/128/1024; strict blocks 1/3, 0/3, 0/3. No gain; not adopted. B2 shifts are placebo noise, not an optimization. Valid guards pass with zero job swap. No serving or full-RL run for this rejected trial.

Sources, raw reports, SASS and guard records are preserved. Compiled libraries are excluded. Paths in historical scripts name the original local fixtures/builds; this archive is an evidence snapshot, not a standalone release.
