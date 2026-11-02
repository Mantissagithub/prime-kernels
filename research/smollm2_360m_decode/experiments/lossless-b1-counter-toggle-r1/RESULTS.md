# Experiments 229–230

No production adoption or new verified speedup. Root implemented and ran the work directly.

## Experiment 229: counter execution is nearly neutral within one binary

A private guarded-profile kernel uses a null pointer to skip clock reads/stores. Both arms call the same loaded function; pointer publication, profile poisoning and synchronization happen outside timed events. Every placement verifies disabled mode preserves all poison values or enabled mode writes every layer/CTA cell. Three processes, three alternating blocks each, exact hidden/KV and captured fallback/refresh pass. B1 median enabled-minus-disabled latency changes are +0.095%, +0.034%, +0.016% at contexts 32/128/1024. There is no useful acceleration from counter execution in this binary.

A separate one-process comparison puts the same guarded kernel with counters disabled against the current package: B1 medians −1.41/−1.40/−1.24%, B2 +0.98/+1.16/+0.91%. Both arms leave the poisoned profile untouched. This localizes the difference to the changed compiled kernel and its remaining checks, not counter writes themselves; it does not establish a pure compiler, instruction-cache or memory-stall cause. No adoption or RL claim.

## Experiment 230: unified PTX reconstruction and compact B1 GU loop rejected

A private helper merges predicated raw loads, compressed reconstruction and raw selection into one inline PTX block with one escape predicate. Arithmetic, PRMT selectors, masks and raw address clamping are unchanged; outputs are written after all inputs are consumed. Both the 3-pass all-BF16-pattern proof and a stronger 4-pass mixed-exponent proof pass over 4,915,200 words/pass. The mixed proof covers all 64 delta pairs at four pair positions over 248 bases, with non-escape encoding required; descriptor stability and refresh lifecycle pass. Full six-case hidden/KV and captured poisoned fallback/refresh are exact. Resource usage and final instruction counts remain 253/246 registers and 8,616/9,048 instructions; machine code differs. B1 medians −0.088/+0.232/+0.048% fail the strict gain gate; no reliable overall improvement, not adopted.

A second one-directive trial requests B1 GU chunk-loop unroll 1, B2 unroll 4. B1 instruction count falls 8,616→8,304 while registers rise 253→255, no spills. The compiler also changes 16 B2 instructions despite the same intended B2 unroll factor. Exact parity and captured checks pass, but B1 latency regresses +1.23/+1.50/+0.96%; all B1 strict blocks fail. B2 is neutral at short context, +0.16% at 1024. Not adopted. All runs use the existing serial guards with zero job swap. No production change or full-RL rerun is warranted. Evidence for both experiments: `prime-kernels/research/smollm2_360m_decode/experiments/lossless-b1-counter-toggle-r1/`.


The mixed-fixture generator coverage is independently enumerated over 63,488 base/pair-position/delta-pair combinations. Native HF comparisons retain their prior diagnostic scope; exact accepted-package parity is the kernel gate. These unadopted trials did not run full RL. The masked profiling comparison localizes the timing difference to the changed compiled kernel and remaining checks, not a proven pure compiler or memory cause.

Source scripts retain original workspace paths. Thin wrappers use the published Experiment227 exact-parity harness. Frozen model/validator dependencies are unchanged from Experiment225. Source, deterministic compressed reports and guard records are included; compiled libraries/weights are excluded. All payload and original hashes verified. Fonts/theme and existing main-panel pixels are retained.
