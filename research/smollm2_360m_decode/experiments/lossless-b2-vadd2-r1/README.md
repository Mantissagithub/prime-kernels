# Experiment 234: B2 packed integer exponent addition rejected

Private candidate applies the accepted B1 __vadd2 exponent addition to B2, with no format, dtype or floating-point arithmetic changes. Each 16-bit field sums to at most 262, within the domain proven for Experiment 233. Full unrolling is retained.

B1 GPU instruction words, including control bits, stay exactly identical (8,720 instructions). B2 changes 9,048 to 9,152 instructions. Registers/shared/spills remain B1 253/38,440/0 and B2 246/44,848/0. No compiler resource regression is the explanation claimed here.

The first uninstrumented decoder process passes exact hidden/KV bits for both batches at contexts 32,128,1024, plus captured poisoned fallback/GU-only refresh at context32. Three alternating blocks, 120 events per arm/case. B2 median latency changes +0.025%, +0.007%, +0.155%; strict blocks 0/3,1/3,0/3. No gain, rejected. B1 timing is a byte-identical placebo, not a gain.

Both guards pass with zero job swap; GPU peak 2,231 MiB. Following the prespecified early rejection rule, no extra decoder or serving sweep is run for this losing variant. Production remains unchanged. Compiled libraries are local only and excluded from the archive.
