# SmolLM2-360M full decoder research checkpoint

This unregistered SM89 prototype executes all 32 decoder layers in one cooperative
launch, with embedding lookup, paged KV and final RMSNorm. It is not accepted for
production. Strict one-step validation passed B1 contexts8/32 and B2 context8;
B2 context32 first diverges after layer15, then KV at layer16. Greedy tokens
still match, but numerical gates fail. Do not loosen those gates.

The next task is per-stage diagnostic capture at layer15 for the failing B2 case.
The current debug buffer retains the last layer only. Optional per-layer output
capture is supported. No performance, multi-step or vLLM claim is made.

Build without fast math:

```sh
nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo --ptxas-options=-v \
  -Xcompiler=-fPIC --shared block360.cu -o libdecode360_r1.so
```

See validation-summary.json and the tiny-prl examples/tiny/HANDOFF.md checkpoint.
