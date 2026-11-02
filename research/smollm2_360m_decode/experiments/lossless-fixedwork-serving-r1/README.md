# Fixed-work lossless serving comparison

`bench_serving.py` runs exactly one arm in one fresh process. Use `baseline` for
the private R30 preload or `candidate` for the packaged lossless decoder. Both
arms use the same seeded token-ID prompts, vLLM engine settings, greedy 64-token
generation, and fixed request shapes. The untimed parity request records token
IDs, sampled-token logprobs, and top-5 logprobs. Two warmups precede five timed
repeats per prompt length and batch size; timed requests disable logprobs.

Example, using the local micro-runner and running one arm at a time:

```sh
bash outputs/megakernel-20261025/runtime/run_micro.sh lossless-fixedwork-a2 \
  outputs/megakernel-20261025/lossless-fixedwork-serving-r1 a2 \
  outputs/megakernel-20261025/lossless-fixedwork-serving-r1/bench_serving.py \
  baseline outputs/megakernel-20261025/lossless-fixedwork-serving-r1/baseline-02.json
```

Repeat with a distinct output filename and `candidate` for each process. The
runner should use the repo's `uv` environment. Output creation refuses to
overwrite an existing JSON. Candidate source, include, and library hashes are
pinned in the harness. Baseline preload traces are written beside its result.

The harness requests 256 GPU blocks to keep KV capacity equal, and sets
`gpu_memory_utilization=0.15`. The requested budget is 15% of device memory;
the launcher must still enforce the 3 GiB GPU and 4 GiB host headroom guards.
No runtime build is intended: R30 and lossless libraries are expected to be
prebuilt. Startup duration is recorded separately from generation timing.

`batch_wall_ms` covers full `LLM.generate` wall time, including scheduling and
output processing. Scheduler-inclusive ITL is unavailable in this harness.
CUDA graph replay can bypass Python decode calls, so route evidence combines
the installed backend, selected model geometry, loaded library identity, and
nonempty runtime/cache state; backend counters are included as diagnostic data
but are not used as a per-repeat proof. Five repeats in one process are not
five independent runs. Do not claim a small speedup until at least five fresh
processes per arm are compared and all parity outputs match.

`compare.py FILE... --output NEW_REPORT.json` checks equal prompt and output
token IDs, engine settings and timed output hashes. Its timing units are
milliseconds per synchronous batch. It summarizes each process separately;
overlapping process ranges remain inconclusive. Guard results are checked
separately. The interrupted A1 attempt has no completion report and contributes
no timing sample.
