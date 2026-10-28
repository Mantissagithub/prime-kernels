# Experimental SmolLM2-360M decode wrapper

This package copies the archived R9 CUDA source and adds a thin Python wrapper
for its `decode360` ABI. The copied CUDA source and include retain their
archived SHA-256 hashes. The R9 kernel passed all 30 held-out cases bitwise for
final hidden state and logits, with exact greedy IDs; the initial four fixtures
also passed bitwise. This wrapper's serving integration remains unverified, and
no speed claim follows from those results.

`decode_step` accepts int32 or int64 input IDs; int64 IDs are cast into a
per-device, graph-stable int32 staging buffer. It accepts the vLLM FLASH_ATTN
KV layout used by the 135M decode path: BF16 `[blocks, 5, 16, 128]`, with K in the first 64 elements and V in
the second 64. It requires batch 1 or 2, 32 layers, and block size 16. The
kernel overwrites its shared output buffer on the next call for that device;
consume or copy the result before calling again. `layer_outputs` and `debug`
are optional caller-owned CUDA buffers; `debug` is contiguous uint8 storage of
the ABI-reported byte capacity.

The first call JIT-builds into the normal `~/.cache/prime_kernels` location.
This wrapper deliberately performs no build until `decode_step` is called.
