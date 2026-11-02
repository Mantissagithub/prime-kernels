#!/usr/bin/env python3
"""GPU proof for encoded GU exactness, fixed-capacity fallback, and refresh."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "prime-kernels"))
import torch

from prime_kernels.lossless_bf16 import COLS, LAYERS, ROWS, EncodedGUCache


def base_weights(device):
    bits = torch.full((ROWS, COLS), 0x3F80, dtype=torch.uint16, device=device)
    return [bits.view(torch.bfloat16).clone() for _ in range(LAYERS)]


def all_bit_patterns(device):
    patterns = torch.arange(1 << 16, dtype=torch.int32, device=device).to(torch.uint16)
    bits = patterns.repeat((ROWS * COLS + patterns.numel() - 1) // patterns.numel())[:ROWS * COLS]
    result = [torch.full((ROWS * COLS,), 0x3F80, dtype=torch.uint16, device=device)
              .view(torch.bfloat16).reshape(ROWS, COLS).contiguous() for _ in range(LAYERS)]
    result[0] = bits.view(torch.bfloat16).reshape(ROWS, COLS).contiguous()
    return result


def wide_escape_weights(device):
    bits = torch.empty((ROWS, COLS), dtype=torch.uint16, device=device)
    bits[:, 0::2] = 0x0000
    bits[:, 1::2] = 0x6400
    return [bits.view(torch.bfloat16).clone() for _ in range(LAYERS)]


def assert_exact(cache, weights, layer):
    actual = cache.decode_layer(layer).view(torch.uint16)
    expected = weights[layer].view(torch.uint16)
    mismatch = int(torch.count_nonzero(actual != expected).item())
    if mismatch:
        raise AssertionError(f"layer {layer} has {mismatch} mismatched BF16 words")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("codec-proof.json"))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.build_only:
        from prime_kernels.lossless_bf16 import _load_library

        _load_library()
        print("lossless codec compiled")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("this proof requires the root CUDA operator's GPU")
    device = torch.device("cuda")
    stable = base_weights(device)
    patterns = all_bit_patterns(device)
    with torch.inference_mode():
        cache = EncodedGUCache(patterns)
    table_ptr = cache.table.data_ptr()
    buffer_ptrs = [[tensor.data_ptr() for key, tensor in table.items() if key in
                    ("plane", "headers", "raw", "row_offsets")] for table in cache._tables]
    initial = cache.statistics()

    # The initial capacities are measured from this encoded input, so this exercises the enabled path.
    assert cache.table.data_ptr() == table_ptr
    assert_exact(cache, patterns, 0)
    pattern_generation = cache.statistics()
    if pattern_generation["active_layers"] != LAYERS:
        raise AssertionError("all-pattern cache unexpectedly selected dense fallback")

    # Same-address native reload: mutate layer 0 in place, refresh the same tensor objects.
    patterns[0].view(torch.uint16).fill_(0xC123)
    cache.refresh(patterns)
    assert_exact(cache, patterns, 0)
    in_place_generation = cache.statistics()

    # Deliberately exceed the initial one-chunk reserve, then ensure disabled layers decode densely.
    overflow = wide_escape_weights(device)
    cache.refresh(overflow)
    overflow_stats = cache.statistics()
    if overflow_stats["active_layers"] != 0:
        raise AssertionError("capacity overflow did not select dense fallback")
    assert_exact(cache, overflow, 0)

    # A later representable reload must republish the encoded mode and exact contents.
    cache.refresh(stable)
    recovered = cache.statistics()
    if recovered["active_layers"] != LAYERS:
        raise AssertionError("cache did not recover after overflow")
    assert_exact(cache, stable, 0)
    final_buffer_ptrs = [[tensor.data_ptr() for key, tensor in table.items() if key in
                          ("plane", "headers", "raw", "row_offsets")] for table in cache._tables]
    if final_buffer_ptrs != buffer_ptrs:
        raise AssertionError("encoded allocation addresses changed across refresh")

    result = {
        "result": "passed",
        "bf16_words_checked_all_patterns": 1 << 16,
        "initial": initial,
        "all_patterns": pattern_generation,
        "in_place_reload": in_place_generation,
        "overflow": overflow_stats,
        "recovered": recovered,
        "descriptor_pointer_stable": cache.table.data_ptr() == table_ptr,
        "encoded_buffer_pointers_stable": final_buffer_ptrs == buffer_ptrs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
