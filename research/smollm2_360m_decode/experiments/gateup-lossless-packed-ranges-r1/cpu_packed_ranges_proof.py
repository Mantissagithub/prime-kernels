#!/usr/bin/env python3
"""Prove packed raw row ranges preserve the CSR prefetch interval on CPU."""
import json
from pathlib import Path

K_CHUNKS = 960 // 8
FIRST_LIMIT = 1 << 24
COUNT_LIMIT = 1 << 8
UINT32_LIMIT = 1 << 32


def pack(first, count):
    assert 0 <= first < FIRST_LIMIT
    assert 0 <= count < COUNT_LIMIT
    assert count <= K_CHUNKS
    value = first | (count << 24)
    assert value < UINT32_LIMIT
    return value


def unpack(value):
    assert 0 <= value < UINT32_LIMIT
    return value & 0xFFFFFF, value >> 24


cases = [
    (0, 0),
    (0, K_CHUNKS),
    (1, 1),
    (FIRST_LIMIT - K_CHUNKS, K_CHUNKS),
    (FIRST_LIMIT - 1, 1),
    (FIRST_LIMIT - 1, 0),
]
for first, count in cases:
    packed = pack(first, count)
    got_first, got_count = unpack(packed)
    assert (got_first, got_count) == (first, count)
    original_begin_bytes = first * 8 * 2
    original_size_bytes = count * 8 * 2
    packed_begin_bytes = got_first * 8 * 2
    packed_size_bytes = got_count * 8 * 2
    assert (packed_begin_bytes, packed_size_bytes) == (original_begin_bytes, original_size_bytes)
    assert list(range(first, first + count)) == list(range(got_first, got_first + got_count))

for first, count in ((FIRST_LIMIT, 0), (0, K_CHUNKS + 1), (0, COUNT_LIMIT)):
    try:
        pack(first, count)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"invalid packed range accepted: {(first, count)}")

result = {
    "rows_checked": len(cases),
    "invalid_ranges_rejected": 3,
    "fixed_row_chunk_bound": K_CHUNKS,
    "first_ordinal_limit_exclusive": FIRST_LIMIT,
    "packed_word_limit_exclusive": UINT32_LIMIT,
    "same_raw_prefetch_begin_and_byte_count": True,
    "cases": [{"first": first, "count": count, "packed_hex": hex(pack(first, count))}
              for first, count in cases],
    "mismatches": 0,
}
Path(__file__).with_name("cpu-packed-ranges-proof.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
