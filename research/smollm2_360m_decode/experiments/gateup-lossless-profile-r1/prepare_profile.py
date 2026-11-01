#!/usr/bin/env python3
"""Generate and verify the profile-only clone from the measured probe source."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gateup-lossless-probe-r1" / "projection_probe.cu"
BASE_SHA256 = "b0218ea40e2f9012c802dde20adfb428d0361ad952761c07a698a7e60e7cf09c"
PROFILE = HERE / "projection_probe.cu"
PHASES = 5
SLOT = PHASES + 1


def extract_function(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for end in range(opening, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[start:end + 1]
    raise ValueError(f"unclosed function at {marker}")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old!r}")
    return source.replace(old, new, 1)


def generate(base):
    decode = extract_function(base, "__device__ __forceinline__ uint4 decode_chunk(")
    profile_decode = decode.replace("decode_chunk(", "decode_chunk_profile(", 1)
    profile_decode = profile_decode.replace(
        "int chunk) {", "int chunk, bool* escaped) {", 1)
    profile_decode = replace_once(
        profile_decode, "const uint32_t base = h >> 24;",
        "const uint32_t base = h >> 24;\n  *escaped = base == 255; /* PROFILE_ESCAPE */")
    stripped_decode = profile_decode.replace(
        "  *escaped = base == 255; /* PROFILE_ESCAPE */\n", "")
    stripped_decode = stripped_decode.replace("decode_chunk_profile(", "decode_chunk(", 1)
    stripped_decode = stripped_decode.replace("int chunk, bool* escaped) {", "int chunk) {")
    if stripped_decode != decode:
        raise ValueError("profile decoder instrumentation stripping changed decode math")

    kernel = extract_function(base, "__global__ void gateup_lossless_kernel(")
    profile_kernel = kernel.replace("gateup_lossless_kernel(",
                                    "gateup_lossless_profile_kernel(", 1)
    profile_kernel = profile_kernel.replace(
        "const bf16* x, float* y) {", "const bf16* x, float* y, uint64_t* profile,\n"
        "                                       int profile_layer) {", 1)
    profile_kernel = replace_once(
        profile_kernel, "const int gw = warp * kCTAs + blockIdx.x;",
        "const int gw = warp * kCTAs + blockIdx.x;\n"
        f"  uint64_t phase_cycles[{PHASES}] = {{}}; /* PROFILE_STATE */\n"
        "  uint32_t escape_chunks = 0; /* PROFILE_STATE */\n"
        "  const uint64_t whole_start = clock64(); /* PROFILE_CLOCK */\n"
        "  const uint64_t prefetch_start = clock64(); /* PROFILE_CLOCK */")
    profile_kernel = replace_once(
        profile_kernel, "  for (int r = gw; r < kN; r += kGlobalWarps) {",
        "  phase_cycles[1] = clock64() - prefetch_start; /* PROFILE_CLOCK */\n"
        "  for (int r = gw; r < kN; r += kGlobalWarps) {")
    profile_kernel = replace_once(
        profile_kernel,
        "        const uint4 wg = decode_chunk(plane + size_t(r) * kK, headers + size_t(r) * kChunks, raw, c);\n"
        "        const uint4 wu = decode_chunk(plane + size_t(kN + r) * kK,\n"
        "                                      headers + size_t(kN + r) * kChunks, raw, c);",
        "        const uint64_t decode_start = clock64(); /* PROFILE_CLOCK */\n"
        "        bool escaped_g, escaped_u; /* PROFILE_STATE */\n"
        "        const uint4 wg = decode_chunk_profile(plane + size_t(r) * kK, headers + size_t(r) * kChunks, raw, c, &escaped_g);\n"
        "        const uint4 wu = decode_chunk_profile(plane + size_t(kN + r) * kK,\n"
        "                                      headers + size_t(kN + r) * kChunks, raw, c, &escaped_u);\n"
        "        phase_cycles[2] += clock64() - decode_start; /* PROFILE_CLOCK */\n"
        "        escape_chunks += escaped_g + escaped_u; /* PROFILE_COUNT */")
    profile_kernel = replace_once(
        profile_kernel, "#pragma unroll\n        for (int b = 0; b < kB; ++b) {",
        "        const uint64_t fma_start = clock64(); /* PROFILE_CLOCK */\n"
        "#pragma unroll\n        for (int b = 0; b < kB; ++b) {")
    profile_kernel = replace_once(
        profile_kernel, "        }\n      }\n    }\n#pragma unroll\n    for (int i = 0; i < 2; ++i)",
        "        }\n        phase_cycles[3] += clock64() - fma_start; /* PROFILE_CLOCK */\n"
        "      }\n    }\n"
        "    const uint64_t reduce_start = clock64(); /* PROFILE_CLOCK */\n"
        "#pragma unroll\n    for (int i = 0; i < 2; ++i)")
    profile_kernel = replace_once(
        profile_kernel, "      }\n  }\n}",
        "      }\n"
        "    phase_cycles[4] += clock64() - reduce_start; /* PROFILE_CLOCK */\n"
        "  }\n"
        "  phase_cycles[0] = clock64() - whole_start; /* PROFILE_CLOCK */\n"
        "#pragma unroll\n  for (int phase = 0; phase < " + str(PHASES) + "; ++phase) {\n"
        "    uint64_t value = phase_cycles[phase];\n"
        "#pragma unroll\n    for (int offset = 16; offset > 0; offset >>= 1) {\n"
        "      const uint64_t peer = __shfl_xor_sync(0xffffffffu, value, offset);\n"
        "      value = value > peer ? value : peer;\n    }\n"
        "    if (lane == 0) profile[(((size_t(profile_layer) * kCTAs + blockIdx.x) * kWarps + warp) * " + str(SLOT) + ") + phase] = value;\n"
        "  }\n  uint32_t count = escape_chunks;\n"
        "#pragma unroll\n  for (int offset = 16; offset > 0; offset >>= 1)\n"
        "    count += __shfl_xor_sync(0xffffffffu, count, offset);\n"
        "  if (lane == 0) profile[(((size_t(profile_layer) * kCTAs + blockIdx.x) * kWarps + warp) * " + str(SLOT) + ") + " + str(PHASES) + "] = count;\n"
        "}")

    # Removing the marked statements and profile-only decoder calls must give
    # the baseline kernel byte for byte. This guards the projection math.
    stripped = profile_kernel
    for marker in (
        "  uint64_t phase_cycles[5] = {}; /* PROFILE_STATE */\n",
        "  uint32_t escape_chunks = 0; /* PROFILE_STATE */\n",
        "  const uint64_t whole_start = clock64(); /* PROFILE_CLOCK */\n",
        "  const uint64_t prefetch_start = clock64(); /* PROFILE_CLOCK */\n",
        "  phase_cycles[1] = clock64() - prefetch_start; /* PROFILE_CLOCK */\n",
        "        const uint64_t decode_start = clock64(); /* PROFILE_CLOCK */\n",
        "        bool escaped_g, escaped_u; /* PROFILE_STATE */\n",
        "        phase_cycles[2] += clock64() - decode_start; /* PROFILE_CLOCK */\n",
        "        escape_chunks += escaped_g + escaped_u; /* PROFILE_COUNT */\n",
        "        const uint64_t fma_start = clock64(); /* PROFILE_CLOCK */\n",
        "        phase_cycles[3] += clock64() - fma_start; /* PROFILE_CLOCK */\n",
        "    const uint64_t reduce_start = clock64(); /* PROFILE_CLOCK */\n",
        "    phase_cycles[4] += clock64() - reduce_start; /* PROFILE_CLOCK */\n",
    ):
        stripped = replace_once(stripped, marker, "")
    tail_start = stripped.index("  phase_cycles[0] = clock64() - whole_start;")
    stripped = stripped[:tail_start] + "}"
    stripped = stripped.replace("gateup_lossless_profile_kernel(", "gateup_lossless_kernel(", 1)
    stripped = stripped.replace("const bf16* x, float* y, uint64_t* profile,\n"
                                "                                       int profile_layer) {",
                                "const bf16* x, float* y) {")
    stripped = stripped.replace("decode_chunk_profile(", "decode_chunk(")
    stripped = stripped.replace(", &escaped_g)", ")").replace(", &escaped_u)", ")")
    if stripped != kernel:
        raise ValueError("profile instrumentation stripping did not reproduce baseline kernel")

    wrapper = '''
extern "C" int run_gateup_lossless_profile(const void* plane, const void* headers,
                                            const void* raw, const void* raw_offsets,
                                            const void* input, void* output,
                                            uint64_t* profile, int layer,
                                            cudaStream_t stream) {
  gateup_lossless_profile_kernel<<<kCTAs, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<const uint32_t*>(raw_offsets),
      static_cast<const bf16*>(input), static_cast<float*>(output), profile, layer);
  return static_cast<int>(cudaGetLastError());
}
'''
    source = base.replace(decode, profile_decode + "\n\n" + decode, 1)
    source = source.replace(kernel, kernel + "\n\n" + profile_kernel, 1)
    source = replace_once(source, 'extern "C" int decode_weights(', wrapper + '\nextern "C" int decode_weights(')
    return source


def main():
    base = BASE.read_text()
    digest = hashlib.sha256(base.encode()).hexdigest()
    if digest != BASE_SHA256:
        raise SystemExit(f"baseline source SHA mismatch: {digest}")
    generated = generate(base)
    PROFILE.write_text(generated)
    print(f"generated {PROFILE.name} from baseline {digest}; stripped kernel matches exactly")


if __name__ == "__main__":
    main()
