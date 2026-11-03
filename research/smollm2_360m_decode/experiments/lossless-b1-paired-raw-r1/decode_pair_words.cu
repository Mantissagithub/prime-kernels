#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ void load_raw_pair(uint32_t hg, uint32_t hu, const uint16_t* raw,
                                             uint4& g, uint4& u) {
  asm volatile(
      "{ .reg .pred eg, eu;\n\t"
      ".reg .b32 bg, bu, ig, iu, g0, g1, g2, g3, u0, u1, u2, u3;\n\t"
      ".reg .b64 ag, au;\n\t"
      "shr.u32 bg, %8, 24;\n\t"
      "shr.u32 bu, %9, 24;\n\t"
      "setp.eq.u32 eg, bg, 255;\n\t"
      "setp.eq.u32 eu, bu, 255;\n\t"
      "and.b32 ig, %8, 0xffffff;\n\t"
      "and.b32 iu, %9, 0xffffff;\n\t"
      "selp.b32 ig, ig, 0, eg;\n\t"
      "selp.b32 iu, iu, 0, eu;\n\t"
      "mad.wide.u32 ag, ig, 16, %10;\n\t"
      "mad.wide.u32 au, iu, 16, %10;\n\t"
      "mov.b32 g0, 0;\n\t"
      "mov.b32 g1, 0;\n\t"
      "mov.b32 g2, 0;\n\t"
      "mov.b32 g3, 0;\n\t"
      "mov.b32 u0, 0;\n\t"
      "mov.b32 u1, 0;\n\t"
      "mov.b32 u2, 0;\n\t"
      "mov.b32 u3, 0;\n\t"
      "@eg ld.global.nc.v4.u32 {g0, g1, g2, g3}, [ag];\n\t"
      "@eu ld.global.nc.v4.u32 {u0, u1, u2, u3}, [au];\n\t"
      "mov.b32 %0, g0;\n\t"
      "mov.b32 %1, g1;\n\t"
      "mov.b32 %2, g2;\n\t"
      "mov.b32 %3, g3;\n\t"
      "mov.b32 %4, u0;\n\t"
      "mov.b32 %5, u1;\n\t"
      "mov.b32 %6, u2;\n\t"
      "mov.b32 %7, u3;\n\t"
      "}\n\t"
      : "=r"(g.x), "=r"(g.y), "=r"(g.z), "=r"(g.w),
        "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
      : "r"(hg), "r"(hu), "l"(raw)
      : "memory");
}

__device__ __forceinline__ uint4 decode_from_raw(const uint2 p, const uint32_t h,
                                                 const uint4 fallback) {
  const uint32_t base = h >> 24;
  const uint32_t raw0 = fallback.x, raw1 = fallback.y, raw2 = fallback.z, raw3 = fallback.w;
  const uint32_t base_packed = base * 0x00010001u;
  uint32_t words[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t d = (h >> (6 * j)) & 63u;
    const uint32_t delta_packed = (d * 0x2001u) & 0x00070007u;
    uint32_t expanded;
    const uint32_t packed = j < 2 ? p.x : p.y;
    if (j % 2)
      asm("prmt.b32 %0, %1, 0, 0xb3a2;" : "=r"(expanded) : "r"(packed));
    else
      asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(expanded) : "r"(packed));
    const uint32_t sign_mant = expanded & 0x807f807fu;
    words[j] = sign_mant | ((base_packed + delta_packed) << 7);
  }
  asm volatile(
      "{ .reg .pred p; .reg .b32 selected0, selected1, selected2, selected3;\n\t"
      "setp.eq.u32 p, %4, 255;\n\t"
      "selp.b32 selected0, %5, %0, p;\n\t"
      "selp.b32 selected1, %6, %1, p;\n\t"
      "selp.b32 selected2, %7, %2, p;\n\t"
      "selp.b32 selected3, %8, %3, p;\n\t"
      "mov.b32 %0, selected0;\n\t"
      "mov.b32 %1, selected1;\n\t"
      "mov.b32 %2, selected2;\n\t"
      "mov.b32 %3, selected3;\n\t"
      "}\n\t"
      : "+r"(words[0]), "+r"(words[1]), "+r"(words[2]), "+r"(words[3])
      : "r"(base), "r"(raw0), "r"(raw1), "r"(raw2), "r"(raw3)
      : "memory");
  return make_uint4(words[0], words[1], words[2], words[3]);
}



__global__ void decode_pair_words(const uint8_t* plane, const uint32_t* headers, const uint16_t* raw,
                                  uint4* g, uint4* u, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int j = (i + 7) % n;
    uint2 pg = __ldg(reinterpret_cast<const uint2*>(plane) + i);
    uint2 pu = __ldg(reinterpret_cast<const uint2*>(plane) + j);
    uint32_t hg = __ldg(headers + i), hu = __ldg(headers + j);
    uint4 rg, ru;
    load_raw_pair(hg, hu, raw, rg, ru);
    g[i] = decode_from_raw(pg, hg, rg);
    u[i] = decode_from_raw(pu, hu, ru);
  }
}
extern "C" int decode_pair_all(void* stream, const void* plane, const void* headers, const void* raw,
                              void* g, void* u, int chunks) {
  decode_pair_words<<<(chunks + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<uint4*>(g), static_cast<uint4*>(u), chunks);
  return static_cast<int>(cudaGetLastError());
}
