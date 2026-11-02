#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ uint4 decode_chunk_predicated(const uint8_t* plane,
                                                          const uint32_t* headers,
                                                          const uint16_t* raw,
                                                          int chunk) {
  const uint2 p = __ldg(reinterpret_cast<const uint2*>(plane) + chunk);
  const uint32_t h = __ldg(headers + chunk);
  const uint32_t base = h >> 24;
  uint32_t raw0 = 0, raw1 = 0, raw2 = 0, raw3 = 0;
  asm volatile(
      "{ .reg .pred escape; .reg .b32 raw_index; .reg .b64 raw_address;\n\t"
      "setp.eq.u32 escape, %4, 255;\n\t"
      "selp.b32 raw_index, %5, 0, escape;\n\t"
      "mad.wide.u32 raw_address, raw_index, 16, %6;\n\t"
      "@escape ld.global.nc.v4.u32 {%0, %1, %2, %3}, [raw_address];\n\t"
      "}\n\t"
      : "+r"(raw0), "+r"(raw1), "+r"(raw2), "+r"(raw3)
      : "r"(base), "r"(h & 0xFFFFFFu), "l"(raw)
      : "memory");

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



__global__ void decode_words(const uint8_t* plane, const uint32_t* headers, const uint16_t* raw, uint4* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = decode_chunk_predicated(plane, headers, raw, i);
}
extern "C" int decode_all(void* stream, const void* plane, const void* headers, const void* raw, void* out, int chunks) {
  decode_words<<<(chunks + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers), static_cast<const uint16_t*>(raw), static_cast<uint4*>(out), chunks);
  return static_cast<int>(cudaGetLastError());
}
