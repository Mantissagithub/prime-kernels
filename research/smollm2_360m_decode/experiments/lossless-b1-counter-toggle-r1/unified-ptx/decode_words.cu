#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ uint4 decode_chunk_predicated(const uint8_t* plane,
                                                          const uint32_t* headers,
                                                          const uint16_t* raw,
                                                          int chunk) {
  const uint2 p = __ldg(reinterpret_cast<const uint2*>(plane) + chunk);
  const uint32_t h = __ldg(headers + chunk);
  uint32_t w0, w1, w2, w3;
  asm volatile(
      "{ .reg .pred escape;\n\t"
      ".reg .b32 base, basepair, index, d, delta, expanded, tmp;\n\t"
      ".reg .b32 r0, r1, r2, r3, w0, w1, w2, w3;\n\t"
      ".reg .b64 address;\n\t"
      "shr.u32 base, %6, 24;\n\t"
      "setp.eq.u32 escape, base, 255;\n\t"
      "and.b32 index, %6, 0xffffff;\n\t"
      "selp.b32 index, index, 0, escape;\n\t"
      "mad.wide.u32 address, index, 16, %7;\n\t"
      "mov.b32 r0, 0;\n\t"
      "mov.b32 r1, 0;\n\t"
      "mov.b32 r2, 0;\n\t"
      "mov.b32 r3, 0;\n\t"
      "@escape ld.global.nc.v4.u32 {r0, r1, r2, r3}, [address];\n\t"
      "mul.lo.u32 basepair, base, 0x00010001;\n\t"
      "shr.u32 d, %6, 0;\n\t"
      "and.b32 d, d, 63;\n\t"
      "mul.lo.u32 delta, d, 0x2001;\n\t"
      "and.b32 delta, delta, 0x00070007;\n\t"
      "prmt.b32 expanded, %4, 0, 0x9180;\n\t"
      "and.b32 expanded, expanded, 0x807f807f;\n\t"
      "add.u32 tmp, basepair, delta;\n\t"
      "shl.b32 tmp, tmp, 7;\n\t"
      "or.b32 tmp, tmp, expanded;\n\t"
      "selp.b32 w0, r0, tmp, escape;\n\t"
      "shr.u32 d, %6, 6;\n\t"
      "and.b32 d, d, 63;\n\t"
      "mul.lo.u32 delta, d, 0x2001;\n\t"
      "and.b32 delta, delta, 0x00070007;\n\t"
      "prmt.b32 expanded, %4, 0, 0xb3a2;\n\t"
      "and.b32 expanded, expanded, 0x807f807f;\n\t"
      "add.u32 tmp, basepair, delta;\n\t"
      "shl.b32 tmp, tmp, 7;\n\t"
      "or.b32 tmp, tmp, expanded;\n\t"
      "selp.b32 w1, r1, tmp, escape;\n\t"
      "shr.u32 d, %6, 12;\n\t"
      "and.b32 d, d, 63;\n\t"
      "mul.lo.u32 delta, d, 0x2001;\n\t"
      "and.b32 delta, delta, 0x00070007;\n\t"
      "prmt.b32 expanded, %5, 0, 0x9180;\n\t"
      "and.b32 expanded, expanded, 0x807f807f;\n\t"
      "add.u32 tmp, basepair, delta;\n\t"
      "shl.b32 tmp, tmp, 7;\n\t"
      "or.b32 tmp, tmp, expanded;\n\t"
      "selp.b32 w2, r2, tmp, escape;\n\t"
      "shr.u32 d, %6, 18;\n\t"
      "and.b32 d, d, 63;\n\t"
      "mul.lo.u32 delta, d, 0x2001;\n\t"
      "and.b32 delta, delta, 0x00070007;\n\t"
      "prmt.b32 expanded, %5, 0, 0xb3a2;\n\t"
      "and.b32 expanded, expanded, 0x807f807f;\n\t"
      "add.u32 tmp, basepair, delta;\n\t"
      "shl.b32 tmp, tmp, 7;\n\t"
      "or.b32 tmp, tmp, expanded;\n\t"
      "selp.b32 w3, r3, tmp, escape;\n\t"
      "mov.b32 %0, w0;\n\t"
      "mov.b32 %1, w1;\n\t"
      "mov.b32 %2, w2;\n\t"
      "mov.b32 %3, w3;\n\t"
      "}\n\t"
      : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
      : "r"(p.x), "r"(p.y), "r"(h), "l"(raw)
      : "memory");
  return make_uint4(w0, w1, w2, w3);
}

__global__ void decode_words(const uint8_t* plane, const uint32_t* headers, const uint16_t* raw, uint4* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = decode_chunk_predicated(plane, headers, raw, i);
}
extern "C" int decode_all(void* stream, const void* plane, const void* headers, const void* raw, void* out, int chunks) {
  decode_words<<<(chunks + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers), static_cast<const uint16_t*>(raw), static_cast<uint4*>(out), chunks);
  return static_cast<int>(cudaGetLastError());
}
