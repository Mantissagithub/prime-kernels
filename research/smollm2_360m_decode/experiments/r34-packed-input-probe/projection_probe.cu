#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <stdint.h>

using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kCTAs = 24;
constexpr int kGlobalWarps = kCTAs * kWarps;

__device__ __forceinline__ float lo_bf16(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

template <int K, int B, int NR, bool Packed>
__device__ __forceinline__ void project_rows(const bf16* const* rows, const float* xf,
                                              const bf16* xb, int lane, float (&res)[NR][B]) {
  constexpr int kChunks = K / 8;
  constexpr int kIters = (kChunks + 31) / 32;
  float acc[NR][B][4];
#pragma unroll
  for (int i = 0; i < NR; ++i)
#pragma unroll
    for (int b = 0; b < B; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][b][e] = 0.f;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int c = lane + 32 * it;
    if (c < kChunks) {
      uint4 wv[NR];
#pragma unroll
      for (int i = 0; i < NR; ++i) wv[i] = __ldg(reinterpret_cast<const uint4*>(rows[i]) + c);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        float xv[8];
        if constexpr (Packed) {
          const uint4 p = *reinterpret_cast<const uint4*>(xb + b * K + c * 8);
          const uint32_t u[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
          for (int e = 0; e < 8; ++e)
            xv[e] = (e & 1) ? hi_bf16(u[e >> 1]) : lo_bf16(u[e >> 1]);
        } else {
          const float4 a = *reinterpret_cast<const float4*>(xf + b * K + c * 8);
          const float4 d = *reinterpret_cast<const float4*>(xf + b * K + c * 8 + 4);
          xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
          xv[4] = d.x; xv[5] = d.y; xv[6] = d.z; xv[7] = d.w;
        }
#pragma unroll
        for (int i = 0; i < NR; ++i) {
          const uint32_t u[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float w = (e & 1) ? hi_bf16(u[e >> 1]) : lo_bf16(u[e >> 1]);
            acc[i][b][e & 3] = fmaf(w, xv[e], acc[i][b][e & 3]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NR; ++i)
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float s = (acc[i][b][0] + acc[i][b][1]) + (acc[i][b][2] + acc[i][b][3]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
      res[i][b] = s;
    }
}

template <int K, int B, int NR, bool Packed>
__global__ void projection_kernel(const bf16* W, const bf16* x, float* y, int N) {
  __shared__ __align__(16) float xf[B * K];
  __shared__ __align__(16) bf16 xb[B * K];
  if constexpr (Packed) {
    const uint4* src = reinterpret_cast<const uint4*>(x);
    uint4* dst = reinterpret_cast<uint4*>(xb);
    for (int i = threadIdx.x; i < B * K / 8; i += kThreads) dst[i] = src[i];
  } else {
    for (int i = threadIdx.x; i < B * K; i += kThreads) xf[i] = __bfloat162float(x[i]);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int gw = B == 2 ? (threadIdx.x >> 5) * kCTAs + blockIdx.x
                        : blockIdx.x * kWarps + (threadIdx.x >> 5);
  for (int r = gw; r < N; r += kGlobalWarps) {
    const bf16* rows[NR];
#pragma unroll
    if constexpr (NR == 2) {
      rows[0] = W + static_cast<size_t>(r) * K;
      rows[1] = W + static_cast<size_t>(N + r) * K;
    } else {
      rows[0] = W + static_cast<size_t>(r) * K;
    }
    float result[NR][B];
    project_rows<K, B, NR, Packed>(rows, xf, xb, lane, result);
    if (lane == 0)
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int b = 0; b < B; ++b) y[b * N * NR + i * N + r] = result[i][b];
  }
}

template <int K, int B, int NR>
static int launch(bool packed, const bf16* W, const bf16* x, float* y, int N, cudaStream_t stream) {
  if (packed) projection_kernel<K, B, NR, true><<<kCTAs, kThreads, 0, stream>>>(W, x, y, N);
  else projection_kernel<K, B, NR, false><<<kCTAs, kThreads, 0, stream>>>(W, x, y, N);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int run_projection(int shape, int batch, int packed, const void* weights,
                              const void* input, void* output, cudaStream_t stream) {
  const auto* W = static_cast<const bf16*>(weights);
  const auto* x = static_cast<const bf16*>(input);
  auto* y = static_cast<float*>(output);
  if (shape == 0) return batch == 1 ? launch<960, 1, 1>(packed, W, x, y, 1600, stream)
                                    : launch<960, 2, 1>(packed, W, x, y, 1600, stream);
  if (shape == 1) return batch == 1 ? launch<960, 1, 1>(packed, W, x, y, 960, stream)
                                    : launch<960, 2, 1>(packed, W, x, y, 960, stream);
  if (shape == 2) return batch == 1 ? launch<960, 1, 2>(packed, W, x, y, 2560, stream)
                                    : launch<960, 2, 2>(packed, W, x, y, 2560, stream);
  if (shape == 3) return batch == 1 ? launch<2560, 1, 1>(packed, W, x, y, 960, stream)
                                    : launch<2560, 2, 1>(packed, W, x, y, 960, stream);
  return static_cast<int>(cudaErrorInvalidValue);
}
