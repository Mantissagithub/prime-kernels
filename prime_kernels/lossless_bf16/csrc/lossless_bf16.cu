#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <cstdint>

#include "encoded_gu.h"

namespace {
constexpr int kRows = 5120;
constexpr int kCols = 960;
constexpr int kChunk = 8;
constexpr int kChunksPerRow = kCols / kChunk;
constexpr int kChunks = kRows * kChunksPerRow;

__global__ void encode_chunks(const uint16_t* input, uint8_t* plane,
                              uint32_t* headers, uint32_t* flags,
                              uint32_t* enabled, uint32_t capacity,
                              uint16_t* raw) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= kChunks) return;
  const uint16_t* words = input + chunk * kChunk;
  uint32_t exponent_min = 255;
  uint32_t exponent_max = 0;
  uint32_t packed_delta = 0;
  uint8_t bytes[kChunk];
  uint16_t raw_words[kChunk];
#pragma unroll
  for (int i = 0; i < kChunk; ++i) {
    const uint16_t word = words[i];
    const uint32_t exponent = (word >> 7) & 255u;
    exponent_min = min(exponent_min, exponent);
    exponent_max = max(exponent_max, exponent);
    bytes[i] = static_cast<uint8_t>(((word >> 8) & 128u) | (word & 127u));
    raw_words[i] = word;
  }
  const bool escape = exponent_min == 255 || exponent_max - exponent_min > 7;
  if (!escape) {
#pragma unroll
    for (int i = 0; i < kChunk; ++i) {
      const uint32_t exponent = (words[i] >> 7) & 255u;
      packed_delta |= (exponent - exponent_min) << (3 * i);
    }
  }
  for (int i = 0; i < kChunk; ++i) plane[chunk * kChunk + i] = bytes[i];
  flags[chunk] = escape ? 1u : 0u;
  // The inclusive scan is launched between this kernel and scatter_chunks.
  // Header ordinals are filled by scatter_chunks after the scan.
  headers[chunk] = escape ? 0xFF000000u : ((exponent_min << 24) | packed_delta);
  if (threadIdx.x == 0 && blockIdx.x == 0) *enabled = 0;
  (void)capacity;
  (void)raw;
}

__global__ void scatter_escapes(const uint16_t* input, const uint32_t* flags,
                                uint32_t* headers,
                                const uint32_t* prefix, uint16_t* raw,
                                uint32_t capacity, uint32_t* enabled) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= kChunks) return;
  if (!flags[chunk]) return;
  const uint32_t ordinal = prefix[chunk] - 1;
  headers[chunk] = 0xFF000000u | ordinal;
  if (ordinal < capacity) {
#pragma unroll
    for (int i = 0; i < kChunk; ++i) raw[ordinal * kChunk + i] = input[chunk * kChunk + i];
  } else {
    atomicExch(enabled, 0u);
  }
}

__global__ void publish_offsets(const uint32_t* prefix, uint32_t* row_offsets) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > kRows) return;
  row_offsets[row] = row == 0 ? 0u : prefix[row * kChunksPerRow - 1];
}

__global__ void publish_mode(const uint32_t* prefix, uint32_t* enabled,
                             uint32_t capacity) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *enabled = prefix[kChunks - 1] <= capacity ? 1u : 0u;
}

__global__ void decode_chunks(const uint8_t* plane, const uint32_t* headers,
                              const uint16_t* raw, uint16_t* output) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= kChunks) return;
  const uint32_t header = headers[chunk];
  if ((header >> 24) == 255) {
    const uint32_t ordinal = header & 0xFFFFFFu;
#pragma unroll
    for (int i = 0; i < kChunk; ++i) output[chunk * kChunk + i] = raw[ordinal * kChunk + i];
  } else {
    const uint32_t base = header >> 24;
#pragma unroll
    for (int i = 0; i < kChunk; ++i) {
      const uint32_t p = plane[chunk * kChunk + i];
      const uint32_t delta = (header >> (3 * i)) & 7u;
      output[chunk * kChunk + i] = static_cast<uint16_t>(((p & 128u) << 8) |
          ((base + delta) << 7) | (p & 127u));
    }
  }
}

}  // namespace

extern "C" size_t lossless_scan_workspace_bytes() {
  size_t bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                                static_cast<uint32_t*>(nullptr), kChunks);
  return bytes;
}

extern "C" int lossless_encode_layer(const void* input, void* plane, void* headers,
                                     void* raw, void* row_offsets, void* flags,
                                     void* prefix, void* scan_workspace,
                                     size_t scan_workspace_bytes, void* enabled,
                                     uint32_t capacity, void* stream_ptr) {
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  constexpr int threads = 256;
  const int blocks = (kChunks + threads - 1) / threads;
  encode_chunks<<<blocks, threads, 0, stream>>>(static_cast<const uint16_t*>(input),
      static_cast<uint8_t*>(plane), static_cast<uint32_t*>(headers),
      static_cast<uint32_t*>(flags), static_cast<uint32_t*>(enabled), capacity,
      static_cast<uint16_t*>(raw));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cub::DeviceScan::InclusiveSum(scan_workspace, scan_workspace_bytes,
      static_cast<const uint32_t*>(flags), static_cast<uint32_t*>(prefix), kChunks, stream);
  if (error != cudaSuccess) return static_cast<int>(error);
  scatter_escapes<<<blocks, threads, 0, stream>>>(static_cast<const uint16_t*>(input),
      static_cast<const uint32_t*>(flags),
      static_cast<uint32_t*>(headers), static_cast<const uint32_t*>(prefix),
      static_cast<uint16_t*>(raw), capacity, static_cast<uint32_t*>(enabled));
  error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  publish_offsets<<<(kRows + 1 + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const uint32_t*>(prefix), static_cast<uint32_t*>(row_offsets));
  error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  publish_mode<<<1, 1, 0, stream>>>(static_cast<const uint32_t*>(prefix),
      static_cast<uint32_t*>(enabled), capacity);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int lossless_decode_layer(const void* plane, const void* headers,
                                     const void* raw, void* output, void* stream_ptr) {
  constexpr int threads = 256;
  const int blocks = (kChunks + threads - 1) / threads;
  decode_chunks<<<blocks, threads, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
      static_cast<const uint8_t*>(plane), static_cast<const uint32_t*>(headers),
      static_cast<const uint16_t*>(raw), static_cast<uint16_t*>(output));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int lossless_write_descriptor(void* table, int index, const void* plane,
                                         const void* headers, const void* raw,
                                         const void* row_offsets, const void* enabled,
                                         void* stream_ptr) {
  EncodedGULayer entry{static_cast<const uint8_t*>(plane),
                       static_cast<const uint32_t*>(headers),
                       static_cast<const uint16_t*>(raw),
                       static_cast<const uint32_t*>(row_offsets),
                       static_cast<const uint32_t*>(enabled)};
  return static_cast<int>(cudaMemcpyAsync(
      static_cast<uint8_t*>(table) + index * sizeof(EncodedGULayer), &entry,
      sizeof(entry), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_ptr)));
}
