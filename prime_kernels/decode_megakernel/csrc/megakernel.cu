// SM89 batched decode megakernel for SmolLM2-135M (R6). One cooperative launch per
// decode step: 24 CTAs x 256 threads, five grid barriers per layer. See NOTES.md and
// ../integration-spec.md. B (1..4) is a template parameter; the host dispatches.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

// Stage B switches (default 0 = stage A, the grid.sync baseline).
//   MK_PREFETCH=1     before each barrier, prefetch.global.L2 the weight rows the next phase reads.
//   MK_COUNTER_SYNC=1 replace grid.sync with a device-side arrival counter per barrier (release/acquire).
#ifndef MK_PREFETCH
#define MK_PREFETCH 1
#endif
#ifndef MK_COUNTER_SYNC
#define MK_COUNTER_SYNC 0
#endif
// R6: MK_ATTN_V2=1 batches 4 positions per lane group (loads for all 4 issued before the math,
// one exp rescale per batch). MK_ADAPT_NC=1 uses ceil(seq/16) chunks per (request, kv head) instead
// of 64/B, so short contexts merge fewer partials.
#ifndef MK_ATTN_V2
#define MK_ATTN_V2 1
#endif
#ifndef MK_ADAPT_NC
#define MK_ADAPT_NC 1
#endif
// MK_GEMV_BATCH=1: load all weight rows of a K=576 GEMV phase (QKV, O) before any dot, and two
// gate/up row pairs at a time, so the phase waits on one memory round trip instead of one per row.
#ifndef MK_GEMV_BATCH
#define MK_GEMV_BATCH 1
#endif
// MK_ATTN_FAST=1 (with MK_ATTN_V2): attention score dots use fused FMA, __expf for the softmax, and
// shift/mask block indexing when block_size is a power of two. Attention has no cuBLAS order to match.
#ifndef MK_ATTN_FAST
#define MK_ATTN_FAST 0
#endif

// Chunks per (request, kv head) for a context of `seq` tokens; nc_max = 64 / B.
__device__ __forceinline__ int chunks_for(int seq, int nc_max) {
#if MK_ADAPT_NC
  const int want = (seq + 15) / 16;
  return want < 1 ? 1 : (want < nc_max ? want : nc_max);
#else
  (void)seq;
  return nc_max;
#endif
}

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

constexpr int kH = 576;
constexpr int kI = 1536;
constexpr int kQH = 9;
constexpr int kKVH = 3;
constexpr int kD = 64;
constexpr int kLayers = 30;
constexpr int kMaxB = 4;
constexpr int kMaxSeq = 8192;
constexpr int kCTAs = 24;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;         // 8
constexpr int kGlobalWarps = kCTAs * kWarps;  // 192
constexpr float kEps = 1e-5f;
constexpr float kAttnScale = 0.125f;

// Global bf16 scratch (elements). Each region holds [B_MAX][width] row-major.
constexpr int kOffX = 0;      // residual stream [4][576]
constexpr int kOffQ = 2304;   // q before RoPE [4][576]
constexpr int kOffK = 4608;   // k before RoPE [4][192]
constexpr int kOffV = 5376;   // v [4][192]
constexpr int kOffH1 = 6144;  // post-attention residual [4][576]
constexpr int kOffAct = 8448; // SwiGLU output [4][1536]
constexpr int kScratchBf16 = 14592;
// Global fp32 scratch (elements) for attention partials, one item per (request, kv head, chunk).
constexpr int kFOffPM = 0;      // [192 items][3 heads] running max
constexpr int kFOffPL = 576;    // [192][3] running sum
constexpr int kFOffPACC = 1152; // [192][3][64] unnormalized P.V
constexpr int kFOffCtr = 1152 + 192 * 3 * 64;  // unsigned[kBarriers + 1]: arrivals per barrier, epoch
constexpr int kBarriers = kLayers * 5;
constexpr int kScratchFloat = kFOffCtr + 160;
constexpr size_t kWorkspaceBytes = size_t(kScratchBf16) * 2 + size_t(kScratchFloat) * 4;

// Host status codes.
constexpr int kErrB = 9101;
constexpr int kErrMaxSeq = 9102;
constexpr int kErrNull = 9103;
constexpr int kErrCoop = 9104;
constexpr int kErrResidency = 9105;
constexpr int kErrSm89 = 9106;
constexpr int kErrAlign = 9107;
constexpr int kErrScratch = 9108;
constexpr int kErrBlock = 9109;

// Prefetch the first `bytes` of a row into L2, one 128-byte line per lane step (warp-cooperative).
__device__ __forceinline__ void prefetch_row(const void* ptr, int bytes) {
  const char* c = reinterpret_cast<const char*>(ptr);
  for (int off = (threadIdx.x & 31) * 128; off < bytes; off += 32 * 128) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(c + off));
  }
}

// Arrival-counter barrier across the cooperative grid. ctr is cumulative over launches: the
// caller passes target = gridDim.x * (launch index + 1). Writers release (fence) before the
// arrival; waiters acquire (fence) after observing the target.
__device__ __forceinline__ void flag_barrier(unsigned* ctr, unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(ctr, 1u);
    while (true) {
      const unsigned v = *reinterpret_cast<volatile unsigned*>(ctr);
      if (v >= target) break;
      __nanosleep(32);
    }
    __threadfence();
  }
  __syncthreads();
}

// Weight table entry: one device struct per layer (device array of 30).
struct LayerW {
  const bf16* norm1;   // [576]
  const bf16* qkv;     // [960, 576]: q rows 0..575, k 576..767, v 768..959
  const bf16* o;       // [576, 576]
  const bf16* norm2;   // [576]
  const bf16* gate_up; // [3072, 576]: gate rows 0..1535, up 1536..3071
  const bf16* down;    // [576, 1536]
};

struct Params {
  const LayerW* layers;
  const bf16* embed;        // [V, 576]
  const bf16* final_norm;   // [576]
  void* const* kv;          // device array of 30 layer base pointers
  const void* input_ids;    // [B] int32 or int64
  int ids_i64;
  const int64_t* positions; // [B] absolute position of the new token
  const int* seq_lens;      // [B] tokens in context including the new one
  const int* block_table;   // [B, bt_stride]
  int bt_stride;
  const int64_t* slot_mapping; // [B]
  const void* cos_sin;      // [max_pos, 64]: cos 0..31, sin 32..63
  int cs_fp32;
  bf16* out_hidden;         // [B, 576]
  bf16* ws;                 // bf16 scratch
  float* wsf;               // fp32 scratch
  long long k_off, v_off, block_stride, token_stride, head_stride;  // elements
  int block_size;
  int bs_shift;  // log2(block_size) when it is a power of two, else -1
};

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rnd(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Halves tree (shfl_down 16..1); lane 0 holds the sum, broadcast to all lanes.
__device__ __forceinline__ float warp_sum(float v) {
  constexpr unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) v = add_rn(v, __shfl_down_sync(mask, v, off));
  return __shfl_sync(mask, v, 0);
}

// CTA-wide sum of one value per thread. All threads must call.
__device__ float block_sum(float v, float* red) {
  const int warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[warp] = v;
  __syncthreads();
  float total = red[0];
  for (int w = 1; w < kWarps; ++w) total = add_rn(total, red[w]);
  __syncthreads();
  return total;
}

// out[i] = bf16(bf16(src[i] * rsqrt(mean(src^2) + eps)) * w[i]) for 576 entries, fp32 out (bf16 values).
__device__ void rms_norm_to_smem(const bf16* src, const bf16* w, float* out, float* red) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < kH; i += kThreads) {
    const float v = bf2f(src[i]);
    ss = add_rn(ss, mul_rn(v, v));
  }
  ss = block_sum(ss, red);
  const float scale = rsqrtf(ss / static_cast<float>(kH) + kEps);
  for (int i = threadIdx.x; i < kH; i += kThreads) {
    out[i] = rnd(mul_rn(rnd(mul_rn(bf2f(src[i]), scale)), bf2f(w[i])));
  }
  __syncthreads();
}

// B dot products of one weight row (K elements, bf16) with B fp32 input vectors vs[b*K..].
// Order follows the cuBLAS M=1 gemv (gemv-order-r1/README.md): K=576 lane L sums elements
// 18L..18L+17 sequentially; K=1536 lane L sums L+32j, j=0..47; then the halves tree.
// Result is valid in every lane (before BF16 rounding).
template <int K, int B>
__device__ __forceinline__ void dots(const bf16* __restrict__ row, const float* vs, int lane, float* out) {
  float s[B];
#pragma unroll
  for (int b = 0; b < B; ++b) s[b] = 0.f;
  if constexpr (K == kH) {
    const bf162* w2 = reinterpret_cast<const bf162*>(row + 18 * lane);
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const float2 f = __bfloat1622float2(w2[i]);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const float2 x = *reinterpret_cast<const float2*>(vs + b * K + 18 * lane + 2 * i);
        s[b] = add_rn(s[b], mul_rn(f.x, x.x));
        s[b] = add_rn(s[b], mul_rn(f.y, x.y));
      }
    }
  } else {
    static_assert(K == kI, "unsupported K");
#pragma unroll 8
    for (int j = 0; j < kI / 32; ++j) {
      const float w = bf2f(row[lane + 32 * j]);
#pragma unroll
      for (int b = 0; b < B; ++b) s[b] = add_rn(s[b], mul_rn(w, vs[b * K + lane + 32 * j]));
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) out[b] = warp_sum(s[b]);
}

// Raw 4-byte words of one K=576 weight row for this lane (elements 18L..18L+17), read-only.
__device__ __forceinline__ void load_row576(const bf16* row, int lane, uint32_t* w) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(row + 18 * lane);
#pragma unroll
  for (int i = 0; i < 9; ++i) w[i] = __ldg(p + i);
}

// Same arithmetic as dots<576,B> on words loaded by load_row576.
template <int B>
__device__ __forceinline__ void dot576_from(const uint32_t* w, const float* vs, int lane, float* out) {
  float s[B];
#pragma unroll
  for (int b = 0; b < B; ++b) s[b] = 0.f;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const bf162*>(&w[i]));
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float2 x = *reinterpret_cast<const float2*>(vs + b * kH + 18 * lane + 2 * i);
      s[b] = add_rn(s[b], mul_rn(f.x, x.x));
      s[b] = add_rn(s[b], mul_rn(f.y, x.y));
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) out[b] = warp_sum(s[b]);
}

// RoPE (neox rotate-half), fp32 math with one BF16 rounding at the end (vLLM rotary kernel).
__device__ __forceinline__ float rope_val(const bf16* head, int d, int pos, const void* cs, int cs_fp32) {
  const int ci = pos * kD + (d & 31);
  const int si = ci + 32;
  float c, s;
  if (cs_fp32) {
    c = reinterpret_cast<const float*>(cs)[ci];
    s = reinterpret_cast<const float*>(cs)[si];
  } else {
    c = bf2f(reinterpret_cast<const bf16*>(cs)[ci]);
    s = bf2f(reinterpret_cast<const bf16*>(cs)[si]);
  }
  const float x = bf2f(head[d]);
  const float y = bf2f(head[d ^ 32]);
  return rnd(d < 32 ? x * c - y * s : x * c + y * s);
}

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const bf162* h = reinterpret_cast<const bf162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// One cooperative decode step for B requests (B = template parameter).
template <int B>
__global__ void __launch_bounds__(kThreads, 1) mk_kernel(const Params p) {
  constexpr int NC = 64 / B;      // chunks per (request, kv head); 3*B*NC <= 192 warps
  constexpr int NI = 3 * B * NC;  // attention work items
  static_assert(NI <= kGlobalWarps, "work items exceed warps");

  cg::grid_group grid = cg::this_grid();
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int gw = blockIdx.x * kWarps + warp;  // 0..191

  __shared__ __align__(16) float vec_s[kMaxB * kI];  // normed / attention / activation vectors per request
  __shared__ __align__(16) float q_s[B * kQH * kD];  // rotated q per request and query head
  __shared__ __align__(16) float k_s[B * kKVH * kD]; // rotated k of this step
  __shared__ __align__(16) float v_s[B * kKVH * kD]; // v of this step
  __shared__ float wt_s[B * kQH * NC];               // merge weights exp(m_c - M)
  __shared__ float inv_s[B * kQH];                   // 1 / merged denominator
  __shared__ float red[kWarps];
  __shared__ int seq_s[B], pos_s[B], nc_s[B];

  bf16* ws = p.ws;
  unsigned* ctrs = reinterpret_cast<unsigned*>(p.wsf + kFOffCtr);
#if MK_COUNTER_SYNC
  const unsigned epoch = *reinterpret_cast<volatile unsigned*>(ctrs + kBarriers);
  const unsigned target = gridDim.x * (epoch + 1u);
#endif
  // Barrier `slot` (0..149): grid.sync, or the arrival counter in MK_COUNTER_SYNC builds.
  auto sync_at = [&](int slot) {
#if MK_COUNTER_SYNC
    flag_barrier(ctrs + slot, target);
#else
    (void)slot;
    grid.sync();
#endif
  };
  float* pm = p.wsf + kFOffPM;
  float* pl = p.wsf + kFOffPL;
  float* pacc = p.wsf + kFOffPACC;

  int tok[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    tok[b] = p.ids_i64 ? static_cast<int>(reinterpret_cast<const int64_t*>(p.input_ids)[b])
                       : reinterpret_cast<const int*>(p.input_ids)[b];
  }
  // Padded rows (slot_mapping < 0) keep a valid position and length so RoPE indices and the
  // softmax stay finite; their K/V write is skipped and their output is ignored by the caller.
  if (t < B) {
    const int sq = max(static_cast<int>(p.seq_lens[t]), 1);
    seq_s[t] = sq;
    pos_s[t] = max(static_cast<int>(p.positions[t]), 0);
    nc_s[t] = chunks_for(sq, NC);
  }
  __syncthreads();

  float o[B], go[B], uo[B];

  for (int layer = 0; layer < kLayers; ++layer) {
    const LayerW L = p.layers[layer];
    bf16* kvw = reinterpret_cast<bf16*>(p.kv[layer]);

    // Phase 1: RMSNorm per request, then QKV rows (960) over 192 warps, one row for all B.
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const bf16* xin = layer == 0 ? p.embed + static_cast<size_t>(tok[b]) * kH : ws + kOffX + b * kH;
      rms_norm_to_smem(xin, L.norm1, vec_s + b * kH, red);
    }
#if MK_GEMV_BATCH
    uint32_t wqkv[5][9];
#pragma unroll
    for (int k = 0; k < 5; ++k) load_row576(L.qkv + static_cast<size_t>(gw + k * kGlobalWarps) * kH, lane, wqkv[k]);
#endif
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      const int r = gw + k * kGlobalWarps;
#if MK_GEMV_BATCH
      dot576_from<B>(wqkv[k], vec_s, lane, o);
#else
      dots<kH, B>(L.qkv + static_cast<size_t>(r) * kH, vec_s, lane, o);
#endif
      if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
          if (r < kH) {
            ws[kOffQ + b * kH + r] = __float2bfloat16_rn(o[b]);
          } else if (r < kH + 192) {
            ws[kOffK + b * 192 + (r - kH)] = __float2bfloat16_rn(o[b]);
          } else {
            ws[kOffV + b * 192 + (r - kH - 192)] = __float2bfloat16_rn(o[b]);
          }
        }
      }
    }
#if MK_PREFETCH
    for (int k = 0; k < 3; ++k) prefetch_row(L.o + static_cast<size_t>(gw + k * kGlobalWarps) * kH, kH * 2);
#endif
    sync_at(layer * 5 + 0);  // B1

    // Phase 2a: RoPE for q and k (all requests, every CTA), v copied; new K/V written by CTA 0.
    for (int idx = t; idx < B * kQH * kD; idx += kThreads) {
      const int b = idx / (kQH * kD), rem = idx % (kQH * kD);
      q_s[idx] = rope_val(ws + kOffQ + b * kH + (rem / kD) * kD, rem % kD, pos_s[b], p.cos_sin, p.cs_fp32);
    }
    for (int idx = t; idx < B * kKVH * kD; idx += kThreads) {
      const int b = idx / (kKVH * kD), rem = idx % (kKVH * kD);
      const int kvh = rem / kD, d = rem % kD;
      k_s[idx] = rope_val(ws + kOffK + b * 192 + kvh * kD, d, pos_s[b], p.cos_sin, p.cs_fp32);
      v_s[idx] = bf2f(ws[kOffV + b * 192 + kvh * kD + d]);
    }
    __syncthreads();
    if (blockIdx.x == 0) {
      for (int idx = t; idx < B * kKVH * kD; idx += kThreads) {
        const int b = idx / (kKVH * kD), rem = idx % (kKVH * kD);
        const int kvh = rem / kD, d = rem % kD;
        const int64_t slot = p.slot_mapping[b];
        if (slot < 0) continue;  // padded row: no K/V write
        const long long blk = slot / p.block_size, off = slot % p.block_size;
        const long long e = blk * p.block_stride + off * p.token_stride + kvh * p.head_stride + d;
        kvw[p.k_off + e] = __float2bfloat16_rn(k_s[idx]);
        kvw[p.v_off + e] = __float2bfloat16_rn(v_s[idx]);
      }
    }

    // Phase 2b: split-KV attention. Item = (request b, kv head kvh, chunk c); warp gw.
    if (gw < NI) {
      const int c = gw % NC, bk = gw / NC, kvh = bk % kKVH, b = bk / kKVH;
      const int seqb = seq_s[b];
      const int ncb = nc_s[b];
      const int Lc = (seqb + ncb - 1) / ncb;
      const int start = c * Lc;
      const int end = min(start + Lc, seqb);
      const int item = (b * kKVH + kvh) * NC + c;
      if (c >= ncb) {
        // inactive chunk for this request: no partial is written or merged
      } else if (start >= end) {
        if (lane < 8) {
#pragma unroll
          for (int hh = 0; hh < 3; ++hh) {
            if (lane == 0) {
              pm[item * 3 + hh] = -INFINITY;
              pl[item * 3 + hh] = 0.f;
            }
            const int dd = (item * 3 + hh) * kD + 8 * lane;
#pragma unroll
            for (int i = 0; i < 8; ++i) pacc[dd + i] = 0.f;
          }
        }
      } else {
        const int j = lane & 7, g = lane >> 3;
        float qr[3][8], acc[3][8], m[3], l[3];
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
          m[hh] = -INFINITY;
          l[hh] = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            qr[hh][i] = q_s[(b * kQH + kvh * 3 + hh) * kD + 8 * j + i];
            acc[hh][i] = 0.f;
          }
        }
        const int len = end - start;
        const int niter = (len + 3) / 4;  // uniform over the warp; group g takes positions start+g+4i

        // One position for this lane group: score for 3 heads, online softmax update.
        auto update = [&](bool valid, const float* kf, const float* vf) {
#pragma unroll
          for (int hh = 0; hh < 3; ++hh) {
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) d = add_rn(d, mul_rn(qr[hh][i], kf[i]));
            d = add_rn(d, __shfl_xor_sync(0xffffffffu, d, 4));
            d = add_rn(d, __shfl_xor_sync(0xffffffffu, d, 2));
            d = add_rn(d, __shfl_xor_sync(0xffffffffu, d, 1));
            const float s = valid ? d * kAttnScale : -INFINITY;
            const float mnew = fmaxf(m[hh], s);
            const float alpha = (mnew == -INFINITY) ? 1.f : expf(m[hh] - mnew);
            const float e = (s == -INFINITY) ? 0.f : expf(s - mnew);
            l[hh] = l[hh] * alpha + e;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[hh][i] = acc[hh][i] * alpha + e * vf[i];
            m[hh] = mnew;
          }
        };
        // Loads one K row slice and V row slice (8 dims for this lane) for position pp.
        auto load = [&](bool valid, int pp, float* kf, float* vf) {
          if (!valid) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { kf[i] = 0.f; vf[i] = 0.f; }
            return;
          }
          if (pp == seqb - 1) {  // current token: rotated k and v from shared memory
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              kf[i] = k_s[(b * kKVH + kvh) * kD + 8 * j + i];
              vf[i] = v_s[(b * kKVH + kvh) * kD + 8 * j + i];
            }
            return;
          }
          const int blk = p.block_table[b * p.bt_stride + pp / p.block_size];
          const int off = pp % p.block_size;
          const long long e = static_cast<long long>(blk) * p.block_stride + off * p.token_stride +
                              kvh * p.head_stride + 8 * j;
          const uint4 ku = *reinterpret_cast<const uint4*>(kvw + p.k_off + e);
          const uint4 vu = *reinterpret_cast<const uint4*>(kvw + p.v_off + e);
          unpack8(ku, kf);
          unpack8(vu, vf);
        };

#if MK_ATTN_V2
        // Software pipeline: raw K/V of batch n+1 are requested before the math of batch n.
        // Each batch is 4 positions per lane group; the current token comes from shared memory.
        uint4 kc[4], vc[4], kn[4], vn[4];
        bool okc[4], curc[4], okn[4], curn[4];
        auto fetch = [&](int it, uint4* kk, uint4* vv, bool* okf, bool* curf) {
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int i = it + u;
            const int pp = start + g + 4 * i;
            const bool valid = (i < niter) && (pp < end);
            okf[u] = valid;
            curf[u] = valid && (pp == seqb - 1);
            kk[u] = make_uint4(0u, 0u, 0u, 0u);
            vv[u] = make_uint4(0u, 0u, 0u, 0u);
            if (valid && !curf[u]) {
#if MK_ATTN_FAST
              const int blk = p.block_table[b * p.bt_stride +
                                            (p.bs_shift >= 0 ? (pp >> p.bs_shift) : pp / p.block_size)];
              const int off = p.bs_shift >= 0 ? (pp & (p.block_size - 1)) : pp % p.block_size;
#else
              const int blk = p.block_table[b * p.bt_stride + pp / p.block_size];
              const int off = pp % p.block_size;
#endif
              const long long e = static_cast<long long>(blk) * p.block_stride + off * p.token_stride +
                                  kvh * p.head_stride + 8 * j;
              kk[u] = *reinterpret_cast<const uint4*>(kvw + p.k_off + e);
              vv[u] = *reinterpret_cast<const uint4*>(kvw + p.v_off + e);
            }
          }
        };
        auto compute = [&](const uint4* kk, const uint4* vv, const bool* okf, const bool* curf) {
          float kf[4][8], vf[4][8];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            if (curf[u]) {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                kf[u][i] = k_s[(b * kKVH + kvh) * kD + 8 * j + i];
                vf[u][i] = v_s[(b * kKVH + kvh) * kD + 8 * j + i];
              }
            } else {
              unpack8(kk[u], kf[u]);
              unpack8(vv[u], vf[u]);
            }
          }
          float sc[4][3];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int hh = 0; hh < 3; ++hh) {
              float d = 0.f;
#if MK_ATTN_FAST
#pragma unroll
              for (int i = 0; i < 8; ++i) d = fmaf(qr[hh][i], kf[u][i], d);
#else
#pragma unroll
              for (int i = 0; i < 8; ++i) d = add_rn(d, mul_rn(qr[hh][i], kf[u][i]));
#endif
              d = add_rn(d, __shfl_xor_sync(0xffffffffu, d, 4));
              d = add_rn(d, __shfl_xor_sync(0xffffffffu, d, 2));
              d = add_rn(d, __shfl_xor_sync(0xffffffffu, d, 1));
              sc[u][hh] = okf[u] ? d * kAttnScale : -INFINITY;
            }
          }
#pragma unroll
          for (int hh = 0; hh < 3; ++hh) {  // batched online softmax: one rescale per 4 positions
            float mx = m[hh];
#pragma unroll
            for (int u = 0; u < 4; ++u) mx = fmaxf(mx, sc[u][hh]);
#if MK_ATTN_FAST
            const float alpha = (mx == -INFINITY) ? 1.f : __expf(m[hh] - mx);
#else
            const float alpha = (mx == -INFINITY) ? 1.f : expf(m[hh] - mx);
#endif
            float ev[4];
            float esum = 0.f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#if MK_ATTN_FAST
              ev[u] = (sc[u][hh] == -INFINITY) ? 0.f : __expf(sc[u][hh] - mx);
#else
              ev[u] = (sc[u][hh] == -INFINITY) ? 0.f : expf(sc[u][hh] - mx);
#endif
              esum += ev[u];
            }
            l[hh] = l[hh] * alpha + esum;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              float av = acc[hh][i] * alpha;
#pragma unroll
              for (int u = 0; u < 4; ++u) av = fmaf(ev[u], vf[u][i], av);
              acc[hh][i] = av;
            }
            m[hh] = mx;
          }
        };
        if (niter > 0) fetch(0, kc, vc, okc, curc);
        for (int it = 0; it < niter; it += 4) {
          if (it + 4 < niter) fetch(it + 4, kn, vn, okn, curn);
          compute(kc, vc, okc, curc);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            kc[u] = kn[u];
            vc[u] = vn[u];
            okc[u] = okn[u];
            curc[u] = curn[u];
          }
        }
#else
        for (int it = 0; it < niter; it += 2) {
          const int pa = start + g + 4 * it;
          const int pb = pa + 4;
          const bool va = pa < end;
          const bool vb = (it + 1 < niter) && pb < end;
          float ka[8], va8[8], kb[8], vb8[8];
          load(va, pa, ka, va8);
          load(vb, pb, kb, vb8);
          update(va, ka, va8);
          update(vb, kb, vb8);
        }
#endif

        // Merge the 4 lane groups of this warp (xor 8 and 16) into one partial per head.
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
          float mm = m[hh];
          mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, 8));
          mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, 16));
          const float f = (m[hh] == -INFINITY) ? 0.f : expf(m[hh] - mm);
          float lt = l[hh] * f;
          lt += __shfl_xor_sync(0xffffffffu, lt, 8);
          lt += __shfl_xor_sync(0xffffffffu, lt, 16);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float a = acc[hh][i] * f;
            a += __shfl_xor_sync(0xffffffffu, a, 8);
            a += __shfl_xor_sync(0xffffffffu, a, 16);
            acc[hh][i] = a;
          }
          if (lane < 8) {
            if (lane == 0) {
              pm[item * 3 + hh] = mm;
              pl[item * 3 + hh] = lt;
            }
            const int dd = (item * 3 + hh) * kD + 8 * lane;
#pragma unroll
            for (int i = 0; i < 8; ++i) pacc[dd + i] = acc[hh][i];
          }
        }
      }
    }
#if MK_PREFETCH
    for (int k = 0; k < 8; ++k) {
      const int r = gw + k * kGlobalWarps;
      prefetch_row(L.gate_up + static_cast<size_t>(r) * kH, kH * 2);
      prefetch_row(L.gate_up + static_cast<size_t>(kI + r) * kH, kH * 2);
    }
#endif
    sync_at(layer * 5 + 1);  // B2

    // Phase 3a: merge partials per (request, query head): weights and denominators.
    if (t < B * kQH) {
      const int b = t / kQH, hq = t % kQH, kvh = hq / 3, hh = hq % 3;
      float M = -INFINITY;
      for (int c = 0; c < nc_s[b]; ++c) M = fmaxf(M, pm[((b * kKVH + kvh) * NC + c) * 3 + hh]);
      float den = 0.f;
      for (int c = 0; c < nc_s[b]; ++c) {
        const int item = (b * kKVH + kvh) * NC + c;
        const float w = expf(pm[item * 3 + hh] - M);
        wt_s[(b * kQH + hq) * NC + c] = w;
        den = add_rn(den, mul_rn(pl[item * 3 + hh], w));
      }
      inv_s[b * kQH + hq] = 1.f / den;
    }
    __syncthreads();
    // Phase 3b: attention vector (bf16 values) for all requests into smem.
    for (int idx = t; idx < B * kH; idx += kThreads) {
      const int b = idx / kH, r = idx % kH, hq = r / kD, d = r % kD;
      const int kvh = hq / 3, hh = hq % 3;
      float sum = 0.f;
      for (int c = 0; c < nc_s[b]; ++c) {
        const int item = (b * kKVH + kvh) * NC + c;
        sum = add_rn(sum, mul_rn(wt_s[(b * kQH + hq) * NC + c], pacc[(item * 3 + hh) * kD + d]));
      }
      vec_s[idx] = rnd(sum * inv_s[b * kQH + hq]);
    }
    __syncthreads();
    // Phase 3c: O projection (576 rows), residual h1 = x + o.
#if MK_GEMV_BATCH
    uint32_t wo[3][9];
#pragma unroll
    for (int k = 0; k < 3; ++k) load_row576(L.o + static_cast<size_t>(gw + k * kGlobalWarps) * kH, lane, wo[k]);
#endif
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const int r = gw + k * kGlobalWarps;
#if MK_GEMV_BATCH
      dot576_from<B>(wo[k], vec_s, lane, o);
#else
      dots<kH, B>(L.o + static_cast<size_t>(r) * kH, vec_s, lane, o);
#endif
      if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
          const bf16* xin = layer == 0 ? p.embed + static_cast<size_t>(tok[b]) * kH : ws + kOffX + b * kH;
          const float h = add_rn(bf2f(xin[r]), rnd(o[b]));
          ws[kOffH1 + b * kH + r] = __float2bfloat16_rn(h);
        }
      }
    }
#if MK_PREFETCH
    for (int k = 0; k < 3; ++k) prefetch_row(L.down + static_cast<size_t>(gw + k * kGlobalWarps) * kI, kI * 2);
#endif
    sync_at(layer * 5 + 2);  // B3

    // Phase 4: RMSNorm(h1) per request, gate/up row pairs (1536), SwiGLU.
#pragma unroll
    for (int b = 0; b < B; ++b) rms_norm_to_smem(ws + kOffH1 + b * kH, L.norm2, vec_s + b * kH, red);
#if MK_GEMV_BATCH
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t wg[2][9], wu[2][9];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int r = gw + (2 * kk + q) * kGlobalWarps;
        load_row576(L.gate_up + static_cast<size_t>(r) * kH, lane, wg[q]);
        load_row576(L.gate_up + static_cast<size_t>(kI + r) * kH, lane, wu[q]);
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int r = gw + (2 * kk + q) * kGlobalWarps;
        dot576_from<B>(wg[q], vec_s, lane, go);
        dot576_from<B>(wu[q], vec_s, lane, uo);
        if (lane == 0) {
#pragma unroll
          for (int b = 0; b < B; ++b) {
            const float g = rnd(go[b]);
            const float u = rnd(uo[b]);
            const float act = g / (1.f + expf(-g)) * u;
            ws[kOffAct + b * kI + r] = __float2bfloat16_rn(act);
          }
        }
      }
    }
#else
    for (int k = 0; k < 8; ++k) {
      const int r = gw + k * kGlobalWarps;
      dots<kH, B>(L.gate_up + static_cast<size_t>(r) * kH, vec_s, lane, go);
      dots<kH, B>(L.gate_up + static_cast<size_t>(kI + r) * kH, vec_s, lane, uo);
      if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
          const float g = rnd(go[b]);
          const float u = rnd(uo[b]);
          const float act = g / (1.f + expf(-g)) * u;
          ws[kOffAct + b * kI + r] = __float2bfloat16_rn(act);
        }
      }
    }
#endif
#if MK_PREFETCH
    if (layer + 1 < kLayers) {
      const LayerW Ln = p.layers[layer + 1];
      for (int k = 0; k < 5; ++k) prefetch_row(Ln.qkv + static_cast<size_t>(gw + k * kGlobalWarps) * kH, kH * 2);
    }
#endif
    sync_at(layer * 5 + 3);  // B4

    // Phase 5: down projection (576 rows over K=1536), residual x = h1 + down.
    for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = bf2f(ws[kOffAct + idx]);
    __syncthreads();
    for (int k = 0; k < 3; ++k) {
      const int r = gw + k * kGlobalWarps;
      dots<kI, B>(L.down + static_cast<size_t>(r) * kI, vec_s, lane, o);
      if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
          const float xv = add_rn(bf2f(ws[kOffH1 + b * kH + r]), rnd(o[b]));
          ws[kOffX + b * kH + r] = __float2bfloat16_rn(xv);
        }
      }
    }
    sync_at(layer * 5 + 4);  // B5
  }

  // Final RMSNorm per request, written by CTA 0 (no LM head; vLLM applies it).
#if MK_COUNTER_SYNC
  if (blockIdx.x == 0 && t == 0) ctrs[kBarriers] = epoch + 1u;  // every CTA read epoch at start
#endif
  if (blockIdx.x == 0) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      rms_norm_to_smem(ws + kOffX + b * kH, p.final_norm, vec_s + b * kH, red);
      for (int i = t; i < kH; i += kThreads) {
        p.out_hidden[b * kH + i] = __float2bfloat16_rn(vec_s[b * kH + i]);
      }
    }
  }
}

// ---------------------------------------------------------------- host API

template <int B>
static cudaError_t launch_b(const Params& p, cudaStream_t stream) {
  void* args[] = {const_cast<Params*>(&p)};
  return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(mk_kernel<B>), dim3(kCTAs),
                                     dim3(kThreads), args, 0, stream);
}

static int query_props(int* sm_count, int* coop, int* active, int* major, int* minor, int* regs,
                       int* smem) {
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return e;
  e = cudaDeviceGetAttribute(coop, cudaDevAttrCooperativeLaunch, device);
  if (e != cudaSuccess) return e;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(active, mk_kernel<kMaxB>, kThreads, 0);
  if (e != cudaSuccess) return e;
  cudaFuncAttributes fa{};
  e = cudaFuncGetAttributes(&fa, mk_kernel<kMaxB>);
  if (e != cudaSuccess) return e;
  *sm_count = prop.multiProcessorCount;
  *major = prop.major;
  *minor = prop.minor;
  *regs = fa.numRegs;
  *smem = static_cast<int>(fa.sharedSizeBytes);
  return cudaSuccess;
}

extern "C" int mk_query(int* sm_count, int* coop, int* active_per_sm, int* major, int* minor,
                        int* regs, int* static_smem) {
  return query_props(sm_count, coop, active_per_sm, major, minor, regs, static_smem);
}

extern "C" size_t mk_workspace_bytes(int B, int max_seq_len) {
  (void)B;
  (void)max_seq_len;  // chunking is fixed per B; the partial buffer does not grow with context
  return kWorkspaceBytes;
}

extern "C" int mk_decode(void* stream, int B, const void* input_ids, int ids_i64,
                         const void* positions, const void* seq_lens, const void* block_table,
                         int bt_stride, const void* slot_mapping, const void* layers,
                         const void* embed, const void* final_norm, void* const* kv_layers,
                         long long k_off, long long v_off, long long block_stride,
                         long long token_stride, long long head_stride, int block_size,
                         const void* cos_sin, int cs_fp32, void* out_hidden, void* scratch,
                         size_t scratch_bytes, int max_seq_len) {
  if (B < 1 || B > kMaxB) return kErrB;
  if (max_seq_len < 1 || max_seq_len > kMaxSeq) return kErrMaxSeq;
  if (!input_ids || !positions || !seq_lens || !block_table || !slot_mapping || !layers ||
      !embed || !final_norm || !kv_layers || !cos_sin || !out_hidden || !scratch) {
    return kErrNull;
  }
  if (scratch_bytes < kWorkspaceBytes) return kErrScratch;
  if (block_size <= 0) return kErrBlock;
  if (bt_stride < 1) return kErrBlock;
  // 16-byte vector loads of K/V rows need 8-element alignment of every offset and stride.
  if (k_off % 8 || v_off % 8 || block_stride % 8 || token_stride % 8 || head_stride % 8)
    return kErrAlign;
  if (k_off < 0 || v_off < 0 || block_stride < 0 || token_stride < 0 || head_stride < 0)
    return kErrAlign;
  if (reinterpret_cast<uintptr_t>(layers) % 16 || reinterpret_cast<uintptr_t>(embed) % 16 ||
      reinterpret_cast<uintptr_t>(final_norm) % 16 || reinterpret_cast<uintptr_t>(scratch) % 16)
    return kErrAlign;

  int sm = 0, coop = 0, active = 0, major = 0, minor = 0, regs = 0, smem = 0;
  int status = query_props(&sm, &coop, &active, &major, &minor, &regs, &smem);
  if (status != cudaSuccess) return status;
  if (major != 8 || minor != 9) return kErrSm89;
  if (!coop) return kErrCoop;
  if (active * sm < kCTAs) return kErrResidency;

  Params p{};
  p.layers = reinterpret_cast<const LayerW*>(layers);
  p.embed = reinterpret_cast<const bf16*>(embed);
  p.final_norm = reinterpret_cast<const bf16*>(final_norm);
  p.kv = kv_layers;
  p.input_ids = input_ids;
  p.ids_i64 = ids_i64 ? 1 : 0;
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.block_table = reinterpret_cast<const int*>(block_table);
  p.bt_stride = bt_stride;
  p.slot_mapping = reinterpret_cast<const int64_t*>(slot_mapping);
  p.cos_sin = cos_sin;
  p.cs_fp32 = cs_fp32 ? 1 : 0;
  p.out_hidden = reinterpret_cast<bf16*>(out_hidden);
  p.ws = reinterpret_cast<bf16*>(scratch);
  p.wsf = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch) + size_t(kScratchBf16) * 2);
  p.k_off = k_off;
  p.v_off = v_off;
  p.block_stride = block_stride;
  p.token_stride = token_stride;
  p.head_stride = head_stride;
  p.block_size = block_size;
  p.bs_shift = (block_size & (block_size - 1)) == 0 ? __builtin_ctz(static_cast<unsigned>(block_size)) : -1;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t e;
  switch (B) {
    case 1: e = launch_b<1>(p, s); break;
    case 2: e = launch_b<2>(p, s); break;
    case 3: e = launch_b<3>(p, s); break;
    default: e = launch_b<4>(p, s); break;
  }
  if (e != cudaSuccess) return e;
  return cudaPeekAtLastError();
}

extern "C" const char* mk_status_string(int status) {
  switch (status) {
    case kErrB: return "batch_must_be_1_to_4";
    case kErrMaxSeq: return "max_seq_len_must_be_1_to_8192";
    case kErrNull: return "null_device_pointer";
    case kErrCoop: return "cooperative_launch_unsupported";
    case kErrResidency: return "grid_exceeds_resident_capacity";
    case kErrSm89: return "kernel_requires_sm89";
    case kErrAlign: return "kv_offsets_or_pointers_not_16_byte_aligned";
    case kErrScratch: return "scratch_smaller_than_mk_workspace_bytes";
    case kErrBlock: return "block_size_or_stride_invalid";
    default: return cudaGetErrorString(static_cast<cudaError_t>(status));
  }
}
