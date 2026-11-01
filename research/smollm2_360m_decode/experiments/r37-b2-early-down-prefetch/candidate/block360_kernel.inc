// R28 kernel phases, included from block360.cu. Projections, norms, RoPE, KV write, barriers and the
// decode360 ABI are R27. Attention is replaced by a single split-KV pass plus a per-CTA log-sum-exp
// merge in the O-projection phase (R6 design, 360M shapes: 15 query heads, 5 KV heads, GQA group 3).
constexpr int kMaxChunks = kGlobalWarps / kKVH;  // 38: chunks per (request, kv head) at B=1

// Debug stage layout is unchanged from R27 so the harness offsets stay valid. R28 does not write the
// QK, scaled-score and probability stages (no materialized scores exist), so those regions are left as
// the caller provided them.
constexpr int kDbgNorm1 = 0;
constexpr int kDbgQKV = kDbgNorm1 + kMaxB * kH;
constexpr int kDbgAttn = kDbgQKV + kMaxB * 1600;
constexpr int kDbgO = kDbgAttn + kMaxB * kH;
constexpr int kDbgH1 = kDbgO + kMaxB * kH;
constexpr int kDbgNorm2 = kDbgH1 + kMaxB * kH;
constexpr int kDbgGate = kDbgNorm2 + kMaxB * kH;
constexpr int kDbgUp = kDbgGate + kMaxB * kI;
constexpr int kDbgSilu = kDbgUp + kMaxB * kI;
constexpr int kDbgAct = kDbgSilu + kMaxB * kI;
constexpr int kDbgDown = kDbgAct + kMaxB * kI;
constexpr int kDbgOut = kDbgDown + kMaxB * kH;
constexpr int kDebugLayerBf16 = kDbgOut + kMaxB * kH;
constexpr size_t kDebugStagesBf16 = size_t(kLayers) * kDebugLayerBf16;
constexpr size_t kDbgAttnQ = kDebugStagesBf16;
constexpr size_t kDbgAttnQK = kDbgAttnQ + size_t(kMaxB) * kH;
constexpr size_t kDbgAttnScores = kDbgAttnQK + size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kDbgAttnProbs = kDbgAttnScores + size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kDbgAttnPv = kDbgAttnProbs + size_t(kMaxB) * kQH * kMaxSeq;
constexpr size_t kDebugBf16 = kDbgAttnPv + size_t(kMaxB) * kH;

// R28 uses only static shared memory (no wmma staging).
template <int B>
constexpr size_t dynamic_smem_bytes() {
  return 0;
}

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
  f[0] = lo_bf16(v.x); f[1] = hi_bf16(v.x);
  f[2] = lo_bf16(v.y); f[3] = hi_bf16(v.y);
  f[4] = lo_bf16(v.z); f[5] = hi_bf16(v.z);
  f[6] = lo_bf16(v.w); f[7] = hi_bf16(v.w);
}

// One warp = one (request, kv head, chunk) item over cache positions [start, end).
// Lane group g = lane >> 3 takes positions start + g + 4i; lane j = lane & 7 owns dims 8j..8j+7, so a
// K or V row of this kv head (128 B) is read by one 8-lane group with 16-byte loads. Each position gives
// three query-head dots (reduced over the 8 lanes), then a batched FP32 online softmax with one rescale
// per 4 positions. The next batch's K/V loads are issued before the current batch's math.
// Outputs: m and l per head, and the unnormalized P.V accumulator (fp32), all in global scratch.
__device__ __forceinline__ void split_kv_item(const Params& p, const bf16* kvw, const float* q_s, int b, int kvh,
                                              int start, int end, int lane, float* pm_item, float* pl_item,
                                              float* pacc_item) {
  constexpr unsigned kFull = 0xffffffffu;
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
  const int niter = (end - start + 3) / 4;  // 4-position steps per lane group, uniform over the warp
  uint4 kc[4], vc[4], kn[4], vn[4];
  bool okc[4], okn[4];
  auto fetch = [&](int it, uint4* kk, uint4* vv, bool* okf) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = it + u;
      const int pp = start + g + 4 * i;
      const bool valid = i < niter && pp < end;
      okf[u] = valid;
      kk[u] = make_uint4(0u, 0u, 0u, 0u);
      vv[u] = make_uint4(0u, 0u, 0u, 0u);
      if (valid) {
        const int blk = p.block_table[b * p.bt_stride + pp / p.block_size];
        const long long e = static_cast<long long>(blk) * p.block_stride + (pp % p.block_size) * p.token_stride +
                            kvh * p.head_stride + 8 * j;
        kk[u] = *reinterpret_cast<const uint4*>(kvw + p.k_off + e);
        vv[u] = *reinterpret_cast<const uint4*>(kvw + p.v_off + e);
      }
    }
  };
  auto compute = [&](const uint4* kk, const uint4* vv, const bool* okf) {
    float sc[4][3];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      float kf[8];
      unpack8(kk[u], kf);
#pragma unroll
      for (int hh = 0; hh < 3; ++hh) {
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) d = fmaf(qr[hh][i], kf[i], d);
        d += __shfl_xor_sync(kFull, d, 4);
        d += __shfl_xor_sync(kFull, d, 2);
        d += __shfl_xor_sync(kFull, d, 1);
        // BF16 rounding of the dot as in the native matmul output; the 0.125 scale is exact.
        sc[u][hh] = okf[u] ? rnd(d) * kAttnScale : -INFINITY;
      }
    }
    float ev[4][3], alpha[3];
#pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
      float mx = m[hh];
#pragma unroll
      for (int u = 0; u < 4; ++u) mx = fmaxf(mx, sc[u][hh]);
      alpha[hh] = (mx == -INFINITY) ? 1.f : expf(m[hh] - mx);
      float es = 0.f;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        ev[u][hh] = (sc[u][hh] == -INFINITY) ? 0.f : expf(sc[u][hh] - mx);
        es += ev[u][hh];
      }
      l[hh] = l[hh] * alpha[hh] + es;
      m[hh] = mx;
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[hh][i] *= alpha[hh];
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      float vf[8];
      unpack8(vv[u], vf);
#pragma unroll
      for (int hh = 0; hh < 3; ++hh)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[hh][i] = fmaf(ev[u][hh], vf[i], acc[hh][i]);
    }
  };
  if (niter > 0) fetch(0, kc, vc, okc);
  for (int it = 0; it < niter; it += 4) {
    if (it + 4 < niter) fetch(it + 4, kn, vn, okn);
    compute(kc, vc, okc);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      kc[u] = kn[u];
      vc[u] = vn[u];
      okc[u] = okn[u];
    }
  }
  // Combine the 4 lane groups of this warp (xor 8, then 16) into one partial per head.
#pragma unroll
  for (int hh = 0; hh < 3; ++hh) {
    float mm = fmaxf(m[hh], __shfl_xor_sync(kFull, m[hh], 8));
    mm = fmaxf(mm, __shfl_xor_sync(kFull, mm, 16));
    const float f = (m[hh] == -INFINITY) ? 0.f : expf(m[hh] - mm);
    float lt = l[hh] * f;
    lt += __shfl_xor_sync(kFull, lt, 8);
    lt += __shfl_xor_sync(kFull, lt, 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float a = acc[hh][i] * f;
      a += __shfl_xor_sync(kFull, a, 8);
      a += __shfl_xor_sync(kFull, a, 16);
      acc[hh][i] = a;
    }
    if (lane < 8) {
      if (lane == 0) {
        pm_item[hh] = mm;
        pl_item[hh] = lt;
      }
      float4* dst = reinterpret_cast<float4*>(pacc_item + hh * kD + 8 * lane);
      dst[0] = make_float4(acc[hh][0], acc[hh][1], acc[hh][2], acc[hh][3]);
      dst[1] = make_float4(acc[hh][4], acc[hh][5], acc[hh][6], acc[hh][7]);
    }
  }
}

template <int B>
__global__ void __launch_bounds__(kThreads, 1) mk_kernel(const Params p) {
  static_assert(B == 1 || B == 2, "the block prototype covers only B=1 and B=2");
  constexpr int NC = kGlobalWarps / (kKVH * B);  // chunks per (request, kv head): 38 at B=1, 19 at B=2
  constexpr int NI = kKVH * B * NC;              // attention items (warps), <= 192
  static_assert(NI <= kGlobalWarps && NC <= kMaxChunks, "attention items exceed warps");

  cg::grid_group grid = cg::this_grid();
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int gw = B == 2 ? warp * kCTAs + blockIdx.x : blockIdx.x * kWarps + warp;

  __shared__ __align__(16) float vec_s[kMaxB * kI];  // normed / attention / activation vectors per request
  __shared__ __align__(16) float q_s[B * kQH * kD];
  __shared__ __align__(16) float k_s[B * kKVH * kD];
  __shared__ __align__(16) float v_s[B * kKVH * kD];
  __shared__ float ms_s[kMaxB * kQH * kMaxChunks];   // merge: running max per (row, chunk), then weights
  __shared__ float ls_s[kMaxB * kQH * kMaxChunks];   // merge: chunk denominators
  __shared__ float inv_s[kMaxB * kQH];               // 1 / merged denominator
  __shared__ float red[128];
  __shared__ int seq_s[B], pos_s[B], nc_s[B];

  bf16* ws = p.ws;
  float* pm = reinterpret_cast<float*>(ws + kScratchBf16) + kFOffPM;
  float* pl = reinterpret_cast<float*>(ws + kScratchBf16) + kFOffPL;
  float* pacc = reinterpret_cast<float*>(ws + kScratchBf16) + kFOffPACC;
  auto sync_grid = [&]() { grid.sync(); };

  if (t < B) {
    seq_s[t] = max(static_cast<int>(p.seq_lens[t]), 1);
    pos_s[t] = max(static_cast<int>(p.positions[t]), 0);
    nc_s[t] = min(NC, (seq_s[t] + 15) / 16);  // chunks of >= 16 positions, adaptive per request
  }
  __syncthreads();

  for (int layer_index = 0; layer_index < kLayers; ++layer_index) {
  bf16* debug_layer = p.debug ? p.debug + static_cast<size_t>(layer_index) * kDebugLayerBf16 : nullptr;
  const LayerW L = p.layers[layer_index];
  bf16* kvw = reinterpret_cast<bf16*>(p.kv[layer_index]);
  const bf16* hidden = p.out_hidden;
  for (int b = 0; b < B; ++b) {
    rms_norm_to_smem((layer_index == 0 ? p.embed + static_cast<size_t>(p.input_ids[b]) * kH : hidden + b * kH), L.norm1, vec_s + b * kH, red);
  }
  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgNorm1 + i] = __float2bfloat16_rn(vec_s[i]);
  }

  // QKV output is packed [Q(960), K(320), V(320)] per request.
  // R27 FAST projection: one output row per warp task, rows gw + k * kGlobalWarps.
  for (int r = gw; r < 1600; r += kGlobalWarps) {
    const bf16* rows[1] = {L.qkv + static_cast<size_t>(r) * kH};
    float res[1][B];
    warp_gemv_rows<kH, B, 1>(rows, vec_s, lane, res);
    if (lane == 0) {
      for (int b = 0; b < B; ++b) ws[kOffQKV + b * 1600 + r] = __float2bfloat16_rn(res[0][b]);
    }
  }
  for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kH) prefetch_row(L.o + static_cast<size_t>(r) * kH, kH * sizeof(bf16));
  }
  sync_grid();
  if (p.debug) {
    const int gi = blockIdx.x * kThreads + t;
    const int stride = kCTAs * kThreads;
    for (int i = gi; i < B * 1600; i += stride)
      debug_layer[kDbgQKV + i] = ws[kOffQKV + i];
  }

  // Rotate Q/K and retain V in shared memory. CTA 0 appends the current K/V cache entry.
  for (int idx = t; idx < B * kQH * kD; idx += kThreads) {
    const int b = idx / (kQH * kD), rem = idx % (kQH * kD);
    const int head = rem / kD, d = rem % kD;
    q_s[idx] = rope_val(ws + kOffQKV + b * 1600 + head * kD, d, pos_s[b], p.cos_sin, p.cs_fp32);
  }
  for (int idx = t; idx < B * kKVH * kD; idx += kThreads) {
    const int b = idx / (kKVH * kD), rem = idx % (kKVH * kD);
    const int head = rem / kD, d = rem % kD;
    const int qkv_k = 960 + head * kD;
    const int qkv_v = 1280 + head * kD;
    k_s[idx] = rope_val(ws + kOffQKV + b * 1600 + qkv_k, d, pos_s[b], p.cos_sin, p.cs_fp32);
    v_s[idx] = bf2f(ws[kOffQKV + b * 1600 + qkv_v + d]);
  }
  __syncthreads();
  if (p.debug && layer_index == p.debug_layer_index && blockIdx.x == 0) {
    bf16* attn_debug = p.debug + kDebugStagesBf16;
    for (int idx = t; idx < B * kH; idx += kThreads)
      attn_debug[kDbgAttnQ - kDebugStagesBf16 + idx] = __float2bfloat16_rn(q_s[idx]);
  }
  if (blockIdx.x == 0) {
    for (int idx = t; idx < B * kKVH * kD; idx += kThreads) {
      const int b = idx / (kKVH * kD), rem = idx % (kKVH * kD);
      const int head = rem / kD, d = rem % kD;
      const int64_t slot = p.slot_mapping[b];
      if (slot < 0) continue;
      const long long blk = slot / p.block_size, off = slot % p.block_size;
      const long long e = blk * p.block_stride + off * p.token_stride + head * p.head_stride + d;
      kvw[p.k_off + e] = __float2bfloat16_rn(k_s[idx]);
      kvw[p.v_off + e] = __float2bfloat16_rn(v_s[idx]);
    }
  }

  // Make CTA 0's current K/V cache write visible before attention reads it.
  sync_grid();

  // Split-KV partials: one warp per (request, kv head, chunk). Each K/V row is read once and used by
  // the three query heads of its GQA group. No grid barrier is needed inside this phase.
  if (gw < NI) {
    const int c = gw % NC, bk = gw / NC, kvh = bk % kKVH, b = bk / kKVH;
    const int ncb = nc_s[b];
    if (c < ncb) {
      const int seqb = seq_s[b];
      const int Lc = (seqb + ncb - 1) / ncb;
      const int start = c * Lc;
      const int end = min(start + Lc, seqb);
      const int item = (b * kKVH + kvh) * NC + c;
      split_kv_item(p, kvw, q_s, b, kvh, start, end, lane, pm + item * 3, pl + item * 3, pacc + item * 3 * kD);
    }
  }
  sync_grid();

  // Merge the chunk partials per (request, query head) with log-sum-exp. Every CTA merges the same
  // data, so no further barrier is needed before the O projection. The merged vector is rounded to
  // BF16, as the R27 attention output was.
  for (int idx = t; idx < B * kQH * NC; idx += kThreads) {
    const int row = idx / NC, c = idx - row * NC;
    const int b = row / kQH, hq = row - b * kQH;
    if (c < nc_s[b]) {
      const int item = (b * kKVH + hq / 3) * NC + c;
      ms_s[idx] = pm[item * 3 + hq % 3];
      ls_s[idx] = pl[item * 3 + hq % 3];
    } else {
      ms_s[idx] = -INFINITY;
      ls_s[idx] = 0.f;
    }
  }
  __syncthreads();
  if (t < B * kQH) {
    const int n = nc_s[t / kQH];
    float M = -INFINITY;
    for (int c = 0; c < n; ++c) M = fmaxf(M, ms_s[t * NC + c]);
    float den = 0.f;
    for (int c = 0; c < n; ++c) {
      const float w = expf(ms_s[t * NC + c] - M);
      ms_s[t * NC + c] = w;
      den = add_rn(den, mul_rn(ls_s[t * NC + c], w));
    }
    inv_s[t] = 1.f / den;
  }
  __syncthreads();
  for (int job = t; job < B * kQH * (kD / 4); job += kThreads) {
    const int row = job / (kD / 4), dq = job % (kD / 4);
    const int b = row / kQH, hq = row % kQH;
    const int kvh = hq / 3, hh = hq % 3;
    const int n = nc_s[b];
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int c = 0; c < n; ++c) {
      const float w = ms_s[row * NC + c];
      const int item = (b * kKVH + kvh) * NC + c;
      const float4 v = reinterpret_cast<const float4*>(pacc + (item * 3 + hh) * kD)[dq];
      a.x = fmaf(w, v.x, a.x);
      a.y = fmaf(w, v.y, a.y);
      a.z = fmaf(w, v.z, a.z);
      a.w = fmaf(w, v.w, a.w);
    }
    const float inv = inv_s[row];
    float* dst = vec_s + b * kH + hq * kD + 4 * dq;
    dst[0] = rnd(a.x * inv);
    dst[1] = rnd(a.y * inv);
    dst[2] = rnd(a.z * inv);
    dst[3] = rnd(a.w * inv);
  }
  __syncthreads();
  if (p.debug && layer_index == p.debug_layer_index && blockIdx.x == 0) {
    bf16* attn_debug = p.debug + kDebugStagesBf16;
    for (int idx = t; idx < B * kH; idx += kThreads)
      attn_debug[kDbgAttnPv - kDebugStagesBf16 + idx] = __float2bfloat16_rn(vec_s[idx]);
  }
  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgAttn + i] = __float2bfloat16_rn(vec_s[i]);
  }

  // O projection plus the first residual. The projection result is rounded to BF16 before the
  // FP32 residual add, as in R25.
  for (int r = gw; r < kH; r += kGlobalWarps) {
    const bf16* rows[1] = {L.o + static_cast<size_t>(r) * kH};
    float res[1][B];
    warp_gemv_rows<kH, B, 1>(rows, vec_s, lane, res);
    if (lane == 0) {
      for (int b = 0; b < B; ++b) {
        const float out = rnd(res[0][b]);
        const float h = add_rn(bf2f(layer_index == 0 ? p.embed[static_cast<size_t>(p.input_ids[b]) * kH + r]
                                                     : hidden[b * kH + r]), out);
        if (p.debug) debug_layer[kDbgO + b * kH + r] = __float2bfloat16_rn(out);
        ws[kOffH1 + b * kH + r] = __float2bfloat16_rn(h);
        if (p.debug) debug_layer[kDbgH1 + b * kH + r] = ws[kOffH1 + b * kH + r];
      }
    }
  }
  for (int k = 0; k < (kI + kGlobalWarps - 1) / kGlobalWarps; ++k) {
    const int r = gw + k * kGlobalWarps;
    if (r < kI) {
      prefetch_row(L.gate_up + static_cast<size_t>(r) * kH, kH * sizeof(bf16));
      prefetch_row(L.gate_up + static_cast<size_t>(kI + r) * kH, kH * sizeof(bf16));
    }
  }
  sync_grid();

  for (int b = 0; b < B; ++b)
    rms_norm_to_smem(ws + kOffH1 + b * kH, L.norm2, vec_s + b * kH, red);
  if (p.debug && blockIdx.x == 0) {
    for (int i = t; i < B * kH; i += kThreads)
      debug_layer[kDbgNorm2 + i] = __float2bfloat16_rn(vec_s[i]);
  }

  if constexpr (B == 2) {
    for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < kH) prefetch_row(L.down + static_cast<size_t>(r) * kI, kI * sizeof(bf16));
    }
  }

  // R27 FAST gate/up: warp task r reads gate row r and up row kI + r together, so SwiGLU is
  // formed in-warp and no gate scratch is staged. Tasks are r = gw + k * kGlobalWarps for r < kI.
  for (int r = gw; r < kI; r += kGlobalWarps) {
    const bf16* rows[2] = {L.gate_up + static_cast<size_t>(r) * kH,
                           L.gate_up + static_cast<size_t>(kI + r) * kH};
    float res[2][B];
    warp_gemv_rows<kH, B, 2>(rows, vec_s, lane, res);
    if (lane == 0) {
      for (int b = 0; b < B; ++b) {
        const float g = rnd(res[0][b]), u = rnd(res[1][b]);
        const float silu = rnd(g / (1.f + expf(-g)));
        if (p.debug) {
          debug_layer[kDbgGate + b * kI + r] = __float2bfloat16_rn(g);
          debug_layer[kDbgUp + b * kI + r] = __float2bfloat16_rn(u);
          debug_layer[kDbgSilu + b * kI + r] = __float2bfloat16_rn(silu);
        }
        ws[kOffAct + b * kI + r] = __float2bfloat16_rn(mul_rn(silu, u));
        if (p.debug) debug_layer[kDbgAct + b * kI + r] = ws[kOffAct + b * kI + r];
      }
    }
  }
  if constexpr (B != 2) {
    for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < kH) prefetch_row(L.down + static_cast<size_t>(r) * kI, kI * sizeof(bf16));
    }
  }
  sync_grid();

  for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = bf2f(ws[kOffAct + idx]);
  __syncthreads();
  // R27 FAST down projection: one output row per warp task, rows gw + k * kGlobalWarps.
  for (int r = gw; r < kH; r += kGlobalWarps) {
    const bf16* rows[1] = {L.down + static_cast<size_t>(r) * kI};
    float res[1][B];
    warp_gemv_rows<kI, B, 1>(rows, vec_s, lane, res);
    if (lane == 0) {
      for (int b = 0; b < B; ++b) {
        const float out = rnd(res[0][b]);
        if (p.debug) debug_layer[kDbgDown + b * kH + r] = __float2bfloat16_rn(out);
        const float y = add_rn(bf2f(ws[kOffH1 + b * kH + r]), out);
        p.out_hidden[b * kH + r] = __float2bfloat16_rn(y);
        if (p.debug) debug_layer[kDbgOut + b * kH + r] = p.out_hidden[b * kH + r];
      }
    }
  }
  if (layer_index + 1 < kLayers) {
    const LayerW Ln = p.layers[layer_index + 1];
    for (int k = 0; k < (1600 + kGlobalWarps - 1) / kGlobalWarps; ++k) {
      const int r = gw + k * kGlobalWarps;
      if (r < 1600) prefetch_row(Ln.qkv + static_cast<size_t>(r) * kH, kH * sizeof(bf16));
    }
  }
  sync_grid(); // layer output is complete before the next layer reads it
  if (p.layer_outputs) {
    for (int i = t; i < B * kH; i += kThreads)
      p.layer_outputs[(static_cast<size_t>(layer_index) * B * kH) + i] = p.out_hidden[i];
    sync_grid();
  }
  }
  if (blockIdx.x == 0) {
    for (int b = 0; b < B; ++b) {
      rms_norm_to_smem(p.out_hidden + b * kH, p.final_norm, vec_s + b * kH, red);
      for (int i = t; i < kH; i += kThreads) p.out_hidden[b * kH + i] = __float2bfloat16_rn(vec_s[b * kH + i]);
    }
  }
}

template <int B>
static int query_props(int* sm_count, int* coop, int* active, int* major, int* minor,
                       int* regs, int* static_smem, int* optin_smem) {
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return e;
  e = cudaDeviceGetAttribute(coop, cudaDevAttrCooperativeLaunch, device);
  if (e != cudaSuccess) return e;
  e = cudaDeviceGetAttribute(optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (e != cudaSuccess) return e;

  constexpr int dynamic_bytes = static_cast<int>(dynamic_smem_bytes<B>());
  e = cudaFuncSetAttribute(mk_kernel<B>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
  if (e != cudaSuccess) return e;
  cudaFuncAttributes fa{};
  e = cudaFuncGetAttributes(&fa, mk_kernel<B>);
  if (e != cudaSuccess) return e;
  if (fa.sharedSizeBytes + dynamic_bytes > *optin_smem) return kErrResidency;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(active, mk_kernel<B>, kThreads, dynamic_bytes);
  if (e != cudaSuccess) return e;
  *sm_count = prop.multiProcessorCount;
  *major = prop.major;
  *minor = prop.minor;
  *regs = fa.numRegs;
  *static_smem = static_cast<int>(fa.sharedSizeBytes);
  return cudaSuccess;
}

extern "C" size_t decode360_workspace_bytes() { return kWorkspaceBytes; }
extern "C" size_t decode360_debug_bytes() { return size_t(kDebugBf16) * sizeof(bf16); }

extern "C" int decode360(void* stream, int B, const void* input_ids, const void* embed, const void* final_norm,
                                const void* positions, const void* seq_lens,
                                const void* block_table, int bt_stride,
                                const void* slot_mapping, const void* layers,
                                void* const* kv, long long k_off, long long v_off,
                                long long block_stride, long long token_stride,
                                long long head_stride, int block_size,
                                const void* cos_sin, int cs_fp32, void* out_hidden, void* layer_outputs,
                                void* scratch, size_t scratch_bytes,
                                void* debug, size_t debug_bytes, int max_seq_len,
                                int debug_layer_index) {
  if (B < 1 || B > 2) return kErrB;
  if (max_seq_len < 1 || max_seq_len > kMaxSeq) return kErrMaxSeq;
  if (debug_layer_index < -1 || debug_layer_index >= kLayers) return kErrBlock;
  if (!input_ids || !embed || !final_norm || !positions || !seq_lens || !block_table || !slot_mapping || !layers ||
      !kv || !cos_sin || !out_hidden || !scratch) return kErrNull;
  if (scratch_bytes < kWorkspaceBytes) return kErrScratch;
  if (debug && debug_bytes < decode360_debug_bytes()) return kErrScratch;
  if (block_size != 16 || bt_stride < 1 ||
      static_cast<long long>(bt_stride) * block_size < max_seq_len) return kErrBlock;
  if (k_off % 8 || v_off % 8 || block_stride % 8 || token_stride % 8 || head_stride % 8 ||
      k_off < 0 || v_off < 0 || block_stride < 0 || token_stride < 0 || head_stride < 0) return kErrAlign;
  if (reinterpret_cast<uintptr_t>(embed) % 16 || reinterpret_cast<uintptr_t>(final_norm) % 16 ||
      reinterpret_cast<uintptr_t>(layers) % 16 || reinterpret_cast<uintptr_t>(kv) % 16 || reinterpret_cast<uintptr_t>(out_hidden) % 16 ||
      (layer_outputs && reinterpret_cast<uintptr_t>(layer_outputs) % 16) ||
      reinterpret_cast<uintptr_t>(scratch) % 16 || (debug && reinterpret_cast<uintptr_t>(debug) % 16))
    return kErrAlign;

  int sm = 0, coop = 0, active = 0, major = 0, minor = 0, regs = 0, static_smem = 0, optin_smem = 0;
  int status = B == 1 ? query_props<1>(&sm, &coop, &active, &major, &minor, &regs, &static_smem, &optin_smem)
                      : query_props<2>(&sm, &coop, &active, &major, &minor, &regs, &static_smem, &optin_smem);
  if (status != cudaSuccess) return status;
  if (major != 8 || minor != 9) return kErrSm89;
  if (!coop) return kErrCoop;
  if (active * sm < kCTAs) return kErrResidency;

  Params p{};
  p.layers = reinterpret_cast<const LayerW*>(layers);
  p.embed = reinterpret_cast<const bf16*>(embed);
  p.final_norm = reinterpret_cast<const bf16*>(final_norm);
  p.input_ids = reinterpret_cast<const int32_t*>(input_ids);
  p.kv = reinterpret_cast<void* const*>(kv);
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.block_table = reinterpret_cast<const int*>(block_table);
  p.bt_stride = bt_stride;
  p.slot_mapping = reinterpret_cast<const int64_t*>(slot_mapping);
  p.cos_sin = cos_sin;
  p.cs_fp32 = cs_fp32 ? 1 : 0;
  p.out_hidden = reinterpret_cast<bf16*>(out_hidden);
  p.layer_outputs = reinterpret_cast<bf16*>(layer_outputs);
  p.debug = reinterpret_cast<bf16*>(debug);
  p.ws = reinterpret_cast<bf16*>(scratch);
  p.k_off = k_off; p.v_off = v_off; p.block_stride = block_stride;
  p.token_stride = token_stride; p.head_stride = head_stride;
  p.block_size = block_size;
  p.max_seq_len = max_seq_len;
  p.debug_layer_index = debug_layer_index;
  void* args[] = {&p};
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t e;
  if (B == 1) {
    e = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(mk_kernel<1>), dim3(kCTAs),
                                    dim3(kThreads), args, dynamic_smem_bytes<1>(), s);
  } else {
    e = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(mk_kernel<2>), dim3(kCTAs),
                                    dim3(kThreads), args, dynamic_smem_bytes<2>(), s);
  }
  if (e != cudaSuccess) return e;
  return cudaPeekAtLastError();
}

extern "C" const char* decode360_status_string(int status) {
  switch (status) {
    case kErrB: return "batch_must_be_1_or_2";
    case kErrMaxSeq: return "max_seq_len_must_be_1_to_8192";
    case kErrNull: return "null_device_pointer";
    case kErrCoop: return "cooperative_launch_unsupported";
    case kErrResidency: return "shared_memory_or_residency_limit";
    case kErrSm89: return "kernel_requires_sm89";
    case kErrAlign: return "kv_offsets_or_pointers_not_16_byte_aligned";
    case kErrScratch: return "scratch_or_debug_buffer_too_small";
    case kErrBlock: return "block_size_or_stride_invalid";
    default: return cudaGetErrorString(static_cast<cudaError_t>(status));
  }
}
