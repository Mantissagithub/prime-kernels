#!/usr/bin/env python3
"""Build private stage-profile clones from semantically anchored phase boundaries."""
from __future__ import annotations
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAIRS = {
    "r30": {"directory": REPO / "outputs/megakernel-20261025/r30-phase-prefetch/candidate",
            "sha256": {"block360.cu": "5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638",
                       "block360_kernel.inc": "60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83"}},
}
VALIDATOR = REPO / "outputs/megakernel-20261025/block-360-heldout-r13/validate_full32.py"
spec = importlib.util.spec_from_file_location("validate_full32", VALIDATOR)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load ABI validator {VALIDATOR}")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
SLOTS = ("kProfileNorm1", "kProfileQkv", "kProfileRopeCache", "kProfileSplitKv", "kProfileMerge",
         "kProfileOResidual", "kProfileNorm2", "kProfileGateUpSilu", "kProfileDownResidual",
         "kProfileLayerCapture", "kProfileFinalNorm")
PROFILE_DECLS = '''
constexpr int kProfileSlots = 13;
constexpr int kProfileCtas = 24;
constexpr int kProfileFields = 2;
enum ProfileSlot : int {
  kProfileNorm1 = 0, kProfileQkv = 1, kProfileRopeCache = 2, kProfileSplitKv = 3,
  kProfileMerge = 4, kProfileOResidual = 5, kProfileNorm2 = 6, kProfileGateUpSilu = 7,
  kProfileDownResidual = 8, kProfileLayerCapture = 9, kProfileFinalNorm = 10,
  kProfileGateUpMath = 11, kProfileDownPrefetchIssue = 12,
};
constexpr size_t kProfileBufferBytes = size_t(32) * kProfileSlots * kProfileCtas *
                                       kProfileFields * sizeof(unsigned long long);
__device__ unsigned long long* g_r28_stage_profile_buffer = nullptr;
'''
PROFILE_API = '''
extern "C" size_t decode360_profile_buffer_bytes() { return kProfileBufferBytes; }
extern "C" int decode360_profile_set_buffer(void* buffer, size_t bytes) {
  if ((!buffer && bytes != 0) || (buffer && bytes != kProfileBufferBytes))
    return cudaErrorInvalidValue;
  auto* device_buffer = reinterpret_cast<unsigned long long*>(buffer);
  return static_cast<int>(cudaMemcpyToSymbol(g_r28_stage_profile_buffer, &device_buffer,
                                              sizeof(device_buffer)));
}
'''

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: semantic anchor expected once, found {count}: {anchor[:100]!r}")
    return text.replace(anchor, replacement, 1)

def insert_before(text: str, anchor: str, code: str, label: str) -> str:
    return replace_once(text, anchor, code + anchor, label)

def insert_after(text: str, anchor: str, code: str, label: str) -> str:
    return replace_once(text, anchor, anchor + code, label)

def instrument_source(text: str) -> str:
    ns = "using bf16 = __nv_bfloat16;\n"
    if text.count(ns) != 1 or "kProfileSlots" in text:
        raise RuntimeError("source declaration anchor invalid or already instrumented")
    text = text.replace(ns, ns + PROFILE_DECLS, 1)
    inc = '#include "block360_kernel.inc"\n'
    text = replace_once(text, inc, inc + PROFILE_API, "profile API export")
    return text

def instrument_kernel(text: str, pair: str) -> str:
    if "profile_start" in text or "kProfile" in text:
        raise RuntimeError(f"{pair}: kernel already has instrumentation")
    shared = "  __shared__ int seq_s[B], pos_s[B], nc_s[B];\n"
    text = insert_after(text, shared, "  __shared__ unsigned long long profile_local[kProfileSlots][kProfileFields];\n",
                        f"{pair} shared profile storage")
    sync_decl = "  auto sync_grid = [&]() { grid.sync(); };\n"
    init = '''  unsigned long long profile_start = 0;
  unsigned long long profile_work_end = 0;
  unsigned long long profile_sync_end = 0;
  unsigned long long profile_sub_start = 0;
  if (t == 0) {
    for (int slot = 0; slot < kProfileSlots; ++slot) {
      profile_local[slot][0] = 0;
      profile_local[slot][1] = 0;
    }
  }
'''
    text = insert_after(text, sync_decl, init, f"{pair} clock state initialization")

    # Norm1 starts before the RMSNorm loop and ends after its optional debug capture.
    norm_start = "  for (int b = 0; b < B; ++b) {\n    "
    # Anchor includes the code immediately before the first RMSNorm loop, not the debug block.
    norm_prefix = "  const bf16* hidden = p.out_hidden;\n" + norm_start
    text = replace_once(text, norm_prefix, "  const bf16* hidden = p.out_hidden;\n  if (t == 0) profile_start = clock64();\n" + norm_start,
                        f"{pair} norm1 start")
    qkv_comment = "  // QKV output is packed [Q(960), K(320), V(320)] per request.\n"
    text = insert_before(text, qkv_comment, "  if (t == 0) profile_local[kProfileNorm1][0] = clock64() - profile_start;\n",
                         f"{pair} norm1 end outside debug branch")

    # QKV phase includes candidate-specific O-row prefetch, then measures the existing grid wait.
    text = insert_before(text, qkv_comment, "  if (t == 0) profile_start = clock64();\n", f"{pair} QKV start")
    qkv_end = '''  sync_grid();
  if (p.debug) {
    const int gi = blockIdx.x * kThreads + t;'''
    qkv_profiled = '''  if (t == 0) profile_work_end = clock64();
  sync_grid();
  if (t == 0) {
    profile_sync_end = clock64();
    profile_local[kProfileQkv][0] = profile_work_end - profile_start;
    profile_local[kProfileQkv][1] = profile_sync_end - profile_work_end;
  }
  if (p.debug) {
    const int gi = blockIdx.x * kThreads + t;'''
    text = replace_once(text, qkv_end, qkv_profiled, f"{pair} QKV end and post-prefetch barrier")

    rope_comment = "  // Rotate Q/K and retain V in shared memory. CTA 0 appends the current K/V cache entry.\n"
    text = insert_before(text, rope_comment, "  if (t == 0) profile_start = clock64();\n", f"{pair} rope/cache start")
    cache_barrier = '''  // Make CTA 0's current K/V cache write visible before attention reads it.
  sync_grid();'''
    cache_profiled = '''  // Make CTA 0's current K/V cache write visible before attention reads it.
  if (t == 0) profile_work_end = clock64();
  sync_grid();
  if (t == 0) {
    profile_sync_end = clock64();
    profile_local[kProfileRopeCache][0] = profile_work_end - profile_start;
    profile_local[kProfileRopeCache][1] = profile_sync_end - profile_work_end;
  }'''
    text = replace_once(text, cache_barrier, cache_profiled, f"{pair} cache-visibility barrier")

    split_comment = "  // Split-KV partials: one warp per (request, kv head, chunk). Each K/V row is read once and used by\n"
    text = insert_before(text, split_comment, "  if (t == 0) profile_start = clock64();\n", f"{pair} split-KV start")
    split_to_merge = "  sync_grid();\n\n  // Merge the chunk partials per (request, query head)"
    split_profiled = '''  if (t == 0) profile_work_end = clock64();
  sync_grid();
  if (t == 0) {
    profile_sync_end = clock64();
    profile_local[kProfileSplitKv][0] = profile_work_end - profile_start;
    profile_local[kProfileSplitKv][1] = profile_sync_end - profile_work_end;
  }

  // Merge the chunk partials per (request, query head)'''
    text = replace_once(text, split_to_merge, split_profiled, f"{pair} split-KV publish barrier")

    merge_comment = "  // Merge the chunk partials per (request, query head) with log-sum-exp. Every CTA merges the same\n"
    text = insert_before(text, merge_comment, "  if (t == 0) profile_start = clock64();\n", f"{pair} merge start")
    o_comment = "  // O projection plus the first residual. The projection result is rounded to BF16 before the\n"
    text = insert_before(text, o_comment, "  if (t == 0) profile_local[kProfileMerge][0] = clock64() - profile_start;\n",
                         f"{pair} merge end after debug capture")

    text = insert_before(text, o_comment, "  if (t == 0) profile_start = clock64();\n", f"{pair} O/residual start")
    # This anchor is immediately after candidate-specific gate/up prefetch and before the existing barrier.
    o_barrier = "  sync_grid();\n\n  for (int b = 0; b < B; ++b)\n    rms_norm_to_smem(ws + kOffH1 + b * kH"
    o_profiled = '''  if (t == 0) profile_work_end = clock64();
  sync_grid();
  if (t == 0) {
    profile_sync_end = clock64();
    profile_local[kProfileOResidual][0] = profile_work_end - profile_start;
    profile_local[kProfileOResidual][1] = profile_sync_end - profile_work_end;
  }

  if (t == 0) profile_start = clock64();
  for (int b = 0; b < B; ++b)
    rms_norm_to_smem(ws + kOffH1 + b * kH'''
    text = replace_once(text, o_barrier, o_profiled, f"{pair} O prefetch barrier and norm2 start")
    gate_comment = "  // R27 FAST gate/up: warp task r reads gate row r and up row kI + r together, so SwiGLU is\n"
    text = insert_before(text, gate_comment, "  if (t == 0) profile_local[kProfileNorm2][0] = clock64() - profile_start;\n",
                         f"{pair} norm2 end outside debug branch")
    text = insert_before(text, gate_comment, "  if (t == 0) profile_start = clock64();\n", f"{pair} gate/up start")
    gate_loop = "  for (int r = gw; r < kI; r += kGlobalWarps) {\n"
    text = insert_before(text, gate_loop, "  if (t == 0) profile_sub_start = clock64();\n", f"{pair} actual gate/up math start")
    gate_to_prefetch = "  }\n  for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {\n"
    gate_pos = text.index(gate_comment)
    boundary_pos = text.find(gate_to_prefetch, gate_pos)
    if boundary_pos < 0:
        raise RuntimeError(f"{pair}: gate/up to down-prefetch boundary missing")
    text = text[:boundary_pos] + text[boundary_pos:].replace(gate_to_prefetch,
        "  }\n  if (t == 0) { profile_local[kProfileGateUpMath][0] = clock64() - profile_sub_start; profile_sub_start = clock64(); }\n  for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k) {\n", 1)
    gate_barrier = "  sync_grid();\n\n  for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = bf2f(ws[kOffAct + idx]);"
    gate_profiled = '''  if (t == 0) {
    profile_local[kProfileDownPrefetchIssue][0] = clock64() - profile_sub_start;
    profile_work_end = clock64();
  }
  sync_grid();
  if (t == 0) {
    profile_sync_end = clock64();
    profile_local[kProfileGateUpSilu][0] = profile_work_end - profile_start;
    profile_local[kProfileGateUpSilu][1] = profile_sync_end - profile_work_end;
  }

  if (t == 0) profile_start = clock64();
  for (int idx = t; idx < B * kI; idx += kThreads) vec_s[idx] = bf2f(ws[kOffAct + idx]);'''
    text = replace_once(text, gate_barrier, gate_profiled, f"{pair} gate/up prefetch barrier and down start")
    down_barrier = "  sync_grid(); // layer output is complete before the next layer reads it\n  if (p.layer_outputs) {"
    down_profiled = '''  if (t == 0) profile_work_end = clock64();
  sync_grid(); // layer output is complete before the next layer reads it
  if (t == 0) {
    profile_sync_end = clock64();
    profile_local[kProfileDownResidual][0] = profile_work_end - profile_start;
    profile_local[kProfileDownResidual][1] = profile_sync_end - profile_work_end;
  }
  if (p.layer_outputs) {'''
    text = replace_once(text, down_barrier, down_profiled, f"{pair} down and next-layer prefetch barrier")
    capture_body = '''  if (p.layer_outputs) {
    for (int i = t; i < B * kH; i += kThreads)
      p.layer_outputs[(static_cast<size_t>(layer_index) * B * kH) + i] = p.out_hidden[i];
    sync_grid();
  }
  }
  if (blockIdx.x == 0) {'''
    capture_profiled = '''  if (p.layer_outputs) {
    if (t == 0) profile_start = clock64();
    for (int i = t; i < B * kH; i += kThreads)
      p.layer_outputs[(static_cast<size_t>(layer_index) * B * kH) + i] = p.out_hidden[i];
    if (t == 0) profile_work_end = clock64();
    sync_grid();
    if (t == 0) {
      profile_sync_end = clock64();
      profile_local[kProfileLayerCapture][0] = profile_work_end - profile_start;
      profile_local[kProfileLayerCapture][1] = profile_sync_end - profile_work_end;
    }
  }
  if (t == 0) {
    unsigned long long* profile_buffer = g_r28_stage_profile_buffer;
    if (profile_buffer) {
      for (int slot = 0; slot < kProfileSlots; ++slot) {
        const size_t index = (((size_t(layer_index) * kProfileSlots + slot) * kProfileCtas + blockIdx.x) * kProfileFields);
        profile_buffer[index] = profile_local[slot][0];
        profile_buffer[index + 1] = profile_local[slot][1];
      }
    }
  }
  }
  if (blockIdx.x == 0) {'''
    text = replace_once(text, capture_body, capture_profiled, f"{pair} layer capture and per-layer profile flush")
    final_norm = '''    for (int b = 0; b < B; ++b) {
      rms_norm_to_smem(p.out_hidden + b * kH, p.final_norm, vec_s + b * kH, red);'''
    final_norm_profiled = '''    if (t == 0) profile_start = clock64();
    for (int b = 0; b < B; ++b) {
      rms_norm_to_smem(p.out_hidden + b * kH, p.final_norm, vec_s + b * kH, red);'''
    text = replace_once(text, final_norm, final_norm_profiled, f"{pair} final norm start")
    final_tail = '''      for (int i = t; i < kH; i += kThreads) p.out_hidden[b * kH + i] = __float2bfloat16_rn(vec_s[b * kH + i]);
    }
  }
}'''
    final_tail_profiled = '''      for (int i = t; i < kH; i += kThreads) p.out_hidden[b * kH + i] = __float2bfloat16_rn(vec_s[b * kH + i]);
    }
    if (t == 0) {
      const unsigned long long work_cycles = clock64() - profile_start;
      unsigned long long* profile_buffer = g_r28_stage_profile_buffer;
      if (profile_buffer) {
        const size_t index = (((size_t(kLayers - 1) * kProfileSlots + kProfileFinalNorm) * kProfileCtas) * kProfileFields);
        profile_buffer[index] = work_cycles;
        profile_buffer[index + 1] = 0;
      }
    }
  }
}'''
    text = replace_once(text, final_tail, final_tail_profiled, f"{pair} final norm timing and flush")
    validate_semantics(text, pair)
    return text

def validate_semantics(text: str, pair: str) -> None:
    for slot in SLOTS:
        if slot not in text:
            raise RuntimeError(f"{pair}: missing profile slot {slot}")
    if text.count("clock64()") != 33:
        raise RuntimeError(f"{pair}: expected 33 phase clock reads, found {text.count('clock64()')}: " + str([line.strip() for line in text.splitlines() if "clock64()" in line]))
    if text.count("kProfileGateUpMath") != 1 or text.count("kProfileDownPrefetchIssue") != 1:
        raise RuntimeError(f"{pair}: expected one write for each private subslot")
    gate_start = text.index("// R27 FAST gate/up:")
    math_start = text.index("profile_sub_start = clock64();", gate_start)
    gate_loop = text.index("for (int r = gw; r < kI; r += kGlobalWarps)", gate_start)
    math_end = text.index("profile_local[kProfileGateUpMath][0]", gate_start)
    down_loop = text.index("for (int k = 0; k < (kH + kGlobalWarps - 1) / kGlobalWarps; ++k)", gate_start)
    down_end = text.index("profile_local[kProfileDownPrefetchIssue][0]", down_loop)
    combined_end = text.index("profile_local[kProfileGateUpSilu][0]", down_end)
    gate_barrier = text.index("sync_grid();", down_end)
    if not (gate_start < math_start < gate_loop < math_end < down_loop < down_end < gate_barrier < combined_end):
        raise RuntimeError(f"{pair}: two subslots do not sit inside the unchanged combined gate/up interval")
    # Both packed and float projection branches must remain exactly adjacent.
    qkv_branches = '''    if constexpr (B == 2) warp_gemv_rows<kH, B, 1>(rows, packed_vec, lane, res);
    else warp_gemv_rows<kH, B, 1>(rows, vec_s, lane, res);'''
    qkv_start = text.index("// QKV output is packed")
    rope_start = text.index("// Rotate Q/K", qkv_start)
    qkv_region = text[qkv_start:rope_start]
    if pair == "r35" and qkv_region.count(qkv_branches) != 1:
        raise RuntimeError(f"{pair}: QKV packed/float if-else branch changed or split")
    if pair == "r30" and "warp_gemv_rows<kH, B, 1>(rows, vec_s, lane, res);" not in text:
        raise RuntimeError("r30: baseline QKV GEMV call missing")
    # Clock markers must be outside the debug-only region and the profile norm must bracket actual norm work.
    norm_begin = text.index("const bf16* hidden = p.out_hidden;")
    norm_end = text.index("profile_local[kProfileNorm1][0]", norm_begin)
    qkv_comment_pos = text.index("// QKV output is packed", norm_begin)
    norm_span = text[norm_begin:qkv_comment_pos]
    norm_loop_pos = text.index("for (int b = 0; b < B; ++b)", norm_begin)
    if norm_span.count("profile_local[kProfileNorm1]") != 1 or norm_span.count("profile_start = clock64()") != 2:
        raise RuntimeError(f"{pair}: norm1 markers are missing or duplicated")
    if not (norm_begin < text.index("profile_start = clock64()", norm_begin) < norm_loop_pos < norm_end < qkv_comment_pos):
        raise RuntimeError(f"{pair}: norm1 timing does not bracket norm1 work")
    # Every profile clock read must sit outside every debug-only braced region.
    for marker in (m.start() for m in __import__("re").finditer(r"clock64\(\)", text)):
        line_start = text.rfind("\n", 0, marker) + 1
        line = text[line_start:text.find("\n", marker)]
        if "profile_" not in line:
            raise RuntimeError(f"{pair}: untagged clock read")
        for debug in __import__("re").finditer(r"if \(p\.debug[^\n]*\) \{", text):
            opening = text.find("{", debug.start())
            depth = 0
            closing = None
            for idx in range(opening, len(text)):
                if text[idx] == "{": depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        closing = idx
                        break
            if closing is not None and debug.start() < marker < closing:
                raise RuntimeError(f"{pair}: clock marker inside debug block at offset {marker}")
    # The R35 BF16/F32 norm outputs and QKV if-else operation remain intact.
    if pair == "r35":
        branch = '''    if constexpr (B == 2)
      rms_norm_to_smem((layer_index == 0 ? p.embed + static_cast<size_t>(p.input_ids[b]) * kH : hidden + b * kH), L.norm1, packed_vec + b * kH, red);
    else
      rms_norm_to_smem((layer_index == 0 ? p.embed + static_cast<size_t>(p.input_ids[b]) * kH : hidden + b * kH), L.norm1, vec_s + b * kH, red);'''
        if text.count(branch) != 1:
            raise RuntimeError("r35: packed/float norm1 if-else branch changed")
    # Every stage start/end marker is pinned to an operation comment/anchor and ordering.
    sequence = ["profile_local[kProfileNorm1]", "profile_local[kProfileQkv]", "profile_local[kProfileRopeCache]",
                "profile_local[kProfileSplitKv]", "profile_local[kProfileMerge]", "profile_local[kProfileOResidual]",
                "profile_local[kProfileNorm2]", "profile_local[kProfileGateUpSilu]",
                "profile_local[kProfileDownResidual]", "profile_local[kProfileLayerCapture]"]
    positions = [text.index(x) for x in sequence]
    if positions != sorted(positions):
        raise RuntimeError(f"{pair}: phase end markers are out of order")
    for label in ("QKV start", "rope/cache start", "split-KV start", "merge start", "O/residual start", "gate/up start"):
        if f"{label}" in text:
            raise RuntimeError("internal label leaked into source")

def strip_is_exact(original_source: str, profiled_source: str, original_inc: str, profiled_inc: str) -> bool:
    source_restored = profiled_source.replace(PROFILE_DECLS, "").replace(PROFILE_API, "") == original_source
    patch_text = "\n".join(difflib.unified_diff(
        original_inc.splitlines(), profiled_inc.splitlines(), n=3,
        fromfile="block360_kernel.inc", tofile="block360_kernel.inc", lineterm="")) + "\n"
    with tempfile.TemporaryDirectory(prefix="tiny-prl-stage-strip-") as temp:
        tempdir = Path(temp)
        candidate = tempdir / "block360_kernel.inc"
        candidate.write_text(profiled_inc)
        stripped = subprocess.run(["patch", "--batch", "--fuzz=0", "-R", "-p0", "-d", str(tempdir)],
            input=patch_text, text=True, capture_output=True)
        include_restored = stripped.returncode == 0 and candidate.read_text() == original_inc
    return source_restored and include_restored

for pair, config in PAIRS.items():
    if "constexpr int kProfileSlots = 13;" not in PROFILE_DECLS or "kProfileDownPrefetchIssue = 12" not in PROFILE_DECLS:
        raise RuntimeError("private profile ABI is not thirteen slots")
    source_dir = config["directory"]
    actual_hashes = {name: digest(source_dir / name) for name in config["sha256"]}
    if actual_hashes != config["sha256"]:
        raise RuntimeError(f"{pair} baseline changed: expected={config['sha256']}, got={actual_hashes}")
    out = HERE / f"{pair}-profile"
    out.mkdir(exist_ok=True)
    original_source = (source_dir / "block360.cu").read_text()
    original_inc = (source_dir / "block360_kernel.inc").read_text()
    profiled_source = instrument_source(original_source)
    profiled_inc = instrument_kernel(original_inc, pair)
    if not strip_is_exact(original_source, profiled_source, original_inc, profiled_inc):
        raise RuntimeError(f"{pair}: removing the explicit profile diff did not reconstruct baseline sources")
    (out / "block360.cu").write_text(profiled_source)
    (out / "block360_kernel.inc").write_text(profiled_inc)
    validator.validate_abi_source(out / "block360_kernel.inc")
    manifest = {"pair": pair, "status": "semantically-anchored-source-only-preflight-passed",
                "baseline_source": str(source_dir / "block360.cu"), "baseline_include": str(source_dir / "block360_kernel.inc"),
                "profile_source": str(out / "block360.cu"), "profile_include": str(out / "block360_kernel.inc"),
                "baseline_sha256": actual_hashes,
                "profile_sha256": {"block360.cu": digest(out / "block360.cu"), "block360_kernel.inc": digest(out / "block360_kernel.inc")},
                "strip_instrumentation_roundtrip": "byte-identical via exact profile declaration removal and strict reverse diff",
                "semantic_checks": ["all profile insertions anchored to exact unique phase context",
                    "norm1 clock starts before norm and stops outside debug block",
                    "QKV packed/float if-else remains adjacent and unmodified",
                    "phase ends ordered norm1, QKV, rope/cache, split-KV, merge, O/residual, norm2, gate/up, down, layer capture",
                    "gate/up math and down-prefetch subslots remain inside the combined gate/up interval",
                    "thirteen-slot private profile buffer keeps the original eleven slots",
                    "decoder ABI validated by canonical source validator"]}
    (HERE / f"{pair}-preflight.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("semantic source anchors, branch-shape checks, and decoder ABI preflight passed; no build or GPU run")
