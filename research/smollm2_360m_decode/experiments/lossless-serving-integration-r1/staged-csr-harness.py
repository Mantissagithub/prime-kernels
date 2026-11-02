#!/usr/bin/env python3
"""Compare dense, rolling depth 1, and staged CSR on one resident GPU session."""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path("/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951")
MODEL_FILE = MODEL / "model.safetensors"
NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
LIB = HERE / "build/libprojection_probe.so"
SOURCE_SHA256 = "aa730f41fdcef3db8a14566a13cf52dacb9b3ac2a640b4ba8eeebe464e6cad26"
LIB_SHA256 = "4f78c509cd9286d6c1b2487eb3c844761d737a6d89c9baacc7498b994f6fbf6e"
LAYERS, N, K, CHUNK = 32, 2560, 960, 8
PLACEMENTS, REPLAYS = 18, 5
MEMORY_LIMIT = 3 * 1024**3


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_rows(tensor, torch):
    """Return byte planes, headers, raw escapes and row offsets for [5120,960]."""
    bits = tensor.contiguous().view(torch.uint16).to(torch.int64)
    words = bits.reshape(N * 2, K // CHUNK, CHUNK)
    exponent = (words >> 7) & 255
    base = exponent.min(dim=2).values
    delta = exponent - base[:, :, None]
    escaped = (base == 255) | (delta.max(dim=2).values > 7)
    packed = torch.zeros_like(base)
    for lane in range(CHUNK):
        packed |= delta[:, :, lane] << (3 * lane)
    per_row_escapes = escaped.sum(dim=1, dtype=torch.int64)
    offsets = torch.empty(N * 2 + 1, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = per_row_escapes.cumsum(0)
    escaped_flat = escaped.reshape(-1)
    escape_ids = torch.nonzero(escaped_flat, as_tuple=False).flatten()
    if escape_ids.numel() >= 1 << 24:
        raise ValueError("escape table exceeds the 24-bit header index")
    headers = ((base << 24) | packed).reshape(-1)
    headers[escaped_flat] = (255 << 24) | torch.arange(escape_ids.numel(), dtype=torch.int64)
    headers = headers.to(torch.uint32).contiguous()
    plane = (((bits >> 8) & 128) | (bits & 127)).to(torch.uint8).contiguous()
    raw = bits.reshape(-1, CHUNK)[escape_ids].to(torch.uint16).contiguous()
    return plane, headers, raw, offsets.to(torch.uint32).contiguous()


def check(rc, label):
    if rc:
        raise RuntimeError(f"{label} returned CUDA error {rc}")


def compare_staged_dense_placements(placement_medians):
    blocks = []
    for block in range(3):
        first = block * 2
        dense_placements = placement_medians["dense"][first:first + 2]
        staged_placements = placement_medians["staged_csr"][first:first + 2]
        blocks.append({
            "block": block + 1,
            "dense_placement_medians_us_per_layer": dense_placements,
            "staged_placement_medians_us_per_layer": staged_placements,
            "staged_both_placements_beat_dense_both_placements": max(staged_placements) < min(dense_placements),
        })
    return blocks


def main():
    parser = argparse.ArgumentParser(description="Compare dense, rolling depth 1, and staged-CSR gate/up projection.")
    parser.add_argument("--output", type=Path, default=HERE / "results-r2.json",
                        help="new results path; existing files are never overwritten")
    args = parser.parse_args()

    if not LIB.is_file():
        raise FileNotFoundError(f"shared library missing: {LIB}")
    source_sha256 = sha256_file(HERE / "projection_probe.cu")
    if source_sha256 != SOURCE_SHA256:
        raise RuntimeError(f"candidate source SHA mismatch: {source_sha256}")
    library_sha256 = sha256_file(LIB)
    if library_sha256 != LIB_SHA256:
        raise RuntimeError(f"shared library SHA mismatch: {library_sha256}")
    result_path = args.output.resolve()
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"cached checkpoint missing: {MODEL_FILE}")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    device = torch.device("cuda")
    lib = ctypes.CDLL(str(LIB))
    vp = ctypes.c_void_p
    lib.run_gateup.argtypes = [ctypes.c_int, vp, vp, vp, vp]
    lib.run_gateup.restype = ctypes.c_int
    lib.run_gateup_lossless_predicated.argtypes = [vp, vp, vp, vp, vp, vp, vp]
    lib.run_gateup_lossless_predicated.restype = ctypes.c_int
    lib.run_gateup_lossless_rolling.argtypes = [ctypes.c_int, vp, vp, vp, vp, vp, vp, vp]
    lib.run_gateup_lossless_rolling.restype = ctypes.c_int
    lib.run_gateup_lossless_staged_csr.argtypes = [vp, vp, vp, vp, vp, vp, vp]
    lib.run_gateup_lossless_staged_csr.restype = ctypes.c_int
    lib.decode_weights_predicated.argtypes = [vp, vp, vp, vp, vp]
    lib.decode_weights_predicated.restype = ctypes.c_int

    dense_cpu, planes_cpu, headers_cpu, raws_cpu, offsets_cpu = [], [], [], [], []
    with safe_open(str(MODEL_FILE), framework="pt", device="cpu") as model:
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}.mlp."
            gate = model.get_tensor(prefix + "gate_proj.weight")
            up = model.get_tensor(prefix + "up_proj.weight")
            if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
                raise TypeError(f"layer {layer} gate/up weights must be BF16")
            if tuple(gate.shape) != (N, K) or tuple(up.shape) != (N, K):
                raise ValueError(f"layer {layer} gate/up shapes must each be [{N},{K}]")
            dense = torch.cat((gate, up), dim=0).contiguous()
            plane, headers, raw, offsets = encode_rows(dense, torch)
            dense_cpu.append(dense)
            planes_cpu.append(plane)
            headers_cpu.append(headers)
            raws_cpu.append(raw)
            offsets_cpu.append(offsets)

    dense = torch.cat(dense_cpu).to(device)
    plane = torch.cat(planes_cpu).to(device)
    headers = torch.cat(headers_cpu).to(device)
    raw = torch.cat(raws_cpu).to(device)
    offsets = torch.cat(offsets_cpu).to(device)
    dense_bytes = sum(t.numel() * t.element_size() for t in dense_cpu)
    plane_bytes = sum(t.numel() * t.element_size() for t in planes_cpu)
    header_bytes = sum(t.numel() * t.element_size() for t in headers_cpu)
    raw_bytes = sum(t.numel() * t.element_size() for t in raws_cpu)
    offset_bytes = sum(t.numel() * t.element_size() for t in offsets_cpu)
    encoded_bytes = plane_bytes + header_bytes + raw_bytes + offset_bytes
    if torch.cuda.memory_allocated(device) >= MEMORY_LIMIT:
        raise RuntimeError("probe weight allocation reached the 3 GiB limit")

    dense_layer_bytes = 2 * N * K * 2
    plane_layer_bytes = 2 * N * K
    header_layer_bytes = 2 * N * (K // CHUNK) * 4
    raw_layer_bytes = [t.numel() * t.element_size() for t in raws_cpu]
    offset_layer_bytes = (2 * N + 1) * 4
    raw_prefix = [0]
    for size in raw_layer_bytes:
        raw_prefix.append(raw_prefix[-1] + size)
    dense_ptr = dense.data_ptr()
    plane_ptr = plane.data_ptr()
    headers_ptr = headers.data_ptr()
    raw_ptr = raw.data_ptr()
    offsets_ptr = offsets.data_ptr()

    def pointer(base, offset):
        return ctypes.c_void_p(base + offset)

    x = torch.randn((2, K), device="cpu", dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(341)).to(device)
    arm_names = ("dense", "rolling_depth_1", "staged_csr")
    outputs = {name: torch.full((LAYERS, 2, 2 * N), float("nan"), device=device,
                                dtype=torch.float32) for name in arm_names}
    decoded = torch.empty((LAYERS, 2 * N, K), device=device, dtype=torch.uint16)

    def enqueue_dense(output):
        stream = torch.cuda.current_stream().cuda_stream
        for layer in range(LAYERS):
            dense_layer = pointer(dense_ptr, layer * dense_layer_bytes)
            check(lib.run_gateup(0, dense_layer, x.data_ptr(),
                                 output[layer].data_ptr(), stream), "run_gateup")

    def enqueue(name, output):
        stream = torch.cuda.current_stream().cuda_stream
        for layer in range(LAYERS):
            if name == "dense":
                dense_layer = pointer(dense_ptr, layer * dense_layer_bytes)
                check(lib.run_gateup(0, dense_layer, x.data_ptr(),
                                     output[layer].data_ptr(), stream), "run_gateup(dense)")
                continue
            args = (pointer(plane_ptr, layer * plane_layer_bytes),
                    pointer(headers_ptr, layer * header_layer_bytes),
                    pointer(raw_ptr, raw_prefix[layer]),
                    pointer(offsets_ptr, layer * offset_layer_bytes),
                    x.data_ptr(), output[layer].data_ptr(), stream)
            if name == "rolling_depth_1":
                check(lib.run_gateup_lossless_rolling(1, *args),
                      "run_gateup_lossless_rolling(depth=1)")
            else:
                check(lib.run_gateup_lossless_staged_csr(*args),
                      "run_gateup_lossless_staged_csr")

    def enqueue_decode():
        stream = torch.cuda.current_stream().cuda_stream
        for layer in range(LAYERS):
            check(lib.decode_weights_predicated(
                pointer(plane_ptr, layer * plane_layer_bytes),
                pointer(headers_ptr, layer * header_layer_bytes),
                pointer(raw_ptr, raw_prefix[layer]), decoded[layer].data_ptr(), stream),
                "decode_weights_predicated")

    enqueue_decode()
    torch.cuda.synchronize()
    dense_words = dense.view(torch.uint16).reshape(LAYERS, 2 * N, K)
    exact_weights = all(torch.equal(decoded[i], dense_words[i]) for i in range(LAYERS))
    if not exact_weights:
        raise RuntimeError("GPU decode did not reproduce every BF16 weight word")

    dense_reference = torch.full((LAYERS, 2, 2 * N), float("nan"), device=device, dtype=torch.float32)
    enqueue_dense(dense_reference)
    torch.cuda.synchronize()
    if not torch.isfinite(dense_reference).all().item():
        raise RuntimeError("untimed dense reference left unwritten output")

    graphs = {}
    for name in arm_names:
        enqueue(name, outputs[name])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            enqueue(name, outputs[name])
        torch.cuda.synchronize()
        graphs[name] = graph

    for output in outputs.values():
        output.fill_(float("nan"))
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    finite = {name: bool(torch.isfinite(output).all().item())
              for name, output in outputs.items()}
    if not all(finite.values()):
        raise RuntimeError(f"NaN-poisoned graph replay left unwritten output: {finite}")
    fp32_exact = {name: torch.equal(dense_reference, output)
                  for name, output in outputs.items()}
    dense_bf16 = dense_reference.to(torch.bfloat16).view(torch.int16)
    bf16_exact = {name: torch.equal(dense_bf16, output.to(torch.bfloat16).view(torch.int16))
                  for name, output in outputs.items()}
    if not all(fp32_exact.values()) or not all(bf16_exact.values()):
        raise RuntimeError(f"compressed projection outputs differ from dense reference: fp32={fp32_exact}, bf16={bf16_exact}")

    def time_graph(graph):
        values = []
        for _ in range(REPLAYS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            values.append(start.elapsed_time(end) * 1000 / LAYERS)
        return values

    times = {name: [] for name in arm_names}
    placement_block = ["dense", "rolling_depth_1", "staged_csr",
                       "staged_csr", "rolling_depth_1", "dense"]
    timed_arm_names = placement_block * 3
    for name in timed_arm_names:
        times[name].append(time_graph(graphs[name]))
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    posttiming_exact = {name: torch.equal(dense_reference, output)
                        for name, output in outputs.items()}
    if not all(posttiming_exact.values()):
        raise RuntimeError("outputs changed or diverged during balanced timing")

    def summary(samples):
        placement_medians = [statistics.median(placement) for placement in samples]
        return [statistics.median(placement_medians), min(min(p) for p in samples),
                max(max(p) for p in samples)]

    placement_medians = {name: [statistics.median(placement) for placement in samples]
                         for name, samples in times.items()}
    strict_block_gate = compare_staged_dense_placements(placement_medians)
    strict_staged_beats_dense = all(
        block["staged_both_placements_beat_dense_both_placements"] for block in strict_block_gate)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    if peak_bytes >= MEMORY_LIMIT:
        raise RuntimeError(f"probe peak allocation {peak_bytes} reached the 3 GiB limit")
    report = {
        "model_file": str(MODEL_FILE), "model_revision": MODEL.name,
        "model_sha256": sha256_file(MODEL_FILE),
        "kernel_source_sha256": source_sha256,
        "harness_sha256": sha256_file(Path(__file__)),
        "shared_library_sha256": library_sha256,
        "layers": LAYERS, "combined_gate_up_shape_per_layer": [2 * N, K],
        "input_shape": [2, K], "output_shape_per_layer": [2, 2 * N],
        "dense_weight_bytes": dense_bytes,
        "encoded_bytes_including_all_metadata": encoded_bytes,
        "encoded_byte_counts": {"planes": plane_bytes, "headers": header_bytes,
                                "raw_escapes": raw_bytes, "csr_row_offsets": offset_bytes},
        "escape_chunks": sum(t.numel() // CHUNK for t in raws_cpu),
        "chunks": LAYERS * 2 * N * (K // CHUNK),
        "gpu_weight_transfers": "one concatenated transfer for each of dense, plane, headers, raw escapes, and CSR row offsets",
        "gpu_decoded_weights_bit_exact": exact_weights,
        "decoded_weight_words_checked": LAYERS * 2 * N * K,
        "graph_captures": len(graphs), "graph_arm_names": list(arm_names),
        "graph_replay_output_write_verified_per_arm": finite,
        "fp32_exact_against_untimed_dense_reference": fp32_exact,
        "bf16_bits_exact_against_untimed_dense_reference": bf16_exact,
        "posttiming_fp32_exact_against_untimed_dense_reference": posttiming_exact,
        "output_elements_checked_per_arm": LAYERS * 2 * 2 * N,
        "timed_placements_per_arm": PLACEMENTS // len(arm_names), "replays_per_placement": REPLAYS,
        "timed_projection_calls_per_arm": LAYERS * (PLACEMENTS // len(arm_names)) * REPLAYS,
        "untimed_warmup_proof_posttiming_projection_calls_per_arm": LAYERS * 3,
        "dense_reference_projection_calls": LAYERS,
        "validation_error_count": 0,
        "per_arm_us_median_range": {name: summary(samples) for name, samples in times.items()},
        "timed_arm_sequence": timed_arm_names,
        "placement_medians_us_per_layer": placement_medians,
        "placement_raw_event_samples_us_per_layer": times,
        "strict_staged_beats_dense_gate_per_block": strict_block_gate,
        "strict_staged_beats_dense_gate_all_blocks": strict_staged_beats_dense,
        "adoption": False,
        "tensor_allocated_peak_bytes": peak_bytes,
        "timing_scope": "same-session dense, rolling depth-1, and staged-CSR projection graphs over 32 distinct resident layers; excludes SwiGLU and surrounding MLP work",
        "hypothesis": "staging each warp-owned CSR row range before the barrier may remove in-loop CSR metadata loads; the strict staged-versus-dense gate is reported without automatic adoption",
        "rolling_shared_bytes_per_cta": 2 * K * 4,
        "staged_csr_added_shared_bytes_per_cta": 8 * 14 * 2 * 8,
        "staged_csr_total_shared_bytes_per_cta": 2 * K * 4 + 8 * 14 * 2 * 8,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
