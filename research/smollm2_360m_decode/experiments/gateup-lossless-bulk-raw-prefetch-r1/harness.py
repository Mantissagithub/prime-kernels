#!/usr/bin/env python3
"""Compare lossless SIMD control and flat raw-prefetch candidate."""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path("/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951")
MODEL_FILE = MODEL / "model.safetensors"
NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
LIB = HERE / "build/libprojection_probe.so"
PTXAS_LOG = HERE / "build/ptxas.log"
LAYERS, N, K, CHUNK = 32, 2560, 960, 8
PLACEMENTS, REPLAYS = 10, 5
MEMORY_LIMIT = 3 * 1024**3


def build_argv(ptxas_verbose=False):
    argv = [NVCC, "-std=c++17", "-O3", "-arch=sm_89", "-Xcompiler=-fPIC", "--shared",
            str(HERE / "projection_probe.cu"), "-o", str(LIB)]
    if ptxas_verbose:
        argv.insert(1, "-Xptxas=-v")
    return argv


def compile_probe(ptxas_verbose=False):
    if LIB.exists():
        raise FileExistsError(f"refusing to overwrite {LIB}")
    if ptxas_verbose and PTXAS_LOG.exists():
        raise FileExistsError(f"refusing to overwrite {PTXAS_LOG}")
    LIB.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(build_argv(ptxas_verbose), check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ptxas_verbose:
        PTXAS_LOG.write_text(result.stdout)
    return result.stdout


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


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true",
                      help="compile and exit without importing torch, loading weights, or using a GPU")
    mode.add_argument("--skip-build", action="store_true", help="use an existing shared library")
    parser.add_argument("--ptxas-verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results.json",
                        help="new results path; existing files are never overwritten")
    args = parser.parse_args()

    build_output = ""
    if not args.skip_build:
        build_output = compile_probe(args.ptxas_verbose)
        if build_output:
            print(build_output, end="")
    if args.build_only:
        return
    if not LIB.is_file():
        raise FileNotFoundError(f"shared library missing: {LIB}")
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
    lib.run_gateup_lossless.argtypes = [vp, vp, vp, vp, vp, vp, vp]
    lib.run_gateup_lossless.restype = ctypes.c_int
    lib.run_gateup_lossless_simd.argtypes = [vp, vp, vp, vp, vp, vp, vp]
    lib.run_gateup_lossless_simd.restype = ctypes.c_int
    lib.run_gateup_lossless_bulk_raw.argtypes = [vp, vp, vp, vp, vp, vp,
                                                 ctypes.c_size_t, vp]
    lib.run_gateup_lossless_bulk_raw.restype = ctypes.c_int
    lib.decode_weights_simd.argtypes = [vp, vp, vp, vp, vp]
    lib.decode_weights_simd.restype = ctypes.c_int

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
    outputs = [torch.full((LAYERS, 2, 2 * N), float("nan"), device=device, dtype=torch.float32)
               for _ in range(3)]
    decoded = torch.empty((LAYERS, 2 * N, K), device=device, dtype=torch.uint16)

    def enqueue(arm, output):
        stream = torch.cuda.current_stream().cuda_stream
        for layer in range(LAYERS):
            dense_layer = pointer(dense_ptr, layer * dense_layer_bytes)
            if arm == 0:
                check(lib.run_gateup(0, dense_layer, x.data_ptr(),
                                     output[layer].data_ptr(), stream), "run_gateup")
            elif arm == 1:
                check(lib.run_gateup_lossless_simd(
                    pointer(plane_ptr, layer * plane_layer_bytes),
                    pointer(headers_ptr, layer * header_layer_bytes),
                    pointer(raw_ptr, raw_prefix[layer]),
                    pointer(offsets_ptr, layer * offset_layer_bytes),
                    x.data_ptr(), output[layer].data_ptr(), stream), "run_gateup_lossless_simd")
            else:
                check(lib.run_gateup_lossless_bulk_raw(
                    pointer(plane_ptr, layer * plane_layer_bytes),
                    pointer(headers_ptr, layer * header_layer_bytes),
                    pointer(raw_ptr, raw_prefix[layer]),
                    pointer(offsets_ptr, layer * offset_layer_bytes),
                    x.data_ptr(), output[layer].data_ptr(), raw_layer_bytes[layer], stream),
                    "run_gateup_lossless_bulk_raw")

    def enqueue_decode():
        stream = torch.cuda.current_stream().cuda_stream
        for layer in range(LAYERS):
            check(lib.decode_weights_simd(
                pointer(plane_ptr, layer * plane_layer_bytes),
                pointer(headers_ptr, layer * header_layer_bytes),
                pointer(raw_ptr, raw_prefix[layer]), decoded[layer].data_ptr(), stream),
                "decode_weights_simd")

    enqueue_decode()
    torch.cuda.synchronize()
    dense_words = dense.view(torch.uint16).reshape(LAYERS, 2 * N, K)
    exact_weights = all(torch.equal(decoded[i], dense_words[i]) for i in range(LAYERS))
    if not exact_weights:
        raise RuntimeError("GPU decode did not reproduce every BF16 weight word")

    enqueue(0, outputs[0])
    torch.cuda.synchronize()
    graphs = []
    for arm in (1, 2):
        enqueue(arm, outputs[arm])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            enqueue(arm, outputs[arm])
        torch.cuda.synchronize()
        graphs.append(graph)

    for output in outputs:
        output.fill_(float("nan"))
    torch.cuda.synchronize()
    enqueue(0, outputs[0])
    graphs[0].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    finite = [bool(torch.isfinite(output).all().item()) for output in outputs]
    graph_arm_finite = finite[1:]
    dense_eager_finite = finite[0]
    if not dense_eager_finite or not all(graph_arm_finite):
        raise RuntimeError(f"NaN-poisoned execution left unwritten output: dense_eager={dense_eager_finite}, graph_arms={graph_arm_finite}")
    fp32_exact = torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[1], outputs[2])
    bf16_exact = all(torch.equal(outputs[a].to(torch.bfloat16).view(torch.int16),
                                 outputs[b].to(torch.bfloat16).view(torch.int16))
                     for a, b in ((0, 1), (1, 2)))
    if not fp32_exact or not bf16_exact:
        raise RuntimeError("dense, SIMD control, and bulk raw-prefetch outputs differ")

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

    times = {"simd_control": [], "bulk_raw_prefetch": []}
    for arm in [1, 2] * (PLACEMENTS // 2):
        times["simd_control" if arm == 1 else "bulk_raw_prefetch"].append(time_graph(graphs[arm - 1]))
    graphs[0].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    if not torch.equal(outputs[1], outputs[2]):
        raise RuntimeError("SIMD control and bulk raw-prefetch outputs diverged during timing")

    def summary(samples):
        placement_medians = [statistics.median(placement) for placement in samples]
        return [statistics.median(placement_medians), min(min(p) for p in samples),
                max(max(p) for p in samples)]

    peak_bytes = torch.cuda.max_memory_allocated(device)
    if peak_bytes >= MEMORY_LIMIT:
        raise RuntimeError(f"probe peak allocation {peak_bytes} reached the 3 GiB limit")
    report = {
        "model_file": str(MODEL_FILE), "model_revision": MODEL.name,
        "model_sha256": sha256_file(MODEL_FILE),
        "kernel_source_sha256": sha256_file(HERE / "projection_probe.cu"),
        "harness_sha256": sha256_file(Path(__file__)),
        "shared_library_sha256": sha256_file(LIB),
        "ptxas_verbose": args.ptxas_verbose,
        "ptxas_log": str(PTXAS_LOG) if args.ptxas_verbose else None,
        "ptxas_output": build_output if args.ptxas_verbose else None,
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
        "graph_captures": 2,
        "graph_replay_output_write_verified_per_arm": graph_arm_finite,
        "dense_eager_output_write_verified": dense_eager_finite,
        "fp32_exact": fp32_exact, "bf16_bits_exact": bf16_exact,
        "output_elements_checked_per_arm": LAYERS * 2 * 2 * N,
        "timed_placements_per_arm": PLACEMENTS // 2, "replays_per_placement": REPLAYS,
        "timed_projection_calls_per_arm": LAYERS * (PLACEMENTS // 2) * REPLAYS,
        "untimed_projection_calls_per_candidate_arm": LAYERS * 3,
        "untimed_dense_control_projection_calls": LAYERS * 2,
        "validation_error_count": 0,
        "simd_control_us_median_range": summary(times["simd_control"]),
        "bulk_raw_prefetch_us_median_range": summary(times["bulk_raw_prefetch"]),
        "placement_raw_event_samples_us_per_layer": times,
        "tensor_allocated_peak_bytes": peak_bytes,
        "timing_scope": "same-session SIMD control versus bulk raw-prefetch candidate; dense weights are retained only for exact decoded-weight and output verification; excludes SwiGLU and surrounding MLP work",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
