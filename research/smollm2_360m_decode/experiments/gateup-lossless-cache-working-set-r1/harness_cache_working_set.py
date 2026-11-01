#!/usr/bin/env python3
"""Measure the frozen predicated projection across four layer working sets."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDICATED = HERE.parent / "gateup-lossless-predicated-decode-r1"
MODEL = Path("/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951")
MODEL_FILE = MODEL / "model.safetensors"
SOURCE = PREDICATED / "projection_probe.cu"
LIB = PREDICATED / "build/libprojection_probe.so"
EXPECTED_SOURCE_SHA256 = "0b56cdc994d445ddcb52f9ffb91565e80cf4de1d9b89723c58a5daa12f8c80f9"
EXPECTED_LIB_SHA256 = "a7578364ff44ad3a6091c0991e59713241ab2ae94dacb76ebedcf060eb46af8c"
FROZEN_HARNESS = PREDICATED / "harness.py"
EXPECTED_FROZEN_HARNESS_SHA256 = "90e85ef40f38dfe01e3211f104efb5eac381deebd7896311c6030ed79740fc09"
LAYERS, N, K, CHUNK = 32, 2560, 960, 8
UNIQUE_COUNTS = (1, 2, 4, 32)
PLACEMENT_ORDER = (1, 2, 4, 32, 32, 4, 2, 1) * 3
REPLAYS = 5
MEMORY_LIMIT = 3 * 1024**3


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_encoder():
    spec = importlib.util.spec_from_file_location("frozen_predicated_harness", FROZEN_HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_rows


def check(rc, label):
    if rc:
        raise RuntimeError(f"{label} returned CUDA error {rc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results-r1.json")
    args = parser.parse_args()
    result_path = args.output.resolve()
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    if not SOURCE.is_file() or not LIB.is_file() or not FROZEN_HARNESS.is_file():
        raise FileNotFoundError("frozen source, DSO, or encoder harness is missing")
    source_sha256 = sha256_file(SOURCE)
    library_sha256 = sha256_file(LIB)
    frozen_harness_sha256 = sha256_file(FROZEN_HARNESS)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"frozen source hash mismatch: {source_sha256}")
    if library_sha256 != EXPECTED_LIB_SHA256:
        raise RuntimeError(f"frozen DSO hash mismatch: {library_sha256}")
    if frozen_harness_sha256 != EXPECTED_FROZEN_HARNESS_SHA256:
        raise RuntimeError(f"frozen encoder harness hash mismatch: {frozen_harness_sha256}")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"cached checkpoint missing: {MODEL_FILE}")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    encode_rows = load_frozen_encoder()
    device = torch.device("cuda")
    lib = ctypes.CDLL(str(LIB))
    vp = ctypes.c_void_p
    lib.run_gateup.argtypes = [ctypes.c_int, vp, vp, vp, vp]
    lib.run_gateup.restype = ctypes.c_int
    lib.run_gateup_lossless_predicated.argtypes = [vp, vp, vp, vp, vp, vp, vp]
    lib.run_gateup_lossless_predicated.restype = ctypes.c_int
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
                raise ValueError(f"layer {layer} gate/up weights must each be [{N},{K}]")
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
    if torch.cuda.memory_allocated(device) >= MEMORY_LIMIT:
        raise RuntimeError("probe weight allocation reached the 3 GiB limit")
    dense_layer_bytes = 2 * N * K * 2
    plane_layer_bytes = 2 * N * K
    header_layer_bytes = 2 * N * (K // CHUNK) * 4
    offset_layer_bytes = (2 * N + 1) * 4
    raw_layer_bytes = [tensor.numel() * tensor.element_size() for tensor in raws_cpu]
    raw_prefix = [0]
    for size in raw_layer_bytes:
        raw_prefix.append(raw_prefix[-1] + size)

    def pointer(base, offset):
        return ctypes.c_void_p(base + offset)

    dense_ptr, plane_ptr = dense.data_ptr(), plane.data_ptr()
    headers_ptr, raw_ptr, offsets_ptr = headers.data_ptr(), raw.data_ptr(), offsets.data_ptr()
    x = torch.randn((2, K), device="cpu", dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(341)).to(device)
    reference = torch.full((LAYERS, 2, 2 * N), float("nan"), device=device,
                           dtype=torch.float32)
    outputs = {count: torch.full((LAYERS, 2, 2 * N), float("nan"), device=device,
                                 dtype=torch.float32) for count in UNIQUE_COUNTS}
    decoded = torch.empty((LAYERS, 2 * N, K), device=device, dtype=torch.uint16)
    stream = torch.cuda.current_stream().cuda_stream

    for layer in range(LAYERS):
        check(lib.run_gateup(0, pointer(dense_ptr, layer * dense_layer_bytes),
                             x.data_ptr(), reference[layer].data_ptr(), stream), "run_gateup")
        check(lib.decode_weights_predicated(
            pointer(plane_ptr, layer * plane_layer_bytes),
            pointer(headers_ptr, layer * header_layer_bytes),
            pointer(raw_ptr, raw_prefix[layer]), decoded[layer].data_ptr(), stream),
            "decode_weights_predicated")

    torch.cuda.synchronize()
    dense_words = dense.view(torch.uint16).reshape(LAYERS, 2 * N, K)
    exact_weights = all(torch.equal(decoded[layer], dense_words[layer])
                        for layer in range(LAYERS))
    if not exact_weights:
        raise RuntimeError("GPU decode did not reproduce every BF16 weight word")

    graphs = {}
    for count in UNIQUE_COUNTS:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_stream = torch.cuda.current_stream().cuda_stream
            for call in range(LAYERS):
                layer = call % count
                check(lib.run_gateup_lossless_predicated(
                    pointer(plane_ptr, layer * plane_layer_bytes),
                    pointer(headers_ptr, layer * header_layer_bytes),
                    pointer(raw_ptr, raw_prefix[layer]),
                    pointer(offsets_ptr, layer * offset_layer_bytes), x.data_ptr(),
                    outputs[count][call].data_ptr(), capture_stream),
                    "run_gateup_lossless_predicated")
        graphs[count] = graph
    torch.cuda.synchronize()

    for output in outputs.values():
        output.fill_(float("nan"))
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    exact_by_case = {}
    bf16_exact_by_case = {}
    finite_by_case = {}
    for count in UNIQUE_COUNTS:
        output = outputs[count]
        finite_by_case[str(count)] = bool(torch.isfinite(output).all().item())
        exact_by_case[str(count)] = all(
            torch.equal(output[call], reference[call % count])
            for call in range(LAYERS))
        bf16_exact_by_case[str(count)] = all(
            torch.equal(output[call].to(torch.bfloat16).view(torch.int16),
                        reference[call % count].to(torch.bfloat16).view(torch.int16))
            for call in range(LAYERS))
        if (not finite_by_case[str(count)] or not exact_by_case[str(count)] or
                not bf16_exact_by_case[str(count)]):
            raise RuntimeError(f"NaN-poisoned graph or output mismatch for {count} unique layers")

    raw_samples = {str(count): [] for count in UNIQUE_COUNTS}
    for count in PLACEMENT_ORDER:
        values = []
        for _ in range(REPLAYS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[count].replay()
            end.record()
            torch.cuda.synchronize()
            values.append(start.elapsed_time(end) * 1000)
        raw_samples[str(count)].append(values)
    torch.cuda.synchronize()

    def timing_summary(samples):
        placement_medians = [statistics.median(placement) for placement in samples]
        return {
            "placement_medians_us_32_calls": placement_medians,
            "placement_medians_us_per_call": [value / LAYERS for value in placement_medians],
            "median_of_placement_medians_us_32_calls": statistics.median(placement_medians),
            "median_of_placement_medians_us_per_call": statistics.median(placement_medians) / LAYERS,
            "placement_median_range_us_32_calls": [min(placement_medians), max(placement_medians)],
            "raw_sample_range_us_32_calls": [min(min(p) for p in samples), max(max(p) for p in samples)],
        }

    dense_bytes = sum(t.numel() * t.element_size() for t in dense_cpu)
    encoded_cases = {}
    for count in UNIQUE_COUNTS:
        encoded_cases[str(count)] = {
            "call_layer_map": [call % count for call in range(LAYERS)],
            "unique_layers_touched": list(range(count)),
            "unique_layer_count": count,
            "distinct_encoded_bytes": {
                "planes": count * 2 * N * K,
                "headers": count * 2 * N * (K // CHUNK) * 4,
                "raw_escapes": sum(raw_layer_bytes[:count]),
                "csr_row_offsets": count * (2 * N + 1) * 4,
                "total_including_metadata": count * (2 * N * K +
                    2 * N * (K // CHUNK) * 4 + (2 * N + 1) * 4) +
                    sum(raw_layer_bytes[:count]),
            },
            "timing": timing_summary(raw_samples[str(count)]),
        }

    peak_bytes = torch.cuda.max_memory_allocated(device)
    if peak_bytes >= MEMORY_LIMIT:
        raise RuntimeError(f"probe peak allocation {peak_bytes} reached the 3 GiB limit")
    report = {
        "diagnostic": "predicated gate/up cache working set; not a full-model or RL speedup",
        "model_file": str(MODEL_FILE),
        "model_revision": MODEL.name,
        "model_sha256": sha256_file(MODEL_FILE),
        "source_file": str(SOURCE),
        "source_sha256": source_sha256,
        "shared_library": str(LIB),
        "shared_library_sha256": library_sha256,
        "harness_sha256": sha256_file(Path(__file__)),
        "frozen_encoder_harness_sha256": frozen_harness_sha256,
        "layers_loaded_to_gpu": LAYERS,
        "calls_per_graph": LAYERS,
        "output_shape_per_case": [LAYERS, 2, 2 * N],
        "unique_layer_counts": list(UNIQUE_COUNTS),
        "placement_order_unique_layer_counts": list(PLACEMENT_ORDER),
        "placements_per_case": {str(count): PLACEMENT_ORDER.count(count) for count in UNIQUE_COUNTS},
        "replays_per_placement": REPLAYS,
        "timed_projection_calls_per_case": len(PLACEMENT_ORDER) // len(UNIQUE_COUNTS) * REPLAYS * LAYERS,
        "gpu_predicated_projection_calls_per_case_including_proof": LAYERS +
            len(PLACEMENT_ORDER) // len(UNIQUE_COUNTS) * REPLAYS * LAYERS,
        "gpu_decoded_weights_bit_exact": exact_weights,
        "decoded_weight_words_checked": LAYERS * 2 * N * K,
        "nan_poisoned_graph_replay_finite_by_case": finite_by_case,
        "fp32_exact_against_dense_reference_by_case": exact_by_case,
        "bf16_bits_exact_against_dense_reference_by_case": bf16_exact_by_case,
        "output_elements_checked_per_case": LAYERS * 2 * 2 * N,
        "dense_weight_bytes_all_32_layers": dense_bytes,
        "encoded_bytes_all_32_layers_including_metadata": sum(
            item["distinct_encoded_bytes"]["total_including_metadata"]
            for item in encoded_cases.values() if item["unique_layer_count"] == LAYERS),
        "working_set_cases": encoded_cases,
        "gpu_tensor_peak_allocated_bytes": peak_bytes,
        "gpu_tensor_allocation_limit_bytes": MEMORY_LIMIT,
        "gpu_device_name": torch.cuda.get_device_name(device),
        "timing_scope": "CUDA-event time for 32 consecutive predicated projection calls; inputs and all 32 encoded GPU layers remain resident; only call-to-layer mapping changes",
        "interpretation_rule": "A large verified advantage for smaller working sets supports cache pressure as an important factor. No such advantage leaves decode or instruction work as candidates. These timings do not assign a causal percentage.",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
