#!/usr/bin/env python3
"""Compare every single-layer anchor with the all-layer working set."""
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
R1_HARNESS = HERE / "harness_cache_working_set.py"
EXPECTED_R1_HARNESS_SHA256 = "7fdadbf5195f94b178e4bf56598c8fac0361f949b332a8c9bdc5e475c9c2b9b3"
LAYERS, N, K, CHUNK = 32, 2560, 960, 8
CASE_NAMES = tuple(f"warm_layer_{layer}" for layer in range(LAYERS)) + ("all_layers",)
PLACEMENT_ORDER = (CASE_NAMES + tuple(reversed(CASE_NAMES))) * 3
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
    parser.add_argument("--output", type=Path, default=HERE / "results-r2.json")
    args = parser.parse_args()
    result_path = args.output.resolve()
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    if not SOURCE.is_file() or not LIB.is_file() or not FROZEN_HARNESS.is_file() or not R1_HARNESS.is_file():
        raise FileNotFoundError("frozen source, DSO, encoder harness, or R1 harness is missing")
    source_sha256 = sha256_file(SOURCE)
    library_sha256 = sha256_file(LIB)
    frozen_harness_sha256 = sha256_file(FROZEN_HARNESS)
    r1_harness_sha256 = sha256_file(R1_HARNESS)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"frozen source hash mismatch: {source_sha256}")
    if library_sha256 != EXPECTED_LIB_SHA256:
        raise RuntimeError(f"frozen DSO hash mismatch: {library_sha256}")
    if frozen_harness_sha256 != EXPECTED_FROZEN_HARNESS_SHA256:
        raise RuntimeError(f"frozen encoder harness hash mismatch: {frozen_harness_sha256}")
    if r1_harness_sha256 != EXPECTED_R1_HARNESS_SHA256:
        raise RuntimeError(f"R1 harness hash mismatch: {r1_harness_sha256}")
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
    outputs = {name: torch.full((LAYERS, 2, 2 * N), float("nan"), device=device,
                                dtype=torch.float32) for name in CASE_NAMES}
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
    for name in CASE_NAMES:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_stream = torch.cuda.current_stream().cuda_stream
            for call in range(LAYERS):
                layer = call if name == "all_layers" else int(name.rsplit("_", 1)[1])
                check(lib.run_gateup_lossless_predicated(
                    pointer(plane_ptr, layer * plane_layer_bytes),
                    pointer(headers_ptr, layer * header_layer_bytes),
                    pointer(raw_ptr, raw_prefix[layer]),
                    pointer(offsets_ptr, layer * offset_layer_bytes), x.data_ptr(),
                    outputs[name][call].data_ptr(), capture_stream),
                    "run_gateup_lossless_predicated")
        graphs[name] = graph
    torch.cuda.synchronize()

    for output in outputs.values():
        output.fill_(float("nan"))
    torch.cuda.synchronize()
    for name in CASE_NAMES:
        graphs[name].replay()
    torch.cuda.synchronize()
    exact_by_case = {}
    bf16_exact_by_case = {}
    finite_by_case = {}
    for name in CASE_NAMES:
        output = outputs[name]
        call_layer_map = list(range(LAYERS)) if name == "all_layers" else [
            int(name.rsplit("_", 1)[1])] * LAYERS
        finite_by_case[name] = bool(torch.isfinite(output).all().item())
        exact_by_case[name] = all(torch.equal(output[call], reference[layer])
                                   for call, layer in enumerate(call_layer_map))
        bf16_exact_by_case[name] = all(
            torch.equal(output[call].to(torch.bfloat16).view(torch.int16),
                        reference[layer].to(torch.bfloat16).view(torch.int16))
            for call, layer in enumerate(call_layer_map))
        if (not finite_by_case[name] or not exact_by_case[name] or
                not bf16_exact_by_case[name]):
            raise RuntimeError(f"NaN-poisoned graph or output mismatch for {name}")

    raw_samples = {name: [] for name in CASE_NAMES}
    for name in PLACEMENT_ORDER:
        graphs[name].replay()
        torch.cuda.synchronize()
        values = []
        for _ in range(REPLAYS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            torch.cuda.synchronize()
            values.append(start.elapsed_time(end) * 1000)
        raw_samples[name].append(values)
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
    for name in CASE_NAMES:
        layers = list(range(LAYERS)) if name == "all_layers" else [int(name.rsplit("_", 1)[1])]
        encoded_cases[name] = {
            "call_layer_map": list(range(LAYERS)) if name == "all_layers" else layers * LAYERS,
            "unique_layers_touched": layers,
            "unique_layer_count": len(layers),
            "distinct_encoded_bytes": {
                "planes": len(layers) * 2 * N * K,
                "headers": len(layers) * 2 * N * (K // CHUNK) * 4,
                "raw_escapes": sum(raw_layer_bytes[layer] for layer in layers),
                "csr_row_offsets": len(layers) * (2 * N + 1) * 4,
                "total_including_metadata": len(layers) * (2 * N * K +
                    2 * N * (K // CHUNK) * 4 + (2 * N + 1) * 4) +
                    sum(raw_layer_bytes[layer] for layer in layers),
            },
            "placement_prewarm_graph_replays": len(raw_samples[name]),
            "placement_prewarm_projection_calls": len(raw_samples[name]) * LAYERS,
            "timed_projection_calls": len(raw_samples[name]) * REPLAYS * LAYERS,
            "timing": timing_summary(raw_samples[name]),
        }

    peak_bytes = torch.cuda.max_memory_allocated(device)
    if peak_bytes >= MEMORY_LIMIT:
        raise RuntimeError(f"probe peak allocation {peak_bytes} reached the 3 GiB limit")
    anchor_medians = [encoded_cases[name]["timing"]["median_of_placement_medians_us_per_call"]
                      for name in CASE_NAMES[:-1]]
    report = {
        "diagnostic": "rotated single-layer anchors versus all layers; not a full-model or RL speedup",
        "model_file": str(MODEL_FILE),
        "model_revision": MODEL.name,
        "model_sha256": sha256_file(MODEL_FILE),
        "source_file": str(SOURCE),
        "source_sha256": source_sha256,
        "shared_library": str(LIB),
        "shared_library_sha256": library_sha256,
        "harness_sha256": sha256_file(Path(__file__)),
        "frozen_encoder_harness_sha256": frozen_harness_sha256,
        "r1_harness_sha256": r1_harness_sha256,
        "layers_loaded_to_gpu": LAYERS,
        "calls_per_graph": LAYERS,
        "output_shape_per_case": [LAYERS, 2, 2 * N],
        "case_names": list(CASE_NAMES),
        "placement_order_cases": list(PLACEMENT_ORDER),
        "placements_per_case": {name: PLACEMENT_ORDER.count(name) for name in CASE_NAMES},
        "replays_per_placement": REPLAYS,
        "prewarm_graph_replays_per_case": 6,
        "prewarm_projection_calls_per_case": 6 * LAYERS,
        "timed_projection_calls_per_case": 6 * REPLAYS * LAYERS,
        "correctness_graph_replays_per_case": 1,
        "correctness_projection_calls_per_case": LAYERS,
        "gpu_predicated_projection_calls_per_case_including_proof_prewarm_timing":
            (1 + 6 + 6 * REPLAYS) * LAYERS,
        "gpu_decoded_weights_bit_exact": exact_weights,
        "decoded_weight_words_checked": LAYERS * 2 * N * K,
        "nan_poisoned_graph_replay_finite_by_case": finite_by_case,
        "fp32_exact_against_dense_reference_by_case": exact_by_case,
        "bf16_bits_exact_against_dense_reference_by_case": bf16_exact_by_case,
        "output_elements_checked_per_case": LAYERS * 2 * 2 * N,
        "dense_weight_bytes_all_32_layers": dense_bytes,
        "all_layers_encoded_bytes_including_metadata": encoded_cases["all_layers"][
            "distinct_encoded_bytes"]["total_including_metadata"],
        "working_set_cases": encoded_cases,
        "warm_anchor_median_us_per_call_min": min(anchor_medians),
        "warm_anchor_median_us_per_call_median": statistics.median(anchor_medians),
        "warm_anchor_median_us_per_call_max": max(anchor_medians),
        "all_layers_median_us_per_call": encoded_cases["all_layers"]["timing"][
            "median_of_placement_medians_us_per_call"],
        "gpu_tensor_peak_allocated_bytes": peak_bytes,
        "gpu_tensor_allocation_limit_bytes": MEMORY_LIMIT,
        "gpu_device_name": torch.cuda.get_device_name(device),
        "timing_scope": "CUDA-event time for 32 consecutive predicated projection calls; all 32 encoded GPU layers remain resident; warm-layer cases repeatedly access one anchored layer and all_layers visits each layer once",
        "interpretation_rule": "Compare all 32 warm-layer anchors with all_layers before interpreting the earlier working-set result. Per-layer variation can reflect weight content or address effects. This diagnostic does not assign a causal cache percentage.",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
