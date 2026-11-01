#!/usr/bin/env python3
"""Compare FP32 and packed-BF16 shared activation staging on real SmolLM2-360M weights."""
import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import os
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
MODEL = Path("/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951")
MODEL_FILE = MODEL / "model.safetensors"
NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
LIB = HERE / "build/libprojection_probe.so"
LAYERS = 32
SHAPES = [("qkv", 0, 1600, 960, 1), ("o", 1, 960, 960, 1),
          ("gate_up_pair", 2, 2560, 960, 2), ("down", 3, 960, 2560, 1)]
PLACEMENTS = 10
REPLAYS = 5
MEMORY_LIMIT = 3 * 1024**3


def build_argv(ptxas_verbose=False):
    argv = [NVCC, "-std=c++17", "-O3", "-arch=sm_89", "-Xcompiler=-fPIC", "--shared",
            str(HERE / "projection_probe.cu"), "-o", str(LIB)]
    if ptxas_verbose:
        argv.insert(1, "-Xptxas=-v")
    return argv


def compile_probe(ptxas_verbose=False):
    LIB.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(build_argv(ptxas_verbose), check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout


def load_lib():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(MODEL_FILE)
    lib = ctypes.CDLL(str(LIB))
    vp = ctypes.c_void_p
    lib.run_projection.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, vp, vp, vp, vp]
    lib.run_projection.restype = ctypes.c_int
    return lib


def load_weights(torch_device):
    weights = {shape[0]: [] for shape in SHAPES}
    with safe_open(str(MODEL_FILE), framework="pt", device="cpu") as model:
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}."
            get = model.get_tensor
            qkv = torch.cat([get(prefix + f"self_attn.{name}_proj.weight")
                             for name in ("q", "k", "v")], dim=0)
            o = get(prefix + "self_attn.o_proj.weight")
            gate_up = torch.cat((get(prefix + "mlp.gate_proj.weight"),
                                 get(prefix + "mlp.up_proj.weight")), dim=0)
            down = get(prefix + "mlp.down_proj.weight")
            for name, tensor in (("qkv", qkv), ("o", o), ("gate_up_pair", gate_up), ("down", down)):
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(f"checkpoint {name} layer {layer} is {tensor.dtype}, expected BF16")
                weights[name].append(tensor.contiguous().to(torch_device))
    return weights


def check(rc, label):
    if rc:
        raise RuntimeError(f"{label} returned CUDA error {rc}")


def graph_for(lib, shape_id, batch, packed, layer_weights, x, output):
    def enqueue():
        stream = torch.cuda.current_stream().cuda_stream
        for layer, weight in enumerate(layer_weights):
            out = output[layer].data_ptr()
            check(lib.run_projection(shape_id, batch, packed, weight.data_ptr(), x.data_ptr(),
                                     out, stream), "run_projection")

    enqueue()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        enqueue()
    torch.cuda.synchronize()
    return graph


def time_graph(graph):
    samples = []
    for _ in range(REPLAYS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / LAYERS)
    return statistics.median(samples), min(samples), max(samples), samples


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true",
                      help="compile the probe and exit without loading the model or using a GPU")
    mode.add_argument("--skip-build", action="store_true",
                      help="use the existing shared library without invoking nvcc")
    parser.add_argument("--ptxas-verbose", action="store_true",
                        help="request ptxas register, spill, and shared-memory reporting")
    args = parser.parse_args()
    build_output = ""
    if not args.skip_build:
        build_output = compile_probe(args.ptxas_verbose)
        if build_output:
            print(build_output, end="")
    if args.build_only:
        return
    lib = load_lib()
    device = torch.device("cuda")
    torch.manual_seed(34)
    weights = load_weights(device)
    allocated = torch.cuda.memory_allocated(device)
    if allocated >= MEMORY_LIMIT:
        raise RuntimeError(f"probe tensor allocation is {allocated} bytes, limit is {MEMORY_LIMIT}")
    report = {"model_file": str(MODEL_FILE), "model_revision": MODEL.name,
              "model_sha256": sha256_file(MODEL_FILE),
              "kernel_source_sha256": sha256_file(HERE / "projection_probe.cu"),
              "harness_sha256": sha256_file(Path(__file__)),
              "shared_library_sha256": sha256_file(LIB),
              "ptxas_verbose": args.ptxas_verbose,
              "ptxas_output": build_output if args.ptxas_verbose else None,
              "layers": LAYERS, "placements_per_arm": PLACEMENTS // 2,
              "replays_per_placement": REPLAYS, "tensor_allocated_bytes": allocated,
              "timing": "CUDA graph with 32 distinct layer weights; microseconds per layer call",
              "activations": "seeded synthetic BF16 inputs; all values exactly BF16-representable",
              "cases": []}

    for name, shape_id, n, k, nr in SHAPES:
        for batch in (1, 2):
            layer_weights = weights[name]
            x = torch.randn((batch, k), device=device, dtype=torch.bfloat16)
            y_fp = torch.full((LAYERS, batch, n * nr), float("nan"), device=device,
                              dtype=torch.float32)
            y_bf = torch.full_like(y_fp, float("nan"))
            ga = graph_for(lib, shape_id, batch, 0, layer_weights, x, y_fp)
            gb = graph_for(lib, shape_id, batch, 1, layer_weights, x, y_bf)
            y_fp.fill_(float("nan"))
            y_bf.fill_(float("nan"))
            torch.cuda.synchronize()
            ga.replay(); torch.cuda.synchronize()
            gb.replay(); torch.cuda.synchronize()
            if not torch.isfinite(y_fp).all() or not torch.isfinite(y_bf).all():
                raise RuntimeError(f"{name} B{batch}: one or more output elements were not written")
            exact = torch.equal(y_fp, y_bf)
            bf16_exact = torch.equal(y_fp.to(torch.bfloat16).view(torch.int16),
                                     y_bf.to(torch.bfloat16).view(torch.int16))
            if not exact or not bf16_exact:
                raise RuntimeError(f"{name} B{batch}: FP32 values or BF16 output bits differ")
            sequence = [0, 1] * (PLACEMENTS // 2)
            times = {0: [], 1: []}
            for arm in sequence:
                times[arm].append(time_graph((ga, gb)[arm]))
            ga.replay(); gb.replay(); torch.cuda.synchronize()
            if not torch.equal(y_fp, y_bf):
                raise RuntimeError(f"{name} B{batch}: alternating timings changed outputs")
            report["cases"].append({"shape": name, "N": n, "K": k, "rows_per_task": nr,
                "B": batch, "fp32_exact": exact, "bf16_bits_exact": bf16_exact,
                "fp32_us_median_range": [statistics.median(v[0] for v in times[0]),
                                          min(v[1] for v in times[0]), max(v[2] for v in times[0])],
                "packed_bf16_us_median_range": [statistics.median(v[0] for v in times[1]),
                                                min(v[1] for v in times[1]), max(v[2] for v in times[1])],
                "placement_medians_us_per_layer": {
                    "fp32": [placement[0] for placement in times[0]],
                    "packed_bf16": [placement[0] for placement in times[1]]},
                "raw_event_samples_us_per_layer_by_placement": {
                    "fp32": [placement[3] for placement in times[0]],
                    "packed_bf16": [placement[3] for placement in times[1]]},
                "completed_graph_replays": {"fp32": len(times[0]) * REPLAYS,
                                             "packed_bf16": len(times[1]) * REPLAYS},
                "output_elements_checked_per_arm": LAYERS * batch * n * nr,
                "graph_replay_output_write_verified_per_arm": True,
                "timed_projection_calls_per_arm": LAYERS * (PLACEMENTS // 2) * REPLAYS,
                "untimed_explicit_projection_calls_per_arm": LAYERS * 3,
                "validation_error_count": 0})
    report["tensor_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(device)
    if report["tensor_allocated_peak_bytes"] >= MEMORY_LIMIT:
        raise RuntimeError("probe peak tensor allocation exceeded 3 GiB")
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
