#!/usr/bin/env python3
"""Compare the R30 paired gate/up path with cross-row cp.async scheduling."""
import argparse
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
PTXAS_LOG = HERE / "ptxas.log"
LAYERS = 32
N, K = 2560, 960
PLACEMENTS = 10
REPLAYS = 5
MEMORY_LIMIT = 3 * 1024**3


def build_argv(ptxas_verbose=False):
    argv = [NVCC, "-std=c++17", "-O3", "-arch=sm_89", "-Xcompiler=-fPIC", "--shared",
            str(HERE / "projection_probe.cu"), "-o", str(LIB)]
    if ptxas_verbose:
        argv.insert(1, "-Xptxas=-v")
    return argv


def compile_probe(ptxas_verbose=False, force=False):
    if LIB.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {LIB}; pass --force to replace it")
    if ptxas_verbose and PTXAS_LOG.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {PTXAS_LOG}; pass --force to replace it")
    LIB.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(build_argv(ptxas_verbose), check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ptxas_verbose:
        PTXAS_LOG.write_text(result.stdout)
    return result.stdout


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(rc, label):
    if rc:
        raise RuntimeError(f"{label} returned CUDA error {rc}")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true",
                      help="compile and exit without importing torch, loading weights, or using a GPU")
    mode.add_argument("--skip-build", action="store_true", help="use existing shared library")
    parser.add_argument("--ptxas-verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow replacing build output or results.json")
    args = parser.parse_args()
    build_output = ""
    if not args.skip_build:
        build_output = compile_probe(args.ptxas_verbose, args.force)
        if build_output:
            print(build_output, end="")
    if args.build_only:
        return
    if not LIB.is_file():
        raise FileNotFoundError(f"shared library missing: {LIB}")
    result_path = HERE / "results.json"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {result_path}; pass --force to replace it")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"cached checkpoint missing: {MODEL_FILE}")

    import ctypes
    import torch
    from safetensors import safe_open

    lib = ctypes.CDLL(str(LIB))
    vp = ctypes.c_void_p
    lib.run_gateup.argtypes = [ctypes.c_int, vp, vp, vp, vp]
    lib.run_gateup.restype = ctypes.c_int
    device = torch.device("cuda")
    torch.manual_seed(341)
    weights = []
    with safe_open(str(MODEL_FILE), framework="pt", device="cpu") as model:
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}.mlp."
            gate = model.get_tensor(prefix + "gate_proj.weight")
            up = model.get_tensor(prefix + "up_proj.weight")
            if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
                raise RuntimeError(f"layer {layer} gate/up weights must be BF16")
            if tuple(gate.shape) != (N, K) or tuple(up.shape) != (N, K):
                raise RuntimeError(f"layer {layer} gate/up shapes must each be [{N},{K}]")
            weights.append(torch.cat((gate, up), dim=0).contiguous().to(device))
    allocated = torch.cuda.memory_allocated(device)
    if allocated >= MEMORY_LIMIT:
        raise RuntimeError(f"probe allocation is {allocated} bytes, limit is {MEMORY_LIMIT}")

    report = {"model_file": str(MODEL_FILE), "model_revision": MODEL.name,
              "model_sha256": sha256_file(MODEL_FILE),
              "kernel_source_sha256": sha256_file(HERE / "projection_probe.cu"),
              "harness_sha256": sha256_file(Path(__file__)),
              "shared_library_sha256": sha256_file(LIB),
              "ptxas_verbose": args.ptxas_verbose,
              "ptxas_output": build_output if args.ptxas_verbose else None,
              "layers": LAYERS, "shape": [N, K], "B": 2,
              "gate_up_weight_bytes": LAYERS * 2 * N * K * 2,
              "tensor_allocated_bytes_before_inputs_outputs": allocated,
              "timing": "CUDA graph over 32 distinct layer gate/up weights; microseconds per layer call",
              "activation": "seeded synthetic BF16 input, expanded to FP32 shared memory",
              "baseline_probe_source_sha256": "05f4c790b240bb8322a559ca3e34a8c84f88d76b40a99a56b80061c4a4aeed27",
              "async_plan": "flatten each warp's row and four K-tile tasks into one persistent pipeline; 2 buffers x 8 warps x 2 rows x 32 lanes x 16 B = 16 KiB static shared",
              "arms": {"baseline": "R30 __ldg uint4 paired G/U rows, NR2, four FP32 FMA accumulators",
                       "cp_async": "PTX cp.async.cg 16 B paired G/U rows; persistent warp task stream prefetches across row boundaries"},
              "cases": []}

    x = torch.randn((2, K), device="cpu", dtype=torch.bfloat16, generator=torch.Generator().manual_seed(341)).to(device)
    outputs = [torch.full((LAYERS, 2, 2 * N), float("nan"), device=device, dtype=torch.float32)
               for _ in range(2)]

    def enqueue(arm, output):
        stream = torch.cuda.current_stream().cuda_stream
        for layer, weight in enumerate(weights):
            check(lib.run_gateup(arm, weight.data_ptr(), x.data_ptr(),
                                 output[layer].data_ptr(), stream), "run_gateup")

    graphs = []
    for arm in (0, 1):
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
    graphs[0].replay()
    torch.cuda.synchronize()
    graphs[1].replay()
    torch.cuda.synchronize()
    finite = [bool(torch.isfinite(output).all().item()) for output in outputs]
    if not all(finite):
        raise RuntimeError(f"NaN-poisoned graph replay left unwritten output: finite={finite}")
    fp32_exact = torch.equal(outputs[0], outputs[1])
    bf16_exact = torch.equal(outputs[0].to(torch.bfloat16).view(torch.int16),
                             outputs[1].to(torch.bfloat16).view(torch.int16))
    if not fp32_exact or not bf16_exact:
        raise RuntimeError("baseline and cp.async outputs differ before timing")

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

    sequence = [0, 1] * (PLACEMENTS // 2)
    times = {0: [], 1: []}
    for arm in sequence:
        times[arm].append(time_graph(graphs[arm]))
    graphs[0].replay(); graphs[1].replay(); torch.cuda.synchronize()
    if not torch.equal(outputs[0], outputs[1]):
        raise RuntimeError("outputs changed or diverged during alternating timing")
    report.update({"graph_replay_output_write_verified_per_arm": finite,
                   "fp32_exact": fp32_exact, "bf16_bits_exact": bf16_exact,
                   "output_elements_checked_per_arm": LAYERS * 2 * 2 * N,
                   "graph_captures": 2, "timed_placements_per_arm": PLACEMENTS // 2,
                   "replays_per_placement": REPLAYS,
                   "timed_projection_calls_per_arm": LAYERS * (PLACEMENTS // 2) * REPLAYS,
                   "untimed_explicit_projection_calls_per_arm": LAYERS * 3,
                   "validation_error_count": 0,
                   "baseline_us_median_range": [statistics.median(map(statistics.median, times[0])),
                                                 min(map(min, times[0])), max(map(max, times[0]))],
                   "cp_async_us_median_range": [statistics.median(map(statistics.median, times[1])),
                                                 min(map(min, times[1])), max(map(max, times[1]))],
                   "placement_raw_event_samples_us_per_layer": {
                       "baseline": times[0], "cp_async": times[1]}})
    report["tensor_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(device)
    if report["tensor_allocated_peak_bytes"] >= MEMORY_LIMIT:
        raise RuntimeError("probe peak allocation exceeded 3 GiB")
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
