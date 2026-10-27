#!/usr/bin/env python3
"""CPU-preflight or run two fixed PV schedules across the frozen 32-layer corpus."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = ROOT / "decode360-pv-native-corpus-r1" / "results-r1"
CORPUS_MANIFEST = CORPUS / "corpus-manifest.json"
NATIVE_CAPTURE = ROOT / "decode360-pv-launch-args-r2" / "results-r1" / "launch-args.jsonl"
NATIVE_VALIDATION = ROOT / "decode360-pv-launch-args-r2" / "results-r1" / "launch-args.validation.json"
R3_SOURCE = ROOT / "decode360-pv-long-probe-r2" / "pv_order_probe.cu"
R3_LIBRARY = ROOT / "decode360-pv-long-probe-r2" / "build" / "libpv_reduce_probe_long_r2.so"
R3_GPU_REPORT = ROOT / "decode360-pv-causal-probe-r3" / "results-r1" / "gpu-probe.json"
R3_REPLAY = ROOT / "decode360-pv-native-corpus-r1" / "results-r1" / "pv-schedule-replay.json"
R21_SOURCE = ROOT / "decode-360-cuda-r21" / "block360_kernel.inc"
R3_SOURCE_SHA256 = "4e43e6aa0fc869d58ec65c89e25ac5f02b1a053fe3764d9e3c4c3aa71ca2b68c"
R3_LIBRARY_SHA256 = "bafc1717c3eab4a2798cad750d0f2ba7df0de6ed3e9f1d0d106889b3f27fdeca"
R3_GPU_REPORT_SHA256 = "777c850183b732af162e6bbf22aea0716992c5bb8328fe92ed6bc6874930f5d1"
R21_SOURCE_SHA256 = "4c1a684303d0329238ab5f3058c7f4058ae0fe36c9b13a44adc3dbb649d8be55"
EXPECTED_SHAPES = {
    "native_probabilities": (2, 15, 8001),
    "native_value": (2, 5, 8001, 64),
    "native_pv_output": (2, 15, 64),
}
CANDIDATE = {"ygroups": 8, "chunk": 64, "accumulators": 1,
             "assignment": 0, "acc_tree": 0, "y_tree": 2, "use_fma": 1}
R21_CONTROL = {"ygroups": 8, "chunk": 8, "accumulators": 4,
              "assignment": 0, "acc_tree": 0, "y_tree": 2, "use_fma": 0}
ABI_NAMES = ["stream", "probabilities", "values", "output", "rows", "tokens", "dims",
             "ygroups", "chunk", "accumulators", "assignment", "acc_tree", "y_tree", "use_fma"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def load_layer(layer: int) -> tuple[dict[str, Any], Path]:
    path = CORPUS / f"layer-{layer:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for name, shape in EXPECTED_SHAPES.items():
        tensor = payload.get(name)
        if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise RuntimeError(f"layer {layer} {name} schema mismatch")
        if tensor.device.type != "cpu":
            raise RuntimeError(f"layer {layer} {name} did not load on CPU")
    if payload.get("case_id") != "seed777-b2-ctx8000" or payload.get("layer") != layer:
        raise RuntimeError(f"layer {layer} corpus identity mismatch")
    return payload, path


def raw_u32(raw: bytes, offset: int) -> int:
    return int.from_bytes(raw[offset:offset + 4], "little")


def native_parameter_map_preflight() -> dict[str, Any]:
    rows = [json.loads(line) for line in NATIVE_CAPTURE.read_text().splitlines() if line.strip()]
    launches = [row for row in rows if row.get("event") == "gemvx_launch"]
    validation = json.loads(NATIVE_VALIDATION.read_text())
    if len(launches) != 1 or validation.get("status") != "capture_validation_ok":
        raise RuntimeError("R2 native GEMVX capture is not a single successful record")
    launch = launches[0]
    formal = launch["formal_args"]
    if (launch.get("formal_arg_query_api"), launch.get("kernel_query_cuda_result"),
            launch.get("query_end_cuda_result"), launch.get("formal_arg_count")) != (
            "cuKernelGetParamInfo", 0, 1, 1):
        raise RuntimeError("R2 parameter query did not capture one CUkernel formal argument")
    if len(formal) != 1 or formal[0].get("device_offset") != 0 or formal[0].get("size") != 152:
        raise RuntimeError("R2 formal argument layout differs from the reviewed 152-byte record")
    raw = bytes.fromhex(formal[0]["hex"])
    mapping = {"0x178": {"raw_offset": 0x18, "expected": 8001},
               "0x1a0": {"raw_offset": 0x40, "expected": 64},
               "0x1a4": {"raw_offset": 0x44, "expected": 8001},
               "0x1a8": {"raw_offset": 0x48, "expected": 64},
               "0x1ec": {"raw_offset": 0x8C, "expected": 64},
               "0x1f0": {"raw_offset": 0x90, "expected": 1}}
    for name, entry in mapping.items():
        entry["captured"] = raw_u32(raw, entry["raw_offset"])
        if entry["captured"] != entry["expected"]:
            raise RuntimeError(f"parameter/SASS map mismatch at c[0][{name}]")
    if raw[0:8] == bytes(8):
        raise RuntimeError("first formal pointer pair unexpectedly zero")
    if launch.get("grid") != [2, 1, 30] or launch.get("block") != [32, 8, 1]:
        raise RuntimeError("R2 launch geometry changed")
    if launch.get("dynamic_shared_bytes") != 2176:
        raise RuntimeError("R2 dynamic shared-memory size changed")
    return {
        "status": validation["status"],
        "formal_argument_offset": formal[0]["device_offset"],
        "formal_argument_size": formal[0]["size"],
        "candidate_c0_formal_base": 0x160,
        "base_status": "cross-checked inference from pointer-pair and six shape/stride fields; cubin nv.info not yet independently decoded",
        "field_map": mapping,
        "tile_span": mapping["0x1ec"]["captured"] * launch["block"][1],
        "native_pv_exact_elements": validation["replay"]["native_exact_elements"],
    }


def cpu_preflight() -> dict[str, Any]:
    manifest = json.loads(CORPUS_MANIFEST.read_text())
    if (manifest.get("schema") != "decode360-pv-native-corpus-r1" or
            manifest.get("status") != "captured all layers" or
            manifest.get("case_id") != "seed777-b2-ctx8000" or
            set(manifest.get("layers", {})) != {str(i) for i in range(32)}):
        raise RuntimeError("native PV corpus is not the expected complete 32-layer capture")
    corpus_files = []
    for layer in range(32):
        payload, path = load_layer(layer)
        expected_hash = manifest["layers"][str(layer)]["file_sha256"]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"layer {layer} corpus hash mismatch")
        corpus_files.append({"layer": layer, "file": path.name, "sha256": actual_hash})
        del payload

    r3_report = json.loads(R3_GPU_REPORT.read_text())
    if sha256(R3_SOURCE) != R3_SOURCE_SHA256 or sha256(R3_LIBRARY) != R3_LIBRARY_SHA256:
        raise RuntimeError("frozen R3 source or probe library hash changed")
    if sha256(R3_GPU_REPORT) != R3_GPU_REPORT_SHA256:
        raise RuntimeError("R3 winner report hash changed")
    if r3_report.get("library_sha256") != R3_LIBRARY_SHA256 or r3_report.get("cuda_source_sha256") != R3_SOURCE_SHA256:
        raise RuntimeError("R3 provenance hashes do not describe the reused probe")
    if r3_report.get("abi_names") != ABI_NAMES:
        raise RuntimeError("R3 probe ABI does not have the expected 14 arguments")
    r3_source = R3_SOURCE.read_text()
    for required in ("chunk < 1", "tile += ygroups * chunk", "tile + y * chunk",
                     "j < chunk", "__fmaf_rn", "float acc[8]", "float* output"):
        if required not in r3_source:
            raise RuntimeError(f"R3 probe source lacks expected generic chunk/FMA feature: {required}")
    if "chunk <= 32" in r3_source or "chunk > 32" in r3_source:
        raise RuntimeError("reused R3 probe has an unexpected chunk upper bound")

    r3_sweep = json.loads(R3_REPLAY.read_text())
    if r3_sweep.get("layer_count") != 32 or r3_sweep.get("schedules_tested_per_layer") != 90:
        raise RuntimeError("R3 32-layer schedule replay provenance changed")
    sweep_labels = r3_report.get("selected_case_intersection_winners", [])
    if len(sweep_labels) != 90 or any("chunk64" in label for label in sweep_labels):
        raise RuntimeError("R3 prior winner set unexpectedly includes the new chunk-64 case")

    if sha256(R21_SOURCE) != R21_SOURCE_SHA256:
        raise RuntimeError("frozen R21 control source hash changed")
    r21_source = R21_SOURCE.read_text()
    for required in ("constexpr int y_chunk = 8", "(j & 3) == 0", "add_rn(acc, mul_rn(prob, value))",
                     "partial = add_rn(partial, acc0)", "const float result = add_rn(left, right)"):
        if required not in r21_source:
            raise RuntimeError(f"R21 control source lacks expected schedule feature: {required}")

    parameter_map = native_parameter_map_preflight()
    if parameter_map["tile_span"] != 512:
        raise RuntimeError("native SASS tile span is not 512")
    return {
        "status": "cpu_preflight_ok",
        "gpu_touched": False,
        "cuda_api_called_by_script": False,
        "corpus_case": manifest["case_id"],
        "corpus_layer_count": len(corpus_files),
        "corpus_files": corpus_files,
        "native_parameter_map": parameter_map,
        "r3_library_sha256": R3_LIBRARY_SHA256,
        "r3_cuda_source_sha256": R3_SOURCE_SHA256,
        "r3_winner_report_sha256": R3_GPU_REPORT_SHA256,
        "r3_probe_abi_arg_count": len(ABI_NAMES),
        "prior_chunk_max": 32,
        "new_native_justified_chunk": 64,
        "candidate": CANDIDATE,
        "r21_control": R21_CONTROL,
        "r21_source_sha256": R21_SOURCE_SHA256,
        "control_note": "R21 is a fixed non-FMA, four-accumulator schedule control; it is not another sweep arm.",
    }


def configure_probe() -> Any:
    library = ctypes.CDLL(str(R3_LIBRARY))
    call = library.pv_reduce_probe
    call.restype = ctypes.c_int
    call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 10
    return call


def call_schedule(call: Any, stream: int, probabilities: torch.Tensor, values: torch.Tensor,
                  arm: dict[str, int]) -> torch.Tensor:
    output = torch.empty((30, 64), dtype=torch.float32, device=probabilities.device)
    status = call(ctypes.c_void_p(stream), ctypes.c_void_p(probabilities.data_ptr()),
                  ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(output.data_ptr()),
                  30, 8001, 64, arm["ygroups"], arm["chunk"], arm["accumulators"],
                  arm["assignment"], arm["acc_tree"], arm["y_tree"], arm["use_fma"])
    if status != 0:
        raise RuntimeError(f"R3 probe returned CUDA status {status} for config {arm}")
    return output


def run_all_layers(output_path: Path) -> dict[str, Any]:
    preflight = cpu_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    call = configure_probe()
    device = torch.device("cuda:0")
    layer_rows = []
    for layer in range(32):
        payload, path = load_layer(layer)
        probabilities = payload["native_probabilities"].contiguous().to(device)
        values = payload["native_value"].repeat_interleave(3, dim=1).contiguous().to(device)
        native = payload["native_pv_output"].reshape(30, 64).contiguous().to(device)
        with torch.no_grad():
            fresh = torch.matmul(probabilities.unsqueeze(-2), values).squeeze(-2).reshape(30, 64)
        torch.cuda.synchronize(device)
        fresh_mismatches = int((fresh != native).sum().item())
        row: dict[str, Any] = {
            "layer": layer,
            "corpus_file": path.name,
            "corpus_sha256": sha256(path),
            "fresh_torch_mismatches": fresh_mismatches,
        }
        if fresh_mismatches == 0:
            for label, arm in (("chunk64_native_candidate", CANDIDATE), ("r21_fixed_control", R21_CONTROL)):
                fp32 = call_schedule(call, torch.cuda.current_stream(device).cuda_stream,
                                     probabilities, values, arm)
                torch.cuda.synchronize(device)
                bf16 = fp32.to(torch.bfloat16)
                row[label] = {
                    "bf16_mismatches": int((bf16 != native).sum().item()),
                    "fp32_checksum": hashlib.sha256(fp32.contiguous().cpu().numpy().tobytes()).hexdigest(),
                }
        else:
            row["candidate_status"] = "skipped_after_fresh_torch_gate_failure"
        layer_rows.append(row)
        del payload, probabilities, values, native

    candidate_exact = sum(row.get("chunk64_native_candidate", {}).get("bf16_mismatches") == 0 for row in layer_rows)
    control_exact = sum(row.get("r21_fixed_control", {}).get("bf16_mismatches") == 0 for row in layer_rows)
    fresh_exact = sum(row.get("fresh_torch_mismatches") == 0 for row in layer_rows)
    report = {
        "status": "complete",
        "device": str(device),
        "torch_profiler_used": False,
        "layer_count": len(layer_rows),
        "fresh_torch_exact_layers": fresh_exact,
        "chunk64_candidate_exact_layers": candidate_exact,
        "r21_control_exact_layers": control_exact,
        "candidate": CANDIDATE,
        "r21_control": R21_CONTROL,
        "native_parameter_map": preflight["native_parameter_map"],
        "r3_library_sha256": R3_LIBRARY_SHA256,
        "r3_cuda_source_sha256": R3_SOURCE_SHA256,
        "r21_source_sha256": R21_SOURCE_SHA256,
        "layer_results": layer_rows,
    }
    write_private(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cpu_preflight and args.run_cuda:
        parser.error("choose either --cpu-preflight or --run-cuda")
    if args.run_cuda:
        output = args.output or HERE / "results-r1" / "chunk64-replay.json"
        report = run_all_layers(output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    report = cpu_preflight()
    if args.output:
        write_private(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
