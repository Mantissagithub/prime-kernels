#!/usr/bin/env python3
"""Profile refresh cost and exactness of the opt-in 360M lossless GU cache."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TINY = REPO.parent
SNAPSHOT = Path(
    "/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct"
    "/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951"
)
PACKAGE = TINY / "prime-kernels/prime_kernels/lossless_bf16/__init__.py"
CUDA_SOURCE = PACKAGE.parent / "csrc/lossless_bf16.cu"
REFRESHES = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_up(torch: Any) -> list[Any]:
    from safetensors import safe_open

    weights = []
    with safe_open(str(SNAPSHOT / "model.safetensors"), framework="pt", device="cpu") as model:
        for layer in range(32):
            prefix = f"model.layers.{layer}.mlp."
            gate = model.get_tensor(prefix + "gate_proj.weight").to(device="cuda", dtype=torch.bfloat16)
            up = model.get_tensor(prefix + "up_proj.weight").to(device="cuda", dtype=torch.bfloat16)
            weights.append(torch.cat((gate, up), dim=0).contiguous())
            del gate, up
    return weights


def exact_decode(cache: Any, weights: list[Any], torch: Any) -> list[bool]:
    results = []
    for layer, expected in enumerate(weights):
        decoded = cache.decode_layer(layer)
        results.append(bool(torch.equal(decoded.view(torch.uint16), expected.view(torch.uint16))))
        del decoded
    return results


def median(values: list[float]) -> float:
    return statistics.median(values)


def cuda_memory(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not SNAPSHOT.is_dir():
        raise FileNotFoundError(f"missing fixed model snapshot: {SNAPSHOT}")

    import torch

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    memory_snapshots = {"before_source_load": cuda_memory(torch)}
    source_started = time.perf_counter()
    weights = load_gate_up(torch)
    torch.cuda.synchronize()
    source_load_seconds = time.perf_counter() - source_started
    memory_snapshots["after_source_load"] = cuda_memory(torch)

    sys.path.insert(0, str(TINY / "prime-kernels"))
    from prime_kernels.lossless_bf16 import EncodedGUCache

    construction_started = time.perf_counter()
    cache = EncodedGUCache(weights)
    torch.cuda.synchronize()
    construction_seconds = time.perf_counter() - construction_started
    memory_snapshots["after_cache_construction"] = cuda_memory(torch)
    initial_statistics = cache.statistics()
    descriptor_ptr = cache.table.data_ptr()
    retained_weight_bytes = sum(tensor.numel() * tensor.element_size() for tensor in weights)

    cache.refresh(weights)
    memory_snapshots["after_untimed_warm_refresh"] = cuda_memory(torch)

    wall_seconds = []
    event_seconds = []
    for _ in range(REFRESHES):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        wall_start = time.perf_counter()
        cache.refresh(weights)
        wall_seconds.append(time.perf_counter() - wall_start)
        end.record(torch.cuda.current_stream())
        end.synchronize()
        event_seconds.append(start.elapsed_time(end) / 1000.0)

    statistics_seconds = []
    for _ in range(REFRESHES):
        started = time.perf_counter()
        stats = cache.statistics()
        statistics_seconds.append(time.perf_counter() - started)
    del stats
    memory_snapshots["after_timed_refreshes_and_statistics"] = cuda_memory(torch)

    before_mutation = [tensor.view(torch.int16)[0, 0].clone() for tensor in weights]
    for tensor in weights:
        tensor.view(torch.int16)[0, 0].bitwise_xor_(1)
    cache.refresh(weights)
    mutated_exact = exact_decode(cache, weights, torch)
    for tensor, original in zip(weights, before_mutation, strict=True):
        tensor.view(torch.int16)[0, 0].copy_(original)
    cache.refresh(weights)
    restored_exact = exact_decode(cache, weights, torch)
    final_statistics = cache.statistics()
    memory_snapshots["after_verification"] = cuda_memory(torch)
    memory_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    checks = {
        "descriptor_pointer_stable": cache.table.data_ptr() == descriptor_ptr,
        "mutated_refresh_all_layers_exact": all(mutated_exact),
        "restored_refresh_all_layers_exact": all(restored_exact),
    }

    result = {
        "schema": "lossless-refresh-profile-r1",
        "model": {
            "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "snapshot_revision": SNAPSHOT.name,
            "weights_sha256": sha256(SNAPSHOT / "model.safetensors"),
        },
        "package": {
            "init_sha256": sha256(PACKAGE),
            "cuda_sha256": sha256(CUDA_SOURCE),
        },
        "device": torch.cuda.get_device_name(),
        "measurement": {
            "source_load_host_wall_seconds": source_load_seconds,
            "cache_construction_including_library_initialization_host_wall_seconds": construction_seconds,
            "refresh_count": REFRESHES,
            "untimed_warm_refresh_count": 1,
            "refresh_host_wall_seconds": wall_seconds,
            "refresh_host_wall_median_seconds": median(wall_seconds),
            "refresh_cuda_event_seconds": event_seconds,
            "refresh_cuda_event_median_seconds": median(event_seconds),
            "statistics_host_wall_seconds": statistics_seconds,
            "statistics_host_wall_median_seconds": median(statistics_seconds),
            "event_note": "Events bracket the synchronous refresh call on its CUDA stream; this is not kernel-only time.",
            "construction_note": "Cache construction includes EncodedGUCache package and CUDA library initialization/loading.",
        },
        "cache": {
            "initial_statistics": initial_statistics,
            "final_statistics": final_statistics,
            "retained_gu_tensor_bytes": retained_weight_bytes,
            "estimated_peak_tensor_bytes": retained_weight_bytes + initial_statistics["allocated_bytes"] +
                                          weights[0].numel() * weights[0].element_size(),
            "estimated_peak_limit_bytes": 3 * 1024**3,
            "memory_snapshots": memory_snapshots,
            "measured_cuda_peak": memory_peak,
            **checks,
            "mutated_refresh_exact_layers": [i for i, exact in enumerate(mutated_exact) if exact],
            "restored_refresh_exact_layers": [i for i, exact in enumerate(restored_exact) if exact],
        },
        "memory_note": "The estimated tensor peak adds retained GU tensors, cache allocations, and one decoded layer; snapshot extraction also uses one temporary CPU tensor at a time.",
        "status": "passed" if all(checks.values()) else "failed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")
    if not all(checks.values()):
        raise RuntimeError(f"lossless cache verification failed; report written to {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: refresh_profile.py OUTPUT.json")
    run(Path(sys.argv[1]).resolve())
