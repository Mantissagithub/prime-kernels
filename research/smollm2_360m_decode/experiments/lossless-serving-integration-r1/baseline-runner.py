#!/usr/bin/env python3
"""Run one fresh R30 full-parameter arm under the canonical GPU guard."""

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[3]
PRIME_RL = REPO / "prime-rl"
UV = "/home/pradheep/.local/bin/uv"
PYTHON = "/home/pradheep/.venv/bin/python"
SOURCE_SHA256 = "5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638"
INCLUDE_SHA256 = "60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83"
DSO_SHA256 = "625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2"
MIN_HOST_BYTES = 8 * 1024**3
MIN_GPU_FREE_MIB = 7680
termination_signal = None
child = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prelaunch_envelope() -> dict:
    available = next(int(line.split()[1]) * 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < MIN_HOST_BYTES:
        raise RuntimeError(f"MemAvailable {available} is below {MIN_HOST_BYTES} bytes")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("canonical GPU runner must expose only CUDA device 0")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    pids = [line.strip() for line in apps.splitlines() if line.strip() and line.strip() != "N/A"]
    if pids:
        raise RuntimeError(f"GPU has active compute processes: {pids}")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = list(csv.reader(query.splitlines(), skipinitialspace=True))
    if not rows or len(rows[0]) != 2:
        raise RuntimeError(f"unexpected GPU query: {query!r}")
    free_mib, utilization = (int(value.strip()) for value in rows[0])
    if free_mib < MIN_GPU_FREE_MIB:
        raise RuntimeError(f"GPU free VRAM {free_mib} MiB is below {MIN_GPU_FREE_MIB} MiB")
    return {"mem_available_bytes": available, "gpu_compute_pids": pids,
            "gpu_free_mib": free_mib, "ambient_gpu_utilization_percent": utilization,
            "minimum_mem_available_bytes": MIN_HOST_BYTES,
            "minimum_gpu_free_mib": MIN_GPU_FREE_MIB}


def config_report(arm: str) -> dict:
    report_path = HERE / "results/config-resolution-check.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "pass":
        raise RuntimeError("CPU-resolved arm comparison is not passing")
    config_path = HERE / "configs" / f"forkserver-{arm}.toml"
    if sha256(config_path) != report["config_sha256"][arm]:
        raise RuntimeError(f"frozen arm config changed after dry resolution: {config_path}")
    artifacts = {
        "block360.cu": (HERE.parents[0] / "r30-phase-prefetch/candidate/block360.cu", SOURCE_SHA256),
        "block360_kernel.inc": (HERE.parents[0] / "r30-phase-prefetch/candidate/block360_kernel.inc", INCLUDE_SHA256),
        "libdecode360_r30_phase_prefetch.so": (
            HERE.parents[0] / "r30-phase-prefetch/build/libdecode360_r30_phase_prefetch.so", DSO_SHA256
        ),
    }
    for name, (path, expected) in artifacts.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned R30 {name} hash mismatch: {actual}")
    for filename, info in report["resolved"].items():
        resolved = Path(info[arm]["path"])
        if sha256(resolved) != info[arm]["sha256"]:
            raise RuntimeError(f"CPU-resolved {arm} config changed: {resolved}")
    return {"report_sha256": sha256(report_path), "arm": arm,
            "config_sha256": report["config_sha256"][arm]}


def differences(left, right, path="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changed = []
        for key in sorted(left.keys() | right.keys()):
            child_path = f"{path}/{key}"
            if key not in left or key not in right:
                changed.append(child_path)
            else:
                changed.extend(differences(left[key], right[key], child_path))
        return changed
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path + "/length"]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, f"{path}/{index}")]
    return [] if left == right else [path]


def verify_case_dry_run(arm: str, metadata: Path, output: Path,
                        route_dir: Path, env: dict) -> dict:
    overlay = metadata / "runtime-identity.toml"
    overlay.write_text("[inference.env_vars]\n"
                       f'PRIME_R30_PRELOAD_TRACE_DIR = "{route_dir}"\n')
    resolved_output = metadata / "dry-resolved"
    command = [UV, "run", "--offline", "--no-project", "--python", PYTHON,
               "rl", "@", str(HERE / "configs" / f"forkserver-{arm}.toml"),
               "@", str(overlay), "--output-dir", str(resolved_output),
               "--dry-run", "--no-dashboard", "--no-ckpt"]
    log_path = metadata / "dry-run.log"
    with log_path.open("w") as log:
        subprocess.run(command, cwd=PRIME_RL, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    run_name = f"r30-full-forkserver-{arm}-2update"
    candidate = resolved_output / run_name / "configs/attempt_1/resolved"
    baseline = HERE / "results/config-dry-resolved-r2" / arm / run_name / "configs/attempt_1/resolved"
    allowed = {
        "trainer.json": {"/output_dir"},
        "orchestrator.json": {"/output_dir"},
        "inference.json": {"/env_vars/PRIME_R30_PRELOAD_TRACE_DIR"},
        "envs/train/alphabet-sort.json": {"/address_file"},
    }
    changed = {}
    for name, whitelist in allowed.items():
        reference = json.loads((baseline / name).read_text())
        actual = json.loads((candidate / name).read_text())
        delta = differences(reference, actual)
        unexpected = sorted(set(delta) - whitelist)
        if unexpected:
            raise RuntimeError(f"runtime config changed unexpected {arm} settings in {name}: {unexpected}")
        changed[name] = delta
    return {"resolved_dir": str(candidate), "allowed_identity_differences": changed,
            "run_output_dir": str(output), "route_trace_dir": str(route_dir),
            "dry_run_log": str(log_path)}


def on_termination(signum, _frame) -> None:
    global termination_signal
    termination_signal = signum


def stop_child() -> None:
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def validate_route(route_dir: Path) -> dict:
    loaded, first_decode = [], []
    for path in sorted(route_dir.glob("worker-*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "private_r30_loaded":
                loaded.append(event)
            elif event.get("event") == "private_r30_first_decode":
                first_decode.append(event)
    valid_loaded = [event for event in loaded
                    if event.get("source_sha256") == SOURCE_SHA256
                    and event.get("library_sha256") == DSO_SHA256]
    valid_first = [event for event in first_decode
                   if event.get("source_sha256") == SOURCE_SHA256
                   and event.get("library_sha256") == DSO_SHA256]
    passed = any(load.get("pid") == first.get("pid")
                 for load in valid_loaded for first in valid_first)
    return {
        "status": "pass" if passed else "missing-or-wrong-r30-first-decode-route",
        "worker_trace_files": [str(path) for path in sorted(route_dir.glob("worker-*.jsonl"))],
        "private_r30_loaded_events": len(loaded),
        "private_r30_first_decode_events": len(first_decode),
        "matched_source_sha256": SOURCE_SHA256 if passed else None,
        "matched_dso_sha256": DSO_SHA256 if passed else None,
    }


def main() -> int:
    global child
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(f"invalid run-id: {args.run_id}")

    envelope = prelaunch_envelope()
    frozen = config_report(args.arm)
    control_id = uuid.uuid4().hex
    output = HERE / "runs" / f"live-{args.run_id}-{args.arm}"
    metadata = HERE / "results" / f"rl-case-{args.run_id}-{args.arm}"
    if output.exists() or metadata.exists():
        raise FileExistsError(f"refusing to reuse output or metadata paths for {args.run_id}")
    route_dir = metadata / "worker-traces"
    metadata.mkdir(parents=True)

    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PRL_RUN_ID": control_id})
    env.pop("PRL_RUN_NAME", None)
    for key in ("PRIME_R30_TRACE_POINTERS", "PRIME_R30_TRACE_FINAL_NORM_HASH",
                "PRIME_R30_TRACE_PHASE_FILE"):
        env.pop(key, None)
    runtime_check = verify_case_dry_run(args.arm, metadata, output, route_dir, env)
    config = HERE / "configs" / f"forkserver-{args.arm}.toml"
    command = [UV, "run", "--offline", "--no-project", "--python", PYTHON,
               "rl", "@", str(config), "@", str(metadata / "runtime-identity.toml"),
               "--output-dir", str(output), "--no-dashboard", "--no-ckpt"]
    launch = {
        "arm": args.arm, "batch_run_id": args.run_id, "prl_run_id": control_id,
        "config": str(config), "output_dir": str(output), "metadata_dir": str(metadata),
        "route_trace_dir": str(route_dir), "prelaunch_envelope": envelope,
        "frozen_config": frozen, "runtime_config_check": runtime_check,
        "command": command, "started_at_unix": time.time(),
    }
    (metadata / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")

    signal.signal(signal.SIGTERM, on_termination)
    signal.signal(signal.SIGINT, on_termination)
    child_started_monotonic = time.monotonic()
    child_started_unix = time.time()
    with (metadata / "rl.log").open("w") as log:
        child = subprocess.Popen(command, cwd=PRIME_RL, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None and termination_signal is None:
                time.sleep(0.25)
            if termination_signal is not None:
                stop_child()
                return_code = 128 + termination_signal
            else:
                return_code = child.returncode
        finally:
            stop_child()
    elapsed = time.monotonic() - child_started_monotonic
    child_returncode = child.returncode
    route = validate_route(route_dir)
    (metadata / "route-validation.json").write_text(json.dumps(route, indent=2) + "\n")
    if route["status"] != "pass" and termination_signal is None:
        return_code = 1
    status = ("terminated" if termination_signal is not None else
              "route-validation-failed" if route["status"] != "pass" else
              "completed" if return_code == 0 else "rl-failed")
    completion = {
        "status": status, "returncode": return_code,
        "rl_child_returncode": child_returncode,
        "rl_elapsed_seconds": elapsed,
        "elapsed_scope": "actual uv rl child only; excludes preflight and config dry-run",
        "child_started_at_unix": child_started_unix,
        "finished_at_unix": time.time(), "route_validation": route,
        "terminal_work_counts": "not parsed; derive updates and rollout work from saved trainer outputs",
    }
    (metadata / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"status": "prelaunch-failed", "error": str(exc)}), file=sys.stderr)
        raise
