"""Run one serving arm and stop it if sampled GPU memory exceeds 3 GiB."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LIMIT_MIB = 3072


def main() -> int:
    artifact = Path(sys.argv[1])
    command = sys.argv[2:]
    if not command:
        raise SystemExit("guard_3gb.py requires a command after its artifact path")
    artifact.mkdir(parents=True, exist_ok=False)
    gpu_log = artifact / "gpu-memory.csv"
    app_log = artifact / "application.log"
    stop_reason = None
    peak_gpu = 0
    proc = subprocess.Popen(command, stdout=app_log.open("w"), stderr=subprocess.STDOUT)
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=gpu_log.open("w"), stderr=subprocess.DEVNULL,
    )

    def terminate(*_args: object) -> None:
        nonlocal stop_reason
        stop_reason = "termination requested"
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        while proc.poll() is None:
            try:
                rows = list(csv.reader(gpu_log.read_text().splitlines()))
                if rows and len(rows[-1]) >= 2 and rows[-1][1].strip().isdigit():
                    peak_gpu = max(peak_gpu, int(rows[-1][1].strip()))
            except FileNotFoundError:
                pass
            if peak_gpu > LIMIT_MIB:
                stop_reason = f"sampled GPU memory exceeded {LIMIT_MIB} MiB"
                proc.terminate()
                break
            if stop_reason:
                break
            time.sleep(0.1)
        try:
            returncode = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
    result = {
        "command": command,
        "returncode": returncode,
        "stop_reason": stop_reason,
        "gpu_memory_limit_mib": LIMIT_MIB,
        "peak_sampled_gpu_memory_mib": peak_gpu,
    }
    (artifact / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if returncode == 0 and stop_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
