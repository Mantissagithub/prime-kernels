#!/usr/bin/env python3
"""Run one fresh baseline or lossless two-update forkserver case."""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE = REPO / "outputs/megakernel-20261025/r30-full-forkserver-r1/configs/forkserver-on.toml"
CANDIDATE = HERE / "configs/forkserver-lossless.toml"
UV = "/home/pradheep/.local/bin/uv"
PYTHON = "/home/pradheep/.venv/bin/python"
BASELINE_DRIVER = REPO / "outputs/megakernel-20261025/r30-full-forkserver-r1/probe/run_case.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(f"invalid run-id: {args.run_id}")
    spec = importlib.util.spec_from_file_location("r30_fullrl_preflight", BASELINE_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical preflight from {BASELINE_DRIVER}")
    baseline_driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_driver)
    envelope = baseline_driver.prelaunch_envelope()
    config = BASELINE if args.arm == "baseline" else CANDIDATE
    output = HERE / "runs" / f"{args.run_id}-{args.arm}"
    metadata = HERE / "results" / f"{args.run_id}-{args.arm}"
    if output.exists() or metadata.exists():
        raise FileExistsError(f"refusing to reuse run paths for {args.run_id}")
    output.mkdir(parents=True)
    metadata.mkdir(parents=True)
    worker_trace_dir = metadata / "worker-traces"
    overlay = metadata / "runtime-identity.toml"
    command = [UV, "run", "--offline", "--no-project", "--python", PYTHON,
               "rl", "@", str(config)]
    if args.arm == "baseline":
        worker_trace_dir.mkdir()
        overlay.write_text("[inference.env_vars]\nPRIME_R30_PRELOAD_TRACE_DIR = "
                           + json.dumps(str(worker_trace_dir)) + "\n")
        command.extend(("@", str(overlay)))
    command.extend(("--output-dir", str(output), "--no-dashboard", "--no-ckpt"))
    launch = {"arm": args.arm, "run_id": args.run_id, "prl_run_id": uuid.uuid4().hex,
              "config": str(config), "prelaunch_envelope": envelope,
              "output_dir": str(output), "command": command, "started_at_unix": time.time(),
              "required_updates": 2}
    (metadata / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PRL_RUN_ID": launch["prl_run_id"]})
    env.pop("PRL_RUN_NAME", None)
    for key in ("PRIME_DECODE_MEGAKERNEL_360M", "PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS",
                "PRIME_R30_PRIVATE_PRELOAD", "PRIME_R30_PRELOAD_TRACE_DIR"):
        env.pop(key, None)
    started = time.monotonic()
    with (metadata / "rl.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    run_logs = list(output.rglob("inference.log"))
    inference_lines = [line.strip() for path in run_logs for line in path.read_text(errors="replace").splitlines()]
    route_events = [line for line in inference_lines
                    if "Lossless 360M decode route active" in line
                    or "Lossless 360M decode route prepared" in line
                    or "Lossless 360M GU cache refreshed" in line]
    metrics = [json.loads(line) for path in output.rglob("metrics.jsonl")
               for line in path.read_text().splitlines() if line.strip()]
    trainer_steps = sorted({int(item["step"]) for item in metrics
                            if item.get("producer") == "trainer" and "step" in item})
    refreshes = sum("Lossless 360M GU cache refreshed" in line for line in route_events)
    active_route = any("Lossless 360M decode route active" in line for line in route_events)
    baseline_route = any(
        json.loads(line).get("event") == "private_r30_first_decode"
        for path in worker_trace_dir.glob("worker-*.jsonl") if worker_trace_dir.exists()
        for line in path.read_text().splitlines() if line.strip()
    )
    passed = (result.returncode == 0 and {1, 2}.issubset(trainer_steps)
              and ((args.arm == "baseline" and baseline_route)
                   or (args.arm == "candidate" and active_route and refreshes >= 2)))
    completion = {"status": "passed" if passed else "failed-validation",
                  "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started,
                  "finished_at_unix": time.time(), "required_updates": 2,
                  "trainer_steps": trainer_steps, "active_lossless_route": active_route,
                  "baseline_r30_route": baseline_route,
                  "lossless_refresh_count": refreshes, "route_events": route_events,
                  "inference_logs": [str(path) for path in run_logs]}
    (metadata / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
