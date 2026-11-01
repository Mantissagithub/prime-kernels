#!/usr/bin/env python3
"""Validate exact harness nvcc arguments without invoking nvcc."""
import os
import subprocess
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness
command = harness.build_argv()
command[0] = sys.executable
command.insert(1, str(HERE / "nvcc_stub.py"))
env = dict(os.environ)
env["PROBE_EXPECTED_SOURCE"] = str(HERE / "projection_probe.cu")
result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
print(result.stdout, end="")
