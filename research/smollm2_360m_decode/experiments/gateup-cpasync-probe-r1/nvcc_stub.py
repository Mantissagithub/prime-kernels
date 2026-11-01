#!/usr/bin/env python3
"""Argument-checking substitute for nvcc. It never creates compiler outputs."""
import json
import os
import sys
from pathlib import Path
args = sys.argv[1:]
source = Path(os.environ["PROBE_EXPECTED_SOURCE"])
expected = ["-std=c++17", "-O3", "-arch=sm_89", "-Xcompiler=-fPIC", "--shared",
            str(source), "-o", str(source.parent / "build/libprojection_probe.so")]
if args != expected:
    raise SystemExit(f"compiler argv mismatch\nexpected={expected!r}\nactual={args!r}")
if not source.is_file():
    raise SystemExit(f"CUDA source missing: {source}")
print(json.dumps({"stub": True, "nvcc_invoked": False, "argv": args}, indent=2))
