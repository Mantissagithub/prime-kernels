#!/usr/bin/env python
"""One-arm-per-process vLLM serving benchmark: R28 vs R30 phase-ahead prefetch for SmolLM2-360M.

Private R30 serving harness adapted from the R28 runner. Each arm preloads its hash-pinned private DSO into the package wrapper before engine creation; package files are untouched.

Run through runtime/run_gpu.sh. Modes:
  --arm r28    preload and route through the pinned R28 DSO.
  --arm r30    preload and route through the private R30 phase-prefetch DSO.

Engine: in-process vLLM (VLLM_ENABLE_V1_MULTIPROCESSING=0), FULL_DECODE_ONLY graphs with capture
sizes [1, 2], prefix caching off, max_model_len 2048. Shapes: prompt length 32/128/1024 x batch 1/2,
64 greedy new tokens with ignore_eos, 3 warmups and 7 measured repeats per shape.

Inter-token latency (ITL) is engine-side per request: (last_token_ts - first_token_ts) / (n - 1), using
RequestOutput.metrics. The engine clock spans scheduling, model execution and output processing, so it
is scheduler-inclusive. Request wall is last_token_ts - submit (submit = time.monotonic() before generate).
"""

import argparse
import ctypes
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIME_RL = Path("/home/pradheep/tiny-prl/prime-rl")
KERNELS = Path("/home/pradheep/tiny-prl/prime-kernels")
MODEL = "/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951"
R28_RESEARCH = Path("/home/pradheep/tiny-prl/prime-kernels/research/smollm2_360m_decode/experiments/r28-flash-decode")
R28_LIBRARY = Path("/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r28-flash-decode/build/libdecode360_r28_flash.so")
R30_RESEARCH = Path("/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r30-phase-prefetch/candidate")
R30_LIBRARY = Path("/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r30-phase-prefetch/build/libdecode360_r30_phase_prefetch.so")
CANDIDATES = {
    "r28": {
        "source": R28_RESEARCH / "block360.cu",
        "include": R28_RESEARCH / "block360_kernel.inc",
        "library": R28_LIBRARY,
        "source_sha256": "1d121b7f89561a955b1fcc665daa7535a13816821920fe4030c4040e7bf61313",
        "include_sha256": "0ac2ba97ed2bfc4684c476987635e98e06affe2323349e4f9c4d9a27ed1247f0",
        "library_sha256": "df3f03f7d680719b2f282fe7ad0a41cc24ff20f04d14809756b9b5e63a7f6603",
    },
    "r30": {
        "source": R30_RESEARCH / "block360.cu",
        "include": R30_RESEARCH / "block360_kernel.inc",
        "library": R30_LIBRARY,
        "source_sha256": "5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638",
        "include_sha256": "60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83",
        "library_sha256": "625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2",
    },
}
PROMPT_LENGTHS = (32, 128, 1024)
BATCHES = (1, 2)
NEW_TOKENS = 64
WARMUPS = 3
REPEATS = 7
TOPK = 5
MAX_MODEL_LEN = 2048
MAX_BATCHED_TOKENS = 2048
GPU_MEMORY_UTILIZATION = 0.3
ENGINE_SEED = 20261021
SAMPLING_SEED = 7
VOCAB_LOW, VOCAB_HIGH = 1000, 49000
TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"  # nvidia-smi timestamp format

ARM = None  # read by the worker RPC; the engine is in-process so this is the same interpreter


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_stamp():
    return datetime.now().strftime(TIME_FORMAT)


def prompt_ids(length, batch, rep, req):
    # Seeded by shape and repeat only, so both arms receive identical prompts.
    rng = random.Random(f"r25-serving|{length}|{batch}|{rep}|{req}")  # same prompts as the R25 serving runs
    return [rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(length)]


def worker_state(worker):
    """Runs inside the in-process worker and verifies its selected private DSO."""
    import inspect
    from pathlib import Path as P

    from prime_rl.inference.vllm import decode_backend
    from vllm.model_executor.models.llama import LlamaModel

    if ARM not in CANDIDATES:
        raise RuntimeError(f"unknown candidate arm {ARM}")
    expected = CANDIDATES[ARM]
    model = worker.get_model().model
    state = {
        "arm": ARM,
        "worker_pid": os.getpid(),
        "model_class": type(model).__name__,
        "forward_patched": bool(getattr(LlamaModel.forward, "_prime_rl_decode_backend", False)),
        "decode_backend_source": str(P(decode_backend.__file__).resolve()),
        "decode_backend_sha256": sha256_file(decode_backend.__file__),
        "selected_model_kernel": decode_backend._model_kernel_name(model),
        "stats": dict(decode_backend.stats),
    }
    import prime_kernels

    module = prime_kernels._LOADED_MODULES.get("decode_megakernel_360m")
    if module is None:
        raise RuntimeError("decode_megakernel_360m was never loaded in the worker")
    package = P(prime_kernels.__file__).resolve()
    module_file = P(module.__file__).resolve()
    if not package.is_relative_to(KERNELS.resolve()) or not module_file.is_relative_to(KERNELS.resolve()):
        raise RuntimeError(f"worker imported prime_kernels outside the sibling checkout: {package}")
    source_hash = sha256_file(expected["source"])
    include_hash = sha256_file(expected["include"])
    if (source_hash, include_hash) != (expected["source_sha256"], expected["include_sha256"]):
        raise RuntimeError(f"{ARM} source/include hash mismatch: source={source_hash} include={include_hash}")
    loaded = module._lib
    if loaded is None or P(loaded._name).resolve() != expected["library"].resolve():
        raise RuntimeError(f"worker wrapper does not hold the {ARM} library: {getattr(loaded, '_name', None)}")
    library_hash = sha256_file(expected["library"])
    if library_hash != expected["library_sha256"]:
        raise RuntimeError(f"{ARM} library hash mismatch: {library_hash}")
    decode_steps = inspect.getclosurevars(LlamaModel.forward).nonlocals.get("decode_steps", {})
    if not state["forward_patched"] or decode_steps.get("decode_megakernel_360m") is not module.decode_step:
        raise RuntimeError("patched LlamaModel.forward does not route to decode_megakernel_360m")
    state.update({
        "prime_kernels_source": str(package),
        "kernel_module": str(module_file),
        "kernel_source": str(expected["source"]),
        "kernel_source_sha256": source_hash,
        "kernel_include": str(expected["include"]),
        "kernel_include_sha256": include_hash,
        "kernel_library": str(expected["library"]),
        "kernel_library_sha256": library_hash,
        "runtime_devices_built": [str(d) for d in module._runtimes],
        "route_is_module_decode_step": True,
    })
    return state


def request_record(out, submit):
    metrics = out.metrics
    if metrics is None:
        raise RuntimeError("RequestOutput.metrics is None; engine stats are required for ITL")
    comp = out.outputs[0]
    tokens = [int(t) for t in comp.token_ids]
    if len(tokens) != NEW_TOKENS:
        raise RuntimeError(f"generated {len(tokens)} tokens, expected {NEW_TOKENS}")
    steps_top, steps_sampled_lp = [], []
    for tid, row in zip(tokens, comp.logprobs):
        if tid not in row:
            raise RuntimeError("sampled token missing from its logprob row")
        top = sorted(
            ([int(k), float(v.logprob), int(v.rank)] for k, v in row.items() if v.rank is not None and v.rank <= TOPK),
            key=lambda item: item[2],
        )
        steps_top.append(top)
        steps_sampled_lp.append(float(row[tid].logprob))
    return {
        "token_ids": tokens,
        "top5": steps_top,
        "sampled_logprob": steps_sampled_lp,
        "first_token_ts": metrics.first_token_ts,
        "last_token_ts": metrics.last_token_ts,
        "itl_ms": (metrics.last_token_ts - metrics.first_token_ts) / (NEW_TOKENS - 1) * 1000,
        "ttft_ms": (metrics.first_token_ts - submit) * 1000,
        "request_wall_ms": (metrics.last_token_ts - submit) * 1000,
    }


def summarize(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def start_clock_sampler(path):
    handle = open(path, "w")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=handle, stderr=subprocess.DEVNULL,
    )
    return proc, handle


def load_candidate_library(arm):
    """Load one of the pinned private DSOs using the wrapper's exact C ABI."""
    candidate = CANDIDATES[arm]
    library_hash = sha256_file(candidate["library"])
    if library_hash != candidate["library_sha256"]:
        raise RuntimeError(f"{arm} library hash mismatch before preload: {library_hash}")
    lib = ctypes.CDLL(str(candidate["library"]))
    vp, ci, ll, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_size_t
    lib.decode360_workspace_bytes.argtypes = []
    lib.decode360_workspace_bytes.restype = sz
    lib.decode360_debug_bytes.argtypes = []
    lib.decode360_debug_bytes.restype = sz
    lib.decode360.argtypes = [vp, ci, vp, vp, vp, vp, vp, vp, ci, vp, vp, vp,
                              ll, ll, ll, ll, ll, ci, vp, ci, vp, vp, vp, sz,
                              vp, sz, ci, ci]
    lib.decode360.restype = ci
    lib.decode360_status_string.argtypes = [ci]
    lib.decode360_status_string.restype = ctypes.c_char_p
    return lib


def main():
    global ARM
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["r28", "r30"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ARM = args.arm
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Both arms use the sibling package wrapper, with their own hash-pinned private DSO.
    sys.path.insert(0, str(KERNELS))
    inherited = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(KERNELS) + (os.pathsep + inherited if inherited else "")
    os.environ["PRIME_DECODE_MEGAKERNEL_360M"] = "1"
    import prime_kernels

    if not Path(prime_kernels.__file__).resolve().is_relative_to(KERNELS.resolve()):
        raise SystemExit(f"prime_kernels resolved outside the sibling checkout: {prime_kernels.__file__}")
    preload_module = prime_kernels.load("decode_megakernel_360m")
    preload_module._lib = load_candidate_library(ARM)
    print("PRELOADED", ARM, preload_module._lib._name, flush=True)

    clocks_path = out_path.with_suffix(".clocks.csv")
    clock_proc, clock_handle = start_clock_sampler(clocks_path)
    try:
        from prime_rl.configs.inference import InferenceConfig
        from prime_rl.inference.server import setup_vllm_env

        setup_vllm_env(InferenceConfig(decode_megakernel=True))
        if os.environ.get("PRIME_DECODE_MEGAKERNEL") != "1":
            raise SystemExit(f"PRIME_DECODE_MEGAKERNEL={os.environ.get('PRIME_DECODE_MEGAKERNEL')} for arm {ARM}")

        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams

        started = time.perf_counter()
        llm = LLM(
            model=MODEL, dtype="auto", max_model_len=MAX_MODEL_LEN, max_num_seqs=2,
            max_num_batched_tokens=MAX_BATCHED_TOKENS, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=False, enforce_eager=False,
            compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2]},
            disable_log_stats=False, seed=ENGINE_SEED, trust_remote_code=False,
        )
        startup_seconds = time.perf_counter() - started
        engine = llm.llm_engine
        mode = {
            "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
            "engine_class": type(engine).__name__,
            "executor_class": type(engine.model_executor).__name__,
            "client_class": type(engine.engine_core).__name__,
            "parent_pid": os.getpid(),
        }
        if mode["executor_class"] != "UniProcExecutor" or mode["client_class"] != "InprocClient":
            raise RuntimeError(f"in-process execution was not selected: {mode}")
        before = llm.collective_rpc(worker_state)
        if any(s["worker_pid"] != os.getpid() for s in before):
            raise RuntimeError("worker RPC did not run in the benchmark process")
        if not all(s["forward_patched"] and s["route_is_module_decode_step"] for s in before):
            raise RuntimeError(f"{ARM} route not installed on every worker: {before}")
        print("WORKER_BEFORE", json.dumps(before), flush=True)

        params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=NEW_TOKENS, min_tokens=NEW_TOKENS,
            ignore_eos=True, logprobs=TOPK, seed=SAMPLING_SEED,
        )

        def run_batch(length, batch, rep):
            prompts = [{"prompt_token_ids": prompt_ids(length, batch, rep, req)} for req in range(batch)]
            submit = time.monotonic()
            stamp_start = now_stamp()
            wall_start = time.perf_counter()
            outs = llm.generate(prompts, params, use_tqdm=False)
            batch_wall_ms = (time.perf_counter() - wall_start) * 1000
            stamp_end = now_stamp()
            return {
                "start": stamp_start, "end": stamp_end, "batch_wall_ms": batch_wall_ms,
                "requests": [request_record(out, submit) for out in outs],
            }

        shapes = []
        measure_start = now_stamp()
        for length in PROMPT_LENGTHS:
            for batch in BATCHES:
                for warm in range(WARMUPS):
                    run_batch(length, batch, 1000 + warm)
                stats_before = llm.collective_rpc(worker_state)[0]["stats"]
                runs = [run_batch(length, batch, rep) for rep in range(REPEATS)]
                stats_after = llm.collective_rpc(worker_state)[0]["stats"]
                delta = {k: stats_after.get(k, 0) - stats_before.get(k, 0) for k in set(stats_after) | set(stats_before)}
                flat = [r for run in runs for r in run["requests"]]
                shape = {
                    "prompt_len": length,
                    "batch": batch,
                    "itl_ms": summarize([r["itl_ms"] for r in flat]),
                    "ttft_ms": summarize([r["ttft_ms"] for r in flat]),
                    "request_wall_ms": summarize([r["request_wall_ms"] for r in flat]),
                    "batch_wall_ms": summarize([r["batch_wall_ms"] for r in runs]),
                    "backend_stats_delta": delta,
                    "runs": runs,
                }
                shapes.append(shape)
                print(
                    f"SHAPE len={length} batch={batch} itl_med={shape['itl_ms']['median']:.3f}ms "
                    f"ttft_med={shape['ttft_ms']['median']:.2f}ms wall_med={shape['request_wall_ms']['median']:.2f}ms "
                    f"delta={delta}",
                    flush=True,
                )
        measure_end = now_stamp()
        after = llm.collective_rpc(worker_state)
        if not any(s["stats"].get("backend_decode", 0) for s in after):
            raise RuntimeError(f"{ARM} arm recorded zero backend_decode steps")

        runtime = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
        }
        payload = {
            "arm": ARM,
            "model": MODEL,
            "harness": "bench_serving_r28_r30.py",
            "candidate": {key: value for key, value in CANDIDATES[ARM].items()
                          if key.endswith("sha256")},
            "harness_sha256": sha256_file(__file__),
            "engine": {
                "max_model_len": MAX_MODEL_LEN, "max_num_seqs": 2, "max_num_batched_tokens": MAX_BATCHED_TOKENS,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION, "enable_prefix_caching": False,
                "enforce_eager": False, "dtype": "auto",
                "compilation_config": {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2]},
                "seed": ENGINE_SEED,
            },
            "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": NEW_TOKENS, "ignore_eos": True,
                         "logprobs": TOPK, "seed": SAMPLING_SEED},
            "new_tokens": NEW_TOKENS, "warmups": WARMUPS, "repeats": REPEATS,
            "prompt_lengths": list(PROMPT_LENGTHS), "batches": list(BATCHES),
            "startup_seconds": startup_seconds,
            "mode": mode,
            "runtime": runtime,
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, check=True).stdout.strip(),
            "worker_before": before,
            "worker_after": after,
            "measure_window": {"start": measure_start, "end": measure_end},
            "clock_log": clocks_path.name,
            "shapes": shapes,
        }
        out_path.write_text(json.dumps(payload, indent=1) + "\n")
        print("WROTE", out_path, flush=True)
    finally:
        clock_proc.terminate()
        clock_proc.wait()
        clock_handle.close()


if __name__ == "__main__":
    main()
