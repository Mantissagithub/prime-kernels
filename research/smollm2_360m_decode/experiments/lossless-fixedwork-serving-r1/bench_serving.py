#!/usr/bin/env python
"""One-arm-per-process fixed-work vLLM serving comparison for R30 vs lossless."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/pradheep/tiny-prl")
RL = ROOT / "prime-rl"
KERNELS = ROOT / "prime-kernels"
HERE = RL / "outputs/megakernel-20261025/lossless-fixedwork-serving-r1"
R30_PRELOAD = RL / "outputs/megakernel-20261025/r30-full-rl-r1/python/sitecustomize.py"
MODEL = "/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951"
PROMPT_LENGTHS = (32, 128, 1024)
BATCHES = (1, 2)
NEW_TOKENS, WARMUPS, REPEATS = 64, 2, 5
MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION = 2048, 0.15
CANDIDATE_HASHES = {
    "source": "fe430f5b84e7d5d829db05ac2b890902d9e31728b9d4a4eb2861a5e5480fd6a8",
    "include": "a4dad52352e4c2e604e83175b166073d636616876d6b9380bfc2e40d7c5f3762",
    "library": "ad9a02c7d7ce1860fe7636bd8ba96e6f375a4b8f0a1d7f1f9e862dd2d232a07d",
}
R30_HASHES = {
    "source": "5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638",
    "include": "60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83",
    "library": "625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2",
}
ARM = None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_ids(length, req):
    rng = random.Random(f"lossless-fixedwork-serving-r1|{length}|{req}")
    return [rng.randrange(1000, 49000) for _ in range(length)]


def worker_state(worker):
    from prime_rl.inference.vllm import decode_backend
    from vllm.model_executor.models.llama import LlamaModel

    model = worker.get_model().model
    module = decode_backend._lossless_module
    state = {
        "pid": os.getpid(),
        "patched": bool(getattr(LlamaModel.forward, "_prime_rl_decode_backend", False)),
        "model_kernel": decode_backend._model_kernel_name(model),
        "stats": dict(decode_backend.stats),
        "backend_source": str(Path(decode_backend.__file__).resolve()),
        "backend_sha256": sha256_file(decode_backend.__file__),
    }
    if not state["patched"] or state["model_kernel"] != "decode_megakernel_360m":
        raise RuntimeError(f"decode backend route unavailable: {state}")
    if ARM == "baseline":
        import prime_kernels

        module = prime_kernels._LOADED_MODULES.get("decode_megakernel_360m")
        if module is None or module._lib is None or not module._runtimes:
            raise RuntimeError("private R30 module or CUDA runtime was not loaded")
        state["kernel_source_sha256"] = module.SOURCE_SHA256
        state["kernel_library_sha256"] = sha256_file(module._lib._name)
        state["kernel_library"] = module._lib._name
        state["runtime_devices"] = [str(x) for x in module._runtimes]
        actual = {"source": state["kernel_source_sha256"],
                  "include": module.INCLUDE_SHA256,
                  "library": state["kernel_library_sha256"]}
        if actual != R30_HASHES:
            raise RuntimeError(f"private R30 identity mismatch: {actual}")
        state["route_module"] = str(Path(module.__file__).resolve())
    else:
        if module is None or module._lib is None or not module._runtimes:
            raise RuntimeError("lossless package module or CUDA runtime was not loaded")
        state["kernel_source"] = str(module.SOURCE.resolve())
        state["kernel_source_sha256"] = sha256_file(module.SOURCE)
        state["kernel_library"] = module._lib._name
        state["kernel_library_sha256"] = sha256_file(module._lib._name)
        state["runtime_devices"] = [str(x) for x in module._runtimes]
        state["encoded_cache"] = module.cache_statistics()
        include = module.SOURCE.with_name("block360_kernel.inc")
        state["kernel_include_sha256"] = sha256_file(include)
        actual = {"source": state["kernel_source_sha256"],
                  "include": state["kernel_include_sha256"],
                  "library": state["kernel_library_sha256"]}
        if actual != CANDIDATE_HASHES:
            raise RuntimeError(f"candidate package identity mismatch: {actual}")
    return state


def summarize(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def main():
    global ARM
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("baseline", "candidate"))
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    ARM = args.arm
    out_path = args.output_json.resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    for key in ("PRIME_DECODE_MEGAKERNEL", "PRIME_DECODE_MEGAKERNEL_360M",
                "PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS", "PRIME_DECODE_REFERENCE",
                "PRIME_R30_PRIVATE_PRELOAD", "PRIME_R30_TRACE_POINTERS",
                "PRIME_R30_TRACE_FINAL_NORM_HASH", "PRIME_R30_TRACE_PHASE_FILE"):
        os.environ.pop(key, None)
    for path in (RL / "src", KERNELS):
        sys.path.insert(0, str(path))
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(RL / "src"), str(KERNELS), inherited]).rstrip(os.pathsep)
    os.environ["PRIME_DECODE_MEGAKERNEL_360M"] = "1"
    if args.arm == "candidate":
        os.environ["PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS"] = "1"
    else:
        os.environ["PRIME_R30_PRIVATE_PRELOAD"] = "1"
        os.environ["PRIME_R30_PRELOAD_TRACE_DIR"] = str(out_path.parent / (out_path.stem + "-r30-trace"))
        spec = importlib.util.spec_from_file_location("r30_private_sitecustomize", R30_PRELOAD)
        preload = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preload)

    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.inference.server import setup_vllm_env

    setup_vllm_env(InferenceConfig(decode_megakernel=True))
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    startup_begin = time.perf_counter()
    llm = LLM(
        model=MODEL, dtype="auto", max_model_len=MAX_MODEL_LEN, max_num_seqs=2,
        max_num_batched_tokens=2048, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False, enforce_eager=False,
        compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": [1, 2]},
        kernel_config={"enable_jit_warmup": False},
        disable_log_stats=False, seed=20261021, trust_remote_code=False,
        num_gpu_blocks_override=256,
    )
    startup_seconds = time.perf_counter() - startup_begin
    mode = {"executor": type(llm.llm_engine.model_executor).__name__,
            "client": type(llm.llm_engine.engine_core).__name__,
            "multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]}
    if mode["executor"] != "UniProcExecutor" or mode["client"] != "InprocClient":
        raise RuntimeError(f"expected in-process vLLM, got {mode}")
    before = llm.collective_rpc(worker_state)
    if any(x["pid"] != os.getpid() for x in before):
        raise RuntimeError(f"worker RPC escaped benchmark process: {before}")

    def generate(length, batch, params):
        ids = [prompt_ids(length, i) for i in range(batch)]
        prompts = [{"prompt_token_ids": prompt} for prompt in ids]
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        wall_ms = (time.perf_counter() - start) * 1000
        if len(outs) != batch:
            raise RuntimeError("vLLM returned a different request count")
        records = []
        for out in outs:
            ids = [int(x) for x in out.outputs[0].token_ids]
            if len(ids) != NEW_TOKENS:
                raise RuntimeError(f"expected {NEW_TOKENS} tokens, got {len(ids)}")
            logprobs = out.outputs[0].logprobs
            top5 = []
            sampled_logprob = []
            for tid, row in zip(ids, logprobs or []):
                if tid not in row:
                    raise RuntimeError("sampled token is missing from its logprob row")
                sampled_logprob.append(float(row[tid].logprob))
                top5.append(sorted(([int(k), float(v.logprob), int(v.rank)]
                                    for k, v in row.items() if v.rank is not None and v.rank <= 5),
                                   key=lambda item: item[2]))
            records.append({"prompt_token_ids": prompts[len(records)]["prompt_token_ids"],
                            "prompt_sha256": hashlib.sha256(json.dumps(
                                prompts[len(records)]["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                            "token_ids": ids, "sampled_logprob": sampled_logprob,
                            "top5": top5, "token_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "prompt_tokens": len(out.prompt_token_ids), "output_tokens": len(ids)})
        return wall_ms, records

    parity_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=NEW_TOKENS,
                                   min_tokens=NEW_TOKENS, ignore_eos=True, logprobs=5, seed=7)
    timed_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=NEW_TOKENS,
                                  min_tokens=NEW_TOKENS, ignore_eos=True, logprobs=None, seed=7)
    shapes = []
    for length in PROMPT_LENGTHS:
        for batch in BATCHES:
            _, parity = generate(length, batch, parity_params)
            for _ in range(WARMUPS):
                generate(length, batch, timed_params)
            times = []
            all_timed = []
            for _ in range(REPEATS):
                elapsed, records = generate(length, batch, timed_params)
                times.append(elapsed)
                all_timed.append(records)
            parity_hashes = [item["token_sha256"] for item in parity]
            if any([item["token_sha256"] for item in run] != parity_hashes for run in all_timed):
                raise RuntimeError(f"timed generation differs from parity generation at length={length} batch={batch}")
            if any(len(item["sampled_logprob"]) != NEW_TOKENS for item in parity):
                raise RuntimeError("parity generation omitted sampled-token logprobs")
            shapes.append({"prompt_length": length, "batch": batch,
                           "input_token_counts": [x["prompt_tokens"] for x in parity],
                           "output_token_counts": [x["output_tokens"] for x in parity],
                           "parity_outputs": parity,
                           "timed_token_hashes": [[x["token_sha256"] for x in run] for run in all_timed],
                           "batch_wall_ms": summarize(times), "batch_wall_ms_samples": times,
                           "scheduler_inclusive_itl_ms": None,
                           "itl_unavailable_reason": "RequestOutput engine timing metrics are not sampled by this harness"})
    after = llm.collective_rpc(worker_state)
    if args.arm == "candidate" and not any(x.get("stats", {}).get("backend_decode", 0) > 0 for x in after):
        raise RuntimeError("candidate recorded no backend_decode calls")
    payload = {
        "schema_version": 1, "arm": args.arm, "model": MODEL,
        "harness_sha256": sha256_file(__file__),
        "engine": {"max_model_len": MAX_MODEL_LEN, "max_num_seqs": 2,
                   "max_num_batched_tokens": 2048, "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                   "num_gpu_blocks_override": 256, "prefix_caching": False,
                   "cudagraph": "FULL_DECODE_ONLY", "capture_sizes": [1, 2],
                   "kernel_config": {"enable_jit_warmup": False}},
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": NEW_TOKENS,
                     "min_tokens": NEW_TOKENS, "ignore_eos": True,
                     "parity_logprobs": 5, "timed_logprobs": None, "seed": 7},
        "warmups": WARMUPS, "repeats": REPEATS, "startup_seconds": startup_seconds,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "vllm": vllm.__version__, "transformers": transformers.__version__},
        "mode": mode, "worker_before": before, "worker_after": after, "shapes": shapes,
    }
    with out_path.open("x") as f:
        f.write(json.dumps(payload, indent=1) + "\n")
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
