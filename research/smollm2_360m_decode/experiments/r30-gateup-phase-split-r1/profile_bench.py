#!/usr/bin/env python3
"""Bounded FAST R28 stage-cycle diagnostic with an uninstrumented R28 control."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TINY = REPO.parent
HELDOUT = REPO / "outputs/megakernel-20261025/block-360-heldout-r13/validate_full32.py"
FROZEN = REPO / "outputs/megakernel-20261025/block-360-reference-r1/capture_native_frozen.py"
FIXTURE_FILES = (
    REPO / "outputs/megakernel-20261025/block-360-heldout-r1/fixtures.json",
    REPO / "outputs/megakernel-20261025/decode-360-wide-validation-r2/fixtures-r1.json",
)
FIXTURE_CASES = {"seed17-b1-ctx32", "seed17-b1-ctx128", "seed17-b2-ctx32", "seed17-b2-ctx128"}
PROFILE_DIR = HERE
PAIR_DIRS = {
    "r30": {"baseline_source": REPO / "outputs/megakernel-20261025/r30-phase-prefetch/candidate",
            "baseline_library": REPO / "outputs/megakernel-20261025/r30-phase-prefetch/build/libdecode360_r30_phase_prefetch.so",
            "profile_source": HERE / "r30-profile", "profile_library": HERE / "r30-build/libdecode360_r30_gateup_phase_split.so"},
}
ARMS: dict[str, dict[str, Path]] = {}
A, B = "A-uninstrumented", "B-instrumented"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_gpu() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                               "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # telemetry only; never gates timing
        return f"unavailable: {error}"


def sample_sm_clock_mhz() -> float | None:
    try:
        value = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        return float(value.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def ordered_bf16(tensor: Any, torch: Any) -> Any:
    bits = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).to(torch.int32)
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits | 0x8000)


def compare(a: Any, b: Any, torch: Any) -> dict[str, Any]:
    a = a.to(torch.bfloat16).contiguous()
    b = b.to(torch.bfloat16).contiguous()
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_equal": False, "reference_shape": list(a.shape), "candidate_shape": list(b.shape),
                "finite": False, "bitwise_equal": False, "different_elements": None, "max_bf16_ulp": None}
    finite = bool(torch.isfinite(a.float()).all()) and bool(torch.isfinite(b.float()).all())
    if not finite:
        return {"shape_equal": True, "finite": False, "bitwise_equal": False,
                "different_elements": None, "max_bf16_ulp": None}
    delta = (ordered_bf16(a, torch) - ordered_bf16(b, torch)).abs()
    return {"shape_equal": True, "finite": True, "bitwise_equal": bool(torch.equal(a, b)),
            "different_elements": int((a != b).sum().item()), "max_bf16_ulp": int(delta.max().item())}


def synthetic_case(case_id: str, batch: int, context: int, torch: Any) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(1024 + batch)
    ids = torch.randint(0, 49152, (batch, context), generator=generator, dtype=torch.long)
    return {"case_id": case_id, "batch": batch, "context": context, "input_ids": ids.tolist(), "source": "synthetic seed"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    global ARMS, A, B
    pair = PAIR_DIRS[args.pair]
    A, B = f"A-{args.pair}-baseline", f"B-{args.pair}-profiled"
    ARMS = {
        A: {"source": pair["baseline_source"] / "block360.cu", "include": pair["baseline_source"] / "block360_kernel.inc",
            "library": pair["baseline_library"]},
        B: {"source": pair["profile_source"] / "block360.cu", "include": pair["profile_source"] / "block360_kernel.inc",
            "library": pair["profile_library"]},
    }
    started_unix = time.time()
    frozen = load_module("capture_native_frozen", FROZEN)
    torch, transformers, config, model_cls, _ = frozen.load_config_and_classes()
    validator = load_module("validate_full32", HELDOUT)
    fixture_cases = []
    for fixture_path in FIXTURE_FILES:
        fixture_cases.extend(json.loads(fixture_path.read_text())["cases"])
    by_id = {case["case_id"]: case for case in fixture_cases if case["case_id"] in FIXTURE_CASES}
    if set(by_id) != FIXTURE_CASES:
        raise RuntimeError(f"expected fixtures {sorted(FIXTURE_CASES)}, found {sorted(by_id)}")
    selected = [by_id["seed17-b1-ctx32"], by_id["seed17-b1-ctx128"], by_id["seed17-b2-ctx32"], by_id["seed17-b2-ctx128"]]
    for context in (1024,):
        selected.append(synthetic_case(f"synthetic-b1-ctx{context}", 1, context, torch))
        selected.append(synthetic_case(f"synthetic-b2-ctx{context}", 2, context, torch))
    device = torch.device("cuda")
    model = frozen.load_model_streaming_to_cuda(device, config, model_cls, torch)
    _, base = frozen.find_llama_base(model)
    weights, weight_table = validator.make_weight_tables(base, torch, device)
    libs = {name: validator.configure_library(paths["library"], ctypes) for name, paths in ARMS.items()}
    libs[B][0].decode360_profile_buffer_bytes.restype = ctypes.c_size_t
    profile_bytes = int(libs[B][0].decode360_profile_buffer_bytes())

    if args.cases != "all":
        wanted = set(args.cases.split(","))
        selected = [c for c in selected if c["case_id"] in wanted]
    records = []
    for case in selected:
        batch, context = int(case["batch"]), int(case["context"])
        input_ids = torch.tensor(case["input_ids"], dtype=torch.long, device=device)
        frozen.set_attention_policy(model, "sdpa")
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1, return_dict=True)
        token_ids = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True).to(torch.int32).contiguous()

        native_cache = validator.clone_dynamic_cache(prefill.past_key_values, torch)
        rope: dict[str, Any] = {}
        handle = base.rotary_emb.register_forward_hook(
            lambda _m, _i, out: rope.update(cos=out[0].detach(), sin=out[1].detach()))
        position_ids = torch.full((batch, 1), context, dtype=torch.long, device=device)
        frozen.set_attention_policy(model, "eager")
        with torch.inference_mode():
            native = base(input_ids=token_ids.to(torch.long), past_key_values=native_cache,
                          position_ids=position_ids, use_cache=True, return_dict=True)
        handle.remove()
        native_hidden = native.last_hidden_state.detach()

        kv_buffers: dict[str, list[Any]] = {name: [] for name in libs}
        layouts: list[dict[str, int]] = []
        block_table = None
        for layer_idx, cache_layer in enumerate(prefill.past_key_values.layers):
            base_kv, layout = validator.make_paged_layer_kv(cache_layer, batch, context, torch)
            layouts.append(layout)
            if block_table is None:
                block_table = torch.empty((batch, layout["blocks_per_request"]), dtype=torch.int32, device=device)
                for b in range(batch):
                    first = b * layout["blocks_per_request"]
                    block_table[b] = torch.arange(first, first + layout["blocks_per_request"], dtype=torch.int32, device=device)
            for name in libs:
                kv_buffers[name].append(base_kv.clone())
        layout = layouts[0]
        kv_ptrs = {name: torch.tensor([kv.data_ptr() for kv in kv_buffers[name]], dtype=torch.int64, device=device)
                   for name in libs}
        positions = torch.full((batch,), context, dtype=torch.int64, device=device)
        seq_lens = torch.full((batch,), context + 1, dtype=torch.int32, device=device)
        slot_mapping = torch.empty((batch,), dtype=torch.int64, device=device)
        for b in range(batch):
            block = int(block_table[b, context // validator.BLOCK_SIZE].item())
            slot_mapping[b] = block * validator.BLOCK_SIZE + context % validator.BLOCK_SIZE
        cos_sin = torch.zeros((context + 1, validator.HEAD_DIM), dtype=torch.bfloat16, device=device)
        cos_sin[context, :32] = rope["cos"].reshape(batch, -1)[0, :32].to(torch.bfloat16)
        cos_sin[context, 32:] = rope["sin"].reshape(batch, -1)[0, :32].to(torch.bfloat16)
        output = {name: torch.empty((batch, validator.HIDDEN), dtype=torch.bfloat16, device=device) for name in libs}
        scratch = {name: torch.empty((values[2],), dtype=torch.uint8, device=device) for name, values in libs.items()}
        profile_buffer = torch.empty((profile_bytes // 8,), dtype=torch.uint64, device=device)
        profile_lib = libs[B][0]
        profile_lib.decode360_profile_set_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        profile_lib.decode360_profile_set_buffer.restype = ctypes.c_int
        profile_status = profile_lib.decode360_profile_set_buffer(ctypes.c_void_p(profile_buffer.data_ptr()), profile_bytes)
        if profile_status != 0:
            raise RuntimeError(f"profile buffer setter returned CUDA status {profile_status}")
        stream = torch.cuda.current_stream(device)
        stream_handle = ctypes.c_void_p(stream.cuda_stream)
        embed = base.embed_tokens.weight.contiguous()
        final_norm = base.norm.weight.contiguous()

        def invoke(name: str) -> None:
            lib, decode, workspace_bytes, debug_bytes = libs[name]
            del debug_bytes
            status = decode(stream_handle, batch, validator.device_ptr(token_ids, ctypes),
                validator.device_ptr(embed, ctypes), validator.device_ptr(final_norm, ctypes),
                validator.device_ptr(positions, ctypes), validator.device_ptr(seq_lens, ctypes),
                validator.device_ptr(block_table, ctypes), layout["blocks_per_request"],
                validator.device_ptr(slot_mapping, ctypes), validator.device_ptr(weight_table, ctypes),
                validator.device_ptr(kv_ptrs[name], ctypes), 0, layout["v_off"], layout["block_stride"],
                layout["token_stride"], layout["head_stride"], validator.BLOCK_SIZE,
                validator.device_ptr(cos_sin, ctypes), 0, validator.device_ptr(output[name], ctypes),
                None, validator.device_ptr(scratch[name], ctypes), workspace_bytes, None, 0, context + 1, 15)
            if status != 0:
                msg = lib.decode360_status_string(status)
                raise RuntimeError(f"{name} decode returned {status}: {msg.decode() if msg else 'unknown'}")

        prefix_snapshots = {name: [kv.clone() for kv in kv_buffers[name]] for name in libs}

        def reset(name: str) -> None:
            for target, prefix in zip(kv_buffers[name], prefix_snapshots[name]):
                target.copy_(prefix)

        for name in (A, B):
            for _ in range(args.warmups):
                reset(name)
                invoke(name)
        torch.cuda.synchronize(device)

        block_rows = []
        for block_index in range(args.blocks):
            order = (A, B, B, A) if block_index % 2 == 0 else (B, A, A, B)
            gpu_before = sample_gpu()
            for name in order:
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
                for start, end in zip(starts, ends):
                    reset(name)
                    if name == B:
                        profile_buffer.zero_()
                    start.record(stream)
                    invoke(name)
                    end.record(stream)
                torch.cuda.synchronize(device)
                elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends)]
                block_rows.append({"arm": name, "block": block_index, "placement": len([r for r in block_rows if r["block"] == block_index]),
                                   "events_ms": elapsed, "median_ms_per_step": sorted(elapsed)[len(elapsed) // 2]})
            block_rows[-1]["gpu_before_block"] = gpu_before
            block_rows[-1]["gpu_after_block"] = sample_gpu()

        parity = {}
        candidate_kv: dict[str, list[tuple[Any, Any]]] = {}
        for name in (A, B):
            reset(name)
            invoke(name)
            torch.cuda.synchronize(device)
            parity[name] = {"final_hidden": compare(native_hidden[:, 0, :], output[name], torch)}
            key_results = []
            candidate_kv[name] = []
            for layer_idx, kv in enumerate(kv_buffers[name]):
                seq = context + 1
                pos = torch.arange(seq, device=device)
                blocks = block_table[:, pos // validator.BLOCK_SIZE].long()
                bases = blocks * layout["block_stride"] + (pos % validator.BLOCK_SIZE) * layout["token_stride"]
                offsets = torch.arange(layout["token_stride"], device=device)
                k = kv[bases[:, :, None] + offsets].reshape(batch, seq, validator.KV_HEADS, validator.HEAD_DIM).permute(0, 2, 1, 3)
                v = kv[layout["v_off"] + bases[:, :, None] + offsets].reshape(batch, seq, validator.KV_HEADS, validator.HEAD_DIM).permute(0, 2, 1, 3)
                target = native.past_key_values.layers[layer_idx]
                candidate_kv[name].append((k, v))
                key_results.append({"layer": layer_idx, "key": compare(target.keys, k, torch),
                                    "value": compare(target.values, v, torch)})
            parity[name]["kv_layers"] = key_results
            checks = [parity[name]["final_hidden"]] + [item[f] for item in key_results for f in ("key", "value")]
            failures = [c for c in checks if not c.get("shape_equal") or not c.get("finite") or c.get("max_bf16_ulp", 2) > 1]
            # Diagnostic for both arms. R25's <=1 ULP gate is the R25 acceptance criterion and is only
            # established for its validated cases; synthetic-b2-ctx1024 is outside that set (see README).
            parity[name]["native_gate"] = "diagnostic only; R25 acceptance is <=1 BF16 ULP on validated cases"
            parity[name]["native_within_1_ulp_all_layers"] = not failures
            parity[name]["native_max_bf16_ulp"] = max([parity[name]["final_hidden"].get("max_bf16_ulp") or 0] +
                                                      [item[f].get("max_bf16_ulp") or 0 for item in key_results for f in ("key", "value")])
        arm_diff = {"final_hidden": compare(output[A], output[B], torch), "kv_layers": []}
        for layer_idx, ((a_k, a_v), (b_k, b_v)) in enumerate(zip(candidate_kv[A], candidate_kv[B], strict=True)):
            arm_diff["kv_layers"].append({"layer": layer_idx, "key": compare(a_k, b_k, torch),
                                           "value": compare(a_v, b_v, torch)})
        arm_checks = [arm_diff["final_hidden"]]
        arm_checks.extend(layer[field] for layer in arm_diff["kv_layers"] for field in ("key", "value"))
        arm_diff["all_hidden_and_kv_bitwise_equal"] = all(
            item.get("shape_equal") and item.get("finite") and item.get("bitwise_equal") for item in arm_checks
        )
        if not arm_diff["all_hidden_and_kv_bitwise_equal"]:
            raise RuntimeError(f"instrumented pair changed hidden/KV values: {arm_diff}")
        reset(B)
        profile_buffer.zero_()
        profile_clock_before = sample_sm_clock_mhz()
        profile_start_event = torch.cuda.Event(enable_timing=True)
        profile_end_event = torch.cuda.Event(enable_timing=True)
        profile_start_event.record(stream)
        invoke(B)
        profile_end_event.record(stream)
        torch.cuda.synchronize(device)
        profile_capture_ms = profile_start_event.elapsed_time(profile_end_event)
        profile_clock_after = sample_sm_clock_mhz()
        profile_cycles = profile_buffer.detach().cpu().reshape(32, 13, 24, 2).tolist()
        profile_lib.decode360_profile_set_buffer(None, 0)
        records.append({"case_id": case["case_id"], "batch": batch, "context": context,
                        "input_ids_sha256": hashlib.sha256(json.dumps(case["input_ids"]).encode()).hexdigest(),
                        "case_source": case.get("source", "frozen fixture"),
                        "prefill_greedy_ids": token_ids.detach().cpu().reshape(-1).tolist(),
                        "parity_outside_timing": {"native": parity, "A_vs_B": arm_diff},
                        "profile_capture_timing": {"cuda_event_ms": profile_capture_ms,
                                                    "sm_clock_mhz_before": profile_clock_before,
                                                    "sm_clock_mhz_after": profile_clock_after},
                        "profile_layout": {"shape": [32, 13, 24, 2], "fields": ["work_cycles", "grid_wait_cycles"],
                                           "slot_names": ["norm1", "qkv", "rope_cache", "split_kv", "merge", "o_residual",
                                                          "norm2", "gate_up_silu", "down_residual", "layer_capture", "final_norm",
                                                          "gate_up_math", "down_prefetch_issue"],
                                           "final_norm_cta": 0, "layer_capture_enabled": False},
                        "profile_cycles": profile_cycles, "blocks": block_rows})
        del native, prefill, native_cache, kv_buffers, prefix_snapshots
        torch.cuda.empty_cache()
    return {"schema_version": 1, "mode": "R30/R35 direct decoder CUDA-event ABBA plus CTA-local clock64 stage diagnostics; LM head excluded",
            "arms": {name: {"source": str(p["source"]), "source_sha256": sha256(p["source"]),
                            "include": str(p["include"]), "include_sha256": sha256(p["include"]),
                            "library": str(p["library"]), "library_sha256": sha256(p["library"])}
                     for name, p in ARMS.items()},
            "started_unix": started_unix, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device), "warmups_per_arm_per_case": args.warmups,
            "repeats_per_placement": args.repeats, "blocks_per_case": args.blocks,
            "placements_per_arm_per_case": 2 * args.blocks,
            "profile_slot_names": ["norm1", "qkv", "rope_cache", "split_kv", "merge", "o_residual",
                                   "norm2", "gate_up_silu", "down_residual", "layer_capture", "final_norm",
                                   "gate_up_math", "down_prefetch_issue"],
            "profile_buffer_bytes": profile_bytes,
            "timing_notes": ["CUDA events recorded on one stream around only decode360 calls; A is the selected uninstrumented DSO and B is its instrumented source clone.",
                             "KV prefix restore, parity, prefill, model work and telemetry are outside event intervals.",
                             "Profile buffer clearing is enqueued before event start; one separate profiled invocation captures the saved stage cycles.",
                             "layer_outputs=null and debug=null; one 32-layer decoder step to final hidden.",
                             "clock64 deltas are same-CTA elapsed cycles; work intervals include local stalls and local barriers, grid-wait intervals include CTA arrival skew; not hardware counters."],
            "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=("r30",), default="r30")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "phase-split.json")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--cases", default="seed17-b2-ctx32,seed17-b2-ctx128", help="comma-separated case_ids, or all")
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.blocks) < 1:
        raise ValueError("warmups, repeats and blocks must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
