#!/usr/bin/env python3
"""Compare frozen R30 and lossless full-decoder outputs on identical inputs and KV prefixes."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path('/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/lossless-full-kernel-integration-r1').resolve()
REPO = HERE.parents[2]
TINY = REPO.parent
sys.path.insert(0, str(TINY / "prime-kernels"))
HELDOUT = REPO / "outputs/megakernel-20261025/block-360-heldout-r13/validate_full32.py"
FROZEN = REPO / "outputs/megakernel-20261025/block-360-reference-r1/capture_native_frozen.py"
FIXTURE_FILES = (
    REPO / "outputs/megakernel-20261025/block-360-heldout-r1/fixtures.json",
    REPO / "outputs/megakernel-20261025/decode-360-wide-validation-r2/fixtures-r1.json",
)
FIXTURE_CASES = {"seed17-b1-ctx32", "seed17-b1-ctx128", "seed17-b2-ctx32", "seed17-b2-ctx128"}
SYNTHETIC_CONTEXT = 1024
R30_DIR = REPO / "outputs/megakernel-20261025/r30-phase-prefetch/candidate"
R30_WORK = REPO / "outputs/megakernel-20261025/r30-phase-prefetch"
LOSSLESS_DIR = TINY / "prime-kernels/prime_kernels/decode_megakernel_360m_lossless/csrc"
LOSSLESS_WORK = REPO / "outputs/megakernel-20261025/lossless-full-kernel-integration-r1"
ARMS = {
    "A-Package": {"source": R30_DIR / "block360.cu", "include": R30_DIR / "block360_kernel.inc",
              "library": R30_WORK / "build" / "libdecode360_r30_phase_prefetch.so"},
    "B-Lossless": {"source": LOSSLESS_DIR / "block360.cu", "include": LOSSLESS_DIR / "block360_kernel.inc",
                   "library": LOSSLESS_WORK / "build" / "prime_decode_megakernel_360m_lossless.so"},
}
ARMS["B-Lossless"] = {"source": Path('/home/pradheep/tiny-prl/outputs/megakernel-20261025/lossless-b1-r1/block360.cu'), "include": Path('/home/pradheep/tiny-prl/outputs/megakernel-20261025/lossless-b1-r1/block360_kernel.inc'), "library": Path('/home/pradheep/tiny-prl/outputs/megakernel-20261025/lossless-b1-r1/candidate.so')}
ARMS["A-Package"] = {"source": LOSSLESS_DIR / "block360.cu", "include": LOSSLESS_DIR / "block360_kernel.inc", "library": LOSSLESS_WORK / "build/prime_decode_megakernel_360m_lossless.so"}
A, B = "A-Package", "B-Lossless"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_lossless(path: Path, ctypes: Any, validator: Any) -> tuple[Any, Any, int, int]:
    lib = ctypes.CDLL(str(path))
    decode = lib.decode360_lossless
    decode.argtypes = validator.ctypes_abi_types(ctypes) + [ctypes.c_void_p]
    decode.restype = ctypes.c_int
    lib.decode360_workspace_bytes.argtypes = []
    lib.decode360_workspace_bytes.restype = ctypes.c_size_t
    lib.decode360_debug_bytes.argtypes = []
    lib.decode360_debug_bytes.restype = ctypes.c_size_t
    lib.decode360_status_string.argtypes = [ctypes.c_int]
    lib.decode360_status_string.restype = ctypes.c_char_p
    return lib, decode, int(lib.decode360_workspace_bytes()), int(lib.decode360_debug_bytes())


def sample_gpu() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                               "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # telemetry only; never gates timing
        return f"unavailable: {error}"


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
    return {"shape_equal": True, "finite": True, "bitwise_equal": bool(torch.equal(a.view(torch.int16), b.view(torch.int16))),
            "different_elements": int((a.view(torch.int16) != b.view(torch.int16)).sum().item()), "max_bf16_ulp": int(delta.max().item())}


def synthetic_case(case_id: str, batch: int, context: int, torch: Any) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(1024 + batch)
    ids = torch.randint(0, 49152, (batch, context), generator=generator, dtype=torch.long)
    return {"case_id": case_id, "batch": batch, "context": context, "input_ids": ids.tolist(), "source": "synthetic seed"}


def run(args: argparse.Namespace) -> dict[str, Any]:
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
    for context in ((1024, 4096, 8000) if args.include_long else (1024,)):
        selected.append(synthetic_case(f"synthetic-b1-ctx{context}", 1, context, torch))
        selected.append(synthetic_case(f"synthetic-b2-ctx{context}", 2, context, torch))
    device = torch.device("cuda")
    model = frozen.load_model_streaming_to_cuda(device, config, model_cls, torch)
    _, base = frozen.find_llama_base(model)
    weights, weight_table = validator.make_weight_tables(base, torch, device)
    libs = {
        A: configure_lossless(ARMS[A]["library"], ctypes, validator),
        B: configure_lossless(ARMS[B]["library"], ctypes, validator),
    }
    from prime_kernels.lossless_bf16 import EncodedGUCache

    encoded_cache = EncodedGUCache([weights[index * 6 + 4] for index in range(32)])
    cache_initial = encoded_cache.statistics()
    if cache_initial["active_layers"] != 32:
        raise RuntimeError(f"checkpoint GU cache has disabled layers: {cache_initial}")
    for index in range(32):
        if not torch.equal(encoded_cache.decode_layer(index).view(torch.uint16),
                           weights[index * 6 + 4].view(torch.uint16)):
            raise RuntimeError(f"checkpoint GU layer {index} is not bitwise lossless")

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
        stream = torch.cuda.current_stream(device)
        embed = base.embed_tokens.weight.contiguous()
        final_norm = base.norm.weight.contiguous()

        def invoke(name: str) -> None:
            lib, decode, workspace_bytes, debug_bytes = libs[name]
            del debug_bytes
            stream_handle = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
            status = decode(stream_handle, batch, validator.device_ptr(token_ids, ctypes),
                validator.device_ptr(embed, ctypes), validator.device_ptr(final_norm, ctypes),
                validator.device_ptr(positions, ctypes), validator.device_ptr(seq_lens, ctypes),
                validator.device_ptr(block_table, ctypes), layout["blocks_per_request"],
                validator.device_ptr(slot_mapping, ctypes), validator.device_ptr(weight_table, ctypes),
                validator.device_ptr(kv_ptrs[name], ctypes), 0, layout["v_off"], layout["block_stride"],
                layout["token_stride"], layout["head_stride"], validator.BLOCK_SIZE,
                validator.device_ptr(cos_sin, ctypes), 0, validator.device_ptr(output[name], ctypes),
                None, validator.device_ptr(scratch[name], ctypes), workspace_bytes, None, 0, context + 1, 15,
                *( [validator.device_ptr(encoded_cache.table, ctypes)] if name in (A, B) else [] ))
            if status != 0:
                msg = lib.decode360_status_string(status)
                raise RuntimeError(f"{name} decode returned {status}: {msg.decode() if msg else 'unknown'}")

        prefix_snapshots = {name: [kv.clone() for kv in kv_buffers[name]] for name in libs}

        def reset(name: str) -> None:
            for target, prefix in zip(kv_buffers[name], prefix_snapshots[name]):
                target.copy_(prefix)

        block_rows = []
        if not args.parity_only:
            for name in (A, B):
                for _ in range(args.warmups):
                    reset(name)
                    invoke(name)
            torch.cuda.synchronize(device)
            for block_index in range(args.blocks):
                order = (A, B, B, A) if block_index % 2 == 0 else (B, A, A, B)
                gpu_before = sample_gpu()
                for name in order:
                    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
                    for start, end in zip(starts, ends):
                        reset(name)
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
            # Diagnostic for both arms; use the R28 relaxed gate, not native bitwise parity.
            # established for its validated cases; synthetic-b2-ctx1024 is outside that set (see README).
            parity[name]["native_gate"] = "diagnostic only; R28 relaxed-gate baseline; native parity is diagnostic"
            parity[name]["native_within_1_ulp_all_layers"] = not failures
            parity[name]["native_max_bf16_ulp"] = max([parity[name]["final_hidden"].get("max_bf16_ulp") or 0] +
                                                      [item[f].get("max_bf16_ulp") or 0 for item in key_results for f in ("key", "value")])
        hidden_diff = compare(output[A], output[B], torch)
        kv_diffs = []
        for layer_idx, ((keys_a, values_a), (keys_b, values_b)) in enumerate(
            zip(candidate_kv[A], candidate_kv[B])
        ):
            kv_diffs.append({
                "layer": layer_idx,
                "key": compare(keys_a, keys_b, torch),
                "value": compare(values_a, values_b, torch),
            })
        exact_checks = [hidden_diff] + [item[field] for item in kv_diffs for field in ("key", "value")]
        if any(not check["finite"] or not check["bitwise_equal"] for check in exact_checks):
            raise RuntimeError(f"packaged/candidate exact hidden+KV parity failed for {case['case_id']}")
        kv_max_abs = max(
            max(float((ka.float() - kb.float()).abs().max()), float((va.float() - vb.float()).abs().max()))
            for (ka, va), (kb, vb) in zip(candidate_kv[A], candidate_kv[B])
        )
        graph_refresh = None
        if context == 32:
            graph = torch.cuda.CUDAGraph()
            reset(B)
            with torch.cuda.graph(graph):
                invoke(B)
            with torch.inference_mode():
                for flag in encoded_cache._enabled:
                    flag.zero_()
                encoded_cache._tables[0]['plane'].zero_()
                encoded_cache._tables[0]['headers'].zero_()
            reset(B)
            graph.replay()
            torch.cuda.synchronize(device)
            if not torch.equal(output[A].view(torch.int16), output[B].view(torch.int16)):
                raise RuntimeError('captured dense fallback used poisoned encoded weights')
            encoded_cache.refresh([weights[index * 6 + 4] for index in range(32)])
            reset(B)
            graph.replay()
            torch.cuda.synchronize(device)
            before = output[B].clone()
            generation_before = encoded_cache.statistics()["generation"]
            original_gate_up = weights[4].clone()
            weights[4].zero_()
            encoded_cache.refresh([weights[index * 6 + 4] for index in range(32)])
            if encoded_cache.statistics()["active_layers"] != 32:
                raise RuntimeError("GU-only in-place refresh disabled a checkpoint layer")
            reset(B)
            graph.replay()
            torch.cuda.synchronize(device)
            graph_output = output[B].clone()
            reset(B)
            invoke(B)
            torch.cuda.synchronize(device)
            replay_cmp = compare(output[B], graph_output, torch)
            changed = not torch.equal(before, graph_output)
            weights[4].copy_(original_gate_up)
            encoded_cache.refresh([weights[index * 6 + 4] for index in range(32)])
            graph_refresh = {
                "generation_before": generation_before,
                "generation_after": encoded_cache.statistics()["generation"],
                "gu_only_in_place_mutation": True,
                "dense_fallback_graph_ignores_poisoned_encoded_weights": True,
                "graph_replay_matches_eager_bf16": replay_cmp["bitwise_equal"],
                "graph_output_changed": changed,
            }
            if not replay_cmp["bitwise_equal"] or not changed:
                raise RuntimeError(f"captured graph missed GU refresh: {graph_refresh}")
        arm_diff = {
            "final_hidden": hidden_diff,
            "kv_layers": kv_diffs,
            "kv_max_abs": kv_max_abs,
            "exact_bitwise_all_layers": True,
            "target": "packaged B2-only lossless decoder",
            "graph_refresh": graph_refresh,
        }
        records.append({"case_id": case["case_id"], "batch": batch, "context": context,
                        "input_ids_sha256": hashlib.sha256(json.dumps(case["input_ids"]).encode()).hexdigest(),
                        "case_source": case.get("source", "frozen fixture"),
                        "prefill_greedy_ids": token_ids.detach().cpu().reshape(-1).tolist(),
                        "parity_outside_timing": {"native": parity, "A_vs_B": arm_diff},
                        "blocks": block_rows})
        del native, prefill, native_cache, kv_buffers, prefix_snapshots
        torch.cuda.empty_cache()
    return {"schema_version": 1, "mode": "packaged versus B1-enabled full-decoder parity",
            "arms": {name: {"source": str(p["source"]), "source_sha256": sha256(p["source"]),
                            "include": str(p["include"]), "include_sha256": sha256(p["include"]),
                            "library": str(p["library"]), "library_sha256": sha256(p["library"])}
                     for name, p in ARMS.items()},
            "started_unix": started_unix, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device), "warmups_per_arm_per_case": args.warmups,
            "repeats_per_placement": args.repeats, "blocks_per_case": args.blocks,
            "placements_per_arm_per_case": 2 * args.blocks,
            "timing_notes": ["CUDA events recorded on one stream around only decode360 calls.",
                             "KV prefix restore, parity, prefill, model work and telemetry are outside event intervals.",
                             "layer_outputs=null and debug=null; one 32-layer decoder step to final hidden."],
            "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results" / "direct-bench-r28-vs-r30-phase-prefetch.json")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--cases", default="all", help="comma-separated case_ids, or all")
    parser.add_argument("--parity-only", action="store_true", help="skip timing and check R30 parity only")
    parser.add_argument("--include-long", action="store_true",
                        help="also benchmark 4096/8000 contexts; use only after confirming the guarded memory budget")
    args = parser.parse_args()
    if not args.parity_only and min(args.warmups, args.repeats, args.blocks) < 1:
        raise ValueError("warmups, repeats and blocks must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
