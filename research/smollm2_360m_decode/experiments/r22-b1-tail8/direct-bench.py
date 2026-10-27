#!/usr/bin/env python3
"""Direct CUDA-event ABBA timing for the R21/R22-tail8 one-step decoder only."""
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
HELDOUT = REPO / "outputs/megakernel-20261025/block-360-heldout-r13/validate_full32.py"
FROZEN = REPO / "outputs/megakernel-20261025/block-360-reference-r1/capture_native_frozen.py"
FIXTURES = REPO / "outputs/megakernel-20261025/block-360-heldout-r1/fixtures.json"
CANDIDATES = {
    "R21": {
        "source": REPO / "outputs/megakernel-20261025/decode-360-cuda-r21/block360.cu",
        "include": REPO / "outputs/megakernel-20261025/decode-360-cuda-r21/block360_kernel.inc",
        "library": REPO / "outputs/megakernel-20261025/decode-360-cuda-r21/build/libdecode360_r21.so",
    },
    "R22-tail8": {
        "source": REPO / "outputs/megakernel-20261025/decode-360-cuda-r22-tail8/block360.cu",
        "include": REPO / "outputs/megakernel-20261025/decode-360-cuda-r22-tail8/block360_kernel.inc",
        "library": REPO / "outputs/megakernel-20261025/decode-360-cuda-r22-tail8/build/libdecode360_r22_tail8.so",
    },
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_preflight() -> dict[str, Any]:
    validator = load_module("validate_full32", HELDOUT)
    result: dict[str, Any] = {"mode": "CPU-only; no CUDA calls", "abi_parameter_count": len(validator.ABI_PARAMETERS),
                              "candidates": {}}
    if len(validator.ABI_PARAMETERS) != 28:
        raise RuntimeError("expected canonical 28-argument decoder ABI")
    for name, paths in CANDIDATES.items():
        source = paths["source"]
        include = paths["include"]
        validator.validate_abi_source(include)
        library = paths["library"]
        lib, _, workspace, debug = validator.configure_library(library, ctypes)
        result["candidates"][name] = {
            "translation_unit_sha256": sha256(source),
            "abi_definition_sha256": sha256(include),
            "library_sha256": sha256(library),
            "workspace_bytes_exported": workspace,
            "debug_bytes_exported": debug,
            "abi_status": "28 source declarations and ctypes types match",
        }
        del lib
    if len({item["workspace_bytes_exported"] for item in result["candidates"].values()}) != 1:
        raise RuntimeError("R21/R22-tail8 workspace sizes differ")
    result["workspace_policy"] = "allocate exact size queried from each library's decode360_workspace_bytes()"
    result["debug_policy"] = "debug=null; decode360_debug_bytes() recorded but not allocated"
    return result


def sample_gpu() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=timestamp,clocks.gr,clocks.mem,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True,
    ).stdout.strip()


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_unix = time.time()
    torch, transformers, config, model_cls, _ = load_module("capture_native_frozen", FROZEN).load_config_and_classes()
    frozen = load_module("capture_native_frozen", FROZEN)
    validator = load_module("validate_full32", HELDOUT)
    import json as json_module

    fixture_cases = json_module.loads(FIXTURES.read_text())["cases"]
    selected = [case for case in fixture_cases if case["batch"] in (1, 2)
                and case["context"] in (32, 128) and case["seed"] == 17]
    if len(selected) != 4:
        raise RuntimeError(f"expected four seed17 B1/B2 context32/128 fixtures, found {len(selected)}")
    device = torch.device("cuda")
    model = frozen.load_model_streaming_to_cuda(device, config, model_cls, torch)
    _, base = frozen.find_llama_base(model)
    weights, weight_table = validator.make_weight_tables(base, torch, device)
    libs = {}
    for name, paths in CANDIDATES.items():
        libs[name] = validator.configure_library(paths["library"], ctypes)

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

        kv_buffers: dict[str, list[Any]] = {}
        layouts: list[dict[str, int]] = []
        for name in libs:
            kv_buffers[name] = []
        block_table = None
        kv_ptrs: dict[str, Any] = {}
        for layer_idx, cache_layer in enumerate(prefill.past_key_values.layers):
            base_kv, layout = validator.make_paged_layer_kv(cache_layer, batch, context, torch)
            layouts.append(layout)
            if block_table is None:
                block_table = torch.empty((batch, layout["blocks_per_request"]), dtype=torch.int32, device=device)
                for b in range(batch):
                    first = b * layout["blocks_per_request"]
                    block_table[b] = torch.arange(first, first + layout["blocks_per_request"], dtype=torch.int32, device=device)
            for name in libs:
                kv = base_kv.clone()
                kv_buffers[name].append(kv)
        layout = layouts[0]
        for name in libs:
            kv_ptrs[name] = torch.tensor([kv.data_ptr() for kv in kv_buffers[name]], dtype=torch.int64, device=device)
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

        # Warm each implementation before measured alternating blocks.
        for name in ("R21", "R22-tail8"):
            for _ in range(args.warmups):
                reset(name)
                invoke(name)
        torch.cuda.synchronize(device)

        block_rows = []
        for block_index in range(args.blocks):
            order = ("R21", "R22-tail8", "R22-tail8", "R21") if block_index % 2 == 0 else ("R22-tail8", "R21", "R21", "R22-tail8")
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
                block_rows.append({"candidate": name, "block": block_index, "events_ms": elapsed,
                                   "median_ms_per_step": sorted(elapsed)[len(elapsed) // 2]})
            # sample telemetry after the whole four-arm ABBA block; timing above remains event-only.
            block_rows[-1]["gpu_after_block"] = sample_gpu()
            block_rows[-1]["gpu_before_block"] = gpu_before

        parity = {}
        candidate_kv: dict[str, list[tuple[Any, Any]]] = {}
        for name in libs:
            reset(name)
            invoke(name)
            torch.cuda.synchronize(device)
            parity[name] = {"final_hidden": compare(native_hidden[:, 0, :], output[name], torch)}
            key_results = []
            candidate_kv[name] = []
            for layer_idx, kv in enumerate(kv_buffers[name]):
                # Unpack only the logical positions through tensor indexing; no timed work here.
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
            checks = [parity[name]["final_hidden"]]
            checks.extend(item[field] for item in key_results for field in ("key", "value"))
            failures = [check for check in checks if not check.get("shape_equal") or not check.get("finite")
                        or check.get("max_bf16_ulp", 2) > 1]
            if failures:
                raise RuntimeError(f"{case['case_id']} {name} failed hidden/KV parity gate: {failures[:3]}")
        r21_r22_tail8_hidden = compare(output["R21"], output["R22-tail8"], torch)
        r21_r22_tail8_kv = []
        for layer_idx, ((r15bk, r15bv), (r20k, r20v)) in enumerate(zip(candidate_kv["R21"], candidate_kv["R22-tail8"])):
            r21_r22_tail8_kv.append({"layer": layer_idx, "key": compare(r15bk, r20k, torch),
                               "value": compare(r15bv, r20v, torch)})
        cross_checks = [r21_r22_tail8_hidden]
        cross_checks.extend(item[field] for item in r21_r22_tail8_kv for field in ("key", "value"))
        cross_failures = [check for check in cross_checks if not check.get("shape_equal") or not check.get("finite")
                          or check.get("max_bf16_ulp", 2) > 1]
        if cross_failures:
            raise RuntimeError(f"{case['case_id']} R21/R22-tail8 cross-candidate parity gate failed: {cross_failures[:3]}")
        records.append({"case_id": case["case_id"], "batch": batch, "context": context,
                        "input_ids_sha256": hashlib.sha256(json.dumps(case["input_ids"]).encode()).hexdigest(),
                        "prefill_greedy_ids": token_ids.detach().cpu().reshape(-1).tolist(),
                        "parity_outside_timing": {"native": parity,
                            "R21_vs_R22-tail8": {"final_hidden": r21_r22_tail8_hidden, "kv_layers": r21_r22_tail8_kv}},
                        "blocks": block_rows})
        del native, prefill, native_cache
    return {"schema_version": 1, "mode": "direct decoder-only CUDA-event ABBA; LM head excluded",
            "started_unix": started_unix, "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device), "warmups_per_library_per_case": args.warmups,
            "repeats_per_arm_per_block": args.repeats, "abba_blocks_per_case": args.blocks,
            "timing_notes": ["CUDA events recorded on one current stream around only decode360 calls.",
                "KV prefix restore, parity, prefill, model/reference work, and telemetry queries are outside event intervals.",
                "layer_outputs=null and debug=null; the measured operation is one 32-layer decoder step ending at final hidden state."],
            "cpu_preflight": cpu_preflight(), "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results-r1.json")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    if args.cpu_preflight:
        result = cpu_preflight()
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if min(args.warmups, args.repeats, args.blocks) < 1:
        raise ValueError("warmups, repeats, and blocks must be positive")
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "case_count": len(result["cases"]), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
