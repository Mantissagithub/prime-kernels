#!/usr/bin/env python3
"""Measure exact BF16 exponent-delta byte-plane encoding on cached gate/up weights."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = Path("/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951")
LAYERS = 32
SHAPE = (2560, 960)
CHUNK = 8


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(bits, torch):
    words = bits.reshape(-1).to(torch.int64).reshape(-1, CHUNK)
    exponents = (words >> 7) & 255
    bases = exponents.min(dim=1).values
    deltas = exponents - bases[:, None]
    escaped = (bases == 255) | (deltas.max(dim=1).values > 7)
    packed = torch.zeros(words.shape[0], dtype=torch.int64)
    for lane in range(CHUNK):
        packed |= deltas[:, lane] << (3 * lane)
    headers = ((bases << 24) | packed).to(torch.int64)
    escaped_ids = torch.nonzero(escaped, as_tuple=False).flatten()
    if escaped_ids.numel() >= 1 << 24:
        raise ValueError("escape table exceeds the 24-bit header index")
    headers[escaped] = (255 << 24) | torch.arange(escaped_ids.numel(), dtype=torch.int64)
    plane = (((words >> 8) & 128) | (words & 127)).to(torch.uint8)
    raw = words[escaped].to(torch.uint16).contiguous()
    return plane, headers.to(torch.uint32), escaped_ids, raw


def decode(plane, headers, raw, torch):
    h = headers.to(torch.int64)
    base = h >> 24
    escaped = base == 255
    shifts = torch.arange(CHUNK, dtype=torch.int64) * 3
    delta = ((h & 0xFFFFFF)[:, None] >> shifts[None, :]) & 7
    exponent = base[:, None] + delta
    p = plane.reshape(-1, CHUNK).to(torch.int64)
    words = ((p & 128) << 8) | (exponent << 7) | (p & 127)
    if escaped.any():
        raw_ids = h[escaped] & 0xFFFFFF
        raw_table = raw.to(torch.int64).reshape(-1, CHUNK)
        if raw_table.shape[0] != raw_ids.numel():
            raise ValueError("raw escape table row count must match escaped chunks")
        if not torch.equal(torch.sort(raw_ids).values,
                           torch.arange(raw_ids.numel(), dtype=torch.int64)):
            raise ValueError("escape header indices must densely cover the raw escape table")
        words[escaped] = raw_table[raw_ids]
    return words.reshape(-1).to(torch.uint16)


def self_test(torch):
    source = torch.tensor([
        0x8000, 0x0000, 0x0001, 0x007F, 0x8001, 0x0002, 0x0003, 0x0004,
        0x0080, 0x0480, 0x0081, 0x0882, 0x0083, 0x0884, 0x0085, 0x0086,
        0x0100, 0x0180, 0x0200, 0x0280, 0x0300, 0x0380, 0x0400, 0x0480,
        0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7F81, 0xFF82, 0x7FFF, 0xFFFF,
        0x8180, 0x8200, 0x8280, 0x8300, 0x8380, 0x8400, 0x8480, 0x8500,
    ], dtype=torch.uint16)
    plane, headers, escaped_ids, raw = encode(source, torch)
    if escaped_ids.tolist() != [1, 3]:
        raise AssertionError(f"expected nonadjacent escapes [1, 3], got {escaped_ids.tolist()}")
    h = headers.to(torch.int64)
    if ((h[[1, 3]] & 0xFFFFFF).tolist() != [0, 1]
            or not bool(((h[[1, 3]] >> 24) == 255).all())):
        raise AssertionError("escape headers do not contain dense raw-table indices")
    restored = decode(plane, headers, raw, torch)
    if not torch.equal(source, restored):
        raise AssertionError("crafted edge-pattern bitwise roundtrip failed")
    permuted_headers = headers.clone()
    permuted_headers[1] = (permuted_headers[1] & 0xFF000000) | 1
    permuted_headers[3] = (permuted_headers[3] & 0xFF000000) | 0
    permuted_restored = decode(plane, permuted_headers, raw.flip(0).contiguous(), torch)
    if not torch.equal(source, permuted_restored):
        raise AssertionError("header-indexed raw-table permutation roundtrip failed")
    return {"chunks": int(headers.numel()), "escaped_chunks": int(escaped_ids.numel()),
            "header_escape_indices": [0, 1], "permuted_table_roundtrip": True,
            "bitwise_roundtrip": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="test crafted CPU bit patterns without reading weights")
    parser.add_argument("--model-file", type=Path, default=MODEL / "model.safetensors")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    import torch
    if args.self_test:
        print(json.dumps(self_test(torch), sort_keys=True))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    from safetensors import safe_open

    report = {"model_file": str(args.model_file), "model_revision": args.model_file.parent.name,
              "model_sha256": sha256(args.model_file), "script_sha256": sha256(Path(__file__)),
              "device": "CPU", "chunk_weights": CHUNK, "shape_per_matrix": list(SHAPE),
              "layers": [], "matrices": [], "total": {"raw_bytes": 0, "encoded_bytes": 0,
                                          "escaped_chunks": 0, "chunks": 0,
                                          "nonfinite_weights": 0}}
    with safe_open(str(args.model_file), framework="pt", device="cpu") as model:
        keys = set(model.keys())
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}.mlp."
            layer_result = {"layer": layer, "raw_bytes": 0, "encoded_bytes": 0,
                            "escaped_chunks": 0, "chunks": 0, "nonfinite_weights": 0}
            for projection in ("gate_proj", "up_proj"):
                name = prefix + projection + ".weight"
                if name not in keys:
                    raise KeyError(name)
                tensor = model.get_tensor(name)
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != SHAPE:
                    raise ValueError(f"{name}: expected BF16 {SHAPE}, got {tensor.dtype} {tuple(tensor.shape)}")
                bits = tensor.view(torch.uint16)
                plane, headers, escaped_ids, raw = encode(bits, torch)
                restored = decode(plane, headers, raw, torch)
                if not torch.equal(bits.reshape(-1), restored):
                    raise AssertionError(f"bitwise roundtrip mismatch: {name}")
                words = bits.reshape(-1).to(torch.int64)
                nonfinite = int((((words >> 7) & 255) == 255).sum().item())
                chunks = headers.numel()
                escaped = escaped_ids.numel()
                raw_bytes = bits.numel() * 2
                encoded_bytes = plane.numel() + chunks * 4 + escaped * 16
                entry = {"name": name, "raw_bytes": raw_bytes, "encoded_bytes": encoded_bytes,
                         "saving_bytes": raw_bytes - encoded_bytes, "chunks": chunks,
                         "escaped_chunks": escaped, "escape_fraction": escaped / chunks,
                         "nonfinite_weights": nonfinite, "bitwise_roundtrip": True}
                report["matrices"].append(entry)
                for key, value in (("raw_bytes", raw_bytes), ("encoded_bytes", encoded_bytes),
                                   ("escaped_chunks", escaped), ("chunks", chunks),
                                   ("nonfinite_weights", nonfinite)):
                    layer_result[key] += value
                    report["total"][key] += value
                del tensor, bits, plane, headers, escaped_ids, raw, restored
            layer_result["saving_bytes"] = layer_result["raw_bytes"] - layer_result["encoded_bytes"]
            layer_result["escape_fraction"] = layer_result["escaped_chunks"] / layer_result["chunks"]
            report["layers"].append(layer_result)
    total = report["total"]
    total["saving_bytes"] = total["raw_bytes"] - total["encoded_bytes"]
    total["saving_fraction"] = total["saving_bytes"] / total["raw_bytes"]
    total["escape_fraction"] = total["escaped_chunks"] / total["chunks"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(total, sort_keys=True))


if __name__ == "__main__":
    main()
