#!/usr/bin/env python3
"""Compare R30 and R35-R2 embedded sm_89 SASS without loading CUDA/PyTorch."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R35 = HERE.parent
MEGA = R35.parent
R30 = MEGA / "r30-phase-prefetch"
ARTIFACTS = {
    "R30": {
        "source": R30 / "candidate/block360.cu",
        "include": R30 / "candidate/block360_kernel.inc",
        "library": R30 / "build/libdecode360_r30_phase_prefetch.so",
    },
    "R35-R2": {
        "source": R35 / "candidate/block360.cu",
        "include": R35 / "candidate/block360_kernel.inc",
        "library": R35 / "r35build-r2/libdecode360_r35_selective_packed.so",
    },
}
SYMBOLS = ("_Z9mk_kernelILi1EEv6Params", "_Z9mk_kernelILi2EEv6Params")
FUNCTION = re.compile(r"^\s*Function : (\S+)\s*$", re.M)
INSTRUCTION_LINE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.+?)\s+/\*\s*0x([0-9a-f]+)\s*\*/\s*$")
ADDRESS_PREFIX = re.compile(r"^\s*/\*([0-9a-f]+)\*/")
CONTROL_WORD_LINE = re.compile(r"^\s*/\*\s*0x([0-9a-f]+)\s*\*/\s*$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_functions(text: str) -> dict[str, list[tuple[str, str, str, str]]]:
    starts = list(FUNCTION.finditer(text))
    result: dict[str, list[tuple[str, str, str, str]]] = {}
    for i, match in enumerate(starts):
        name = match.group(1)
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[match.end():end]
        lines = body.splitlines()
        instructions = []
        for i, line in enumerate(lines):
            if ADDRESS_PREFIX.match(line) is None:
                continue
            match = INSTRUCTION_LINE.match(line)
            if match is None:
                raise SystemExit(f"unsupported SASS instruction line in {name}: {line.strip()}")
            if i + 1 == len(lines):
                raise SystemExit(f"missing control word after {name} instruction at {match.group(1)}")
            control = CONTROL_WORD_LINE.fullmatch(lines[i + 1])
            if control is None:
                raise SystemExit(f"unsupported SASS encoding after {name} instruction at {match.group(1)}")
            address, operation, instruction_word, control_word = match.groups() + (control.group(1),)
            if len(instruction_word) != 16 or len(control_word) != 16:
                raise SystemExit(f"expected two 16-hex-digit words for {name} instruction at {address}")
            instructions.append((address, operation.strip(), instruction_word, control_word))
        if not instructions:
            raise SystemExit(f"no instructions found for {name}")
        result[name] = instructions
    return result


def summarize(instructions: list[tuple[str, str, str, str]]) -> dict[str, object]:
    normalized = "\n".join(f"{address} {operation}" for address, operation, _, _ in instructions)
    encoded = b"".join(
        int(word, 16).to_bytes(8, byteorder="little")
        for _, _, instruction_word, control_word in instructions
        for word in (instruction_word, control_word)
    )
    return {
        "instruction_count": len(instructions),
        "first": instructions[0][:2] if instructions else None,
        "last": instructions[-1][:2] if instructions else None,
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "encoded_byte_count": len(encoded),
        "encoded_little_endian_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main() -> None:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise SystemExit("cuobjdump not found; install/use CUDA toolkit with sm_89 support")
    output: dict[str, object] = {"cuobjdump": cuobjdump, "architecture": "sm_89", "arms": {}}
    with tempfile.TemporaryDirectory(prefix="r35-sass-audit-") as tmp:
        for arm, files in ARTIFACTS.items():
            sass_path = Path(tmp) / f"{arm}.sass"
            command = [cuobjdump, "--dump-sass", "--gpu-architecture", "sm_89", str(files["library"])]
            sass_path.write_text(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            text = sass_path.read_text()
            functions = extract_functions(text)
            arm_result: dict[str, object] = {"artifacts": {}, "functions": {}}
            for kind, path in files.items():
                arm_result["artifacts"][kind] = {"path": str(path), "sha256": digest(path)}
            for symbol in SYMBOLS:
                if symbol not in functions:
                    raise SystemExit(f"{symbol} not found in {files['library']}")
                arm_result["functions"][symbol] = summarize(functions[symbol])
            output["arms"][arm] = arm_result

    r30 = output["arms"]["R30"]["functions"]
    r35 = output["arms"]["R35-R2"]["functions"]
    output["comparison"] = {
        "B1_same_normalized_sass": r30[SYMBOLS[0]]["normalized_sha256"] == r35[SYMBOLS[0]]["normalized_sha256"],
        "B2_same_normalized_sass": r30[SYMBOLS[1]]["normalized_sha256"] == r35[SYMBOLS[1]]["normalized_sha256"],
        "B1_same_encoded_bytes": r30[SYMBOLS[0]]["encoded_little_endian_sha256"] == r35[SYMBOLS[0]]["encoded_little_endian_sha256"],
        "B2_same_encoded_bytes": r30[SYMBOLS[1]]["encoded_little_endian_sha256"] == r35[SYMBOLS[1]]["encoded_little_endian_sha256"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
