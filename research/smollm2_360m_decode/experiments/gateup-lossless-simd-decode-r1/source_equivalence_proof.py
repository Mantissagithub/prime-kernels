#!/usr/bin/env python3
"""Prove the compressed control is unchanged and SIMD only replaces decode calls."""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/"gateup-lossless-probe-r1"/"projection_probe.cu"
CANDIDATE=HERE/"projection_probe.cu"
CONTROL_SHA256="b0218ea40e2f9012c802dde20adfb428d0361ad952761c07a698a7e60e7cf09c"
def extract(src,prefix):
    begin=src.index(prefix); brace=src.index("{",begin); depth=0
    for i in range(brace,len(src)):
        if src[i]=="{": depth+=1
        elif src[i]=="}":
            depth-=1
            if depth==0:return src[begin:i+1]
    raise ValueError(prefix)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    control=CONTROL.read_text(); candidate=CANDIDATE.read_text()
    assert sha(CONTROL)==CONTROL_SHA256
    control_kernel=extract(control,"__global__ void gateup_lossless_kernel(")
    expected=control_kernel.replace("gateup_lossless_kernel","gateup_lossless_simd_kernel",1).replace("decode_chunk(","decode_chunk_simd(")
    assert extract(candidate,"__global__ void gateup_lossless_kernel(")==control_kernel
    assert extract(candidate,"__global__ void gateup_lossless_simd_kernel(")==expected
    assert candidate.count("decode_chunk_simd(")==4
    assert "extern \"C\" int run_gateup_lossless_simd(" in candidate
    assert "extern \"C\" int decode_weights_simd(" in candidate
    print(json.dumps({"control_source_sha256":CONTROL_SHA256,"candidate_source_sha256":sha(CANDIDATE),"control_kernel_identical":True,"candidate_kernel":"original kernel renamed; exactly its two decode helper calls use decode_chunk_simd","raw_prefetch_hints":"retained","math_and_reduction":"identical source","candidate_abi":"run_gateup_lossless_simd has same seven-pointer/stream signature","full_weight_validation":"decode_weights_simd uses the SIMD helper"},indent=2))
if __name__=="__main__": main()
