"""Use the published exact-parity harness for one isolated row-map change."""
import importlib.util
from pathlib import Path

ROOT = Path("/home/pradheep/tiny-prl")
HERE = Path(__file__).resolve().parent
PATH = ROOT / "prime-kernels/research/smollm2_360m_decode/experiments/lossless-b1-r1/packaged_benchmark.py"
spec = importlib.util.spec_from_file_location("interleave_exact_benchmark", PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.ARMS[bench.A] = {"source": bench.LOSSLESS_DIR / "block360.cu",
                       "include": bench.LOSSLESS_DIR / "block360_kernel.inc",
                       "library": bench.LOSSLESS_WORK / "build/prime_decode_megakernel_360m_lossless.so"}
bench.ARMS[bench.B] = {"source": HERE / "block360.cu", "include": HERE / "block360_kernel.inc",
                       "library": HERE / "interleaved.so"}
original_run = bench.run

def run(args):
    result = original_run(args)
    result["mode"] = "current package versus uninstrumented B1 CTA interleaving"
    for case in result["cases"]:
        case["parity_outside_timing"]["A_vs_B"]["target"] = "current B1+B2 lossless package"
    return result

bench.run = run
if __name__ == "__main__":
    bench.main()
