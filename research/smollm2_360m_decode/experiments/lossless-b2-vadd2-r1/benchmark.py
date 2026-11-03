"""Use the published exact-parity harness for one packed integer-add change."""
import os
import importlib.util
from pathlib import Path

ROOT = Path("/home/pradheep/tiny-prl")
HERE = Path(__file__).resolve().parent
os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / "codec-build")
PATH = ROOT / "prime-kernels/research/smollm2_360m_decode/experiments/lossless-b1-r1/packaged_benchmark.py"
spec = importlib.util.spec_from_file_location("vadd2_exact_benchmark", PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.ARMS[bench.A] = {"source": HERE / "frozen-package/block360.cu",
                       "include": HERE / "frozen-package/block360_kernel.inc",
                       "library": HERE / "frozen-package/baseline.so"}
bench.ARMS[bench.B] = {"source": HERE / "block360.cu", "include": HERE / "block360_kernel.inc",
                       "library": HERE / "candidate.so"}
original_run = bench.run

def run(args):
    result = original_run(args)
    result["mode"] = "accepted B1 package versus B2 packed integer exponent addition"
    for case in result["cases"]:
        case["parity_outside_timing"]["A_vs_B"]["target"] = "current B1+B2 lossless package"
    return result

bench.run = run
if __name__ == "__main__":
    bench.main()
