"""Screen B1-only shared scratch sizing with the published exact-parity harness."""
import os
import importlib.util
from pathlib import Path

ROOT = Path("/home/pradheep/tiny-prl")
HERE = Path(__file__).resolve().parent
os.environ["TORCH_EXTENSIONS_DIR"] = str(Path("/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/lossless-b2-vadd2-r1/codec-build"))
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
    result["mode"] = "accepted package versus B1 shared scratch sized to template batch"
    for case in result["cases"]:
        case["parity_outside_timing"]["A_vs_B"]["target"] = "current B1+B2 lossless package"
    return result

bench.run = run
if __name__ == "__main__":
    bench.main()
