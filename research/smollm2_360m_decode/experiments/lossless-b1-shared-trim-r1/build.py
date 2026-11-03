from pathlib import Path
import shutil
from torch.utils.cpp_extension import load
p=Path(__file__).resolve().parent
lib=load(name="prime_decode_shared_trim",sources=[str(p/"block360.cu")],extra_cuda_cflags=["-O3","-arch=sm_89","-std=c++17","-lineinfo","-Xptxas=-v"],is_python_module=False,verbose=False)
shutil.copy2(lib,p/"candidate.so")
print("built",lib)
