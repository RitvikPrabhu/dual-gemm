from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os, shutil

root = Path(__file__).parent.resolve()

CUTLASS_DIR = Path(os.environ.get("CUTLASS_DIR", root / "third_party" / "cutlass")).resolve()

def find_cuda_home():
    for key in ("CUDA_HOME", "CUDATOOLKIT_HOME", "CUDA_PATH"):
        ch = os.environ.get(key)
        if ch:
            return Path(ch)

    nvcc = shutil.which("nvcc")
    if nvcc:
        p = Path(nvcc)
        parts = p.parts
        if "hpc_sdk" in parts and "compilers" in parts:
            return p.parents[2] / "cuda"

        if "cuda" in parts:
            return p.parents[1]

    return Path("/usr/local/cuda")

CUDA_HOME = find_cuda_home()
CUDA_INCLUDE = CUDA_HOME / "include"
CUDA_LIB64  = CUDA_HOME / "lib64"

if not CUDA_LIB64.exists():
    raise RuntimeError(f"Could not find CUDA lib64 at {CUDA_LIB64}. "
                       f"Set CUDA_HOME or edit setup.py to point to your toolkit.")

def parse_arch_list():
    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    archs = [a.strip() for a in raw.replace(",", ";").split(";") if a.strip()] if raw else ["80"]
    flags = []
    for a in archs:
        a = a.lower().replace(".", "")
        if a.endswith("a"):
            flags.append(f"-gencode=arch=compute_{a},code=sm_{a}")
        else:
            flags.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    return flags

def choose_ccbin():
    return os.environ.get("CUDAHOSTCXX") or os.environ.get("CXX") or "/usr/bin/g++"

include_dirs = [
    str(root / "csrc" / "cutlass_dual"),                 
    str(CUTLASS_DIR / "include"),                        
    str(CUTLASS_DIR / "tools" / "util" / "include"),     
    str(CUDA_INCLUDE),                                   
]

cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-std=c++17",
    f"-ccbin={choose_ccbin()}",
]
nvcc_flags += parse_arch_list()

ext = CUDAExtension(
    name="dual_gemm",
    sources=[
        "csrc/binding.cu",   
    ],
    include_dirs=include_dirs,
    library_dirs=[str(CUDA_LIB64)],                  
    runtime_library_dirs=[str(CUDA_LIB64)],          
    extra_link_args=[f"-Wl,-rpath,{CUDA_LIB64}"],    
    extra_compile_args={
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    },
)

setup(
    name="dual-gemm-cutlass",
    version="0.0.1",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)

