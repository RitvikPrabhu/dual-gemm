import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;9.0"

setup(
    name="dual-gemm-cutlass",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="dual_gemm",
            sources=[str(root / "csrc" / "binding.cu")],  # use .cpp if that's your file
            include_dirs=[
                str(root / "csrc" / "cutlass_dual"),
                str(root / "third_party" / "cutlass" / "include"),
                str(root / "third_party" / "cutlass" / "tools" / "util" / "include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
