from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cutlass_inc = ["third_party/cutlass/include", "third_party/cutlass/tools/util/include"]

ext = CUDAExtension(
    name="llama4_dualgemm_cuda",
    sources=["csrc/binding.cpp", "csrc/dual_gemm_kernel.cu"],
    include_dirs=cutlass_inc,
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-gencode=arch=compute_90,code=sm_90",  # H100
        ],
    },
)

setup(
    name="llama4-dualgemm",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)

