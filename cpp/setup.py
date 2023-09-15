from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# detect avx512f
import re
from subprocess import run

# get supported cpu flags
cmd = run(
    "lscpu -J | jq -r '.lscpu[] | select(.field == \"Flags:\") | .data'",
    shell=True, capture_output=True
)
flags = cmd.stdout.decode('utf-8').split()

# add avx512f if supported
opts = ['-fopenmp', '-std=c++17']
if 'avx512f' in flags:
    opts.append('-mavx512f')

setup(
    name='matmul_quant',
    ext_modules=[
        CUDAExtension(
            'matmul_quant',
            sources=['extension.cpp', 'matmul_quant_cpu.cpp', 'matmul_quant_cuda.cu'],
            extra_compile_args={
                'cxx': opts,
                'nvcc': ['--compiler-bindir=/home/doug/programs/cuda-gcc/bin']
            },
            extra_link_args=['-lgomp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
