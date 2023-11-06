from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import os
import re
from subprocess import run

# get cuda bindir
if 'CUDAHOSTCXX' in os.environ:
    cudahostcxx = os.environ['CUDAHOSTCXX']
    extra_compile_args = {'nvcc': [f'--compiler-bindir={cudahostcxx}']}

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
            extra_compile_args={'cxx': opts, **extra_compile_args},
            extra_link_args=['-lgomp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
