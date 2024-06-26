from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

import os
from subprocess import run

# default compiler args
extra_compile_args = {}
extra_link_args = ['-lgomp']

# get supported cpu flags
cmd = run(
    "lscpu -J | jq -r '.lscpu[] | select(.field == \"Flags:\") | .data'",
    shell=True, capture_output=True
)
flags = cmd.stdout.decode('utf-8').split()

# add avx512f if supported
cxx_flags = ['-fopenmp', '-std=c++17']
if 'avx512f' in flags:
    cxx_flags.append('-mavx512f')

# select CPU or CUDA build
cpu_only = os.environ.get('ZIGGY_CPU_ONLY', '').lower() in ('1', 'true', 'yes')
has_cuda = 'CUDA_HOME' in os.environ
if not cpu_only and has_cuda:
    Extension = CUDAExtension
    sources = ['extension_cuda.cpp', 'matmul_quant_cpu.cpp', 'matmul_quant_cuda.cu']

    # handle non-standard compiler
    if 'CUDAHOSTCXX' in os.environ:
        cudahostcxx = os.environ['CUDAHOSTCXX']
        extra_compile_args['nvcc'] = [f'--compiler-bindir={cudahostcxx}']
else:
    if has_cuda:
        print('WARNING: Environment variable CUDA_HOME not found, building in CPU-only mode.')

    Extension = CppExtension
    sources = ['extension_cpu.cpp', 'matmul_quant_cpu.cpp']

setup(
    name='matmul_quant',
    ext_modules=[
        Extension(
            'matmul_quant',
            sources=sources,
            extra_compile_args={'cxx': cxx_flags, **extra_compile_args},
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
