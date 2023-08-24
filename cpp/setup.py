from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='matmul_quant',
    ext_modules=[
        CUDAExtension(
            'matmul_quant',
            sources=['extension.cpp', 'matmul_quant.cpp', 'matmul_kernels.cu'],
            extra_compile_args={
                'cxx': ['-fopenmp', '-mavx512f', '-std=c++17'],
                'nvcc': ['--compiler-bindir=/usr/bin/cuda']
            },
            extra_link_args=['-lgomp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
