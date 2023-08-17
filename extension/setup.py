from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='matmul_quant',
    ext_modules=[
        CppExtension('matmul_quant', ['matmul_quant.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
