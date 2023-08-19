from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='matmul_quant',
    ext_modules=[
        CppExtension(
            'matmul_quant',
            sources=['matmul_quant.cpp'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-lgomp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
