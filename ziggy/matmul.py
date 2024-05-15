# combined matmul interface

import torch

from .utils import MissingModule

##
## load available submodules
##

try:
    import matmul_quant as matmul_extension

    # monkey patch dequantize to use standard dtype
    matmul_extension._dequantize = matmul_extension.dequantize
    def dequantize(x, bits, scale, zero_point, dtype):
        dtype_str = 'half' if dtype == torch.float16 else 'float'
        return matmul_extension._dequantize(x, bits, scale, zero_point, dtype_str)
    matmul_extension.dequantize = dequantize

    # assign extension for both devices
    matmul_cpu = matmul_extension
    matmul_cuda = matmul_extension
except ImportError:
    from . import matmul_torch as matmul_cpu
    try:
        from . import matmul_triton as matmul_cuda
    except ImportError:
        matmul_cuda = MissingModule('Failed to import triton matmul module')

##
## define general routing routines
##

def quantize(x, bits, scale, zero_point):
    if x.device.type == 'cuda':
        return matmul_cuda.quantize(x, bits, scale, zero_point)
    else:
        return matmul_cpu.quantize(x, bits, scale, zero_point)

def dequantize(x, bits, scale, zero_point, dtype):
    if x.device.type == 'cuda':
        return matmul_cuda.dequantize(x, bits, scale, zero_point, dtype)
    else:
        return matmul_cpu.dequantize(x, bits, scale, zero_point, dtype)

def matmul_float(x, y):
    if x.device != y.device:
        y = y.to(x.device)
    if x.device.type == 'cuda':
        return matmul_cuda.matmul_float(x, y)
    else:
        return matmul_cpu.matmul_float(x, y)

def matmul_quant(x, y, bits, scale, zero_point):
    if x.device != y.device:
        y = y.to(x.device)
    if x.device.type == 'cuda':
        return matmul_cuda.matmul_quant(x, y, bits, scale, zero_point)
    else:
        return matmul_cpu.matmul_quant(x, y, bits, scale, zero_point)
