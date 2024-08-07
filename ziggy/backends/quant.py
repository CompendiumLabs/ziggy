# combined matmul interface

import os
import torch

from ..utils import MissingModule

##
## load available submodules
##

def get_bool_env(name, default=False):
    if name in os.environ:
        value = os.environ[name].lower()
        return value in ['true', '1']
    else:
        return default

try:
    disable_extension = get_bool_env('ZIGGY_DISABLE_EXTENSION')
    assert(not disable_extension)

    from . import quant_extension
    quant_cpu = quant_extension
    quant_cuda = quant_extension
except (ImportError, AssertionError) as e:
    from . import quant_torch as quant_cpu
    try:
        from . import quant_triton as quant_cuda
    except ImportError:
        quant_cuda = MissingModule('Failed to import triton quantize module')

##
## define general routing routines
##

def quantize(x, bits, scale, zero_point):
    if x.device.type == 'cuda':
        return quant_cuda.quantize(x, bits, scale, zero_point)
    else:
        return quant_cpu.quantize(x, bits, scale, zero_point)

def dequantize(x, bits, scale, zero_point, dtype):
    if x.device.type == 'cuda':
        return quant_cuda.dequantize(x, bits, scale, zero_point, dtype)
    else:
        return quant_cpu.dequantize(x, bits, scale, zero_point, dtype)

def matmul_float(x, y):
    if x.device != y.device:
        y = y.to(x.device)
    if x.device.type == 'cuda':
        return quant_cuda.matmul_float(x, y)
    else:
        return quant_cpu.matmul_float(x, y)

def matmul_quant(x, y, bits, scale, zero_point):
    if x.device != y.device:
        y = y.to(x.device)
    if x.device.type == 'cuda':
        return quant_cuda.matmul_quant(x, y, bits, scale, zero_point)
    else:
        return quant_cpu.matmul_quant(x, y, bits, scale, zero_point)
