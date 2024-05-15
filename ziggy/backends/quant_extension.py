# wrapper for quantize extensions

import torch

from matmul_quant import quantize, dequantize as _dequantize, matmul_quant
from .quant_torch import matmul_float

def dequantize(x, bits, scale, zero_point, dtype):
    dtype_str = 'half' if dtype == torch.float16 else 'float'
    return _dequantize(x, bits, scale, zero_point, dtype_str)
