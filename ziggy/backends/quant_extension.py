# wrapper for quantize extension

import torch

from matmul_quant import (
    quantize as _quantize,
    dequantize as _dequantize,
    matmul_quant as _matmul_quant,
)
from .quant_torch import matmul_float

def quantize(x, bits, scale, zero_point):
    if x.device.type == 'cpu':
        assert(x.dtype == torch.float32)

    return _quantize(x, bits, scale, zero_point)

def dequantize(x, bits, scale, zero_point, dtype):
    dtype_str = 'half' if dtype == torch.float16 else 'float'

    assert(x.dtype == torch.uint8)
    if x.device.type == 'cpu':
        assert(dtype == torch.float32)

    return _dequantize(x, bits, scale, zero_point, dtype_str)

def matmul_quant(x, y, bits, scale, zero_point):
    N, K1 = x.shape
    K, M = y.shape

    QFACT = 8 // bits

    assert(x.device == y.device)
    assert(x.dtype == torch.uint8)
    assert(QFACT * K1 == K)
    if y.device.type == 'cpu':
        assert(y.dtype == torch.float32)

    return _matmul_quant(x, y, bits, scale, zero_point)
