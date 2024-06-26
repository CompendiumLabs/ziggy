# pure torch (quantized) matmul implementation

import torch

from ..utils import batch_indices

# ensure that half-half matmul uses fp32 accumulation
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

def _quantize(x, bits, scale, zero_point):
    N, K = x.shape
    device = x.device

    QFACT = 8 // bits
    K1 = K // QFACT

    # unswizzle indices
    index = torch.arange(K, device=device)
    col = index % QFACT
    shift = bits * col

    # quantize values
    xf = torch.clamp(x / scale + zero_point, 0.0, 255.0)
    xi = torch.round(xf).to(torch.uint8)

    # pack bits
    xq = xi << shift[None,:]
    xp = xq.reshape((N, K1, QFACT)).sum(dim=2, dtype=torch.uint8)

    return xp

def _dequantize(x, bits, scale, zero_point, dtype):
    N, K1 = x.shape
    device = x.device

    QMASK = (1 << bits) - 1
    QFACT = 8 // bits
    K = QFACT * K1

    # unswizzle indices
    index = torch.arange(K, device=device)
    row, col = index // QFACT, index % QFACT
    shift = bits * col

    # unpack bits
    xi = (x[:,row] >> shift[None,:]) & QMASK

    # dequantize values
    xf = scale * (xi.to(dtype) - zero_point)

    return xf

# expects x: [N, K], y: [K, M] with N >> M
def _matmul_float(x, y):
    # compute matmul
    z = x @ y

    return z

# expects x: [N, K1] quantized, y: [K, M] unquantized with N >> M
def _matmul_quant(x, y, bits, scale, zero_point):
    # tensor shape
    N, Kx = x.shape
    Ky, M = y.shape

    # quant params
    QFACT = 8 // bits
    QMASK = (1 << bits) - 1
    K, K1 = Ky, Kx

    # output type
    device = x.device
    dtype = y.dtype

    # unswizzle indices
    index = torch.arange(K, device=device)
    row, col = index // QFACT, index % QFACT
    shift = bits * col

    # unpack bits
    xi = (x[:,row] >> shift[None,:]) & QMASK

    # dequantize and multiply
    xf = xi.to(dtype) - zero_point
    z = scale * (xf @ y)

    return z

##
## batchifried
##

# these wrappers are too varied to be simplified with a decorator

BATCH_SIZE = 1024

def quantize(x, bits, scale, zero_point, batch_size=BATCH_SIZE):
    N, K = x.shape
    device = x.device
    QFACT = 8 // bits
    K1 = K // QFACT

    assert(x.is_contiguous())
    assert(8 % bits == 0)
    assert(K % 8 == 0)

    out = torch.empty((N, K1), device=device, dtype=torch.uint8)
    for i1, i2 in batch_indices(N, batch_size):
        out[i1:i2] = _quantize(x[i1:i2], bits, scale, zero_point)

    return out

def dequantize(x, bits, scale, zero_point, dtype, batch_size=BATCH_SIZE):
    N, K1 = x.shape
    device = x.device
    QFACT = 8 // bits
    K = QFACT * K1

    assert(x.is_contiguous())
    assert(8 % bits == 0)
    assert(x.dtype == torch.uint8)

    out = torch.empty((N, K), device=device, dtype=dtype)
    for i1, i2 in batch_indices(N, batch_size):
        out[i1:i2] = _dequantize(x[i1:i2], bits, scale, zero_point, dtype)

    return out

def matmul_float(x, y, batch_size=BATCH_SIZE):
    N, Kx = x.shape
    Ky, M = y.shape
    device = x.device

    assert(Kx == Ky)
    assert(y.device == device)

    out = torch.empty((N, M), device=device, dtype=torch.float32)
    for i1, i2 in batch_indices(N, batch_size):
        out[i1:i2] = _matmul_float(x[i1:i2], y)

    return out

def matmul_quant(x, y, bits, scale, zero_point, batch_size=BATCH_SIZE):
    N, Kx = x.shape
    Ky, M = y.shape
    QFACT = 8 // bits
    device = x.device

    assert(x.is_contiguous())
    assert(8 % bits == 0)
    assert(QFACT * Kx == Ky)
    assert(y.device == device)

    out = torch.empty((N, M), device=device, dtype=torch.float32)
    for i1, i2 in batch_indices(N, batch_size):
        out[i1:i2] = _matmul_quant(x[i1:i2], y, bits, scale, zero_point)

    return out
