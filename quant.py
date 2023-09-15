# handle quantized embedding matrices

import torch
from enum import Enum

##
## Load fast quantization extension
##

import matmul_quant

##
## QuantizedEmbedding
##

class QType(Enum):
    float = 0
    half = 1
    qint8 = 2
    qint4 = 3

def is_quantized(qtype):
    return qtype in (QType.qint8, QType.qint4)

def qtype_to_dtype(qtype):
    if is_quantized(qtype):
        return torch.uint8
    elif qtype == QType.float:
        return torch.float
    elif qtype == QType.half:
        return torch.half
    else:
        raise ValueError(f'Invalid quantization type: {qtype}')

def qtype_to_bits(qtype):
    if qtype == QType.qint8:
        return 8
    elif qtype == QType.qint4:
        return 4
    else:
        raise ValueError(f'Invalid quantization type: {qtype}')

class QuantizedEmbedding:
    def __init__(self, size, dims, qtype=QType.float, scale=None, zero_point=None, device='cuda', allocate=True):
        if is_quantized(qtype) and (scale is None or zero_point is None):
            raise ValueError('`scale` and `zero_point` must be specified for quantized embeddings')

        # set runtime options
        self.device = device
        self.dtype = qtype_to_dtype(qtype)

        # set up quantization
        self.qtype = qtype
        self.is_quantized = is_quantized(qtype)
        if self.is_quantized:
            self.bits = qtype_to_bits(qtype)
            self.scale = scale
            if zero_point is None:
                self.zero_point = (1 << (self.bits - 1)) - 0.5
            else:
                self.zero_point = zero_point

        # set up storage
        self.size = size
        self.dims = dims

        # allocate storage
        if allocate:
            self.data = torch.empty(size, dims, dtype=self.dtype)

    # here `a` is considered big and immovable
    # transpose pattern is to keep `a` contiguous
    def similarity(self, vecs):
        # move `b` to same device as `a`
        vecs = vecs.to(device=self.device)

        # break down by quantization and device type
        if self.is_quantized:
            if self.device.type == 'cpu':
                return matmul_quant.matmul_quant_cpu(
                    self.data, vecs.float().T, self.bits, self.scale, self.zero_point
                ).T
            elif self.device.type == 'cuda':
                return matmul_quant.matmul_quant_cuda(
                    self.data, vecs.T, self.bits, self.scale, self.zero_point
                ).T
        else:
            return (self.data @ vecs.to(dtype=vecs.dtype).T).T
