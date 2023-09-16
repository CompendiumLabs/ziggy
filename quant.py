# handle quantized embedding matrices

from typing import Any
import torch
from enum import Enum

from utils import resize_alloc

##
## Load fast quantization extension
##

import matmul_quant

##
## QuantizedEmbedding
##

class QuantType(Enum):
    float = 0
    half = 1
    qint8 = 2
    qint4 = 3

def is_quantized(qtype):
    return qtype in (QuantType.qint8, QuantType.qint4)

def qtype_to_dtype(qtype):
    if is_quantized(qtype):
        return torch.uint8
    elif qtype == QuantType.float:
        return torch.float
    elif qtype == QuantType.half:
        return torch.half
    else:
        raise ValueError(f'Invalid quantization type: {qtype}')

def qtype_to_bits(qtype):
    if qtype == QuantType.float:
        return 32
    elif qtype == QuantType.half:
        return 16
    elif qtype == QuantType.qint8:
        return 8
    elif qtype == QuantType.qint4:
        return 4
    else:
        raise ValueError(f'Invalid quantization type: {qtype}')

class QuantSpec:
    def __init__(self, qtype, scale=None, zero_point=None):
        # core spec
        self.qtype = qtype
        self.scale = scale
        self.zero_point = zero_point

        # derived stats
        self.is_quantized = is_quantized(qtype)
        self.bits = qtype_to_bits(qtype)
        self.dtype = qtype_to_dtype(qtype)

        # ensure valid
        if self.is_quantized and (scale is None or zero_point is None):
            raise ValueError('`scale` and `zero_point` must be specified for quantized embeddings')

    @classmethod
    def from_width(cls, qtype, zero, width):
        bits = qtype_to_bits(qtype)
        imax = (1 << bits) - 1
        scale = (2*width)/imax
        zero_point = (width-zero)/scale
        return cls(qtype, scale, zero_point)

    @classmethod
    def from_range(cls, qtype, min, max):
        width = 0.5*(max-min)
        zero = 0.5*(max+min)
        return cls.from_width(qtype, zero, width)

Half = QuantSpec(QuantType.half)
Float = QuantSpec(QuantType.float)

class RowAccessor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx,:]

    def __setitem__(self, idx, vec):
        self.data[idx,:] = vec

class QuantizedEmbedding:
    def __init__(self, size, dims, qspec=Float, device='cuda', data=None):
        # set up options
        self.device = device
        self.qspec = qspec

        # allocate storage
        if data is None:
            self.data = torch.empty(size, dims, device=device, dtype=self.qspec.dtype)
        else:
            self.data = data

        # set up raw access
        self.raw = RowAccessor(self.data)

    @classmethod
    def load(cls, data):
        size, dims = data['data'].size()
        return cls(size, dims, qspec=data['qspec'], data=data['data'])

    def save(self):
        return {
            'qspec': self.qspec,
            'data': self.data,
        }

    def size(self):
        return self.data.size(0)

    def dims(self):
        return self.data.size(1)

    def resize(self, size):
        resize_alloc(self.data, size)

    def __setitem__(self, idx, vec):
        if self.qspec.is_quantized:
            if vec.device == 'cpu':
                qvec = matmul_quant.quantize_and_pack_cpu(vec)
            elif vec.device == 'cuda':
                qvec = matmul_quant.quantize_and_pack_cuda(vec)
        else:
            qvec = vec
        self.data[idx,:] = qvec

    def __getitem__(self, idx):
        raise NotImplementedError('Need to implement __getitem__')

    # here `a` is considered big and immovable
    # transpose pattern is to keep `a` contiguous
    def similarity(self, vecs, mask=None):
        # move `b` to same device as `a`
        vecs = vecs.to(device=self.device)

        # mask in desired entries
        if mask is None:
            data = self.data
        elif type(mask) is int:
            data = self.data[:mask]
        elif type(mask) is torch.Tensor and mask.dtype == torch.bool:
            data = self.data.index_select(0, mask)
        else:
            raise ValueError(f'Invalid mask type: {type(mask)}')

        # break down by quantization and device type
        if self.is_quantized:
            if self.device.type == 'cpu':
                return matmul_quant.matmul_quant_cpu(
                    data, vecs.float().T, self.qspec.bits,
                    self.qspec.scale, self.qspec.zero_point
                ).T
            elif self.device.type == 'cuda':
                return matmul_quant.matmul_quant_cuda(
                    data, vecs.T, self.qspec.bits,
                    self.qspec.scale, self.qspec.zero_point
                ).T
        else:
            return (data @ vecs.to(dtype=vecs.dtype).T).T
