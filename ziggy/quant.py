# handle quantized embedding matrices

from typing import Any
from enum import Enum
import torch

from .utils import resize_alloc, MissingModule

##
## Load fast quantization extension
##

try:
    import matmul_quant
except ImportError:
    matmul_quant = MissingModule(
        'You need to compile the "matmul_quant" extension for quantization support.'
    )

##
## QuantizedEmbedding
##

class QuantType(Enum):
    float = 0
    half = 1
    qint8 = 2
    qint4 = 3
    qint2 = 4
    qint1 = 5

def is_quantized(qtype):
    return qtype in (QuantType.qint8, QuantType.qint4, QuantType.qint2, QuantType.qint1)

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
    elif qtype == QuantType.qint2:
        return 2
    elif qtype == QuantType.qint1:
        return 1
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

    def __repr__(self):
        if self.is_quantized:
            return f'{self.qtype.name}(scale={self.scale:.5g}, zero_point={self.zero_point:.5g})'
        else:
            return f'{self.qtype.name}'

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

    @classmethod
    def load(cls, data):
        return cls(data['qtype'], scale=data['scale'], zero_point=data['zero_point'])

    def save(self):
        return {
            'qtype': self.qtype,
            'scale': self.scale,
            'zero_point': self.zero_point,
        }

    def quantize(self, vec):
        if self.is_quantized:
            return matmul_quant.quantize_and_pack(
                vec, self.bits, self.scale, self.zero_point
            )
        else:
            return vec

    def dequantize(self, vec, dtype=None):
        if self.is_quantized:
            if dtype == None:
                dtype_str = 'half' if vec.device == 'cuda' else 'float'
            elif dtype == torch.half:
                dtype_str = 'half'
            elif dtype == torch.float:
                dtype_str = 'float'
            else:
                raise ValueError(f'Unsupported dtype: {dtype}')
            return matmul_quant.dequantize_and_unpack(
                vec, dtype_str, self.bits, self.scale, self.zero_point
            )
        else:
            return vec

    def matmul(self, a, b):
        if self.is_quantized:
            return matmul_quant.matmul_quant(
                a, b, self.bits, self.scale, self.zero_point
            )
        else:
            return a @ b

Half = QuantSpec(QuantType.half)
Float = QuantSpec(QuantType.float)

class Accessor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vec):
        self.data[idx] = vec

class QuantizedEmbedding:
    def __init__(self, size, dims, qspec=Float, device='cuda', data=None, qdata=None):
        # set up options
        self.device = device
        self.qspec = qspec

        # set up dims
        self.dims = dims
        if self.qspec.is_quantized:
            qfact = 8 // qspec.bits
            assert(dims % qfact == 0)
            qdims = dims // qfact
        else:
            qdims = dims

        # allocate storage
        if data is not None:
            self.data = self.quantize(data.to(device=device))
        elif qdata is not None:
            self.data = qdata.to(device=device)
        else:
            self.data = torch.empty(size, qdims, device=device, dtype=self.qspec.dtype)

        # set up raw access
        self.raw = Accessor(self.data)

    @classmethod
    def load(cls, data, device='cuda'):
        size, dims = data['data'].size()
        qspec = QuantSpec.load(data['qspec'])
        return cls(size, dims, qspec=qspec, qdata=data['data'], device=device)

    def save(self):
        return {
            'qspec': self.qspec.save(),
            'data': self.data,
        }

    def __len__(self):
        return self.data.size(0)

    def resize(self, size):
        resize_alloc(self.data, size)

    def quantize(self, vec):
        return self.qspec.quantize(vec)

    def dequantize(self, vec):
        return self.qspec.dequantize(vec)

    def __setitem__(self, idx, vec):
        self.data[idx] = self.quantize(vec)

    def __getitem__(self, idx):
        return self.dequantize(self.data[idx])

    def zero_(self):
        self.data.zero_()

    # here `data` is considered big and immovable
    # transpose pattern is to keep `data` contiguous
    def similarity(self, vecs, mask=None):
        # move `b` to same device as `a`
        vecs = vecs.to(device=self.device)

        # mask in desired entries
        if mask is None:
            data = self.data
        elif type(mask) is int:
            data = self.data[:mask]
        elif type(mask) is torch.Tensor:
            data = self.data.index_select(0, mask)
        else:
            raise ValueError(f'Invalid mask type: {type(mask)}')

        # break down by quantization and device type
        return self.qspec.matmul(data, vecs.T).T
