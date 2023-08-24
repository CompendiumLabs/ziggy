# vector index code

import faiss
import torch

from math import ceil, log2

from utils import IndexDict

##
## Load fast quantization extension
##

import matmul_quant

# here `a` is considered big and immovable
# transpose pattern is to keep `a` contiguous
def similarity(a, b):
    # move `b` to same device as `a`
    b = b.to(device=a.device)

    # break down by quantization and device
    if a.is_quantized:
        if a.device.type == 'cpu' and a.dtype == torch.qint8:
            return matmul_quant.matmul_qint8_float32_cpu(a, b.T.float()).T
        elif a.device.type == 'cuda' and a.dtype == torch.qint8:
            return matmul_quant.matmul_qint8_float16_cuda(a, b.T.half()).T
        else:
            raise Exception(f'Unsupported device/dtype: {a.device}/{a.dtype}')
    else:
        return (a @ b.to(dtype=a.dtype)).T

##
## Utils
##

def next_power_of_2(x):
    return pow(2, round(ceil(log2(x))))

def resize_alloc(a, size):
    a.resize_(size, *a.shape[1:])

def double_alloc(a):
    resize_alloc(a, 2*a.size(0))

##
## Pure Torch
##

class TorchVectorIndex:
    def __init__(
        self, dims=None, size=1024, load=None, device='cuda', dtype=None,
        scale=4.0/128, zero_point=0
    ):
        # default datatype
        if dtype is None:
            dtype = torch.float16 if device == 'cuda' else torch.float32

        # set runtime options
        self.device = device

        # init state
        if load is not None:
            self.load(load)
        else:
            # set up storage
            self.labels = []
            self.grpids = IndexDict()
            self.allocate(
                size, dims, dtype=dtype, scale=scale, zero_point=zero_point
            )

    def allocate(self, size, dims, dtype=torch.float32, scale=None, zero_point=None):
        if dtype == torch.qint8:
            self.values = torch._empty_affine_quantized(
                size, dims, device=self.device, dtype=torch.qint8,
                scale=scale, zero_point=zero_point
            )
        else:
            self.values = torch.empty(size, dims, device=self.device, dtype=dtype)
        self.groups = torch.empty(size, device=self.device, dtype=torch.int32)

    def size(self):
        return len(self.labels)

    def load(self, path):
        data = torch.load(path) if type(path) is str else path
        self.labels = data['labels']
        self.grpids = IndexDict(data['grpids'])
        self.values = data['values']
        self.groups = data['groups']

    def save(self, path=None):
        size = self.size()
        data = {
            'labels': self.labels,
            'grpids': dict(self.grpids),
            'values': self.values[:size,:],
            'groups': self.groups[:size],
        }
        if path is not None:
            torch.save(data, path)
        else:
            return data

    def expand(self, size):
        if size > self.values.size(0):
            resize_alloc(self.values, size)
            resize_alloc(self.groups, size)

    def compress(self, min_size=1024):
        size = max(min_size, self.size())
        if size < self.values.size(0):
            resize_alloc(self.values, size)
            resize_alloc(self.groups, size)

    def add(self, labs, vecs, groups=None, strict=False):
        # ensure on device
        vecs = vecs.to(self.device)

        # validate input size
        nlabs = len(labs)
        nv, dv = vecs.shape
        _, d0 = self.values.shape
        assert(nv == nlabs)
        assert(dv == d0)

        # get breakdown of new vs old
        slabs = set(labs)
        exist = slabs.intersection(self.labels)
        novel = slabs - exist

        # raise if trying invalid strict add
        if strict and len(exist) > 0:
            raise Exception(f'Trying to add existing labels in strict mode.')

        # expand groups if needed
        self.grpids.add(groups)
        gidx = self.grpids.idx(groups)
        if type(groups) is list:
            gids = torch.tensor(gidx, device=self.device, dtype=torch.int32)
        else:
            gids = torch.full((nlabs,), gidx, device=self.device, dtype=torch.int32)

        if len(exist) > 0:
            # update existing
            elocs, idxs = map(list, zip(*[
                (i, self.labels.index(x)) for i, x in enumerate(labs) if x in exist
            ]))
            self.values[idxs,:] = vecs[elocs,:]
            self.groups[idxs] = gids[elocs]

        if len(novel) > 0:
            # get new labels in input order
            xlocs, xlabs = map(list, zip(*[
                (i, x) for i, x in enumerate(labs) if x in novel
            ]))

            # expand size if needed
            nlabels0 = self.size()
            nlabels1 = nlabels0 + len(novel)
            self.expand(nlabels1)

            # add in new labels and vectors
            self.labels.extend(xlabs)
            self.values[nlabels0:nlabels1,:] = vecs[xlocs,:]
            self.groups[nlabels0:nlabels1] = gids[xlocs]

    def remove(self, labs=None, func=None):
        labs = [l for l in self.labels if func(l)] if func is not None else labs
        for lab in set(labs).intersection(self.labels):
            idx = self.labels.index(lab)
            self.labels.pop(idx)
            self.values[idx,:] = self.values[self.size(),:]
            self.groups[idx] = self.groups[self.size(),:]

    def clear(self):
        self.labels = []

    def get(self, labels):
        idxs = [self.labels.index(l) for l in labels]
        return self.values[idxs,:]

    def search(self, vecs, k, groups=None, return_simil=True):
        # allow for single vec
        squeeze = vecs.ndim == 1
        if squeeze:
            vecs = vecs.unsqueeze(0)

        # clamp k to max size
        num = self.size()
        k1 = min(k, num)

        # get compare values
        if groups is None:
            labs = self.labels
            vals = self.values[:num,:]
        else:
            ids = torch.tensor(self.grpids.idx(groups), device=self.device, dtype=torch.int32)
            sel = torch.isin(self.groups[:num], ids)
            idx = torch.nonzero(sel).squeeze()
            labs = [self.labels[i] for i in idx]
            vals = self.values[idx,:]

        # compute similarity matrix
        sims = similarity(vals, vecs)

        # get top results
        tops = sims.topk(k1)
        klab = [[labs[i] for i in row] for row in tops.indices]
        kval = tops.values

        # return single vec if needed
        if squeeze:
            klab, kval = klab[0], kval[0]

        # return labels/simils
        return (klab, kval) if return_simil else klab

##
## FAISS
##

# this won't handle deletion
class FaissIndex:
    def __init__(self, dims, spec='Flat', device='cuda'):
        self.dims = dims
        self.labels = []
        self.values = faiss.index_factory(dims, spec)

        # move to gpu if needed
        if device == 'cuda':
            res = faiss.StandardGpuResources()
            self.values = faiss.index_cpu_to_gpu(res, 0, self.values)

    def size(self):
        return len(self.labels)

    def load(self, path):
        data = torch.load(path) if type(path) is str else path
        self.labels = data['labels']
        self.values = self.values.add(data['values'])

    def save(self, path):
        data = {
            'labels': self.labels,
            'values': self.values.reconstruct_n(0, self.size())
        }
        if path is not None:
            torch.save(data, path)
        else:
            return data

    def add(self, labs, vecs):
        # validate input size
        nlabs = len(labs)
        nv, dv = vecs.shape
        assert(nv == nlabs)
        assert(dv == self.dims)

        # reject adding existing labels
        exist = set(labs).intersection(self.labels)
        if len(exist) > 0:
            raise Exception(f'Adding existing labels not supported.')

        # construct label ids
        size0 = self.size()
        size1 = size0 + nlabs
        ids = torch.arange(size0, size1)

        # add to index
        self.labels.extend(labs)
        self.values.add(vecs)

    def search(self, vecs, k, return_simil=True):
        # allow for single vec
        squeeze = vecs.ndim == 1
        if squeeze:
            vecs = vecs.unsqueeze(0)

        # exectute search
        vals, ids = self.values.search(vecs, k)
        labs = [[self.labels[i] for i in row] for row in ids]

        # return single vec if needed
        if squeeze:
            labs, vals = labs[0], vals[0]

        # return labels/simils
        return labs, vals if return_simil else labs
