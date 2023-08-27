# document databases

import os
import re
import json
import torch

from math import ceil, inf
from operator import itemgetter
from itertools import chain, groupby, islice, accumulate
from pathlib import Path
from torch.nn.functional import normalize

from llm import DEFAULT_EMBED, HuggingfaceEmbedding
from index import TorchVectorIndex
from utils import process

##
## Utils
##

# default paragraph splitter
def paragraph_splitter(text, delim='\n{2,}', minlen=1):
    paras = [para.strip() for para in re.split(delim, text)]
    return [para for para in paras if len(para) >= minlen]

# group tuples by `idx` element, preserving other orders
def groupby_dict(tups, idx=0):
    getter = itemgetter(idx)
    tups = sorted(tups, key=getter)
    return {
        i: [k for _, k in j] for i, j in groupby(tups, key=getter)
    }

# robust text reader (for encoding errors)
def robust_read(path):
    with open(path, 'r', errors='ignore') as fid:
        return fid.read()

# get batch indices
def batch_indices(length, batch_size):
    return [(i, min(i+batch_size, length)) for i in range(0, length, batch_size)]

# cumulative sum
def cumsum(lengths):
    return list(chain([0], accumulate(lengths)))

# get cumulative indices
def cumul_indices(lengths):
    sums = cumsum(lengths)
    return [(i, j) for i, j in zip(sums[:-1], sums[1:])]

# generate loader for jsonl file
def stream_jsonl(path, maxrows=None):
    with open(path) as fid:
        for i, line in enumerate(fid):
            if maxrows is not None and i >= maxrows:
                break
            yield json.loads(line)

# generate (resolved) batches from generator
def batch_generator(gen, batch_size):
    while (batch := list(islice(gen, batch_size))) != []:
        yield batch

# data storage:
# chunks: dict {name: [chunk1, chunk2, ...]}
# index: TorchVectorIndex {(name, chunk_idx): vec}
class DocumentDatabase:
    def __init__(
            self, embed=DEFAULT_EMBED, delim='\n{2,}', minlen=1, batch_size=1024,
            model_device='cuda', index_device='cpu', doc_index=True, dims=None, **kwargs
        ):
        # instantiate model and embedding
        self.embed = HuggingfaceEmbedding(embed, device=model_device) if type(embed) is str else embed
        self.dims = dims if dims is not None else self.embed.dims

        # set up options
        self.splitter = lambda s: paragraph_splitter(s, delim=delim, minlen=minlen)
        self.batch_size = batch_size

        # initalize index
        self.chunks = {}
        self.cindex = TorchVectorIndex(self.dims, device=index_device, **kwargs)
        self.dindex = TorchVectorIndex(self.dims, device=index_device, **kwargs) if doc_index else None

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', doc_batch=1024, maxrows=None,
        progress=True, maxsize=10, **kwargs
    ):
        self = cls(**kwargs)
        def loader(path):
            lines = stream_jsonl(path, maxrows=maxrows)
            for batch in batch_generator(lines, doc_batch):
                yield [(row[name_col], row[text_col]) for row in batch]
        def indexer(data):
            self.index_docs(data)
            if progress:
                print('█', end='', flush=True)
        process(loader(path), indexer, maxsize=maxsize)
        return self

    @classmethod
    def from_torch(cls, path, **kwargs):
        self = cls(**kwargs)
        data = torch.load(path)
        self.chunks = data['chunks']
        self.cindex.load(data['cindex'])
        self.dindex.load(data['dindex'])
        return self

    def save(self, path=None, compress=True):
        if compress:
            self.cindex.compress()
            self.dindex.compress()
        data = {
            'chunks': self.chunks,
            'cindex': self.cindex.save(),
            'dindex': self.dindex.save()
        }
        if path is None:
            return data
        else:
            torch.save(data, path)

    def index_docs(self, texts):
        # split into chunks dict
        targ = texts.items() if type(texts) is dict else texts
        chunks = {k: self.splitter(v) for k, v in targ}
        chunks = {k: v for k, v in chunks.items() if len(v) > 0}

        # get names and labels
        names = list(chunks)
        labels = [(n, j) for n, c in chunks.items() for j in range(len(c))]
        groups = [n for n, _ in labels]

        # assess chunk information
        chunk_sizes = [len(c) for c in chunks.values()]
        nchunks = sum(chunk_sizes)
        nbatch = int(ceil(nchunks/self.batch_size))

        # embed chunks with chosen batch_size
        chunk_iter = chain(*chunks.values())
        embeds = torch.cat([
            self.embed.embed(list(islice(chunk_iter, self.batch_size))) for i in range(nbatch)
        ], dim=0)

        # update chunks and add to index
        self.chunks.update(chunks)
        self.cindex.add(labels, embeds, groups=groups)

        # make document level embeddings
        if self.dindex is not None:
            docemb = normalize(torch.stack([
                embeds[i:j,:].mean(0) for i, j in cumul_indices(chunk_sizes)
            ], dim=0))
            self.dindex.add(names, docemb)

    def search(self, query, kd=25, kc=10, cutoff=-torch.inf):
        # embed query string
        qvec = self.embed.embed(query).squeeze()

        # search document index
        docs = self.dindex.search(qvec, kd, return_simil=False) if self.dindex is not None else None
        labs, sims = self.cindex.search(qvec, kc, groups=docs)
        match = list(zip(labs, sims.tolist()))

        # group by document and filter by cutoff
        locs = groupby_dict([l for l, v in match if v > cutoff])
        text = {k: [self.chunks[k][i] for i in v] for k, v in locs.items()}

        # return text
        return text

# index documents in a specified directory
class FilesystemDatabase(DocumentDatabase):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path)
        self.reindex()

    def save(self, path):
        data = super().save()
        data['times'] = {n: self.get_mtime(n) for n in self.get_names()}
        torch.save(data, path)

    def load(self, path):
        data = super().load(path)
        self.times = data['times']

    def get_names(self):
        return sorted(os.listdir(self.path))

    def get_mtime(self, name):
        return os.path.getmtime(self.path / name)

    def get_text(self, name):
        return robust_read(self.path / name)

    def clear(self, names=None):
        names = self.get_names() if names is None else names
        self.cindex.remove(func=lambda x: x[0] in names)
        self.dindex.remove(labs=names)

    def reindex(self, names=None):
        names = self.get_names() if names is None else names
        self.clear(names=names)
        self.times = {n: self.get_mtime(n) for n in names}
        self.index_docs((n, self.get_text(n)) for n in names)

    def refresh(self, names=None):
        names = self.get_names() if names is None else names
        update = [
            name for name in names if self.get_mtime(name) > self.times.get(name, -inf)
        ]
        self.reindex(names=update)
