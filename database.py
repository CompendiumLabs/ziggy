# document Databases


import os
import re
import json
import torch

from math import ceil, inf
from operator import itemgetter
from itertools import chain, groupby, islice, accumulate
from pathlib import Path
from torch.nn.functional import normalize

from llm import (
    DEFAULT_MODEL, DEFAULT_EMBED, DEFAULT_SYSTEM_PROMPT,
    sprint, HuggingfaceModel, HuggingfaceEmbedding
)
from index import TorchVectorIndex

##
## Utils
##

# default paragraph splitter
def paragraph_splitter(text, delim='\n{2,}', minlen=1):
    paras = [para.strip() for para in re.split(delim, text)]
    return [para for para in paras if len(para) >= minlen]

# group tuples by first element
def groupby_dict(tups, idx=0):
    return {
        i: [k for _, k in j] for i, j in groupby(tups, key=itemgetter(idx))
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
            self, model=DEFAULT_MODEL, embed=DEFAULT_EMBED, index=TorchVectorIndex,
            delim='\n{2,}', minlen=1, batch_size=8192, device='cuda', dtype=torch.float16, **kwargs
        ):
        # instantiate model and embedding
        self.model = HuggingfaceModel(model, device=device, **kwargs) if type(model) is str else model
        self.embed = HuggingfaceEmbedding(embed, device=device) if type(embed) is str else embed

        # set up options
        self.splitter = lambda s: paragraph_splitter(s, delim=delim, minlen=minlen)
        self.batch_size = batch_size

        # initalize index
        self.chunks = {}
        self.cindex = index(self.embed.dims, device=device, dtype=dtype)
        self.dindex = index(self.embed.dims, device=device, dtype=dtype)

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', doc_batch=1024, maxrows=None, progress=True, **kwargs
    ):
        self = cls(**kwargs)
        stream = stream_jsonl(path, maxrows=maxrows)
        for batch in batch_generator(stream, doc_batch):
            self.index_docs((row[name_col], row[text_col]) for row in batch)
            if progress:
                print('█', end='', flush=True)
        return self

    @classmethod
    def from_torch(cls, path, **kwargs):
        self = cls(**kwargs)
        data = torch.load(path)
        self.chunks = data['chunks']
        self.cindex.load(data['cindex'])
        self.dindex.load(data['dindex'])
        return self

    def save(self, path=None):
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
            self.embed.embed(islice(chunk_iter, self.batch_size)) for i in range(nbatch)
        ], dim=0)

        # make document level embeddings
        docemb = normalize(torch.stack([
            embeds[i:j,:].mean(0) for i, j in cumul_indices(chunk_sizes)
        ], dim=0))

        # update chunks and add to index
        self.chunks.update(chunks)
        self.cindex.add(labels, embeds, groups=groups)
        self.dindex.add(names, docemb)

    def search(self, query, k=10, groups=None, cutoff=0.0):
        # get relevant chunks
        qvec = self.embed.embed(query).squeeze()
        docs = self.dindex.search(qvec, k, return_simil=False)
        labs, sims = self.cindex.search(qvec, k, groups=docs)
        match = list(zip(labs, sims.tolist()))

        # group by document and filter by cutoff
        locs = groupby_dict([l for l, v in match if v > cutoff])
        text = {k: [self.chunks[k][i] for i in v] for k, v in locs.items()}

        # return text
        return text

    def query(self, query, context=2048, maxlen=2048, **kwargs):
        # search db and get some context
        matches = self.search(query, **kwargs)
        chunks = {k: '; '.join(v) for k, v in matches.items()}
        notes = '\n'.join([f'{k}: {v}' for k, v in chunks.items()])

        # construct prompt
        meta = 'Using a synthesis of your general knowledge and the text given below, answer the question posed at the end concisely.'
        system = f'{DEFAULT_SYSTEM_PROMPT}\n\n{meta}'
        user = f'TEXT:\n{notes}\n\nQUESTION: {query}'

        # generate response
        yield from self.model.generate(user, chat=system, context=context, maxlen=maxlen)

    def iquery(self, query, **kwargs):
        for s in self.query(query, **kwargs):
            sprint(s)

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