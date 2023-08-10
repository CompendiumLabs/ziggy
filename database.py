# document Databases


import os
import re
import json
import torch

from math import ceil, inf
from operator import itemgetter
from itertools import chain, groupby, islice
from pathlib import Path

from llm import (
    DEFAULT_MODEL, DEFAULT_EMBED, DEFAULT_SYSTEM_PROMPT,
    sprint, HuggingfaceModel, HuggingfaceEmbedding
)
from index import TorchVectorIndex
from utils import IndexedDict

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

# generate loader for jsonl file
def stream_jsonl(path):
    with open(path) as fid:
        for line in fid:
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
            self, model=DEFAULT_MODEL, embed=DEFAULT_EMBED, index=None,
            delim='\n{2,}', minlen=1, batch_size=8192, **kwargs
        ):
        # instantiate model and embedding
        self.model = HuggingfaceModel(model, **kwargs) if type(model) is str else model
        self.embed = HuggingfaceEmbedding(embed) if type(embed) is str else embed

        # set up options
        self.splitter = lambda s: paragraph_splitter(s, delim=delim, minlen=minlen)
        self.batch_size = batch_size

        # initalize index
        self.chunks = IndexedDict()
        self.index = index if index is not None else TorchVectorIndex(self.embed.dims)

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', doc_batch=1024, progress=True, **kwargs
    ):
        self = cls(**kwargs)
        for batch in batch_generator(stream_jsonl(path), doc_batch):
            self.index_docs((row[name_col], row[text_col]) for row in batch)
            if progress:
                print('█', end='', flush=True)
        return self

    @classmethod
    def from_torch(cls, path, **kwargs):
        self = cls(**kwargs)
        data = torch.load(path)
        self.chunks = data['chunks']
        self.index.load(data['index'])
        return self

    def save(self, path=None):
        data = {
            'chunks': self.chunks,
            'index': self.index.save()
        }
        if path is None:
            return data
        else:
            torch.save(data, path)

    def index_docs(self, texts):
        # split into chunks dict
        targ = texts.items() if type(texts) is dict else texts
        chunks = {k: self.splitter(v) for k, v in targ}
        labels = [(n, j) for n, c in chunks.items() for j in range(len(c))]

        # assess chunk information
        chunk_sizes = [len(c) for c in chunks.values()]
        nchunks = sum(chunk_sizes)
        nbatch = int(ceil(nchunks/self.batch_size))

        # embed chunks with chosen batch_size
        chunk_iter = chain(*chunks.values())
        embeds = torch.cat([
            self.embed.embed(islice(chunk_iter, self.batch_size)) for i in range(nbatch)
        ], dim=0)

        # update chunks and get group indices
        self.chunks.update(chunks)
        chunk_groups = [self.chunks.index(n) for n in chunks]
        groups = torch.tensor([
            g for g, s in zip(chunk_groups, chunk_sizes) for _ in range(s)
        ], dtype=torch.int32)

        # add to the index
        self.index.add(labels, embeds, groups=groups)

    def search(self, query, k=10, cutoff=0.0):
        # get relevant chunks
        qvec = self.embed.embed(query).squeeze()
        labs, vecs = self.index.search(qvec, k)
        match = list(zip(labs, vecs.tolist()))

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
        meta = 'Below is some relevant text from my person notes. Using a synthesis of your general knowledge and my notes, answer the question posed at the end concisely. Try to provide quotes from my notes as evidence when possible.'
        system = f'{DEFAULT_SYSTEM_PROMPT}\n\n{meta}'
        user = f'NOTES:\n{notes}\n\nQUESTION: {query}'

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
        self.index.remove(func=lambda x: x[0] in names)

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
