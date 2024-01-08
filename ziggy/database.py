# document databases

import os
import re
import json
import torch
import mimetypes

from math import inf
from itertools import chain
from operator import itemgetter
from pathlib import Path
from glob import glob
from torch.nn.functional import normalize

from .embed import HuggingfaceEmbedding, DEFAULT_EMBED
from .index import TorchVectorIndex
from .quant import Float
from .utils import batch_generator, cumul_indices, list_splitter

##
## Utils
##

# default paragraph splitter
def text_splitter(text, delim, min_len=1, max_len=None):
    if delim is not None:
        paras = [p.strip() for p in re.split(delim, text)]
    else:
        paras = [text]
    paras = [p for p in paras if len(p) >= min_len]
    if max_len is not None:
        paras = list(chain.from_iterable(
            list_splitter(p, max_len) for p in paras
        ))
    return paras

# robust text reader (for encoding errors)
def read_text(path):
    with open(path, 'r', errors='ignore') as fid:
        return fid.read()

# read a pdf file in text
def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return '\n\n'.join([
        page.extract_text() for page in reader.pages
    ])

# generate loader for jsonl file
def stream_jsonl(path, maxrows=None):
    with open(path) as fid:
        for i, line in enumerate(fid):
            if maxrows is not None and i >= maxrows:
                break
            yield json.loads(line)

##
## generic text database
##

# data storage
# text — {label: text}
# index — TorchVectorIndex {label: vec}
class TextDatabase:
    def __init__(
            self, embed=DEFAULT_EMBED, embed_device='cuda', index_device='cuda',
            batch_size=128, allocate=True, dims=None, qspec=Float, **kwargs
        ):
        # instantiate embedding model
        if type(embed) is str:
            self.embed = HuggingfaceEmbedding(embed, device=embed_device, batch_size=batch_size)
        else:
            self.embed = embed

        # initalize index
        if allocate:
            self.dims = self.embed.dims if dims is None else dims
            self.text = {}
            self.index = TorchVectorIndex(self.dims, device=index_device, qspec=qspec, **kwargs)

    @classmethod
    def load(cls, path, device='cuda', **kwargs):
        data = torch.load(path, map_location=device) if type(path) is str else path
        self = cls(allocate=False, **kwargs)
        self.text = data['text']
        self.index = TorchVectorIndex.load(data['index'], device=device)
        return self

    def save(self, path=None):
        data = {
            'text': self.text,
            'index': self.index.save(),
        }
        if path is None:
            return data
        else:
            torch.save(data, path)

    def __len__(self):
        return len(self.text)

    def embed_text(self, text, threaded=True):
        return self.embed.embed(text, threaded=threaded, batch_size=self.batch_size)

    def index_text(self, labels, text):
        self.text.update(zip(labels, text))

    def index_vecs(self, labels, vecs, groups=None):
        self.index.add(labels, vecs, groups=groups)

    def add(self, labels, text, groups=None, threaded=True):
        vecs = self.embed.embed(text, threaded=threaded)
        self.index_text(labels, text)
        self.index_vecs(labels, vecs, groups=groups)
        return vecs

    def remove(self, labels):
        for l in labels:
            del self.text[l]
        self.index.remove(labels)

    def clear(self, zero=False):
        self.text = {}
        self.index.clear(zero=zero)

    def get_text(self, labels):
        if type(labels) is not list:
            labels = [labels]
        return [self.text[l] for l in labels]

    def get_vecs(self, labels):
        return self.index.get(labels)

    def search(self, query, groups=None, top_k=10, cutoff=-torch.inf, return_simil=False):
        qvec = self.embed.embed(query).squeeze() if type(query) is str else query
        labs, sims = self.index.search(qvec, top_k, groups=groups)
        match = [(l, v) for l, v in zip(labs, sims.tolist()) if v > cutoff]
        order = sorted(match, key=itemgetter(1), reverse=True)
        return order if return_simil else [l for l, v in order]

##
## document oriented database
##

# labels: [name, idx]
# dindex: TorchVectorIndex {name: vec}
class DocumentDatabase(TextDatabase):
    def __init__(
        self, index_device='cuda', allocate=True, dims=None, qspec=Float,
        delim='\n', min_len=1, max_len=None, **kwargs
    ):
        # init core text database
        super().__init__(
            index_device=index_device, allocate=allocate, dims=dims, qspec=qspec, **kwargs
        )

        # set up document parsing
        self.splitter = lambda s: text_splitter(s, delim, min_len=min_len, max_len=max_len)

        # possibly allocated document index
        if allocate:
            self.dindex = TorchVectorIndex(self.dims, device=index_device, qspec=qspec)

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', doc_batch=1024, maxrows=None,
        progress=True, threaded=True, **kwargs
    ):
        self = cls(**kwargs)
        lines = stream_jsonl(path, maxrows=maxrows)
        for i, batch in enumerate(batch_generator(lines, doc_batch)):
            self.add_docs([
                (row[name_col], row[text_col]) for row in batch
            ], threaded=threaded)
            if progress:
                print('█', end='', flush=True)
        return self

    @classmethod
    def load(cls, path, index_device='cuda', **kwargs):
        self = super().load(data, index_device=index_device, **kwargs)
        self.dindex = TorchVectorIndex.load(data['dindex'], device=index_device)
        return self

    def save(self, path=None):
        data = super().save()
        data['dindex'] = self.dindex.save()
        if path is None:
            return data
        else:
            torch.save(data, path)

    def process_docs(self, docs):
        targ = docs.items() if type(docs) is dict else docs
        chunks = {k: self.splitter(v) for k, v in targ}
        return {k: v for k, v in chunks.items() if len(v) > 0}

    def index_chunks(self, chunks, threaded=True):
        # convert to flat and add
        labels = [(k, i) for k, v in chunks.items() for i in range(len(v))]
        text = list(chain.from_iterable(chunks.values()))
        groups = [d for d, _ in labels]
        vecs = self.add(labels, text, groups=groups, threaded=threaded)

        # add aggregated document embeddings
        sizes = [len(v) for v in chunks.values()]
        dvecs = normalize(torch.stack([
            vecs[i:j,:].mean(dim=0) for i, j in cumul_indices(sizes)
        ], dim=0), dim=-1)
        self.dindex.add(list(chunks), dvecs)

    def add_docs(self, texts, threaded=True):
        chunks = self.process_docs(texts)
        self.index_chunks(chunks, threaded=threaded)

    def remove_docs(self, names):
        # remove from index
        labels = [(n, i) for n, i in self.text if n in names]
        self.remove(labels)

        # remove from document index
        if self.dindex is not None:
            self.dindex.remove(names)

    def search_docs(self, query, top_d=25, **kwargs):
        qvec = self.embed.embed(query).squeeze()
        docs = self.dindex.search(qvec, top_d, return_simil=False)
        return self.search(qvec, groups=docs, **kwargs)

##
## filesystem interface
##

# this relies solely on the extension
def load_document(path):
    # null on non-existent or directory
    if not os.path.exists(path) or os.path.isdir(path):
        return None

    # get extension and mimetype
    name, direc = os.path.split(path)
    base, ext = os.path.splitext(name)
    ext = ext.lstrip('.')
    mime, _ = mimetypes.guess_type(path)
    mtype = mime.split('/')[0] if mime is not None else None

    # dispatch to readers
    if mime == 'application/pdf':
        return read_pdf(path)
    elif mtype == 'text' or mtype is None:
        return read_text(path)

# index documents in a specified directory
class FilesystemDatabase(DocumentDatabase):
    def __init__(self, path='.', pattern='*', **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path)
        self.pattern = pattern
        self.reindex()

    def save(self, path):
        data = super().save()
        data['times'] = {n: self.get_mtime(n) for n in self.get_names()}
        torch.save(data, path)

    def load(self, path):
        data = super().load(path)
        self.times = data['times']

    def get_names(self):
        names = glob(str(self.path / self.pattern), recursive=True)
        return [os.path.relpath(n, self.path) for n in sorted(names)]

    def get_mtime(self, name):
        return os.path.getmtime(self.path / name)

    def get_doc(self, name):
        return load_document(self.path / name)

    def reindex(self, names=None):
        if names is None:
            names = self.get_names()
            self.clear()
        else:
            self.remove_docs(names=names)
        self.times = {n: self.get_mtime(n) for n in names}
        texts = [(n, self.get_doc(n)) for n in names]
        self.add_docs([(n, t) for n, t in texts if t is not None])

    def refresh(self, names=None):
        names = self.get_names() if names is None else names
        update = [
            name for name in names if self.get_mtime(name) > self.times.get(name, -inf)
        ]
        self.reindex(names=update)
