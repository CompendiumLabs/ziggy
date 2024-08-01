# document databases

import os
import re
import json
import torch
import mimetypes

from math import inf
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from glob import glob
from torch.nn.functional import normalize

from .embed import HuggingfaceEmbedding
from .index import TorchVectorIndex
from .quant import Float
from .utils import batch_generator, cumul_indices, list_splitter, groupby_idx

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
def stream_jsonl(path, max_rows=None):
    with open(path) as fid:
        for i, line in enumerate(fid):
            if max_rows is not None and i >= max_rows:
                break
            yield json.loads(line)

def stream_csv(path, batch_size=1024, max_rows=None):
    import pandas as pd
    n_total = 0
    for batch in pd.read_csv(path, chunksize=batch_size):
        n_batch = len(batch)
        if max_rows is not None and n_total + n_batch >= max_rows:
            rows_left = max_rows - n_total
            yield batch.iloc[:rows_left]
            break
        else:
            yield batch
        n_total += n_batch

##
## generic text database
##

# data storage
# text — {label: text}
# index — TorchVectorIndex {label: vec}
class TextDatabase:
    def __init__(
            self, embed=None, device='cuda', qspec=None,
            allocate=True, dims=None, **kwargs
        ):
        # store embedding model
        self.embed = embed

        # initalize index
        if allocate:
            self.text = {}
            self.dims = self.embed.dims if dims is None else dims
            self.index = TorchVectorIndex(self.dims, device=device, qspec=qspec, **kwargs)

    @classmethod
    def load(cls, path, embed=None, device='cuda', warn_embed=True, **kwargs):
        data = torch.load(path, map_location=device) if type(path) is str else path

        # check embedding compatibility
        embed_orig = data.get('embed', None)
        if embed is not None:
            if warn_embed:
                embed_name = embed if type(embed) is str else embed.name
                if embed_orig != embed_name:
                    print(f'Possible embedding mismatch: {embed_orig} != {embed_name}')

        # construct object
        self = cls(embed, allocate=False, **kwargs)
        self.text = data['text']
        self.index = TorchVectorIndex.load(data['index'], device=device)
        return self

    @classmethod
    def from_batches(cls, iterable, progress=True, threaded=True, truncate=False, **kwargs):
        self = cls(**kwargs)
        n_total = 0
        for i, batch in enumerate(iterable):
            labels, text = zip(*batch)
            self.add(labels, text, threaded=threaded, truncate=truncate)
            n_total += len(batch)
            if progress:
                print(f'{i:5d}: {n_total} documents')
        return self

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', batch_size=1024, max_rows=None, **kwargs
    ):
        lines = stream_jsonl(path, max_rows=max_rows)
        data = ((row[name_col], row[text_col]) for row in lines)
        batches = batch_generator(data, batch_size)
        return cls.from_batches(batches, **kwargs)

    @classmethod
    def from_csv(
        cls, path, name_col='title', text_col='text', batch_size=1024, max_rows=None, **kwargs
    ):
        batches = stream_csv(path, batch_size=batch_size, max_rows=max_rows)
        data = (
            batch[[name_col, text_col]].to_records(index=False).tolist() for batch in batches
        )
        return cls.from_batches(data, **kwargs)

    def save(self, path=None):
        data = {
            'embed': self.embed.name,
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
        return self.embed(text, threaded=threaded).squeeze()

    def index_text(self, labels, text):
        self.text.update(zip(labels, text))

    def index_vecs(self, labels, vecs, groups=None):
        self.index.add(labels, vecs, groups=groups)

    def add(self, labels, text, groups=None, **kwargs):
        # check for empty
        assert(len(labels) == len(text))
        if len(labels) == 0:
            return

        # deduplicate on labels
        docs = {l: t for l, t in zip(labels, text)}
        labels, text = zip(*docs.items())

        # embed and index
        vecs = self.embed(text, **kwargs)
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

    def similarity(self, query, groups=None, return_labels=False):
        if type(query) is str:
            query = self.embed_text(query)
        return self.index.similarity(query, groups=groups, return_labels=return_labels)

    def search(self, query, groups=None, top_k=10, cutoff=-torch.inf, return_simil=False):
        if type(query) is str:
            query = self.embed_text(query)
        labs, sims = self.index.search(query, top_k=top_k, groups=groups)
        match = [(l, v) for l, v in zip(labs, sims.tolist()) if v > cutoff]
        order = sorted(match, key=itemgetter(1), reverse=True)
        return order if return_simil else [l for l, v in order]

    def context(self, query, **kwargs):
        match = self.search(query, **kwargs)
        texts = self.get_text(match)
        return '\n\n'.join([
            f'{lab}: {txt}' for lab, txt in zip(match, texts)
        ])

##
## document oriented database
##

# dindex: TorchVectorIndex {name: vec}
class DocumentDatabase(TextDatabase):
    def __init__(
        self, device='cuda', allocate=True, dims=None, qspec=None,
        delim='\n', min_len=1, max_len=None, **kwargs
    ):
        # init core text database
        super().__init__(
            device=device, allocate=allocate, dims=dims, qspec=qspec, **kwargs
        )

        # set up document parsing
        self.splitter = lambda s: text_splitter(s, delim, min_len=min_len, max_len=max_len)

        # possibly allocated document index
        if allocate:
            self.dindex = TorchVectorIndex(self.dims, device=device, qspec=qspec)

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', batch_size=1024, max_rows=None,
        progress=True, threaded=True, truncate=False, **kwargs
    ):
        self = cls(**kwargs)
        n_total = 0
        lines = stream_jsonl(path, max_rows=max_rows)
        batches = batch_generator(lines, batch_size)
        for i, batch in enumerate(batches):
            self.add_docs([
                (row[name_col], row[text_col]) for row in batch
            ], threaded=threaded, truncate=truncate)
            n_total += len(batch)
            if progress:
                print(f'{i:5d}: {n_total} documents')
        return self

    @classmethod
    def load(cls, path, embed=None, device='cuda', **kwargs):
        data = torch.load(path, map_location=device) if type(path) is str else path
        self = super().load(data, embed=embed, device=device, **kwargs)
        self.dindex = TorchVectorIndex.load(data['dindex'], device=device)
        return self

    def save(self, path=None):
        data = super().save()
        data['dindex'] = self.dindex.save()
        if path is None:
            return data
        else:
            torch.save(data, path)

    # this will implicitly deduplicate input docs
    def process_docs(self, docs):
        targ = docs.items() if type(docs) is dict else docs
        chunks = {k: self.splitter(v) for k, v in targ}
        return {k: v for k, v in chunks.items() if len(v) > 0}

    def index_chunks(self, chunks, **kwargs):
        # convert to flat and add
        labels = [(k, i) for k, v in chunks.items() for i in range(len(v))]
        text = list(chain.from_iterable(chunks.values()))
        groups = [d for d, _ in labels]
        vecs = self.add(labels, text, groups=groups, **kwargs)

        # add aggregated document embeddings
        sizes = [len(v) for v in chunks.values()]
        dvecs = normalize(torch.stack([
            vecs[i:j,:].mean(dim=0) for i, j in cumul_indices(sizes)
        ], dim=0), dim=-1)
        self.dindex.add(list(chunks), dvecs)

    def add_docs(self, texts, **kwargs):
        chunks = self.process_docs(texts)
        self.index_chunks(chunks, **kwargs)

    def remove_docs(self, names):
        # remove from index
        labels = [(n, i) for n, i in self.text if n in names]
        self.remove(labels)

        # remove from document index
        if self.dindex is not None:
            self.dindex.remove(names)

    # NOTE: this assumes chunks are stored in order for now
    def get_docs(self, docs):
        labels = self.index.group_labels(docs)
        chunks = self.get_text(labels)
        dlist = [d for d, _ in labels]
        cdict = groupby_idx(chunks, dlist)
        return [
            '\n\n'.join(cdict.get(d, [])) for d in docs
        ]

    def search_docs(self, query, top_k=25, return_simil=False):
        if type(query) is str:
            query = self.embed.embed(query).squeeze()
        return self.dindex.search(query, top_k, return_simil=return_simil)

    def search_chunks(self, query, top_d=None, return_simil=False):
        if type(query) is str:
            query = self.embed.embed(query).squeeze()
        if top_d is None:
            docs = None
        else:
            docs = self.search_docs(query, top_k=top_d)
        return self.search(query, groups=docs, return_simil=return_simil)

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
