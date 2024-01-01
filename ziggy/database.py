# document databases

import os
import re
import json
import torch
import mimetypes

from math import ceil, inf
from itertools import chain, islice
from operator import itemgetter
from pathlib import Path
from glob import glob
from torch.nn.functional import normalize

from .llm import DEFAULT_EMBED, HuggingfaceEmbedding
from .index import TorchVectorIndex
from .quant import Float
from .utils import batch_generator, cumul_indices, groupby_idx, string_splitter

##
## Utils
##

# default paragraph splitter
def paragraph_splitter(text, delim='\n{2,}', minlen=1, maxlen=None):
    if delim is not None:
        paras = [p.strip() for p in re.split(delim, text)]
    else:
        paras = [text]
    paras = [p for p in paras if len(p) >= minlen]
    if maxlen is not None:
        paras = list(chain.from_iterable(
            string_splitter(p, maxlen) for p in paras
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
## generic document database
##

# data storage:
# chunks: dict {name: [chunk1, chunk2, ...]}
# cindex: TorchVectorIndex {(name, chunk_idx): vec}
# dindex: TorchVectorIndex {name: vec}
class DocumentDatabase:
    def __init__(
            self, embed=DEFAULT_EMBED, delim='\n{2,}', minlen=1, maxlen=None, batch_size=4096,
            model_device='cuda', index_device='cuda', doc_index=True, allocate=True, qspec=Float,
            dims=None, **kwargs
        ):
        # instantiate model and embedding
        self.embed = HuggingfaceEmbedding(embed, device=model_device) if type(embed) is str else embed

        # set up options
        self.splitter = lambda s: paragraph_splitter(s, delim=delim, minlen=minlen, maxlen=maxlen)
        self.batch_size = batch_size

        # initalize index
        if allocate:
            dims = self.embed.dims if dims is None else dims
            self.chunks = {}
            self.cindex = TorchVectorIndex(dims, device=index_device, qspec=qspec, **kwargs)
            self.dindex = TorchVectorIndex(dims, device=index_device, qspec=qspec, **kwargs) if doc_index else None

    @classmethod
    def from_jsonl(
        cls, path, name_col='title', text_col='text', doc_batch=1024, maxrows=None,
        progress=True, threaded=True, **kwargs
    ):
        self = cls(**kwargs)
        lines = stream_jsonl(path, maxrows=maxrows)
        for i, batch in enumerate(batch_generator(lines, doc_batch)):
            self.index_docs([
                (row[name_col], row[text_col]) for row in batch
            ], threaded=threaded)
            if progress:
                print('█', end='', flush=True)
        return self

    @classmethod
    def load(cls, path, device='cuda', **kwargs):
        data = torch.load(path, map_location=device) if type(path) is str else path
        self = cls(allocate=False, **kwargs)
        self.chunks = data['chunks']
        self.cindex = TorchVectorIndex.load(data['cindex'], device=device)
        self.dindex = TorchVectorIndex.load(data['dindex'], device=device) if 'dindex' in data else None
        return self

    def save(self, path=None):
        data = {
            'chunks': self.chunks,
            'cindex': self.cindex.save(),
        }
        if self.dindex is not None:
            data['dindex'] = self.dindex.save()
        if path is None:
            return data
        else:
            torch.save(data, path)

    def process_docs(self, texts):
        # split into chunks dict
        targ = texts.items() if type(texts) is dict else texts
        chunks = {k: self.splitter(v) for k, v in targ}
        chunks = {k: v for k, v in chunks.items() if len(v) > 0}
        # chunks = {k: [f'{k}: {c}' for c in v] for k, v in chunks.items()}
        return chunks

    def embed_chunks(self, chunks, threaded=True):
        # assess chunk information
        nchunks = sum([len(c) for c in chunks.values()])
        nbatch = int(ceil(nchunks/self.batch_size))

        # embed chunks with chosen batch_size
        chunk_iter = chain(*chunks.values())
        embeds = torch.cat([
            self.embed.embed(list(islice(chunk_iter, self.batch_size)), threaded=threaded) for _ in range(nbatch)
        ], dim=0)

        return embeds

    def index_embeds(self, chunks, embeds):
        # assess chunk information
        chunk_sizes = [len(c) for c in chunks.values()]
        nchunks = sum(chunk_sizes)

        # get names and labels
        labels = [(n, j) for n, s in zip(chunks, chunk_sizes) for j in range(s)]
        groups = [n for n, _ in labels]

        # update chunks and add to index
        self.chunks.update(chunks)
        self.cindex.add(labels, embeds, groups=groups)

        # make document level embeddings
        if self.dindex is not None:
            docemb = normalize(torch.stack([
                embeds[i:j,:].mean(dim=0) for i, j in cumul_indices(chunk_sizes)
            ], dim=0), dim=-1)
            self.dindex.add(list(chunks), docemb)

    def index_docs(self, texts, threaded=True):
        chunks = self.process_docs(texts)
        embeds = self.embed_chunks(chunks, threaded=threaded)
        self.index_embeds(chunks, embeds)

    def remove_docs(self, names):
        # remove from chunk dict
        for name in names:
            del self.chunks[name]

        # remove from index
        self.cindex.remove(func=lambda x: x[0] in names)
        if self.dindex is not None:
            self.dindex.remove(labs=names)

    def clear(self, zero=False):
        self.chunks = {}
        self.cindex.clear(zero=zero)
        if self.dindex is not None:
            self.dindex.clear(zero=zero)

    def search(self, query, max_docs=25, max_chunks=10, cluster=True, diffuse=None, cutoff=-torch.inf):
        # embed query string
        qvec = self.embed.embed(query).squeeze()

        # search document index first
        if cluster:
            docs = self.dindex.search(qvec, max_docs, return_simil=False)
        else:
            docs = None

        # search within matching documents
        labs, sims = self.cindex.search(qvec, max_chunks, groups=docs)

        # introduce kernel diffusion of similarity
        if diffuse is not None:
            pass

        # extract matched labels with cutoff
        match = [l for l, v in zip(labs, sims.tolist()) if v > cutoff]

        # return if no good matches
        if len(match) == 0:
            return {}

        # group by document
        docs, idxs = zip(*match)
        text = {
            k: [self.chunks[k][i] for i in v] for k, v in groupby_idx(idxs, docs).items()
        }

        # return text
        return text

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

    def get_text(self, name):
        return load_document(self.path / name)

    def reindex(self, names=None):
        if names is None:
            names = self.get_names()
            self.clear()
        else:
            self.remove_docs(names=names)
        self.times = {n: self.get_mtime(n) for n in names}
        texts = [(n, self.get_text(n)) for n in names]
        self.index_docs([(n, t) for n, t in texts if t is not None])

    def refresh(self, names=None):
        names = self.get_names() if names is None else names
        update = [
            name for name in names if self.get_mtime(name) > self.times.get(name, -inf)
        ]
        self.reindex(names=update)
