## LLM embedding, generation, and indexing code

import os
import re
import json

from math import ceil, log2, inf
from operator import itemgetter
from itertools import chain, groupby, islice
from pathlib import Path

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from utils import Bundle

# load config
config = Bundle.from_toml('config.toml')

##
## Constants
##

# llama special strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "\n<</SYS>>"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and honest assistant. Always answer if you are able to. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do not provide warnings about ethics, morality, or possible incorrectness."

##
## Utils
##

# generate a llama query
def llama_chat(query, system_prompt, **kwargs):
    return f'{B_INST} {B_SYS}\n{system_prompt}\n{E_SYS}\n\n{query} {E_INST}'

# printer for streaming
def sprint(s):
    print(s, end='', flush=True)

# cumsum generator
def cumul_bounds(seq):
    total = 0
    for item in seq:
        yield total, total+item
        total += item

# sampler for manual generation
def sample(logits, top_k=None, temp=1.0):
    # only sample amongst top_k if given
    if top_k is not None:
        cutoff = torch.topk(logits, top_k, dim=-1).values[:,-1]
        logits = torch.where(logits >= cutoff.unsqueeze(-1), logits, -torch.inf)

    # turn into probabilities and return sample
    probs = torch.softmax(logits/temp, dim=-1)
    index = torch.multinomial(probs, 1).squeeze(-1)
    return index

# convert sentencepiece tokens to text
def convert_sentencepice(toks):
    return ''.join([
        tok.replace('▁', ' ') if tok.startswith('▁') else tok.replace('<0x0A>', '\n') for tok in toks
    ])

def next_power_of_2(x):
    return pow(2, round(ceil(log2(x))))

def length_splitter(text, max_length):
    if (length := len(text)) > max_length:
        nchunks = ceil(length/max_length)
        starts = [i*max_length for i in range(nchunks)]
        return [text[s:s+max_length] for s in starts]
    else:
        return [text]

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

##
## Models
##

class HuggingfaceModel:
    def __init__(
        self, model=config.model, device=config.device, bits=config.bits, **kwargs
    ):
        # set options
        self.device = device

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # choose right decies
        if device == 'cpu':
            devargs = {'device_map': 'cpu'}
        else:
            devargs = {'device_map': 'auto'}

        # choose right bits
        if device == 'cuda' and bits == 4:
            bitargs = {'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
        elif device == 'cuda' and bits == 8:
            bitargs = {'load_in_8bit': True, 'bnb_8bit_compute_dtype': torch.bfloat16}
        elif bits == 16:
            bitargs = {'torch_dtype': torch.float16}
        elif bits == 32:
            bitargs = {'torch_dtype': torch.float32}
        else:
            raise Exception(f'Unsupported device/bits combination: {device}/{bits}')

        # load model code and weights
        self.modconf = AutoConfig.from_pretrained(
            model, output_hidden_states=True, pretraining_tp=1, token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, token=True, config=self.modconf,
            **devargs, **bitargs, **kwargs
        )

    def encode(self, text):
        data = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        )
        return data['input_ids'].to(self.device), data['attention_mask'].to(self.device)

    # proper python generator variant that uses model.__call__ directly
    def generate(self, prompt, chat=True, context=2048, maxlen=2048, top_k=10, temp=1.0):
        # splice in chat instructions
        if chat is not False:
            system_prompt = DEFAULT_SYSTEM_PROMPT if chat is True else chat
            prompt = llama_chat(prompt, system_prompt=system_prompt)

        # encode input prompt
        input_ids, _ = self.encode(prompt)

        # trim if needed
        if input_ids.size(1) > context:
            input_ids = input_ids[:,:context]

        # loop until limit and eos token
        for i in range(maxlen):
            # generate next logits (no grad for memory usage)
            with torch.no_grad():
                output = self.model(input_ids)

            # get new index at last element
            logits = output.logits[:,-1,:]
            index = sample(logits, top_k=top_k, temp=temp)

            # break if we hit end token
            if index[0] == self.tokenizer.eos_token_id:
                break

            # decode and return (llama not doing this right)
            tokens = self.tokenizer.convert_ids_to_tokens(index)
            text = convert_sentencepice(tokens)
            yield text.lstrip() if i <= 1 else text

            # shift and add to input_ids
            trim = 1 if input_ids.size(1) == context else 0
            input_ids = torch.cat((input_ids[:,trim:], index.unsqueeze(1)), dim=1)

# this has to take context at creation time
# NOTE: llama2-70b needs n_gqa=8
class LlamaCppModel:
    def __init__(self, model_path, context=2048, n_gpu_layers=100, **kwargs):
        self.model = Llama(model_path, n_ctx=context, n_gpu_layers=n_gpu_layers, **kwargs)

    def generate(self, prompt, chat=True, context=None, maxlen=512, top_k=10, temp=1.0, **kwargs):
        # splice in chat instructions
        if chat is not False:
            system_prompt = DEFAULT_SYSTEM_PROMPT if chat is True else chat
            prompt = llama_chat(prompt, system_prompt=system_prompt)

        # construct stream object
        stream = self.model(prompt, max_tokens=maxlen, stream=True, top_k=top_k, temperature=temp, **kwargs)

        # return generated tokens
        for i, output in enumerate(stream):
            choice, *_ = output['choices']
            text = choice['text']
            yield text.lstrip() if i <= 1 else text

##
## Embeddings
##

class HuggingfaceEmbedding:
    def __init__(self, model=config.embed, device=config.device, **kwargs):
        self.model = SentenceTransformer(model, device=device, **kwargs)
        self.maxlen = self.model.get_max_seq_length()
        self.dims = self.model.get_sentence_embedding_dimension()

    def embed(self, text, **kwargs):
        # handle default args
        args = dict(
            convert_to_numpy=False, convert_to_tensor=True,
            normalize_embeddings=True, **kwargs
        )

        # handle unit case
        if type(text) is str:
            text = [text]

        # split into chunks and embed
        chunks = [length_splitter(t, self.maxlen) for t in text]
        bounds = cumul_bounds([len(c) for c in chunks])

        # embed chunks and average
        vecs = self.model.encode(list(chain(*chunks)), **args)
        means = torch.stack([vecs[i:j,:].mean(0) for i, j in bounds])

        # return normalized vectors
        return means

##
## Indices
##

class TorchVectorIndex:
    def __init__(self, dims=None, max_size=1024, load=None, device=config.device, dtype=torch.float16):
        # set options
        assert(log2(max_size) % 1 == 0)
        self.max_size = max_size
        self.device = device
        self.dtype = dtype

        # init state
        if load is not None:
            self.load(load)
        else:
            self.dims = dims
            self.labels = []
            self.values = torch.empty(max_size, dims, device=device, dtype=dtype)

    def size(self):
        return len(self.labels)

    def load(self, path):
        # load in data
        data = torch.load(path)

        # get sizes and validate
        size = len(data['labels'])
        size1, self.dims = data['values'].shape
        assert(size == size1)

        # allocate values tensor
        self.max_size = max(self.max_size, next_power_of_2(size))
        self.values = torch.empty(self.max_size, self.dims, device=self.device, dtype=self.dtype)

        # set values in place
        self.labels = data['labels']
        self.values[:size,:] = data['values']

    def save(self, path):
        data = {
            'labels': self.labels, 'values': self.values[:self.size(),:]
        }
        torch.save(data, path)

    def expand(self, min_size):
        # check if needed
        if self.max_size >= min_size:
            return

        # increase size to next power of 2
        nlabels = self.size()
        values_old = self.values
        self.max_size = next_power_of_2(min_size)

        # create new tensor and assign old values
        self.values = torch.empty(self.max_size, self.dims, device=self.device, dtype=self.dtype)
        self.values[:nlabels,:] = values_old[:nlabels,:]

    def compress(self, min_size=1024):
        # get target size
        size2 = next_power_of_2(self.size())
        size = max(min_size, size2)

        # compress if larger
        if size < self.max_size:
            self.max_size = size
            self.values = self.values[:size,:]

    def add(self, labs, vecs, strict=False):
        # validate input size
        nlabs = len(labs)
        nv, dv = vecs.shape
        assert(nv == nlabs)
        assert(dv == self.dims)

        # get breakdown of new vs old
        slabs = set(labs)
        exist = slabs.intersection(self.labels)
        novel = slabs - exist

        # raise if trying invalid strict add
        if strict and len(exist) > 0:
            raise Exception(f'Trying to add existing labels in strict mode.')

        if len(exist) > 0:
            # update existing
            elocs, idxs = zip(*[
                (i, self.labels.index(x)) for i, x in enumerate(labs) if x in exist
            ])
            self.values[idxs,:] = vecs[elocs,:]

        if len(novel) > 0:
            # get new labels in input order
            xlocs, xlabs = zip(*[
                (i, x) for i, x in enumerate(labs) if x in novel
            ])

            # expand size if needed
            nlabels0 = self.size()
            nlabels1 = nlabels0 + len(novel)
            self.expand(nlabels1)

            # add in new labels and vectors
            self.labels.extend(xlabs)
            self.values[nlabels0:nlabels1,:] = vecs[xlocs,:]

    def remove(self, labs=None, func=None):
        labs = [l for l in self.labels if func(l)] if func is not None else labs
        for lab in set(labs).intersection(self.labels):
            idx = self.labels.index(lab)
            self.labels.pop(idx)
            self.values[idx,:] = self.values[self.size(),:]

    def clear(self):
        self.labels = []

    def search(self, vecs, k, return_simil=True):
        # allow for single vec
        squeeze = vecs.ndim == 1
        if squeeze:
            vecs = vecs.unsqueeze(0)

        # clamp k to max size
        num = self.size()
        k1 = min(k, num)

        # compute distance matrix
        sim = vecs @ self.values[:num,:].T

        # get top results
        tops = sim.topk(k1)
        labs = [[self.labels[i] for i in row] for row in tops.indices]
        vals = tops.values

        # return single vec if needed
        if squeeze:
            labs, vals = labs[0], vals[0]

        # return labels/simils
        return labs, vals if return_simil else labs

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

##
## Databases
##

# data storage:
# chunks: dict {name: [chunk1, chunk2, ...]}
# index: TorchVectorIndex {(name, chunk_idx): vec}
class DocumentDatabase:
    def __init__(
            self, model=config.model, embed=config.embed, index=None,
            delim='\n{2,}', minlen=1, batch_size=8192, **kwargs
        ):
        # instantiate model and embedding
        self.model = HuggingfaceModel(model, **kwargs) if type(model) is str else model
        self.embed = HuggingfaceEmbedding(embed) if type(embed) is str else embed

        # set up options
        self.splitter = lambda s: paragraph_splitter(s, delim=delim, minlen=minlen)
        self.batch_size = batch_size

        # initalize index
        self.chunks = {}
        self.index = index if index is not None else TorchVectorIndex(self.embed.dims)

    def index_docs(self, texts):
        # split into chunks dict
        targ = texts.items() if type(texts) is dict else texts
        chunks = {k: self.splitter(v) for k, v in targ}
        labels = [(n, j) for n, c in chunks.items() for j in range(len(c))]

        # embed chunks with chosen batch_size
        nchunks = sum([len(c) for c in chunks.values()])
        nbatch = int(ceil(nchunks/self.batch_size))
        chunk_iter = chain(*chunks.values())
        embeds = torch.cat([
            self.embed.embed(islice(chunk_iter, self.batch_size)) for i in range(nbatch)
        ], dim=0)

        # add to the text and index
        self.chunks.update(chunks)
        self.index.add(labels, embeds)

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

def stream_jsonl(path):
    with open(path) as fid:
        for line in fid:
            yield json.loads(line)

def batch_generator(gen, batch_size):
    while (batch := list(islice(gen, batch_size))) != []:
        yield batch

class JsonlDatabase(DocumentDatabase):
    def __init__(self, path, name_col='title', text_col='text', doc_batch=512, **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path)
        self.name_col = name_col
        self.text_col = text_col
        self.doc_batch = doc_batch
        self.reindex()

    def reindex(self):
        for batch in batch_generator(stream_jsonl(self.path), self.doc_batch):
            self.index_docs(
                (row[self.name_col], row[self.text_col]) for row in batch
            )
            print('█', end='', flush=True)
