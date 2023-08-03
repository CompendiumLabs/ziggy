## LLM embedding, generation, and indexing code

import os
import re
import torch
import torch.nn.functional as F

from math import ceil, log2
from operator import itemgetter
from itertools import chain, groupby
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from utils import Bundle

# load config
config = Bundle.from_toml('config.toml')

# llama special strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "\n<</SYS>>"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and honest assistant. Always answer if you are able to. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do not provide warnings about ethics, morality, or possible incorrectness."

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
    def generate(self, prompt, chat=True, context=512, maxlen=512, top_k=10, temp=1.0):
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

class LlamaCppModel:
    def __init__(self, model_path, **kwargs):
        self.model = Llama(model_path, **kwargs)

    def generate(self, prompt, chat=True, context=512, maxlen=512, **kwargs):
        # splice in chat instructions
        if chat is not False:
            system_prompt = DEFAULT_SYSTEM_PROMPT if chat is True else chat
            prompt = llama_chat(prompt, system_prompt=system_prompt)

        # construct stream object
        stream = self.model(prompt, max_tokens=maxlen, stream=True, **kwargs)

        # return generated tokens
        for output in stream:
            choice, *_ = output['choices']
            yield choice['text']

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

class TorchVectorIndex:
    def __init__(self, dims, max_size=1024, load=None, device=config.device):
        # set options
        assert(log2(max_size) % 1 == 0)
        self.max_size = max_size
        self.device = device
        self.dims = dims

        # init state
        if load is not None:
            self.load(load)
        else:
            self.labels = []
            self.values = torch.empty(max_size, dims, device=device)

    def size(self):
        return len(self.labels)

    def load(self, path):
        pass

    def save(self, path):
        pass

    def expand(self, min_size):
        # check if needed
        if self.max_size >= min_size:
            return

        # increase size to next power of 2
        nlabels = self.size()
        values_old = self.values
        self.max_size = pow(2, round(ceil(log2(min_size))))

        # create new tensor and assign old values
        self.values = torch.empty(self.max_size, self.dims, device=self.device)
        self.values[:nlabels,:] = values_old[:nlabels,:]

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

    def remove(self, labs):
        pass

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

def length_splitter(text, max_length):
    if (length := len(text)) > max_length:
        nchunks = ceil(length/max_length)
        starts = [i*max_length for i in range(nchunks)]
        return [text[s:s+max_length] for s in starts]
    else:
        return [text]

# default paragraph splitter
def paragraph_splitter(text):
    return [
        para for para in re.split('\n{2,}', text) if len(para.strip()) > 0
    ]

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

# index documents in a specified directory
class FilesystemDatabase:
    def __init__(
            self, path, model=config.model, embed=config.embed, index=None,
            splitter=paragraph_splitter, batch_size=config.batch_size, load=None
        ):
        # set options
        self.path = path
        self.splitter = splitter
        self.batch_size = batch_size

        # instantiate model and embedding
        self.model = HuggingfaceModel(model) if type(model) is str else model
        self.embed = HuggingfaceEmbedding(embed) if type(embed) is str else embed
        self.index = index if index is not None else TorchVectorIndex(self.embed.dims)

        # load if given
        if load is not None:
            self.load(load)
        else:
            self.reindex()

    def reindex(self):
        # clear existing entries
        self.index.clear()

        # read in all files and split into chunks
        names = sorted(os.listdir(self.path))
        text = {n: robust_read(os.path.join(self.path, n)) for n in names}
        self.chunks = {k: self.splitter(v) for k, v in text.items()}

        # flatten file chunks and make labels
        labels = [(n, j) for n, c in self.chunks.items() for j in range(len(c))]
        chunks = list(chain(*self.chunks.values()))

        # embed chunks with chosen batch_size
        indices = batch_indices(len(chunks), self.batch_size)
        embeds = torch.cat([self.embed.embed(chunks[i1:i2]) for i1, i2 in indices], dim=0)

        # add to the index
        self.index.add(labels, embeds)

    def load(self, path):
        pass

    def save(self, path):
        pass

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

    def query(self, query, context=512, maxlen=512, **kwargs):
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

# example usage
# model = HuggingfaceModel('meta-llama/Llama-2-7b-chat-hf')
# db = FilesystemDatabase('notes', model=model)
# db.iquery('Give me a concise summary of my research ideas')
