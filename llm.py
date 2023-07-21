## LLM embedding, generation, and indexing code

import os
import torch
import torch.nn.functional as F

from math import ceil, log2
from operator import itemgetter
from itertools import chain, groupby
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer

from utils import Bundle

# load config
config = Bundle.from_toml('config.toml')
auth = Bundle.from_toml('auth.toml')

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
    def __init__(self, model=config.model, context_size=config.context_size, **kwargs):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model code and weights
        self.context_size = context_size
        self.modconf = AutoConfig.from_pretrained(
            model, output_hidden_states=True, pretraining_tp=1, token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map='auto', trust_remote_code=True, load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, token=True, config=self.modconf,
            **kwargs
        )

        # get embedding dimension
        self.dims = self.model.config.hidden_size

    def encode(self, text):
        data = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        )
        return data['input_ids'].to('cuda'), data['attention_mask'].to('cuda')

    def embed(self, text):
        # encode input text
        input_ids, attn_mask = self.encode(text)

        # get model output (no grad for memory usage)
        with torch.no_grad():
            output = self.model(input_ids)

        # get masked embedding
        state = output.hidden_states[0]
        mask = attn_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        return F.normalize(embed, dim=-1)

    # proper python generator variant that uses model.__call__ directly
    def generate(self, prompt, chat=True, max_new_tokens=200, top_k=10, temp=1.0):
        # splice in chat instructions
        if chat is not False:
            system_prompt = DEFAULT_SYSTEM_PROMPT if chat is True else chat
            prompt = llama_chat(prompt, system_prompt=system_prompt)

        # encode input prompt
        input_ids, _ = self.encode(prompt)

        # trim if needed
        if input_ids.size(1) > self.context_size:
            input_ids = input_ids[:,:self.context_size]

        # loop until limit and eos token
        for i in range(max_new_tokens):
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
            trim = 1 if input_ids.size(1) == self.context_size else 0
            input_ids = torch.cat((input_ids[:,trim:], index.unsqueeze(1)), dim=1)

class HuggingfaceEmbedding:
    def __init__(self, model=config.embed, **kwargs):
        self.model = SentenceTransformer(model, device='cuda', **kwargs)
        self.dims = self.model.get_sentence_embedding_dimension()

    def embed(self, text):
        vecs = self.model.encode(text, convert_to_numpy=False, convert_to_tensor=True)
        return F.normalize(vecs, dim=-1)

class VectorIndex:
    def __init__(self, dims, dtype=torch.float32, device='cuda'):
        self.dims = dims
        self.dtype = dtype
        self.device = device

class TorchVectorIndex(VectorIndex):
    def __init__(self, dims, max_size=1024, dtype=torch.float32, device='cuda'):
        assert(log2(max_size) % 1 == 0)
        super().__init__(dims, dtype=dtype, device=device)

        # empty state
        self.max_size = max_size
        self.labels = []
        self.values = torch.empty(max_size, dims, dtype=dtype, device=device)

    def size(self):
        return len(self.labels)

    def expand(self, min_size):
        # check if needed
        if self.max_size >= min_size:
            return

        # increase size to next power of 2
        nlabels = self.size()
        values_old = self.values
        self.max_size = pow(2, round(ceil(log2(min_size))))

        # create new tensor and assign old values
        self.values = torch.empty(
            self.max_size, self.dims, dtype=self.dtype, device=self.device
        )
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
def paragraph_splitter(text, max_length):
    paras = [
        para for para in text.split('\n\n') if len(para.strip()) > 0
    ]
    return list(chain.from_iterable(
        length_splitter(para, max_length) for para in paras
    ))

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
            self, path, model=config.model, embed=config.embed, index=None, splitter=paragraph_splitter,
            context_size=config.context_size, batch_size=config.batch_size
        ):
        self.path = path
        self.model = HuggingfaceModel(model) if type(model) is str else model
        self.embed = HuggingfaceEmbedding(embed) if type(embed) is str else embed
        self.index = index if index is not None else TorchVectorIndex(self.embed.dims)
        self.splitter = lambda c: splitter(c, context_size)
        self.batch_size = batch_size
        self.reindex()

    def reindex(self):
        # clear existing entries
        self.index.clear()

        # get files in directory
        self.names = sorted(os.listdir(self.path))

        # read in all files and split into chunks
        paths = [os.path.join(self.path, x) for x in self.names]
        text = [robust_read(x) for x in paths]
        self.chunks = [self.splitter(x) for x in text]

        # flatten file chunks and make labels
        labels = [(i, j) for i, c in enumerate(self.chunks) for j in range(len(c))]
        chunks = list(chain.from_iterable(self.chunks))

        # embed chunks with chosen batch_size
        indices = batch_indices(len(chunks), self.batch_size)
        embeds = torch.cat([self.embed.embed(chunks[i1:i2]) for i1, i2 in indices], dim=0)

        # add to the index
        self.index.add(labels, embeds)

    def search(self, query, k=10, cutoff=0.0):
        # get relevant chunks
        qvec = self.embed.embed(query).squeeze()
        labs, vecs = self.index.search(qvec, k)
        match = list(zip(labs, vecs.tolist()))

        # group by document and filter by cutoff
        locs = groupby_dict([l for l, v in match if v > cutoff])
        text = {self.names[k]: [self.chunks[k][i] for i in v] for k, v in locs.items()}

        # return text
        return text

    def query(self, query, **kwargs):
        # search db and get some context
        chunks = {k: '\n'.join(v) for k, v in self.search(query, **kwargs).items()}
        context = '\n\n'.join([f"{k}:\n{v}" for k, v in chunks.items()])

        # construct prompt
        system = f'{DEFAULT_SYSTEM_PROMPT}. Below are some relevant snippets of text from my person notes. Using a synthesis of your general knowledge and my notes, answer the question posed at the end concisely. Try to quote specific lines from my notes where possible.'
        user = f'{context}\n\n{query}'
        
        # generate response
        yield from self.model.generate(user, chat=system)

# example usage
# gen = HuggingfaceModel('tiiuae/falcon-7b-instruct')
# for s in gen.generator('Write a poem about Valencia.'):
#     sprint(s)
