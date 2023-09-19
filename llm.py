## LLM generation and embeddings

from math import ceil
from itertools import chain, islice
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from utils import l2_mean

##
## Constants
##

# default settings
DEFAULT_MODEL = 'meta-llama/Llama-2-7b-chat-hf'
DEFAULT_EMBED = 'all-MiniLM-L6-v2'

# llama special strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "\n<</SYS>>"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and honest assistant. Always answer if you are able to. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do not provide warnings about ethics, morality, illegality, safety, or possible incorrectness."

##
## Utils
##

# generate a llama query
def llama_chat(query, system_prompt):
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

def length_splitter(text, max_length):
    if (length := len(text)) > max_length:
        nchunks = ceil(length/max_length)
        starts = [i*max_length for i in range(nchunks)]
        return [text[s:s+max_length] for s in starts]
    else:
        return [text]

##
## Models
##

class HuggingfaceModel:
    def __init__(
        self, model=DEFAULT_MODEL, device='cuda', bits=None, compile=False, **kwargs
    ):
        # set options
        self.device = device

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # choose right bits
        if bits is None:
            bits = 16 if device == 'cuda' else 32
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
            device_map=device, **bitargs, **kwargs
        )

        # compile model if needed
        if compile:
            self.model = torch.compile(self.model)

    def encode(self, text, **kwargs):
        targs = {'padding': True, 'truncation': True, **kwargs}
        data = self.tokenizer(text, return_tensors='pt', **targs)
        return data['input_ids'].to(self.device), data['attention_mask'].to(self.device)

    def embed(self, text, **kwargs):
        # encode input text
        input_ids, attn_mask = self.encode(text, **kwargs)

        # get model output (no grad for memory usage)
        with torch.no_grad():
            output = self.model(input_ids, attn_mask)

        # get masked embedding
        state = output.hidden_states[-1]
        mask = attn_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        return F.normalize(embed, dim=-1)

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
    def __init__(self, model_path, context=2048, n_gpu_layers=100, verbose=False, **kwargs):
        self.model = Llama(
            model_path, n_ctx=context, n_gpu_layers=n_gpu_layers, verbose=verbose, **kwargs
        )

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

    def igenerate(self, query, **kwargs):
        for s in self.generate(query, **kwargs):
            sprint(s)

##
## Embeddings
##

class HuggingfaceEmbedding:
    def __init__(self, model=DEFAULT_EMBED, device='cuda', **kwargs):
        self.model = SentenceTransformer(model, **kwargs).to(device)
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
        with torch.no_grad():
            vecs = self.model.encode(list(chain(*chunks)), **args)
        means = torch.stack([l2_mean(vecs[i:j,:], dim=0) for i, j in bounds])

        # return normalized vectors
        return means

##
## ONNX
##

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# this gets a 60% speedup on GPU!!!
class HuggingfaceEmbeddingONNX:
    def __init__(self, model_id=f'sentence-transformers/{DEFAULT_EMBED}', save_path='testing/onnx', device='cuda'):
        self.device = device

        # initial load
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

        self.optimization_config = OptimizationConfig(
            optimization_level=99, optimize_for_gpu=True, fp16=True,
        )
        optimizer = ORTOptimizer.from_pretrained(self.model)

        # Export the optimized model
        optimizer.optimize(save_dir=save_path, optimization_config=self.optimization_config)

        # load optimized
        self.tokenizer = AutoTokenizer.from_pretrained(save_path)
        self.model = ORTModelForFeatureExtraction.from_pretrained(save_path).to(device)

    def embed_batch(self, text, normalize=True, **kwargs):
        targs = {'padding': True, 'truncation': True, **kwargs}
        encode = self.tokenizer(text, return_tensors='pt', **targs)

        input_ids = encode['input_ids'].to(self.device)
        attention_mask = encode['attention_mask'].to(self.device)
        token_type_ids = torch.zeros(encode['input_ids'].shape, dtype=torch.int64, device=self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # get masked embedding
        state = output[0]
        mask = attention_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        return F.normalize(embed, dim=-1)

    def embed(self, text, maxlen=512, batch_size=128):
        # handle unit case
        if type(text) is str:
            text = [text]

        # split into chunks and embed
        chunks = [length_splitter(t, maxlen) for t in text]
        bounds = cumul_bounds([len(c) for c in chunks])
        chunk_iter = chain(*chunks)

        # assess chunk information
        nchunks = sum([len(c) for c in chunks])
        nbatch = int(ceil(nchunks/batch_size))

        # embed chunks and average
        embed = torch.cat([
            self.embed_batch(list(islice(chunk_iter, batch_size))) for i in range(nbatch)
        ], dim=0)
        means = torch.stack([l2_mean(embed[i:j,:], dim=0) for i, j in bounds])

        # return normalized vectors
        return F.normalize(means, dim=-1)
