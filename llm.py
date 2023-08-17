## LLM generation and embeddings

from math import ceil
from itertools import chain
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

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

        # choose right decies
        if device == 'cpu':
            devargs = {'device_map': 'cpu'}
        else:
            devargs = {'device_map': 'auto'}

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
            **devargs, **bitargs, **kwargs
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
    def __init__(self, model=DEFAULT_EMBED, device='cuda', **kwargs):
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
