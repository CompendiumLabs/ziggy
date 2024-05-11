## Embedding models

import os
import json
import numpy as np
from math import ceil
from itertools import chain
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.functional import normalize as norm
from transformers import AutoTokenizer, AutoModel
import huggingface_hub as hub
from huggingface_hub.utils import EntryNotFoundError

from .utils import (
    pipeline_threads, batch_generator, batch_indices, list_splitter, cumsum, cumul_bounds, RequestTracker, l2mean
)

##
## ONNX Wrapper
##

class ONNXEmbedding:
    def __init__(self, model_dir, device):
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        # device settings
        provider = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

        # create model
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_dir, provider=provider, provider_options=provider_options
        )

        # store config
        self.session = self.model.model
        self.config = self.model.config
        self.device = device

    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        output_names = [out.name for out in self.session.get_outputs()]

        if self.device == 'cuda':
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif self.device == 'cpu':
            # pass in correct inputs
            input_names = [inp.name for inp in self.session.get_inputs()]
            onnx_inputs = {'input_ids': input_ids.numpy(), 'attention_mask': attention_mask.numpy()}
            if 'token_type_ids' in input_names:
                onnx_inputs['token_type_ids'] = token_type_ids.numpy()
            outputs = self.session.run(None, onnx_inputs)

            # try to find output location
            output_names = [out.name for out in self.session.get_outputs()]
            if 'token_embeddings' in output_names:
                output_loc = output_names.index('token_embeddings')
            elif 'last_hidden_state' in output_names:
                output_loc = output_names.index('last_hidden_state')
            output = [torch.from_numpy(outputs[output_loc])]
        return output

def compile_onnx(model, save_dir, device, trust_remote_code=False):
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
    from optimum.onnxruntime.configuration import OptimizationConfig

    optim_args = dict(optimize_for_gpu=True, fp16=True) if device == 'cuda' else {}
    model = ORTModelForFeatureExtraction.from_pretrained(
        model, export=True, trust_remote_code=trust_remote_code
    )
    optimization_config = OptimizationConfig(
        optimization_level=99, **optim_args
    )
    optimizer = ORTOptimizer.from_pretrained(model)
    optimizer.optimize(
        save_dir=save_dir, optimization_config=optimization_config
    )

##
## Embeddings
##

def detect_pooling_type(repo_id):
    # get modules file
    try:
        mod_path = hub.hf_hub_download(repo_id, filename='modules.json')
    except EntryNotFoundError:
        return
    with open(mod_path) as f:
        mod_data = json.load(f)

    # pick off pooling layer
    pool_rel = None
    for mod in mod_data:
        if mod['type'] == 'sentence_transformers.models.Pooling':
            pool_rel = mod['path']
            break
    if pool_rel is None:
        return

    # get pooling layer conf
    try:
        pool_path = hub.hf_hub_download(repo_id, filename=f'{pool_rel}/config.json')
    except EntryNotFoundError:
        return
    with open(pool_path) as f:
        pool_data = json.load(f)

    # get pooling type
    if pool_data['pooling_mode_cls_token']:
        return 'cls'
    elif pool_data['pooling_mode_mean_tokens']:
        return 'mean'
    elif pool_data['pooling_mode_max_tokens']:
        raise NotImplementedError('max pooling not supported')
    elif pool_data['pooling_mode_mean_sqrt_len_tokens']:
        raise NotImplementedError('mean-sqrt-len pooling not supported')

class HuggingfaceEmbedding:
    def __init__(
        self, model_id, tokenize_id=None, max_len=None, batch_size=128,
        queue_size=256, device='cuda', dtype=None, onnx=None, save_dir=None, compile=False,
        pooling_type=None, trust_remote_code=False
    ):
        # get env config
        ONNX_DIR = os.environ.get('ZIGGY_ONNX_DIR', 'onnx')
        save_dir = save_dir if save_dir is not None else ONNX_DIR

        # runtime options
        self.device = device
        self.queue_size = queue_size

        # detect onnx
        if onnx is None:
            optim_path = f'{model_id}/model_optimized.onnx'
            onnx = os.path.isdir(model_id) and os.path.exists(optim_path)

        # load model onnx or not
        if onnx:
            if os.path.isdir(model_id):
                model_path = model_id
                model_id = os.path.basename(model_path)
            else:
                model_path = os.path.join(save_dir, f'{model_id}-{device}')
                if compile or not os.path.isdir(model_path):
                    compile_onnx(
                        model_id, model_path, device, trust_remote_code=trust_remote_code
                    )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = ONNXEmbedding(model_path, device)
        else:
            device_map = {'': 0} if device == 'cuda' else device
            if dtype is None:
                dtype = torch.float16 if device == 'cuda' else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)

        # get pooling type
        if pooling_type is None:
            self.pooling_type = detect_pooling_type(model_id)
            if self.pooling_type is None:
                raise ValueError('Pooling type not detected, must specify')
        else:
            self.pooling_type = pooling_type

        # get model info
        self.name = model_id
        self.batch_size = batch_size
        self.max_len = self.model.config.max_position_embeddings if max_len is None else max_len
        self.dims = self.model.config.hidden_size

    def tokenize(self, text, truncate=False):
        encode = self.tokenizer(
            text, max_length=self.max_len, padding='max_length', truncation=True,
            return_overflowing_tokens=not truncate, return_tensors='pt'
        )
        if truncate:
            n_text, _ = encode.input_ids.shape
            overflow_to_sample_mapping = torch.arange(n_text)
        else:
            overflow_to_sample_mapping = encode.overflow_to_sample_mapping
        return overflow_to_sample_mapping, encode.input_ids, encode.attention_mask

    def forward_batch(self, input_ids, attention_mask, pooling_type=None, normalize=True):
        # prepare model inputs on device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64, device=self.device)

        # get model output
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        state = output[0]

        # get sentence embeddings
        pooling_type = self.pooling_type if pooling_type is None else pooling_type
        if pooling_type == 'mean':
            mask = attention_mask.float().unsqueeze(-1)
            embed = (state*mask).sum(1)/mask.sum(1)
        elif pooling_type == 'cls':
            embed = state[:,0,:]
        elif pooling_type == 'none':
            embed = torch.where(attention_mask.unsqueeze(-1) == 1, state, torch.nan)

        # normalized embedding
        if normalize:
            embed = norm(embed, dim=-1)

        return embed

    def embed_batch(self, text, truncate=False, **kwargs):
        doc_indices, input_ids, attention_mask = self.tokenize(text, truncate=truncate)
        indices = batch_indices(input_ids.size(0), self.batch_size)
        embed = torch.cat([
            self.forward_batch(input_ids[i1:i2], attention_mask[i1:i2], **kwargs) for i1, i2 in indices
        ])
        return doc_indices, embed

    def embed(self, text, threaded=False, truncate=False, **kwargs):
        # handle unit case
        if type(text) is str:
            text = [text]

        if threaded:
            # make workers
            results = []
            def loader():
                yield from batch_generator(text, self.batch_size)
            def tokenizer(texts):
                return self.tokenize(texts, truncate=truncate)
            def forwarder(data):
                doc_indices, input_ids, attention_mask = data
                indices = batch_indices(input_ids.size(0), self.batch_size)
                embed = torch.cat([
                    self.forward_batch(input_ids[i1:i2], attention_mask[i1:i2], **kwargs)
                    for i1, i2 in indices
                ])
                results.append((doc_indices, embed))

            # embed chunks and average
            pipeline_threads(loader(), tokenizer, forwarder, maxsize=self.queue_size)
        else:
            # embed chunks and average
            results = [
                self.embed_batch(
                    chunk, truncate=truncate, **kwargs
                ) for chunk in batch_generator(text, self.batch_size)
            ]

        # unpack offsets and embeds
        offsets, embeds = zip(*results)

        # document offsets come zero-indexed within batch
        nchunks = [o.max().item()+1 for o in offsets]
        indices = torch.cat([o+n for o, n in zip(offsets, cumsum(nchunks))])
        _, sizes = torch.unique(indices, return_counts=True)

        # aggregate to document level
        embed0 = torch.cat(embeds)
        embed = torch.stack([
            l2mean(embed0[i:j], dim=0) for i, j in cumul_bounds(sizes)
        ])

        # return normalized vectors
        return embed

##
## llama.cpp
##

# = defaultdict(list)
# + handles popping off maximal list
# + handles deletion on empty list
class SizeDist(dict):
    def __init__(self, data):
        sdist = defaultdict(list)
        for i, size in enumerate(data):
            sdist[size].append(i)
        super().__init__(sdist)

    def pop(self, max_size=None):
        if max_size is None:
            size = max(self, default=None)
        else:
            size = max((s for s in self if s <= max_size), default=None)
        if size is None:
            return
        ret = self[size].pop(0)
        if len(self[size]) == 0:
            del self[size]
        return ret

def pack_batches(sizes, max_len):
    # get size distribution
    n_seq = len(sizes)
    sdist = SizeDist(sizes)
    assert max(sdist) <= max_len

    # plan batch contents
    batches = []
    bidxs = []
    bsize = 0
    for _ in range(n_seq):
        # get a maximal sample
        idx = sdist.pop(max_len-bsize)

        # if none we commit batch and retry
        if idx is None:
            batches.append(bidxs)
            bidxs = []
            bsize = 0
            idx = sdist.pop(max_len)

        # append to batch
        bidxs.append(idx)
        bsize += sizes[idx]

    # append final batch
    batches.append(bidxs)

    return batches

class LlamaCppEmbedding:
    def __init__(self, model_path, max_len=512, pooling_type=None, device='cuda', verbose=False, **kwargs):
        from llama_cpp import Llama, llama_pooling_type, LLAMA_POOLING_TYPE_UNSPECIFIED

        # set up device
        ngl = 0 if device == 'cpu' else 99
        self.device = device

        # get pooling type
        pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED if pooling_type is None else pooling_type

        # load model
        self.model = Llama(
            model_path, embedding=True, n_batch=max_len, n_ctx=max_len,
            pooling_type=pooling_type, n_gpu_layers=ngl, verbose=verbose
        )

        # get metadata
        self.name = os.path.basename(model_path)
        self.max_len = max_len
        self.pooling_type = llama_pooling_type(self.model._ctx.ctx)
        self.dims = self.model.n_embd()

    def tokenize(self, text, special=False):
        if type(text) is str:
            text = [text]
            squeeze = True
        else:
            squeeze = False
        ids = [
            self.model.tokenize(s.encode('utf-8'), special=special)
            for s in text
        ]
        return ids[0] if squeeze else ids

    def forward_batch(self, tokens):
        from llama_cpp import (
            llama_kv_cache_clear, llama_get_embeddings, llama_get_embeddings_seq,
            LLAMA_POOLING_TYPE_NONE
        )

        ctx = self.model._ctx
        n_seq = len(tokens)
        n_embd = self.model.n_embd()

        # check total batch size
        n_toks = [len(toks) for toks in tokens]
        assert sum(n_toks) <= self.max_len

        # add tokens to batch
        self.model._batch.reset()
        for seq_id, toks in enumerate(tokens):
            self.model._batch.add_sequence(toks, seq_id, False)

        # run model on batch
        llama_kv_cache_clear(ctx.ctx)
        ctx.decode(self.model._batch)

        # store embeddings
        if self.pooling_type == LLAMA_POOLING_TYPE_NONE:
            embeds_ptr = llama_get_embeddings(ctx.ctx)
            embeds = [
                [embeds_ptr[k*n_embd:(k+1)*n_embd] for k in range(i, j)]
                for i, j in cumul_bounds(n_toks)
            ]
        else:
            embeds = [
                llama_get_embeddings_seq(ctx.ctx, i)[:n_embd]
                for i in range(n_seq)
            ]

        # return as lists
        return embeds

    def forward(self, tokens):
        from llama_cpp import LLAMA_POOLING_TYPE_NONE
        assert self.pooling_type != LLAMA_POOLING_TYPE_NONE

        # plan batch contents
        sizes = [len(toks) for toks in tokens]
        batches = pack_batches(sizes, self.max_len)

        # allocate output tensor
        embeds = torch.empty(len(tokens), self.dims, device=self.device, dtype=torch.float32)

        # compute embeddings
        for idxs in batches:
            toks = [tokens[i] for i in idxs]
            embs = self.forward_batch(toks)
            embeds[idxs] = torch.tensor(embs, device=self.device, dtype=torch.float32)

        return embeds

    def embed(self, text, threaded=None, truncate=False, normalize=True, special=False):
        if type(text) is str:
            text = [text]

        # get tokens and length distribution
        tokens = self.tokenize(text, special=special)

        # handle truncation
        if truncate:
            chunks = [t[:self.max_len] for t in tokens]
        else:
            splits = [ceil(len(t) / self.max_len) for t in tokens]
            chunks = list(chain.from_iterable((list_splitter(t, self.max_len) for t in tokens)))

        # run forward compute
        embeds = self.forward(chunks)

        # return normalized vectors
        if normalize:
            embeds = norm(embeds, dim=-1)

        # aggreate to document level
        if not truncate and max(splits) > 1:
            embeds = torch.stack([
                l2mean(embeds[i:j], dim=0) for i, j in cumul_bounds(splits)
            ])

        return embeds

##
## OpenAI
##

DEFAULT_OPENAI_EMBED = 'text-embedding-3-large'

openai_config = {
    'text-embedding-ada-002': {
        'dims': 1536,
        'max_len': 8191,
        'req_limit': 10_000,
        'tok_limit': 5_000_000,
    },
    'text-embedding-3-small': {
        'dims': 1536,
        'max_len': 8192,
        'req_limit': 10_000,
        'tok_limit': 5_000_000,
    },
    'text-embedding-3-large': {
        'dims': 3072,
        'max_len': 8192,
        'req_limit': 10_000,
        'tok_limit': 5_000_000,
    }
}

class OpenAIEmbedding:
    def __init__(
        self, model_id, tokenizer_id=None, dims=None, max_len=None, batch_size=1024,
        dtype=torch.half, device='cuda', tok_limit=None, req_limit=None, timepad=10, **kwargs
    ):
        import tiktoken
        from openai import OpenAI

        # runtime options
        self.dtype = dtype
        self.device = device

        # embed general options
        self.tok_limit = tok_limit
        self.req_limit = req_limit
        self.timepad = timepad

        # store sizing info
        config = openai_config[model_id]
        self.batch_size = batch_size
        self.max_len = config['max_len'] if max_len is None else max_len
        self.dims = config['dims'] if dims is None else dims

        # get tokenizer
        self.tokenizer = tiktoken.encoding_for_model(model_id)
        self.client = OpenAI()
        self.model = lambda t: self.client.embeddings.create(input=t, model=model_id)

        # usage tracking
        req_limit = config['req_limit'] if req_limit is None else req_limit
        tok_limit = config['tok_limit'] if tok_limit is None else tok_limit
        self.requests = RequestTracker((req_limit, tok_limit), 60+timepad)

    def truncate_batch(self, text):
        encs = [e[:self.max_len] for e in self.tokenizer.encode_batch(text) if len(e) > 0]
        text1 = [self.tokenizer.decode(e) for e in encs]
        sizes = [len(e) for e in encs]
        return text1, sizes

    def embed_batch(self, text):
        # truncate batches and log request
        text1, sizes = self.truncate_batch(text)
        self.requests.add(1, sum(sizes))

        # wait until clear on rate limits
        self.requests.ensure()

        # fetch embeddings and turn into tensor
        rets = self.model(text1)
        embeds = torch.tensor(
            [d.embedding for d in rets.data], dtype=self.dtype, device=self.device
        )

        # return on device tensor
        return embeds

    def embed(self, text, threaded=False):
        if type(text) is str:
            text = [text]
        return torch.cat([
            self.embed_batch(chunk) for chunk in batch_generator(iter(text), self.batch_size)
        ], dim=0)
