## Embedding models

import os
import numpy as np
from math import ceil
from itertools import chain
from pathlib import Path

import torch
from torch.nn.functional import normalize as norm
from transformers import AutoTokenizer, AutoModel

from .utils import (
    pipeline_threads, batch_generator, batch_indices, list_splitter, cumsum, cumul_bounds, RequestTracker
)

##
## Constants
##

DEFAULT_EMBED = 'BAAI/bge-large-en-v1.5'

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

class HuggingfaceEmbedding:
    def __init__(
        self, model_id=DEFAULT_EMBED, tokenize_id=None, max_len=None, batch_size=128,
        queue_size=256, device='cuda', dtype=None, onnx=None, save_dir=None, compile=False,
        pooling_type='cls', trust_remote_code=False
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

        # get model info
        self.name = model_id
        self.batch_size = batch_size
        self.max_len = self.model.config.max_position_embeddings if max_len is None else max_len
        self.dims = self.model.config.hidden_size
        self.pooling_type = pooling_type

    def tokenize_batch(self, text, truncate=False):
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

    def forward_batch(self, input_ids, attention_mask):
        # prepare model inputs on device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64, device=self.device)

        # get model output
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        state = output[0]

        # get sentence embeddings
        if self.pooling_type == 'mean':
            mask = attention_mask.float().unsqueeze(-1)
            embed = (state*mask).sum(1)/mask.sum(1)
        elif self.pooling_type == 'cls':
            embed = state[:,0,:]

        # return normalized embedding
        return norm(embed, dim=-1)

    def embed_batch(self, text, truncate=False):
        doc_indices, input_ids, attention_mask = self.tokenize_batch(text, truncate=truncate)
        indices = batch_indices(input_ids.size(0), self.batch_size)
        embed = torch.cat([
            self.forward_batch(input_ids[i1:i2], attention_mask[i1:i2]) for i1, i2 in indices
        ])
        return doc_indices, embed

    def embed(self, text, threaded=False, truncate=False, normalize=True):
        # handle unit case
        if type(text) is str:
            text = [text]

        if threaded:
            # make workers
            results = []
            def loader():
                yield from batch_generator(text, self.batch_size)
            def tokenizer(texts):
                return self.tokenize_batch(texts, truncate=truncate)
            def forwarder(data):
                doc_indices, input_ids, attention_mask = data
                indices = batch_indices(input_ids.size(0), self.batch_size)
                embed = torch.cat([
                    self.forward_batch(input_ids[i1:i2], attention_mask[i1:i2]) for i1, i2 in indices
                ])
                results.append((doc_indices, embed))

            # embed chunks and average
            pipeline_threads(loader(), tokenizer, forwarder, maxsize=self.queue_size)
        else:
            # embed chunks and average
            results = [
                self.embed_batch(chunk, truncate=truncate) for chunk in batch_generator(text, self.batch_size)
            ]

        # unpack offsets and embeds
        offsets, embeds = zip(*results)

        # document offsets come zero-indexed within batch
        nchunks = [o.max().item()+1 for o in offsets]
        indices = torch.cat([o+n for o, n in zip(offsets, cumsum(nchunks))])
        _, sizes = torch.unique(indices, return_counts=True)

        # aggregate to document level
        embed0 = torch.cat(embeds, dim=0)
        embed = torch.stack([
            embed0[i:j,:].mean(dim=0) for i, j in cumul_bounds(sizes)
        ], dim=0)

        # return normalized vectors
        if normalize:
            embed = norm(embed, dim=-1)

        # return normalized vectors
        return embed

##
## llama.cpp
##

class LlamaCppEmbedding:
    def __init__(self, model_path, max_len=512, device='cuda', verbose=False, **kwargs):
        from llama_cpp import Llama

        # set up device
        ngl = 0 if device == 'cpu' else 99
        self.device = device

        # load model
        self.model = Llama(
            model_path, embedding=True, n_batch=max_len, n_gpu_layers=ngl, verbose=verbose
        )

        # get metadata
        self.max_len = max_len
        self.dims = self.model.n_embd()

    def tokenize(self, text):
        return self.model.tokenize(text.encode('utf-8'))

    def forward_batch(self, tokens):
        from llama_cpp import llama_kv_cache_clear, llama_get_embeddings_ith

        ctx = self.model._ctx
        n_seq = len(tokens)
        n_embd = self.model.n_embd()
        n_toks = sum(len(toks) for toks in tokens)

        # add tokens to batch
        self.model._batch.reset()
        for seq_id, toks in enumerate(tokens):
            assert len(toks) <= self.max_len
            self.model._batch.add_sequence(toks, seq_id, False)

        # run model on batch
        llama_kv_cache_clear(ctx.ctx)
        ctx.decode(self.model._batch)

        # store embeddings
        embeds = [
            llama_get_embeddings_ith(ctx.ctx, i)[:n_embd]
            for i in range(n_seq)
        ]

        # return on device
        return torch.tensor(embeds, device=self.device, dtype=torch.float32)

    def forward(self, tokens, size=None):
        # set up output
        size = len(tokens) if size is None else size
        embeds = torch.empty(size, self.dims, device=self.device, dtype=torch.float32)
        pos = 0

        # int batch
        batch = []
        batch_seq = 0
        batch_tok = 0

        # accumulate and forward
        for toks in tokens:
            ntoks = len(toks)

            # forward if we're full
            if batch_tok + ntoks > self.max_len:
                embeds[pos:pos+batch_seq] = self.forward_batch(batch)
                pos += batch_seq

                # reset batch
                batch = []
                batch_seq = 0
                batch_tok = 0

            # append for later
            batch.append(toks)
            batch_seq += 1
            batch_tok += ntoks

        # forward any remaining
        embeds[pos:pos+batch_seq] = self.forward_batch(batch)
        return embeds

    # this is (good) lazy — zero copy with truncate, one copy with split
    def embed(self, text, threaded=None, truncate=False, normalize=True):
        if type(text) is str:
            text = [text]

        # get tokens and handle truncation
        tokens = [self.tokenize(t) for t in text]

        # forward based on case
        if truncate:
            embeds = self.forward((t[:self.max_len] for t in tokens), size=len(tokens))
        else:
            # flatten into chunks and forward
            splits = [ceil(len(t) / self.max_len) for t in tokens]
            chunks = chain.from_iterable(list_splitter(t, self.max_len) for t in tokens)
            embeds0 = self.forward(chunks, size=sum(splits))

            # aggreate to document level
            embeds = torch.stack([
                embeds0[i:j,:].mean(dim=0) if j > i + 1 else embeds0[i,:]
                for i, j in cumul_bounds(splits)
            ], dim=0)

        # return normalized vectors
        if normalize:
            embeds = norm(embeds, dim=-1)

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
        self, model_id=DEFAULT_OPENAI_EMBED, tokenizer_id=None, dims=None, max_len=None, batch_size=1024,
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
