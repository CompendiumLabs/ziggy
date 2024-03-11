#!/usr/bin/env python

import os
import time
import json
from itertools import chain
from subprocess import run
from glob import glob

import torch

from .embed import HuggingfaceEmbedding, LlamaCppEmbedding
from .database import stream_jsonl, text_splitter
from .quant import Half, Float, QuantType

TASKS_CQAD = [
    'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval',
    'CQADupstackMathematicaRetrieval', 'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval',
    'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval', 'CQADupstackWordpressRetrieval'
]

TASKS_RETRIEVAL = [
    'ArguAna', 'ClimateFEVER', *TASKS_CQAD, 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO',
    'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID'
]

def task_split(task):
    return 'dev' if task == 'MSMARCO' else 'test'

def load_model(model, cpu=False, onnx=True, max_len=512, n_threads=None, verbose=False):
    if type(model) is str:
        if model.endswith('.gguf'):
            device = 'cpu' if cpu else 'cuda'
            emb = LlamaCppEmbedding(model, max_len=max_len, device=device, n_threads=n_threads, verbose=verbose)
        else:
            device = 'cpu' if cpu else 'cuda'
            emb = HuggingfaceEmbedding(model_id=model, max_len=max_len, device=device, onnx=onnx)
    else:
        emb = model
    return emb

def load_data(path, max_rows=None):
    if type(path) is str:
        _, ext = os.path.splitext(path)
        if ext == '.jsonl':
            data = (line['text'] for line in stream_jsonl(path, max_rows=max_rows))
        else:
            data = (line[:-1] for line in open(path))
    else:
        data = path
    return data

def profile_embed(
        model, path, cpu=False, max_len=512, delim='\n', min_len=100, max_rows=None,
        onnx=True, truncate=True, threaded=True, n_threads=None, verbose=False
    ):
    emb = load_model(model, cpu=cpu, onnx=onnx, max_len=max_len, n_threads=n_threads, verbose=verbose)

    # split data into chunks
    data = list(load_data(path, max_rows=max_rows))

    # do the embedding
    start = time.time()
    vecs = emb.embed(data, truncate=truncate, threaded=threaded)
    delta = time.time() - start

    # get document stats
    nchunks = len(data)
    length = torch.tensor([len(d) for d in data]).float()
    size_avg = length.mean()
    size_std = length.std()
    speed = nchunks/delta

    # get memory stats
    pid = os.getpid()
    cmd = run(
        f'nvidia-smi -q -x | xq -x \"/nvidia_smi_log/gpu/processes/process_info[./pid = \'{pid}\']/used_memory\"',
        shell=True, capture_output=True
    )
    mem = cmd.stdout.decode('utf-8').strip()

    # print our results
    print()
    print(f'Chunks: {nchunks}')
    print(f'Size: {size_avg:.2f} ± {size_std:.2f}')
    print(f'Time: {delta:.2f} seconds')
    print(f'Speed: {speed:.2f} chunks/second')
    print(f'Memory: {mem}')

def profile_tokenizer(
        model, path, cpu=False, max_len=512, delim='\n', min_len=100, max_rows=None,
        onnx=True, truncate=True, threaded=True, n_threads=None, verbose=False
):
    emb = load_model(model, cpu=cpu, onnx=onnx, max_len=max_len, n_threads=n_threads, verbose=verbose)

    # load data
    data = list(load_data(path, max_rows=max_rows))
    if max_len is not None:
        data = [d[:max_len] for d in data]

    # do the embedding
    start = time.time()
    toks = emb.tokenize(data)
    delta = time.time() - start

    if type(toks) is tuple:
        toks, attn = toks[1:]
        length = attn.sum(dim=-1).float()
    else:
        length = torch.tensor([len(d) for d in toks]).float()

    # get document stats
    nchunks = len(data)
    size_avg = length.mean()
    size_std = length.std()
    speed = nchunks/delta

    # print our results
    print()
    print(f'Chunks: {nchunks}')
    print(f'Size: {size_avg:.2f} ± {size_std:.2f}')
    print(f'Time: {delta:.2f} seconds')
    print(f'Speed: {speed:.2f} chunks/second')

def check_embed(mod_ll, mod_st, path, normalize=True, cpu=False, max_len=512, max_rows=None, trust_remote_code=False):
    from sentence_transformers import SentenceTransformer

    # set up device
    ngl = 0 if cpu else 99

    # load models
    if type(mod_ll) is str:
        mod_ll = LlamaCppEmbedding(mod_ll, max_len=max_len, n_gpu_layers=ngl)
    if type(mod_st) is str:
        mod_st = SentenceTransformer(mod_st, trust_remote_code=trust_remote_code)

    # load data
    data = list(load_data(path, max_rows=max_rows))
    if max_len is not None:
        data = [d[:max_len] for d in data]

    # compute embeddings
    emb_ll = mod_ll.embed(data, normalize=normalize, truncate=True).cpu().numpy()
    emb_st = mod_st.encode(data, normalize_embeddings=normalize)

    # compare embeddings
    sim = (emb_ll * emb_st).sum(axis=1)

    return sim

# check tokenizer individually
def check_tokenizer(mod_ll, mod_hf, path, max_rows=None, max_len=512):
    from llama_cpp import Llama
    from transformers import AutoTokenizer
    from Levenshtein import editops
    from termcolor import cprint

    # load models
    if type(mod_ll) is str:
        mod_ll = Llama(mod_ll, verbose=False)
    if type(mod_hf) is str:
        mod_hf = AutoTokenizer.from_pretrained(mod_hf)

    # load data
    data = list(load_data(path, max_rows=max_rows))
    if max_len is not None:
        data = [d[:max_len] for d in data]

    # compute token ids
    ids_ll = [mod_ll.tokenize(text.encode('utf-8')) for text in data]
    ids_st = [mod_hf.encode(text) for text in data]

    def tokmap(i, replace=False):
        tok = mod_hf._tokenizer.id_to_token(i)
        if tok.startswith('##'):
            return tok[2:]
        else:
            pre = '_' if replace else ' '
            return f'{pre}{tok}'

    # compare token ids
    for i, (id_ll, id_st) in enumerate(zip(ids_ll, ids_st)):
        if id_ll != id_st:
            print(f'Mismatch at index {i}')
            ops = {
                i1: (op, i2) for op, i1, i2 in editops(id_ll, id_st)
            }
            for pos1, id1 in enumerate(id_ll):
                if pos1 in ops:
                    op, pos2 = ops[pos1]
                    id2 = id_st[pos2]
                    tok1 = tokmap(id1, replace=True)
                    tok2 = tokmap(id2, replace=True)
                    if op == 'insert':
                        cprint(f'[+{tok1}]', color='green', attrs=['bold'], end='')
                    elif op == 'delete':
                        cprint(f'[-{tok1}]', color='red', attrs=['bold'], end='')
                    elif op == 'replace':
                        print('[', end='')
                        cprint(f'{tok1}', color='red', attrs=['bold'], end='')
                        cprint(f'→{tok2}', color='green', attrs=['bold'], end='')
                        print(']', end='')
                else:
                    tok1 = tokmap(id1, replace=False)
                    print(tok1, end='')
            print('\n')

# wrapper for quantization tests
class MtebWrapper:
    def __init__(self, model, qspec=None):
        self.model = model
        self.qspec = qspec

    def encode(self, text, **kwargs):
        print(f'encode: {len(text)}')
        vecs = self.model.embed(text, threaded=True)
        if self.qspec is None:
            vec1 = vecs
        else:
            qvec = self.qspec.quantize(vecs)
            vec1 = self.qspec.dequantize(qvec)
        return vec1.cpu().numpy()

def profile_mteb(model, maxlen=256, qspec=Half, savedir='benchmarks', onnx=True, compile=False):
    from mteb import MTEB

    if type(model) is str:
        emb = HuggingfaceEmbedding(model_id=model, maxlen=maxlen, onnx=onnx, compile=compile)
    else:
        emb = model

    # set up quantization
    emb1 = MtebWrapper(emb, qspec=qspec)
    quant = qspec.qtype.name

    # run benchmarks
    for task in TASKS_RETRIEVAL:
        split = task_split(task)
        evaluation = MTEB(tasks=[task], task_langs=['en'])
        results = evaluation.run(
            emb1, output_folder=f'{savedir}/{model}-{maxlen}-{quant}', eval_splits=[split]
        )

def path_to_spec(path):
    org, tag = path.split('/')
    return org, *tag.rsplit('-', maxsplit=2)

def extract_json(path, keys):
    name, _ = os.path.splitext(path)
    with open(path) as f:
        data = json.load(f)
    for k in keys:
        data = data[k]
    return data

def aggregate_mteb(savedir='benchmarks', metric='ndcg_at_10', display=True):
    import numpy as np
    import pandas as pd

    # get model specs
    models = glob(f'*/*', root_dir=savedir)
    specs = [path_to_spec(path) for path in models]

    # get benchmark results
    tasks = [[
        os.path.splitext(b)[0] for b in glob('*.json', root_dir=f'{savedir}/{p}')
    ] for p in models]
    results = [[
        (t, extract_json(f'{savedir}/{m}/{t}.json', (task_split(t), metric))) for t in ts
    ] for m, ts in zip(models, tasks)]

    # combine results
    info = chain(*[[(*m, *b) for b in bs] for m, bs in zip(specs, results)])

    # return dataframe
    columns = ['org', 'model', 'maxlen', 'quant']
    data = pd.DataFrame.from_records(info, columns=[*columns, 'bench', metric])
    data[metric] = 100*data[metric].astype(np.float32)

    # aggregate over CQAD tasks
    cqad = data[data['bench'].isin(TASKS_CQAD)]
    cqad = cqad.groupby(columns)[metric].mean().reset_index()
    cqad['bench'] = 'CQADupstackRetrieval'
    data = pd.concat([data, cqad], ignore_index=True)
    data = data[~data['bench'].isin(TASKS_CQAD)]

    # compute averages
    avgs = data.groupby(columns)[metric].mean().reset_index()
    avgs['bench'] = 'Average'
    data = pd.concat([data, avgs], ignore_index=True)

    # sort and reshape
    data = data.set_index([*columns, 'bench'])[metric].unstack('bench')
    data = data.sort_index(
        axis=1, key=lambda s: s.to_series().replace('Average', '')
    )

    # display or return
    if display:
        print(data.T.to_string(float_format='%.2f'))
    else:
        return data

# main entry point
if __name__ == '__main__':
    import fire

    fire.Fire({
        'profile': profile_embed,
        'mteb': profile_mteb,
        'aggregate': aggregate_mteb,
    })
