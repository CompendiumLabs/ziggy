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

def profile_embed(
        model, path, cpu=False, max_len=512, delim='\n', min_len=100, max_rows=None,
        onnx=True, threaded=True, n_threads=1
    ):
    if type(model) is str:
        if model.endswith('.gguf'):
            ngl = 0 if cpu else 99
            emb = LlamaCppEmbedding(model, max_len=max_len, n_gpu_layers=ngl, n_threads=n_threads)
        else:
            device = 'cpu' if cpu else 'cuda'
            emb = HuggingfaceEmbedding(model_id=model, max_len=max_len, device=device, onnx=onnx)
    else:
        emb = model

    # split data into chunks
    if type(path) is str:
        _, ext = os.path.splitext(path)
        if ext == '.jsonl':
            data = (line['text'] for line in stream_jsonl(path, max_rows=max_rows))
        else:
            data = (line[:-1] for line in open(path))
    else:
        data = path

    splitter = lambda text: text_splitter(text, delim, min_len=min_len, max_len=max_len)
    chunks = sum((splitter(t) for t in data), [])

    # do the embedding
    start = time.time()
    vecs = emb.embed(chunks, truncate=True, threaded=threaded)
    delta = time.time() - start

    # get document stats
    nchunks = len(chunks)
    length = torch.tensor([len(d) for d in chunks]).float()
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

def check_embed(gguf, repo_id, path, normalize=True, cpu=False, max_len=512, max_rows=None, trust_remote_code=False):
    from .embed import LlamaCppEmbedding
    from sentence_transformers import SentenceTransformer

    # set up device
    ngl = 0 if cpu else 99

    # load models
    mod_ll = LlamaCppEmbedding(gguf, max_len=max_len, n_gpu_layers=ngl)
    mod_st = SentenceTransformer(repo_id, trust_remote_code=trust_remote_code)

    # load data
    data = open(path).read().splitlines()
    if max_rows is not None:
        data = data[:max_rows]

    # compute embeddings
    emb_ll = mod_ll.embed(data, normalize=normalize, truncate=True).cpu().numpy()
    emb_st = mod_st.encode(data, normalize_embeddings=normalize)

    # compare embeddings
    sim = (emb_ll * emb_st).sum(axis=1)

    return sim

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
