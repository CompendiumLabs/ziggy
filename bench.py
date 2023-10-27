#!/usr/bin/env python

import os
import time
import json
from itertools import chain
from subprocess import run
from glob import glob

import torch
from mteb import MTEB

from llm import HuggingfaceEmbedding
from database import stream_jsonl, paragraph_splitter, DocumentDatabase
from quant import Half, Float, QuantType

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

# load jsonl chunks directly
def load_data(path, delim='\n', minlen=100, nrows=None):
    docs = [line['text'] for line in stream_jsonl(path, maxrows=nrows)]
    chunks = list(chain(*[paragraph_splitter(doc, delim=delim, minlen=minlen) for doc in docs]))
    return chunks

def profile_embed(model, path, maxlen=256, delim='\n', minlen=100, maxrows=None, onnx=True, compile=False):
    if type(model) is str:
        emb = HuggingfaceEmbedding(model_id=model, maxlen=maxlen, onnx=onnx, compile=compile)
    else:
        emb = model

    # index data
    start = time.time()
    db = DocumentDatabase.from_jsonl(path, embed=emb, delim=delim, minlen=minlen, maxrows=maxrows)
    delta = time.time() - start

    # get document stats
    ndocs = db.dindex.size()
    nchunks = db.cindex.size()
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
    print(f'Documents: {ndocs}')
    print(f'Chunks: {nchunks}')
    print(f'Time: {delta:.2f} seconds')
    print(f'Speed: {speed:.2f} chunks/second')
    print(f'Memory: {mem}')

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
