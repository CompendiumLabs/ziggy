#!/usr/bin/env python

import os
import time
from itertools import chain
from subprocess import run

import torch
from mteb import MTEB

from llm import HuggingfaceEmbedding
from database import stream_jsonl, paragraph_splitter, DocumentDatabase

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

def profile_mteb(model, maxlen=256, onnx=True, compile=False):
    if type(model) is str:
        emb = HuggingfaceEmbedding(model_id=model, maxlen=maxlen, onnx=onnx, compile=compile)
    else:
        emb = model
    emb1 = MtebWrapper(emb)

    # run benchmarks
    evaluation = MTEB(task_types=['Retrieval'])
    results = evaluation.run(emb1, output_folder=f'benchmarks/{model}-{maxlen}')

    # print our results
    print(results)

# main entry point
if __name__ == '__main__':
    import fire

    fire.Fire({
        'profile': profile_embed,
        'mteb': profile_mteb,
    })
