import time
from itertools import chain
from subprocess import run

import torch
import mteb

from llm import HuggingfaceEmbedding
from database import stream_jsonl, paragraph_splitter, DocumentDatabase

# load jsonl chunks directly
def load_data(path, delim='\n', minlen=100, nrows=None):
    docs = [line['text'] for line in stream_jsonl(path, maxrows=nrows)]
    chunks = list(chain(*[paragraph_splitter(doc, delim=delim, minlen=minlen) for doc in docs]))
    return chunks

def profile_embed(model, path, delim='\n', minlen=100, maxlen=256, maxrows=None, onnx=True, compile=False):
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
    cmd = run(
        'nvidia-smi --format=csv,noheader,nounits --query-gpu=memory.used',
        shell=True, capture_output=True
    )
    mem = int(cmd.stdout.decode('utf-8').strip())

    print()
    print(f'Documents: {ndocs}')
    print(f'Chunks: {nchunks}')
    print(f'Time: {delta:.2f} seconds')
    print(f'Speed: {speed:.2f} chunks/second')
    print(f'Memory: {mem} MiB')

# wrapper for quantization tests
class QuantizationWrapper:
    def __init__(model, qspec):
        self.model = model
        self.qspec = qspec

    def embed(self, text):
        vecs = self.model.embed(text)
        qvec = self.qspec.quantize(vecs)
        dvec = self.qspec.dequantize(qvec)
        return dvec

# main entry point
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Profile LLM embedding')
    parser.add_argument('model', type=str, help='model id or path')
    parser.add_argument('path', type=str, help='path to jsonl file')
    parser.add_argument('--maxlen', type=int, default=256, help='maximum paragraph length')
    parser.add_argument('--delim', type=str, default='\n', help='paragraph delimiter')
    parser.add_argument('--minlen', type=int, default=100, help='minimum paragraph length')
    parser.add_argument('--maxrows', type=int, default=None, help='number of rows to load')
    parser.add_argument('--no-onnx', action='store_true', help='use onnx model')
    parser.add_argument('--compile', action='store_true', help='compile onnx model')
    args = parser.parse_args()

    profile_embed(
        args.model, args.path, maxlen=args.maxlen, delim=args.delim, minlen=args.minlen,
        maxrows=args.maxrows, onnx=not args.no_onnx, compile=args.compile
    )
