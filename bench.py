import torch
from itertools import chain

import mteb

from database import stream_jsonl, paragraph_splitter

# load jsonl chunks directly
def load_data(path, delim='\n', minlen=100, nrows=None):
    docs = [line['text'] for line in stream_jsonl(path, maxrows=nrows)]
    chunks = list(chain(*[paragraph_splitter(doc, delim=delim, minlen=minlen) for doc in docs]))
    return chunks

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
