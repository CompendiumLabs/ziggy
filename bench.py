import torch
import torch.nn.functional as F
from itertools import chain

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

from database import stream_jsonl, paragraph_splitter

def load_data(path, nrows=1024):
    docs = [line['text'] for line in stream_jsonl(path, maxrows=nrows)]
    chunks = list(chain(*[paragraph_splitter(doc, delim='\n', minlen=100) for doc in docs]))
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def save_onnx(model, path):
    # save onnx tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.save_pretrained(path)

    # load vanilla transformers and convert to onnx
    model = ORTModelForFeatureExtraction.from_pretrained(model, from_transformers=True)
    optimizer = ORTOptimizer.from_pretrained(model)
    config = OptimizationConfig(optimization_level=99)

    # apply the optimization configuration to the model
    optimizer.optimize(save_dir=path, optimization_config=config)

class ONNXModel:
    def __init__(self, path, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = ORTModelForFeatureExtraction.from_pretrained(path, file_name='model_optimized.onnx')

    def encode(self, text, **kwargs):
        args = {'padding': True, 'truncation': True, **kwargs}
        data = self.tokenizer(text, return_tensors='pt', **args)
        return data

    def embed(self, inputs):
        model_inputs = self.encode(inputs)
        model_outputs = self.model(**model_inputs)
        embeddings = mean_pooling(model_outputs, model_inputs['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)
