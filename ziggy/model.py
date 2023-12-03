# gpt-fast tools

import torch
from pathlib import Path

from model import Transformer
from quantize import WeightOnlyInt8QuantHandler, WeightOnlyInt4QuantHandler

import torch._inductor.config
import torch._dynamo.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

# Experimental feature to reduce compilation times, will be on by default in future (requires nightly right now)
# torch._inductor.config.fx_graph_cache = True 

##
## Loading
##

def load_torch_model(checkpoint_path, device='cuda', bits=8, precision=torch.bfloat16):
    if type(checkpoint_path) is str:
        checkpoint_path = Path(checkpoint_path)

    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if bits == 16:
        pass
    elif bits == 8:
        print('Using int8 weight-only quantization!')
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()
    elif bits == 4:
        print('Using int4 quantization!')
        path_comps = checkpoint_path.name.split('.')
        assert path_comps[-2].startswith('g')
        groupsize = int(path_comps[-2][1:])
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()
    else:
        raise Exception(f'Unsupported bits value [{bits}]')

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()

##
## Sampling
##

def multinomial_sample(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    i = torch.argmax(probs_sort / q, dim=-1, keepdim=True)
    return i.to(dtype=torch.int)

def logits_to_probs(logits, temp=1.0, top_k=None):
    logits = logits / max(temp, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -torch.inf, logits)
    return torch.softmax(logits, dim=-1)

def sample(logits, temp=1.0, top_k=None):
    probs = logits_to_probs(logits[0, -1], temp=temp, top_k=top_k)
    return multinomial_sample(probs)

def decode_token(model, x, input_pos, **kwargs):
    with torch.no_grad():
        logits = model(x.view(1, -1), input_pos)
    return sample(logits, **kwargs)
