# gpt-fast tools

import torch
from pathlib import Path

from gptfast.model import Transformer
from quantize import WeightOnlyInt8QuantHandler, WeightOnlyInt4QuantHandler

import torch._inductor.config
import torch._dynamo.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

from ziggy.utils import sprint, convert_sentencepice

# Experimental feature to reduce compilation times, will be on by default in future (requires nightly right now)
# torch._inductor.config.fx_graph_cache = True 

##
## loading
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
## sampling
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

##
## model
##

class TorchLlamaModel:
    def __init__(self, checkpoint_path, tokenizer_path=None, max_seq_length=1024, bits=8, compile=True, device='cuda'):
        if type(checkpoint_path) is str:
            checkpoint_path = Path(checkpoint_path)
        if tokenizer_path is None:
            tokenizer_path = checkpoint_path.parent / 'tokenizer.model'

        self.device = device
        self.toker = SentencePieceProcessor(model_file=str(tokenizer_path))

        print('Loading model...')
        self.model = load_torch_model(checkpoint_path, device=device, bits=bits)

        if compile:
            print('Compiling model...')
            self.decode = torch.compile(decode_token, mode='reduce-overhead', fullgraph=True)
        else:
            self.decode = decode_token

        with torch.device(device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    def encode(self, text, padding=None):
        tokens = [self.toker.bos_id()] + self.toker.encode(text)
        return torch.tensor(tokens, dtype=torch.int, device=self.device)

    def generate(self, prompt, num_new_tokens=256, **kwargs):
        # tokenize input prompt
        tokens = self.encode(prompt)
        T = tokens.size(0)

        # do first token gen
        input_pos = torch.arange(0, T, device=self.device)
        next_token = decode_token(self.model, tokens, input_pos)[0]

        # yield the first token
        piece = self.toker.id_to_piece(next_token.item())
        yield convert_sentencepice([piece])

        # set up for one at a time
        input_pos = torch.tensor([T], device=self.device, dtype=torch.int)

        # loop until limit or eos token
        for i in range(num_new_tokens):
            # `clone` is needed to prevent torch.compile errors!
            cur_token = next_token.clone()

            # decode next token
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                next_token = self.decode(self.model, cur_token, input_pos, **kwargs)

            # check for end of string token or advance
            if next_token == self.toker.eos_id():
                break
            else:
                input_pos += 1

            # yield property formatted next token
            piece = self.toker.id_to_piece(next_token.item())
            yield convert_sentencepice([piece])

    def igenerate(self, prompt, **kwargs):
        for s in self.generate(prompt, **kwargs):
            sprint(s)
