import torch
import faiss
import faiss.contrib.torch_utils
from math import ceil, log2
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

## GOOD MODEL OPTIONS
# lmsys/vicuna-7b-v1.3
# tiiuae/falcon-7b-instruct

# printer for streaming
def sprint(s):
    print(s, end='', flush=True)

# sampler for manual generation
def sample(logits, top_k=None, temp=1.0):
    # only sample amongst top_k if given
    if top_k is not None:
        cutoff = torch.topk(logits, top_k, dim=-1).values[:,-1]
        logits = torch.where(logits >= cutoff.unsqueeze(-1), logits, -torch.inf)

    # turn into probabilities and return sample
    probs = torch.softmax(logits/temp, dim=-1)
    index = torch.multinomial(probs, 1).squeeze(-1)
    return index

class HuggingfaceModel:
    def __init__(self, model, block_size=1024, **kwargs):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model code and weights
        self.block_size = block_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map='auto', trust_remote_code=True, output_hidden_states=True,
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            **kwargs
        )

    def encode(self, text):
        data = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        )
        return data['input_ids'].to('cuda'), data['attention_mask'].to('cuda')

    def embed(self, text):
        # encode input text
        input_ids, attn_mask = self.encode(text)

        # get model output
        output = self.model(input_ids)

        # get masked embedding
        state = output.hidden_states[-1]
        mask = attn_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        norm = embed.square().sum(1).sqrt()
        return embed/norm.unsqueeze(-1)

    def generate(self, prompt, max_new_tokens=200, top_k=10, stream=False):
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True) if stream else None

        # encode input prompt
        input_ids, _ = self.encode(prompt)

        # generate output ids
        output_ids = self.model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            streamer=streamer, top_k=top_k
        )

        if not stream:
            # decode output text
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text

    # proper python generator variant that uses model.__call__ directly
    def generator(self, prompt, max_new_tokens=200, top_k=10, temp=1.0):
        # encode input prompt
        input_ids, _ = self.encode(prompt)

        # trim if needed
        if input_ids.size(1) > self.block_size:
            input_ids = input_ids[:,:self.block_size]

        # loop until limit and eos token
        for i in range(max_new_tokens):
            # get new index at last element
            output = self.model(input_ids)
            logits = output.logits[:,-1,:]
            index = sample(logits, top_k=top_k, temp=temp)

            # break if we hit end token
            if index[0] == self.tokenizer.eos_token_id:
                break

            # decode and return
            yield self.tokenizer.decode(index)

            # shift and add to input_ids
            trim = 1 if input_ids.size(1) == self.block_size else 0
            input_ids = torch.cat((input_ids[:,trim:], index.unsqueeze(1)), dim=1)

class VectorIndex:
    def __init__(self, dims, dtype=torch.float32, device='cuda'):
        self.dims = dims
        self.dtype = dtype
        self.device = device

class TorchVectorIndex(VectorIndex):
    def __init__(self, dims, max_size=1024, dtype=torch.float32, device='cuda'):
        assert(log2(max_size) % 1 == 0)
        super().__init__(dims, dtype=dtype, device=device)

        # empty state
        self.max_size = max_size
        self.labels = []
        self.values = torch.empty(max_size, dims, dtype=dtype, device=device)

    def size(self):
        return len(self.labels)

    def expand(self, min_size):
        # check if needed
        if self.max_size >= min_size:
            return

        # increase size to next power of 2
        values_old = self.values
        self.max_size = pow(2, round(ceil(log2(min_size))))

        # create new tensor and assign old values
        self.values = torch.empty(
            self.max_size, self.dims, dtype=self.dtype, device=self.device
        )
        self.values[:nlabels,:] = values_old[:nlabels,:]

    def add(self, labs, vecs, strict=False):
        # validate input size
        nlabs = len(labs)
        nv, dv = vecs.shape
        assert(nv == nlabs)
        assert(dv == self.dims)

        # get breakdown of new vs old
        slabs = set(labs)
        exist = slabs.intersection(self.labels)
        novel = slabs - exist

        # raise if trying invalid strict add
        if strict and len(exist) > 0:
            raise Exception(f'Label {lab} is already in index.')

        if len(exist) > 0:
            # update existing
            elocs, idxs = zip([
                (i, self.labels.index(x)) for i, x in enumerate(labs) if x in exist
            ])
            self.values[idxs,:] = vals[elocs,:]

        if len(novel) > 0:
            # add in new labels
            nlabels0 = self.size()
            self.labels.extend(novel)

            # expand size if needed
            nlabels1 = self.size()
            self.expand(nlabels1)

            # assign new vector values
            nlocs = [i for i, x in enumerate(labs) if x in novel]
            self.values[nlabels0:nlabels1,:] = vecs[nlocs,:]

    def remove(self, labs):
        pass

    def search(self, vecs, k, return_simil=True):
        # allow for single vec
        if vecs.ndim == 1:
            vecs = vecs.unsqueeze(0)

        # compute distance matrix
        num = self.size()
        sim = vecs @ self.values[:num,:].T

        # get top results
        tops = torch.topk(sim, k)
        labs = [[self.labels[i] for i in row] for row in tops.indices]
        vals = tops.values

        # return labels/simils
        return labs, vals if return_simil else labs

# faiss doesn't work great with ids and removal!
class FaissVectorIndex(VectorIndex):
    def __init__(self, dims, spec='Flat', dtype=torch.float32, device='cuda'):
        assert(dtype == torch.float32)
        super().__init__(dims, dtype=dtype, device=device)

        # create faiss index
        self.labels = []
        self.index = faiss.index_factory(dims, spec)

    def train(self, vecs):
        self.index.train(vecs)

    def add(self, labs, vecs):
        self.index.add_with_ids(vecs, labs)

    def remove(self, labs):
        self.index.remove_ids(labs)

    def search(self, vec, k):
        return self.index.search(vec, k)

# example usage
# gen = HuggingfaceModel('tiiuae/falcon-7b-instruct')
# gen.generate('Write a poem about Valencia.', stream=True)
