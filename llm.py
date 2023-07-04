import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from math import ceil, log2

## GOOD MODEL OPTIONS
# lmsys/vicuna-7b-v1.3
# tiiuae/falcon-7b-instruct

# sampler for manual generation
def sample(logits, top_k=None, temp=1.0):
    # only sample amongst top_k if given
    if top_k is not None:
        cutoff = torch.topk(logits, top_k, dim=-1).values[:,-1]
        logits = torch.where(logits >= cutoff.unsqueeze(-1), logits, -torch.inf)

    # turn into probabilities and return sample
    probs = torch.softmax(temp*logits, dim=-1)
    index = torch.multinomial(probs, 1).squeeze(-1)
    return index

class HuggingfaceModel:
    def __init__(self, model, **kwargs):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model code and weights
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
        output = self.model(input_ids, attn_mask)

        # get masked embedding
        state = output.hidden_states[-1]
        mask = attn_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        norm = embed.square().sum(1).sqrt()
        return embed/norm.unsqueeze(-1)

    def generate(self, prompt, num_samples=None, max_length=200, top_k=10, stream=False):
        num_return_sequences = num_samples if num_samples is not None else 1
        streamer = TextStreamer(self.tokenizer) if stream else None

        # encode input prompt
        input_ids, attn_mask = self.encode(prompt)

        # generate output ids
        output_ids = self.model.generate(
            input_ids, max_length=max_length, do_sample=True, streamer=streamer,
            top_k=top_k, num_return_sequences=num_return_sequences
        )

        if not stream:
            # decode output text
            output_text = [
                self.tokenizer.decode(oids, skip_special_tokens=True) for oids in output_ids
            ]

            # return output text
            return output_text[0] if num_samples is None else output_text

    def generator(self, prompt, num_samples=None, max_length=200, top_k=10, temp=1.0):
        num_return_sequences = num_samples if num_samples is not None else 1

        # encode input prompt
        input_ids, attn_mask = self.encode(prompt)

        # get new logit at last element
        output = self.model(input_ids, attn_mask)
        logits = output.logits[:,-1,:]
        index = sample(logits, top_k=top_k, temp=temp)

        # decode and return
        toks = self.tokenizer.decode(index)

        # shift and add to input_ids
        input_ids = torch.cat(input_ids[:,1:,:], index.unsqueeze(1), 1)

class TorchVectorIndex:
    def __init__(self, dims, size=1024, dtype=torch.float32, device='cuda'):
        # configuration
        self.dims = dims
        self.size = size
        self.dtype = dtype
        self.device = device
        assert(log2(size) % 1 == 0)

        # empty state
        self.labels = []
        self.values = torch.empty(size, dims, dtype=dtype, device=device)

    def expand(self, min_size):
        # check if needed
        if self.size >= min_size:
            return

        # increase size to next power of 2
        values_old = self.values
        self.size = pow(2, round(ceil(log2(min_size))))

        # create new tensor and assign old values
        self.values = torch.empty(
            self.size, self.dims, dtype=self.dtype, device=self.device
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

        # update existing
        locs, idxs = zip([
            (i, self.labels.index(x)) for i, x in enumerate(labs) if x in labs
        ])
        self.values[idxs,:] = vals[locs,:]

        # add in new labels and track size
        nlabels0 = len(self.labels)
        self.labels.append(novel)
        nlabels1 = len(self.labels)

        # expand tensor and assign new values
        self.expand(nlabels1)
        self.values[nlabels0:nlabels1,:] = vals

    def remove(self, labs):
        pass

    def search(self, vec):
        pass

class FaissVectorIndex:
    pass

# example usage
# gen = HuggingfaceModel('tiiuae/falcon-7b-instruct')
# gen.generate('Write a poem about Valencia.', stream=True)
