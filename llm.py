import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

## GOOD MODEL OPTIONS
# lmsys/vicuna-7b-v1.3
# tiiuae/falcon-7b-instruct

class TextGenerator:
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

        # get embedding
        state = output.hidden_states[-1]
        mask = attn_mask.unsqueeze(-1).float()
        embed = (state*mask).sum(1)/mask.sum(1)

        # return embedding
        return embed

    def generate(self, prompt, num_samples=None, max_length=200, top_k=10):
        # encode input prompt
        input_ids, attn_mask = self.encode(prompt)

        # generate output ids
        num_return_sequences = num_samples if num_samples is not None else 1
        output_ids = self.model.generate(
            input_ids, max_length=max_length, do_sample=True,
            top_k=top_k, num_return_sequences=num_return_sequences
        )

        # decode output text
        output_text = [
            self.tokenizer.decode(oids, skip_special_tokens=True) for oids in output_ids
        ]

        # return output text
        return output_text[0] if num_samples is None else output_text

# example usage
# gen = TextGenerator('tiiuae/falcon-7b-instruct')
# print(gen.generate('Write a poem about Valencia.'))

