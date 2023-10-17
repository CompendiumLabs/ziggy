## LLM generation and embeddings

from math import ceil
from itertools import chain
import os
import torch

from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from utils import pipeline_threads, batch_generator, cumsum, cumul_bounds, sprint

##
## Constants
##

# default settings
DEFAULT_MODEL = 'meta-llama/Llama-2-7b-chat-hf'
DEFAULT_EMBED = 'sentence-transformers/all-MiniLM-L6-v2'

# llama special strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "\n<</SYS>>"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and honest assistant. Always answer if you are able to. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do not provide warnings about ethics, morality, illegality, safety, or possible incorrectness."

##
## Utils
##

# generate a llama query
def llama_chat(query, system_prompt):
    return f'{B_INST} {B_SYS}\n{system_prompt}\n{E_SYS}\n\n{query} {E_INST}'

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

# convert sentencepiece tokens to text
def convert_sentencepice(toks):
    return ''.join([
        tok.replace('▁', ' ') if tok.startswith('▁') else tok.replace('<0x0A>', '\n') for tok in toks
    ])

def length_splitter(text, max_length):
    if (length := len(text)) > max_length:
        nchunks = ceil(length/max_length)
        starts = [i*max_length for i in range(nchunks)]
        return [text[s:s+max_length] for s in starts]
    else:
        return [text]

##
## Models
##

class HuggingfaceModel:
    def __init__(
        self, model=DEFAULT_MODEL, device='cuda', bits=None, compile=False, **kwargs
    ):
        # set options
        self.device = device

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # choose right bits
        if bits is None:
            bits = 16 if device == 'cuda' else 32
        if device == 'cuda' and bits == 4:
            bitargs = {'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
        elif device == 'cuda' and bits == 8:
            bitargs = {'load_in_8bit': True, 'bnb_8bit_compute_dtype': torch.bfloat16}
        elif bits == 16:
            bitargs = {'torch_dtype': torch.float16}
        elif bits == 32:
            bitargs = {'torch_dtype': torch.float32}
        else:
            raise Exception(f'Unsupported device/bits combination: {device}/{bits}')

        # load model code and weights
        self.modconf = AutoConfig.from_pretrained(
            model, output_hidden_states=True, pretraining_tp=1, token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, config=self.modconf,
            device_map=device, **bitargs, **kwargs
        )

        # compile model if needed
        if compile:
            self.model = torch.compile(self.model)

    def encode(self, text, **kwargs):
        targs = {'padding': True, 'truncation': True, **kwargs}
        data = self.tokenizer(text, return_tensors='pt', **targs)
        return data['input_ids'].to(self.device), data['attention_mask'].to(self.device)

    def embed(self, text, **kwargs):
        # encode input text
        input_ids, attn_mask = self.encode(text, **kwargs)

        # get model output (no grad for memory usage)
        with torch.no_grad():
            output = self.model(input_ids, attn_mask)

        # get masked embedding
        state = output.hidden_states[-1]
        mask = attn_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        return normalize(embed, dim=-1)

    # proper python generator variant that uses model.__call__ directly
    def generate(self, prompt, chat=True, context=2048, maxlen=2048, top_k=10, temp=1.0):
        # splice in chat instructions
        if chat is not False:
            system_prompt = DEFAULT_SYSTEM_PROMPT if chat is True else chat
            prompt = llama_chat(prompt, system_prompt=system_prompt)

        # encode input prompt
        input_ids, _ = self.encode(prompt)

        # trim if needed
        if input_ids.size(1) > context:
            input_ids = input_ids[:,:context]

        # loop until limit and eos token
        for i in range(maxlen):
            # generate next logits (no grad for memory usage)
            with torch.no_grad():
                output = self.model(input_ids)

            # get new index at last element
            logits = output.logits[:,-1,:]
            index = sample(logits, top_k=top_k, temp=temp)

            # break if we hit end token
            if index[0] == self.tokenizer.eos_token_id:
                break

            # decode and return (llama not doing this right)
            tokens = self.tokenizer.convert_ids_to_tokens(index)
            text = convert_sentencepice(tokens)
            yield text.lstrip() if i <= 1 else text

            # shift and add to input_ids
            trim = 1 if input_ids.size(1) == context else 0
            input_ids = torch.cat((input_ids[:,trim:], index.unsqueeze(1)), dim=1)

# this has to take context at creation time
# NOTE: llama2-70b needs n_gqa=8
class LlamaCppModel:
    def __init__(self, model_path, context=2048, n_gpu_layers=100, verbose=False, **kwargs):
        self.model = Llama(
            model_path, n_ctx=context, n_gpu_layers=n_gpu_layers, verbose=verbose, **kwargs
        )

    def generate(self, prompt, chat=True, context=None, maxlen=512, top_k=10, temp=1.0, **kwargs):
        # splice in chat instructions
        if chat is not False:
            system_prompt = DEFAULT_SYSTEM_PROMPT if chat is True else chat
            prompt = llama_chat(prompt, system_prompt=system_prompt)

        # construct stream object
        stream = self.model(prompt, max_tokens=maxlen, stream=True, top_k=top_k, temperature=temp, **kwargs)

        # return generated tokens
        for i, output in enumerate(stream):
            choice, *_ = output['choices']
            text = choice['text']
            yield text.lstrip() if i <= 1 else text

    def igenerate(self, query, **kwargs):
        for s in self.generate(query, **kwargs):
            sprint(s)

##
## Embeddings
##

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
    from optimum.onnxruntime.configuration import OptimizationConfig
except:
    print('ONNX not available.')

class HuggingfaceEmbedding:
    def __init__(self, model_id=DEFAULT_EMBED, maxlen=None, batch_size=128, save_dir='onnx', device='cuda', onnx=False, compile=False):
        # runtime options
        self.device = device

        # load params
        if onnx:
            ModelConstructor = ORTModelForFeatureExtraction.from_pretrained
            model_path = os.path.join(save_dir, model_id)
        else:
            ModelConstructor = AutoModel.from_pretrained
            model_path = model_id

        # compile if needed
        if onnx and (compile or not os.path.isdir(model_id)):
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_id, export=True
            )
            optimization_config = OptimizationConfig(
                optimization_level=99, optimize_for_gpu=True, fp16=True
            )
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(
                save_dir=model_path, optimization_config=optimization_config
            )

        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = ModelConstructor(model_path).to(device)

        # get model info
        self.batch_size = batch_size
        self.maxlen = self.model.config.max_position_embeddings if maxlen is None else maxlen
        self.dims = self.model.config.hidden_size

    def tokenize_batch(self, text, maxlen=None, **kwargs):
        maxlen = maxlen if maxlen is not None else self.maxlen
        targs = {
            'padding': 'max_length', 'truncation': True, 'max_length': maxlen,
            'return_overflowing_tokens': True, **kwargs
        }
        encode = self.tokenizer(text, return_tensors='pt', **targs)
        return encode.overflow_to_sample_mapping, encode.input_ids, encode.attention_mask

    def forward_batch(self, input_ids, attention_mask):
        # prepare model inputs on device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64, device=self.device)

        # get model output
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # get masked embedding
        state = output[0]
        mask = attention_mask.float().unsqueeze(-1)
        embed = (state*mask).sum(1)/mask.sum(1)

        # return normalized embedding
        return normalize(embed, dim=-1)

    def embed_batch(self, text, **kwargs):
        doc_indices, input_ids, attention_mask = self.tokenize_batch(text, **kwargs)
        embed = self.forward_batch(input_ids, attention_mask)
        return doc_indices, embed

    def embed(self, text, maxlen=None, batch_size=None, queue_size=256, threaded=False):
        batch_size = batch_size if batch_size is not None else self.batch_size

        # handle unit case
        if type(text) is str:
            text = [text]

        if threaded:
            # make workers
            results = []
            def loader():
                yield from batch_generator(iter(text), batch_size)
            def tokenizer(texts):
                return self.tokenize_batch(texts, maxlen=maxlen)
            def forwarder(data):
                doc_indices, input_ids, attention_mask = data
                embed = self.forward_batch(input_ids, attention_mask)
                results.append((doc_indices, embed))

            # embed chunks and average
            pipeline_threads(loader(), tokenizer, forwarder, maxsize=queue_size)
        else:
            # embed chunks and average
            results = [
                self.embed_batch(chunk) for chunk in batch_generator(iter(text), batch_size)
            ]

        # unpack offsets and embeds
        offsets, embeds = zip(*results)

        # document offsets come zero-indexed within batch
        nchunks = [o.max().item()+1 for o in offsets]
        indices = torch.cat([o+n for o, n in zip(offsets, cumsum(nchunks))])
        _, sizes = torch.unique(indices, return_counts=True)

        # aggregate to document level
        embed = torch.cat(embeds, dim=0)
        means = normalize(torch.stack([
            embed[i:j,:].mean(dim=0) for i, j in cumul_bounds(sizes)
        ], dim=0), dim=-1)

        # return normalized vectors
        return means

##
## meta seamless model
##

try:
    from seamless_communication.models.inference import Translator
    import onnxruntime as ort
except:
    print('Seamless not available.')

class SeamlessModel:
    def __init__(
            self, model_size='large', vocoder='vocoder_36langs', device='cuda', lang='eng',
            onnx=False, save_dir='onnx/seamless'
        ):
        # options
        self.device = device
        self.lang = lang

        # create seamless model
        self.translator = Translator(
            f'seamlessM4T_{model_size}', vocoder_name_or_card=vocoder, device=torch.device(device)
        )

        # store model params
        self.dims = self.translator.model.text_encoder_frontend.model_dim
        self.maxlen = self.translator.model.text_encoder_frontend.pos_encoder.max_seq_len

        if onnx:
            seqs = torch.zeros(1, 1, dtype=torch.int64, device=device)
            seq_lens = torch.zeros(1, dtype=torch.int64, device=device)
            embedf = torch.zeros(1, 1, self.dims, dtype=torch.half, device=device)
            maskf = torch.zeros(1, 1, dtype=torch.half, device=device)

            enc_front_path = os.path.join(save_dir, 'enc_front.onnx')
            enc_path = os.path.join(save_dir, 'enc.onnx')

            torch.onnx.export(
                self.translator.model.text_encoder_frontend, (seqs, seq_lens), enc_front_path,
                input_names=['seqs', 'seq_lens'], output_names=['embedf', 'maskf'],
                dynamic_axes={'seqs': [0, 1], 'seq_lens': [0]}, opset_version=16
            )
            torch.onnx.export(
                self.translator.model.text_encoder, (embedf, maskf), enc_path,
                input_names=['embedf', 'maskf'], output_names=['embed', 'mask'],
                dynamic_axes={'embedf': [0, 1], 'maskf': [0]}, opset_version=16
            )

            self.onnx = True
            self.enc_front_onnx = ort.InferenceSession(
                enc_front_path, providers=['CUDAExecutionProvider']
            )
            self.enc_onnx = ort.InferenceSession(
                enc_path, providers=['CUDAExecutionProvider']
            )
        else:
            self.onnx = False

    def forward(self, seqs, seq_lens):
        if self.enc_front_onnx is None:
            embedf, maskf = self.translator.model.text_encoder_frontend(seqs, seq_lens)
            embed, mask = self.translator.model.text_encoder(embedf, maskf)
            return embed, mask
        else:
            embedf, maskf = self.enc_front_onnx.run(
                None, {'seqs': seqs.cpu().numpy(), 'seq_lens': seq_lens.cpu().numpy()}
            )
            embed, mask = self.enc_onnx.run(
                None, {'embedf': embedf, 'maskf': maskf}
            )
            return torch.from_numpy(embed).to(self.device), torch.from_numpy(mask).to(self.device)

    def encode(self, text, lang=None):
        # use default language
        lang = self.lang if lang is None else lang

        # handle unit case
        text = [text] if type(text) is str else text

        # make encoder for lang
        token_encoder = self.translator.text_tokenizer.create_encoder(
            task='translation', lang=lang, mode='source', device=torch.device(self.device)
        )

        # encode into input ids
        toks = [token_encoder(s) for s in text]

        # compute sequence lengths and clip to maxlen
        src = self.translator.collate(toks)
        seqs = src['seqs'][:,:self.maxlen]
        seq_lens = src['seq_lens'].clip(max=self.maxlen)

        # return pair
        return seqs, seq_lens

    def embed_batch(self, text, lang=None):
        # encode into input ids
        src, src_len = self.encode(text, lang)

        # run through model
        with torch.no_grad():
            outf, maskf = self.translator.model.text_encoder_frontend(src, src_len)
            out, mask = self.translator.model.text_encoder(outf, maskf)

        # make mask if none
        if mask is None:
            mask = torch.zeros(
                out.size(0), out.size(1), device=out.device, dtype=out.dtype
            )

        # aggregate over sequence length
        weight = mask.exp().unsqueeze(-1)
        embed = (out*weight).sum(1)/weight.sum(1)

        # return normalized embedding
        return normalize(embed, dim=-1)

    def embed(self, text, lang=None, batch_size=32):
        return torch.cat([
            self.embed_batch(batch, lang=lang) for batch in batch_generator(iter(text), batch_size)
        ], dim=0)

    def translate(self, text, src_lang, tgt_lang):
        trans, _, _ = self.translator.predict(text, 't2tt', tgt_lang, src_lang)
        return str(trans)
