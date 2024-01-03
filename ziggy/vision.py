# vision — image+text to text

# example usage (query='What is shown in this image?')
# tokenizer, model, image_processor, context_len = load_llava_model('liuhaotian/llava-v1.5-13b')
# output = query_llava_images(model, tokenizer, image_processor, image_path, query)

import re
import torch
import base64
from PIL import Image
from io import BytesIO

from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from .utils import sprint
from .llm import get_gguf_meta_string

##
## LlavaCpp
##

# convert PIL Image to Data URL (data:image/png;base64,...)
def pil_to_dataurl(img):
    with BytesIO() as bf:
        img.save(bf, format='png')
        bdat = bf.getvalue()
    bstr = base64.b64encode(bdat).decode('utf-8')
    return f'data:image/png;base64,{bstr}'

class LlavaCppModel:
    def __init__(self, model_path, chat_path, context=2048, n_gpu_layers=100, verbose=False, prompt_type='llama', **kwargs):
        # store options
        self.context = context
        self.prompt_type = prompt_type

        # load patched chat handler
        self.chat = Llava15ChatHandler(clip_model_path=chat_path)

        # load llama model
        self.model = Llama(
            model_path, chat_handler=self.chat, n_ctx=context, logits_all=True,
            n_gpu_layers=n_gpu_layers, verbose=verbose, **kwargs
        )

        # get model info
        self.arch = get_gguf_meta_string(self.model, 'general.architecture')
        self.name = get_gguf_meta_string(self.model, 'general.name')

    def generate(self, query, image, system=None, maxgen=None, temp=1.0, top_k=0, **kwargs):
        # ensure PIL Image then data URL
        if type(image) is str:
            image = Image.open(image).convert('RGB')
        image_data = pil_to_dataurl(image)

        # get prompt generator
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': image_data}},
                {'type': 'text', 'text': query}
            ]},
        ]

        # construct stream object
        stream = self.model.create_chat_completion(
            messages=messages, max_tokens=maxgen, stream=True, temperature=temp, top_k=top_k, **kwargs
        )

        # return generated tokens
        for i, output in enumerate(stream):
            choice, *_ = output['choices']
            delta = choice['delta']
            if 'content' in delta:
                yield delta['content']

    def igenerate(self, query, image_path, **kwargs):
        for s in self.generate(query, image_path, **kwargs):
            sprint(s)

##
## Llava (original)
##

class LlavaModel:
    def __init__(self, model_id='liuhaotian/llava-v1.5-13b', device='cuda'):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        # get config options
        self.device = device
        self.model_id = model_id
        self.model_name = get_model_name_from_path(model_id)

        # load tokenizer and model
        self.tokenizer, self.model, self.processor, self.context = load_pretrained_model(
            model_path=model_id, model_base=None, model_name=self.model_name
        )
        self.model = self.model.to(device=device)

    def query_llava_images(
        self, images, query, conv_mode='llava_v1', temp=1.0, top_p=None, num_beams=1,
        max_new_tokens=512
    ):
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        # ensure list
        if type(images) is not list:
            images = [images]

        # load from path if needed
        images = [Image.open(i) if type(i) is str else i for i in images]
        images_tensor = process_images(
            images, self.processor, self.model.config
        ).to(self.device, dtype=torch.float16)

        # make input tokens
        query = DEFAULT_IMAGE_TOKEN + '\n' + query
        conv = conv_templates[conv_mode].copy()
        conv.append_message('USER', query)
        conv.append_message('ASSISTANT', None)
        prompt = conv.get_prompt()

        # make input ids
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .to(self.device)
        )

        # set up stopping criterion
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if temp > 0 else False,
                temperature=temp,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # decode output
        _, input_len = input_ids
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()

        # trim outputs
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]

        # return outputs
        return outputs.strip()

##
## Nougat
##

def gen_page_output(prediction, repeat, page_num, skipping=True):
    # check if model output is faulty
    if prediction.strip() == '[MISSING_PAGE_POST]':
        # uncaught repetitions -- most likely empty page
        return f'\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n'
    elif skipping and repeat is not None:
        if repeat > 0:
            # If we end up here, it means the output is most likely not complete and was truncated.
            print(f'Skipping page {page_num} due to repetitions.')
            return f'\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n'
        else:
            # If we end up here, it means the document page is too different from the training domain.
            # This can happen e.g. for cover pages.
            return f'\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n'
    else:
        return prediction

class NougatConvert:
    def __init__(self, model_id='0.1.0-small', bf16=True, cuda=True):
        from nougat import NougatModel
        from nougat.utils.checkpoint import get_checkpoint
        from nougat.utils.device import move_to_device

        checkpoint = get_checkpoint(model_id)
        self.model = NougatModel.from_pretrained(checkpoint)
        self.model = move_to_device(self.model, bf16=bf16, cuda=cuda)
        self.model.eval()

    # convert pdf to markdown text
    def convert_pdf(self, path, batch_size=8, skipping=True, markdown=True):
        from nougat.utils.dataset import LazyDataset
        from nougat.postprocessing import markdown_compatible

        # load pdf
        prepare = partial(self.model.encoder.prepare_input, random_padding=False)
        dataset = LazyDataset(path, prepare)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # document state
        pages = []
        page_num = 0

        # iterate over batches
        for i, (sample, is_last_page) in enumerate(tqdm(loader)):
            # run inference model
            results = self.model.inference(image_tensors=sample, early_stopping=skipping)
            predictions = results['predictions']
            repetition = results.get('repeat', repeat(None))

            # iterate over batch
            for pred, reps in zip(predictions, repetition):
                page_num += 1
                output = gen_page_output(pred, None, page_num, skipping=skipping)
                pages.append(markdown_compatible(output) if markdown else output)

        # return full document
        document = ''.join(pages).strip()
        document = re.sub(r'\n{3,}', '\n\n', document).strip()
        return document
