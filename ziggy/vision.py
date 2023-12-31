# vision — image+text to text

# example usage (query='What is shown in this image?')
# tokenizer, model, image_processor, context_len = load_llava_model('liuhaotian/llava-v1.5-13b')
# output = query_llava_images(model, tokenizer, image_processor, image_path, query)

import re
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast

##
## LayoutLMv3
##

class Line:
    def __init__(self):
        self.boxes = []
        self.words = []
        self.hull = None

    def __len__(self):
        return len(self.boxes)

    def add(self, box, word):
        self.boxes.append(box)
        self.words.append(word)

        if len(self.boxes) == 1:
            self.hull = box
        else:
            hl, ht, hr, hb = self.hull
            bl, bt, br, bb = box
            self.hull = [
                min(hl, bl), min(ht, bt),
                max(hr, br), max(hb, bb),
            ]

    # is this horizontally consistent with the line
    def aligned(self, box, tol=0):
        hl, ht, hr, hb = self.hull
        bl, bt, br, bb = box
        return (
            bl < hr + tol and
            br > hl - tol and
            bt > ht - tol and
            bb < hb + tol
        )

    # is this close to the line in any direction
    def close(self, box, tol=0):
        hl, ht, hr, hb = self.hull
        bl, bt, br, bb = box
        return (
            bl < hr + tol and
            br > hl - tol and
            bt < hb + tol and
            bb > ht - tol
        )

    def text(self):
        return ' '.join(self.words)

class Block:
    def __init__(self):
        self.lines = []

    def __len__(self):
        return len(self.lines)

    def add(self, box, word, tol=0):
        # handle empty case
        if len(self.lines) == 0:
            line = Line()
            line.add(box, word)
            self.lines.append(line)
            return True

        # look for aligned lines
        for line in reversed(self.lines):
            if line.aligned(box, tol=tol):
                line.add(box, word)
                return True

        # look for close lines
        for line in reversed(self.lines):
            if line.close(box, tol=tol):
                line1 = Line()
                line1.add(box, word)
                self.lines.append(line1)
                return True

        # no match
        return False

    def hull(self):
        hl, ht, hr, hb = zip(*[l.hull for l in self.lines])
        return [min(hl), min(ht), max(hr), max(hb)]

    def text(self):
        return '\n'.join([l.text() for l in self.lines])

def glom_boxes(boxes, words, tol=0.75):
    # init state
    blocks = []

    # get inherent scale
    tboxes = torch.tensor(boxes)
    hscale = (tboxes[:, 3] - tboxes[:, 1]).mean()
    htol = hscale*tol

    for box, wrd in zip(boxes, words):
        # handle first iteration
        if len(blocks) == 0:
            block = Block()
            block.add(box, wrd)
            blocks.append(block)
            continue

        # append or create as needed
        for block in reversed(blocks):
            if block.add(box, wrd, tol=htol):
                break
        else:
            block = Block()
            block.add(box, wrd)
            blocks.append(block)

    # return blocks
    return blocks

class LayoutModel:
    def __init__(self, proc_id='microsoft/layoutlmv3-base'):
        self.proc_id = proc_id
        self.proc = LayoutLMv3ImageProcessor.from_pretrained(proc_id)

    def process_image(self, image, markup=False):
        # handle path case
        if type(image) is str:
            image = read_image(image, mode=ImageReadMode.RGB)

        # run model on input image
        with torch.inference_mode():
            result = self.proc(image)

        # get boxes and words
        boxes, = result.boxes
        words, = result.words
        boxes = torch.tensor(boxes)

        # un-normalize boxes
        c, h, w = image.shape
        boxes1 = (boxes/1000)*torch.tensor([[w, h, w, h]])

        if markup:
            return draw_bounding_boxes(image, boxes1)
        else:
            return boxes1, words

##
## Nougat
##

##
## Llava
##

class LlavaModel:
    def __init__(self, model_id='liuhaotian/llava-v1.5-13b', device='cuda'):
        self.device = device
        self.model_id = model_id
        self.model_name = get_model_name_from_path(model_id)
        self.tokenizer, self.model, self.processor, self.context = load_pretrained_model(
            model_path=model_id, model_base=None, model_name=self.model_name
        )
        self.model = self.model.to(device=device)

    def query_llava_images(
        self, images, query, conv_mode='llava_v1', temp=1.0, top_p=None, num_beams=1,
        max_new_tokens=512
    ):
        # ensure list
        if type(images) is not list:
            images = [images]

        # load from path if needed
        images = [
            Image.open(i) if type(i) is str else i for i in images
        ]

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in query:
            if self.model.config.mm_use_im_start_end:
                query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        else:
            if self.model.config.mm_use_im_start_end:
                query = image_token_se + '\n' + query
            else:
                query = DEFAULT_IMAGE_TOKEN + '\n' + query

        # make input tokens
        conv = conv_templates[conv_mode].copy()
        conv.append_message('USER', query)
        conv.append_message('ASSISTANT', None)
        prompt = conv.get_prompt()

        # convert image to on device tensor
        images_tensor = process_images(
            images, self.processor, self.model.config
        ).to(self.device, dtype=torch.float16)

        # make input ids
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .to(self.device)
        )

        # set up stopping criterion
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

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

        # validate output
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )

        # decode output
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()

        # trim outputs
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        # return outputs
        return outputs
