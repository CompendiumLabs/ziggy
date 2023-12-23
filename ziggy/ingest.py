# ingestion tools (mostly PDF)

import re
from tqdm import tqdm
from functools import partial
from itertools import repeat
from torch.utils.data import DataLoader

from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible

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
        checkpoint = get_checkpoint(model_id)
        self.model = NougatModel.from_pretrained(checkpoint)
        self.model = move_to_device(self.model, bf16=bf16, cuda=cuda)
        self.model.eval()

    # convert pdf to markdown text
    def convert_pdf(self, path, batch_size=8, skipping=True, markdown=True):
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
