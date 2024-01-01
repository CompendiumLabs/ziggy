# ingestion tools (mostly PDF)

import os
import re
import json
import torch
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path
from functools import partial
from itertools import repeat

from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible

import pytesseract
import fitz as pymupdf
from PIL import Image

from .utils import groupby_key

##
## Boxes
##

class Hull:
    def __init__(self, hull=None):
        self.hull = hull

    def __len__(self):
        return 4

    def __iter__(self):
        return iter(self.hull)

    def __call__(self):
        return self.hull

    def __repr__(self):
        if self.hull is None:
            return '[Empty]'
        else:
            hl, ht, hr, hb = self.hull
            return f'[H {hl:.0f} → {hr:.0f}, V {ht:.0f} → {hb:.0f}]'

    def add(self, box):
        if self.hull is None:
            self.hull = box
        else:
            hl, ht, hr, hb = self.hull
            bl, bt, br, bb = box
            self.hull = [
                min(hl, bl), min(ht, bt),
                max(hr, br), max(hb, bb),
            ]

    def close(self, box, tol=0):
        hl, ht, hr, hb = self.hull
        bl, bt, br, bb = box
        return (
            bl < hr + tol and
            br > hl - tol and
            bt < hb + tol and
            bb > ht - tol
        )

class Container:
    def __init__(self, sep):
        self.sep = sep
        self.items = []

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __repr__(self):
        return self.sep.join(str(i) for i in self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, item):
        self.items.append(item)

class Span:
    def __init__(self, box, word):
        self.box = box
        self.word = word

    def __repr__(self):
        return self.word

    @classmethod
    def load(cls, data):
        box, word = data['box'], data['word']
        return cls(box, word)

    def save(self):
        return {
            'box': self.box,
            'word': self.word,
        }

class Line(Container):
    def __init__(self):
        super().__init__(' ')
        self.hull = Hull()

    @classmethod
    def load(cls, data):
        self = cls()
        self.items = [Span.load(s) for s in data['spans']]
        self.hull = Hull(data['hull'])
        return self

    def save(self):
        return {
            'spans': [s.save() for s in self],
            'hull': self.hull(),
        }

    def add(self, box, word, tol=0):
        if len(self) == 0 or self.aligned(box, tol=tol):
            span = Span(box, word)
            self.append(span)
            self.hull.add(box)
            return True
        else:
            return False

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
        return self.hull.close(box, tol=tol)

class Block(Container):
    def __init__(self):
        super().__init__('\n')
        self.hull = Hull()

    @classmethod
    def load(cls, data):
        self = cls()
        self.items = [Line.load(l) for l in data['lines']]
        self.hull = Hull(data['hull'])
        return self

    def save(self):
        return {
            'lines': [l.save() for l in self],
            'hull': self.hull(),
        }

    def add(self, box, word, tol=0):
        # handle empty case
        if len(self) == 0:
            line = Line()
            line.add(box, word)
            self.append(line)
            self.hull.add(box)
            return True

        # reject if not close to outer hull
        if not self.hull.close(box, tol=tol):
            return False

        # add to existing line if aligned
        for line in reversed(self):
            if line.add(box, word, tol=tol):
                self.hull.add(box)
                return True

        # create new line if close
        for line in reversed(self):
            if line.close(box, tol=tol):
                line1 = Line()
                line1.add(box, word)
                self.append(line1)
                self.hull.add(box)
                return True

        # no match
        return False

class Figure:
    def __init__(self, box, cap, img):
        self.box = Hull(box)
        self.cap = cap
        self.img = img

    def __repr__(self):
        return f'{self.cap} {self.box}'

    @classmethod
    def load(cls, data):
        box, cap = Hull(data['box']), data['cap']
        img = Image.frombytes('RGB', data['size'], data['img'])
        return cls(box, cap, img)

    def save(self):
        return {
            'box': self.box(),
            'cap': self.cap,
            'img': self.img.tobytes(),
            'size': self.img.size,
        }

class Page(Container):
    def __init__(self):
        super().__init__('\n\n')
        self.figures = []

    @classmethod
    def load(cls, data):
        self = cls()
        self.items = [Block.load(b) for b in data['blocks']]
        self.figures = [Figure.load(f) for f in data['figures']]
        return self

    def save(self):
        return {
            'blocks': [b.save() for b in self],
            'figures': [f.save() for f in self.figures],
        }

    def add(self, box, word, tol=0):
        # handle empty case
        if len(self) == 0:
            block = Block()
            block.add(box, word)
            self.append(block)
            return

        # append or create as needed
        for block in reversed(self):
            if block.add(box, word, tol=tol):
                break
        else:
            block = Block()
            block.add(box, word)
            self.append(block)

    def add_figure(self, box, cap, img):
        fig = Figure(box, cap, img)
        self.figures.append(fig)

class Document(Container):
    def __init__(self):
        super().__init__('\n--------------------------------\n')

    @classmethod
    def load(cls, path):
        data = torch.load(path) if type(path) is str else path
        self = cls()
        self.items = [Page.load(p) for p in data['pages']]
        return self

    def save(self, path=None):
        data = {
            'pages': [p.save() for p in self],
        }
        if path is None:
            return data
        else:
            torch.save(data, path)

    def add(self, page):
        self.append(page)

def glom_boxes(boxes, words, tol=1):
    # get scale
    tboxes = torch.tensor(boxes).float()
    hscale = (tboxes[:, 3] - tboxes[:, 1]).mean()
    htol = hscale*tol

    # feed in boxen
    page = Page()
    for box, word in zip(boxes, words):
        page.add(box, word, tol=htol)

    # return page
    return page

##
## Layout
##

class Tesseract:
    def __init__(self, config=''):
        self.config = config

    def process_image(self, image, markup=False):
        # handle path case
        if type(image) is str:
            image = Image.open(image).convert(mode='RGB')

        # run tesseract engine
        data = pytesseract.image_to_data(image, output_type='dict', config=self.config)
        words = data['text']
        boxes = list(zip(data['left'], data['top'], data['width'], data['height']))

        # get non-empty text boxes
        istxt = [w.strip() != '' for w in words]
        words = [w for t, w in zip(istxt, words) if t]
        boxes = [b for t, b in zip(istxt, boxes) if t]

        # convert boxes to hull format
        boxes = [(x, y, x+w, y+h) for x, y, w, h in boxes]

        # return results
        if markup:
            image = pil_to_tensor(image)
            boxes = torch.tensor(boxes)
            return draw_bounding_boxes(image, boxes)
        else:
            return boxes, words

##
## pdffigures2
##

def detect_figures(pdf_path, timeout=30):
    # get env config
    PDFFIGURES_JARPATH = os.environ.get('PDFFIGURES_JARPATH', None)
    PDFFIGURES_TEMPDIR = os.environ.get('PDFFIGURES_TEMPDIR', '/tmp/pdffigures2')

    # ensure jar path
    if PDFFIGURES_JARPATH is None:
        print('PDFFIGURES_JARPATH not set')
        return

    # ensure output dirs
    output_dir = Path(PDFFIGURES_TEMPDIR) / 'output'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # make the call
    process = subprocess.Popen(
        f'java -jar {PDFFIGURES_JARPATH} -m {output_dir}/ -d {output_dir}/ -s {output_dir}/stats.json -q {pdf_path}',
        shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )

    # run with timeout
    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f'pdffigures2 timed out on {pdf_path} ({timeout} seconds)')
        process.terminate()
        return False

    # get stats path
    fname = os.path.basename(pdf_path)
    base, _ = os.path.splitext(fname)
    stats_path = output_dir / f'{base}.json'

    # read back results
    with open(stats_path) as fid:
        stats = json.load(fid)

    # load in figures
    figures = []
    for fig in stats:
        page = fig['page']
        cap = fig['caption']
        bnd = fig['regionBoundary']
        box = [bnd['x1'], bnd['y1'], bnd['x2'], bnd['y2']]
        img = Image.open(fig['renderURL']).convert(mode='RGB')
        figures.append((page, box, cap, img))

    # return results
    return figures

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

##
## Pipeline
##

class Ingest:
    def __init__(self):
        self.tess = Tesseract()

    # this get the layout for text and figures
    def segment_pdf(self, pdf_path):
        # detect figures first
        figures = detect_figures(pdf_path)
        figs_map = groupby_key(figures, 0)

        # open pdf
        pdf =  pymupdf.open(pdf_path)

        # loop through pages
        doc = Document()
        for i, p in enumerate(pdf):
            # get relevant figs
            figs = figs_map[i]

            # get page image
            pix = p.get_pixmap(dpi=150)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

            # get pages layout
            boxes, words = self.tess.process_image(img)
            page = glom_boxes(boxes, words)

            # add figures and append
            for _, box, cap, img in figs:
                page.add_figure(box, cap, img)
            doc.add(page)

        # return parsed document
        return doc
