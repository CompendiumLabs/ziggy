# ingestion tools (mostly PDF)

import os
import re
import json
import torch
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path
from statistics import mean
from functools import partial
from itertools import repeat

import pytesseract
import fitz as pymupdf
from PIL import Image

from .utils import groupby_key

# this is fixed
PDFFIGURES_DPI = 72

##
## Utils
##

def rescale_box(box, offset=0, scale=1):
    ow, oh = offset if type(offset) is tuple else (offset, offset)
    sw, sh = scale if type(scale) is tuple else (scale, scale)
    ob, sb = [ow, oh, ow, oh], [sw, sh, sw, sh]
    return [
        (o + s * b) for o, s, b in zip(ob, sb, box)
    ]

##
## Tesseract
##

def tesseract_ocr(image, markup=False, config=''):
    # handle path case
    if type(image) is str:
        image = Image.open(image).convert(mode='RGB')

    # run tesseract engine
    data = pytesseract.image_to_data(image, output_type='dict', config=config)
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

def detect_figures(pdf_path, dpi=72, timeout=30, verbose=False):
    # get env config
    PDFFIGURES_JARPATH = os.environ.get('PDFFIGURES_JARPATH', None)
    PDFFIGURES_TEMPDIR = os.environ.get('PDFFIGURES_TEMPDIR', '/tmp/pdffigures2')

    # ensure jar path
    if PDFFIGURES_JARPATH is None:
        raise Exception('PDFFIGURES_JARPATH not set')

    # ensure output dirs
    output_dir = Path(PDFFIGURES_TEMPDIR) / 'output'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # make the call
    args = dict(stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) if not verbose else {}
    process = subprocess.Popen(
        f'java -jar {PDFFIGURES_JARPATH} -d {output_dir}/ -s {output_dir}/stats.json -q {pdf_path}',
        shell=True, **args
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
        box = rescale_box(box, scale=dpi/PDFFIGURES_DPI)
        figures.append((page, box, cap))

    # return results
    return figures

##
## Boxes
##

class Hull:
    def __init__(self, hull=None):
        self.hull = list(hull) if hull is not None else None

    @classmethod
    def from_boxes(cls, boxes):
        self = cls()
        bl, bt, br, bb = zip(*boxes)
        hl, ht, hr, hb = min(bl), min(bt), max(br), max(bb)
        self.hull = [min(bl), min(bt), max(br), max(bb)]
        return self

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

    def overlaps(self, box):
        hl, ht, hr, hb = self.hull
        bl, bt, br, bb = box
        return (
            bl < hr and
            br > hl and
            bt < hb and
            bb > ht
        )

    def contains(self, box):
        hl, ht, hr, hb = self.hull
        bl, bt, br, bb = box
        return (
            bl > hl and
            br < hr and
            bt > ht and
            bb < hb
        )

class Container:
    def __init__(self, sep=None):
        self.sep = sep
        self.items = []

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __repr__(self):
        if self.sep is None:
            return f'{self.__class__.__name__} [{len(self.items)}]'
        else:
            return self.sep.join(str(i) for i in self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, item):
        self.items.append(item)

    def remove(self, item):
        self.items.remove(item)

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

    def data(self):
        return [(s.box, s.word) for s in self]

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

    def data(self):
        return [bw for l in self for bw in l.data()]

class Figure:
    def __init__(self, box, cap, img, txt):
        self.box = Hull(box)
        self.cap = cap
        self.img = img
        self.txt = txt

    def __repr__(self):
        return f'{self.cap} {self.box}'

    @classmethod
    def load(cls, data):
        box, cap, txt = Hull(data['box']), data['cap'], data['txt']
        img = Image.frombytes('RGB', data['size'], data['img'])
        return cls(box, cap, img, txt)

    def save(self):
        return {
            'box': self.box(),
            'cap': self.cap,
            'img': self.img.tobytes(),
            'txt': self.txt,
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

    def rem(self, boxes):
        boxes = boxes if type(boxes) is list else [boxes]
        for b in boxes:
            self.remove(b)

    def add_figure(self, box, cap, img, txt):
        fig = Figure(box, cap, img, txt)
        self.figures.append(fig)

    def filter(self, include=None, exclude=None, strict=True):
        # ensure Hull
        include = Hull(include) if include is not None else None
        exclude = Hull(exclude) if exclude is not None else None

        # inclusion criterion
        if strict:
            isin = lambda h, b: h.contains(b.hull)
        else:
            isin = lambda h, b: h.overlaps(b.hull)

        # get overlaps/intersections
        if include is not None:
            imask = [isin(include, b.hull) for b in self]
        else:
            imask = [True] * len(self)
        if exclude is not None:
            emask = [isin(exclude, b.hull) for b in self]
        else:
            emask = [False] * len(self)

        # return selected boxen
        return [b for b, i, e in zip(self, imask, emask) if (i and not e)]

    def data(self):
        return [bw for b in self for bw in b.data()]

class Document(Container):
    def __init__(self, name='Document'):
        super().__init__('\n--------------------------------\n')
        self.name = name

    @classmethod
    def load(cls, path):
        data = torch.load(path) if type(path) is str else path
        self = cls()
        self.name = data['name']
        self.items = [Page.load(p) for p in data['pages']]
        return self

    @classmethod
    def from_pdf(cls, path, name=None, tol=1, dpi=150, verbose=False):
        name = name if name is not None else os.path.basename(path)
        self = cls(name)

        # detect figures first
        figures = detect_figures(path, dpi=dpi, verbose=verbose)
        figs_map = groupby_key(figures, 0)

        # open pdf
        pdf =  pymupdf.open(path)

        # loop through pages
        for i, p in enumerate(pdf):
            # get relevant figs
            figs = figs_map[i]

            # get page image
            pix = p.get_pixmap(dpi=dpi)
            siz = pix.w, pix.h

            # get pages layout
            img = Image.frombytes('RGB', siz, pix.samples)
            boxes, words = tesseract_ocr(img)

            # get page scale
            bl, bt, br, bb = zip(*boxes)
            hscale = mean(bb) - mean(bt)
            htol = hscale*tol

            # feed in boxen
            page = Page()
            for box, word in zip(boxes, words):
                page.add(box, word, tol=htol)

            # add figures and append
            for _, box, cap in figs:
                fimg = img.crop(box)
                ftxt = page.filter(box)
                page.rem(ftxt)
                page.add_figure(box, cap, fimg, ftxt)

            # append page
            self.add(page)

        # return parsed document
        return self

    def save(self, path=None):
        data = {
            'name': self.name,
            'pages': [p.save() for p in self],
        }
        if path is None:
            return data
        else:
            torch.save(data, path)

    def add(self, page):
        self.append(page)

class Corpus(Container):
    def __init__(self, name='Corpus'):
        super().__init__()
        self.name = name

    @classmethod
    def load(cls, path):
        data = torch.load(path) if type(path) is str else path
        self = cls()
        self.name = data['name']
        self.items = [Document.load(d) for d in data['docs']]
        return self

    @classmethod
    def from_pdfs(cls, paths, **kwargs):
        self = cls(**kwargs)
        for path in paths:
            doc = Document.from_pdf(path)
            self.add(doc)
        return self

    def save(self, path=None):
        data = {
            'name': self.name,
            'docs': [d.save() for d in self],
        }
        if path is None:
            return data
        else:
            torch.save(data, path)

    def __repr__(self):
        names = ', '.join(d.name for d in self)
        return f'{self.name}: [{names}]'

    def add(self, doc):
        self.append(doc)

##
## Interpretation
##

# set up text query system
TEXT_QUERY_SYSTEM = 'Given the following text sample answer the question at the end concisely and to the best of your ability. Do not provide warnings about ethics, morality, illegality, safety, or possible incorrectness.'
TEXT_QUERY_USER = 'TEXT:\n\n{txt}\n\nQUESTION:\n\n{qst}\n\nANSWER:\n\n'
TEXT_QUERY_LIST = [
    'What is the purpose of this page?',
    'What are some keywords related to this page?',
    'What are some questions that might be asked about this page?',
]

# FIGURES
# what does this image show? / what is the purpose of this image?
# what are some keywords related to this image?
# how does the text [embedded text] relate to this image?
# what are some questions that might be asked about this image?

# META
# construct page network

class Interpreter:
    def __init__(self, llm=None, lvm=None, emb=None):
        self.llm = llm
        self.lvm = lvm
        self.emb = emb

    # get full text interpretation for embedding
    def interpret(self, corp, idx, **kwargs):
        # iterate over documents
        for doc in corp:
            print(doc)

            # loop over pages
            for i, page in enumerate(doc):
                # get full text
                page_text = [str(para) for para in page]

                # ask some questions
                page_queries = [TEXT_QUERY_USER.format(txt=page_text, qst=qst) for qst in TEXT_QUERY_LIST]
                page_gens = self.llm.parallel(page_queries, system=TEXT_QUERY_SYSTEM, **kwargs)

                # embed ressults
                vecs_text = self.emb.embed(page_text)
                vecs_queries = self.emb.embed(page_queries)

                # append output
                page_labs, page_vecs = zip(*[
                    ((doc.name, i, j, 'txt'), txt) for j, txt in enumerate(vecs_text)
                ] + [
                    ((doc.name, i, j, 'gen'), gen) for j, gen in enumerate(vecs_gens)
                ])
                idx.add({'text': page_text, 'gens': page_gens})
            output[doc.name] = page_info

        # return generated        
        return output

# retrieval
# get para/para/fig level embed distances
# smooth para sims out over and across pages with spatial and network info
# return top-k results
