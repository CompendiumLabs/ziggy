# ziggy

The documentation is pretty sparse right now, but most of the action is in `llm.py`, `index.py`, and `database.py`. Note that there are custom `torch` kernels for quantization, dequantization, and quantized matmul, so you'll need to compile and install those first for vector index capabilities.

##  Install

To build `torch` extensions:
```bash
python setup.py install --user
```

To build testing cpp get `libtorch` and run:
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/home/doug/mlai/ziggy/cpp/libtorch_cuda -DCMAKE_CUDA_HOST_COMPILER=/home/doug/programs/cuda-gcc/bin/c++
cmake --build . --config Release
```

## Usage

To load up a `huggingface` embedding model just run

```python
import llm
mod = llm.HuggingfaceEmbedding('TaylorAI/bge-micro-v2')
```

By default it does not optimization and uses the maximum token window length. If you want to say use ONNX optimization and set a 256 length token window

```python
mod = llm.HuggingfaceEmbedding('TaylorAI/bge-micro-v2', onnx=True, maxlen=256)
```

Then we can embed text with the `mod.embed` function with either a string or a list of strings. To load in big corpora, I first convert them to a single JSONL file to load in. To load the file `path` into an index

```python
db = database.DocumentDatabase.from_jsonl(path, embed=mod, delim='\n', minlen=100)
```

Here we set the within record delimiter to a single newline and discard any chunks with less than `100` characters. This is a pretty good approach for Wikipedia.
