# ziggy

The documentation is pretty sparse right now, but most of the action is in `llm.py`, `index.py`, and `database.py`. Note that there are custom `torch` kernels for quantization, dequantization, and quantized matmul, so you'll need to compile and install those first for vector index capabilities.

##  Install

For simple usage, no additional steps are needed beyond cloning the repository. If you want quantization support (optional), you need to build the custom `torch` extensions, in the `extension` directory:
```bash
cd extension
python setup.py install --user
```
For CUDA builds, the environment variable `CUDA_HOME` must be set to your CUDA directory. To build in CPU-only mode, set the environment variable `ZIGGY_CPU_ONLY=1`.

*If you need to use an older C++ compiler with `nvcc`, set `CUDAHOSTCXX` to the directory containing the `c++` binary.*

To build standalone testing binaries (very optional), get `libtorch` for CUDA, assign its absolute path to `LIBTORCH_PATH`, and run:
```bash
mkdir build
cmake -B build -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH
cmake --build build --config Release
```

## Usage

To load up a `huggingface` embedding model just run

```python
model_id = 'TaylorAI/bge-micro-v2'
mod = ziggy.HuggingfaceEmbedding(model_id)
```

By default it does not optimization and uses the maximum token window length. If you want to say use ONNX optimization and set a 256 length token window

```python
mod = ziggy.HuggingfaceEmbedding(model_id, onnx=True, max_len=256)
```

You can also use `llama.cpp` via `llama-cpp-python` with
```python
mod = ziggy.LlamaCppPython(path_to_gguf)
```

Then we can embed text with the `mod.embed` function with either a string or a list of strings. To load in big corpora, I first convert them to a single JSONL file to load in. To load the file `path` into an index

```python
db = ziggy.DocumentDatabase.from_jsonl(path, embed=mod, delim='\n', min_len=100)
```

Here we set the within record delimiter to a single newline and discard any chunks with less than `100` characters. This is a pretty good approach for Wikipedia. If you want to quantize the stored vectors, you can do something like the following

```python
qspec = ziggy.QuantSpec.from_width(ziggy.QuantType.qint4, 0.0, 0.1)
db = ziggy.DocumentDatabase.from_jsonl(path, embed=mod, qspec=qspec, delim='\n', min_len=100)
```

Now you can do top-k semantic search on the database with something like

```python
db.search('Anarchism', 5)
```

By default the indexing operation will have generated average embeddings at the document level and these are used as a first pass when searching.

# Text-Embeddings-Inference

Command to start up `Qwen3-Embedding-0.6B` locally.

```bash
text-embeddings-router --model-id "Qwen/Qwen3-Embedding-0.6B" --hostname 127.0.0.1 --port 8080 --max-client-batch-size 32768 --max-batch-tokens 32768 --payload-limit 16777216
```

Send a test message to the server

```bash
xh http://127.0.0.1:8080/v1/embeddings input="hello world"
```
