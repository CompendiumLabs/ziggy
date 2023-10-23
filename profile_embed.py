# load profiler

import llm
import quant
import database

# options
model = 'TaylorAI/bge-micro-v2'
path = '/home/doug/data/wikidumps/parsed/enwiki-20230801-articles.jsonl'
qpsec = qspec = quant.QuantSpec.from_width(quant.QuantType.qint4, 0.0, 0.1)

# run that baby
emb = llm.HuggingfaceEmbedding(model_id=model, onnx=True, maxlen=256)
db = database.DocumentDatabase.from_jsonl(path, embed=emb, qspec=qspec, delim='\n', minlen=100, progress=False)
