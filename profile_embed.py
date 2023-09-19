# load profiler

import llm
import quant
import database

# options
dims = 384
path = 'testing/wiki/enwiki-testing.jsonl'
qpsec = qspec = quant.QuantSpec.from_width(quant.QuantType.qint4, 0.0, 4.0)

# run that baby
emb = llm.HuggingfaceEmbeddingONNX()
db = database.DocumentDatabase.from_jsonl(
    path, embed=emb, dims=dims, qspec=qspec,
    delim='\n', minlen=100, maxrows=1024
)
