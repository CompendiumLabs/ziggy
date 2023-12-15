# streaming llm server

import os
import json
from itertools import chain, accumulate
from time import sleep

from .utils import cumsum, cumul_indices, split_list

# z score normalization
def uniform_scale(vals):
    min_val, max_val = vals.min(), vals.max()
    return (vals - min_val) / (max_val - min_val)

# buffer stream so we can yield batches
def buffer_stream(stream, min_size):
    buf = []
    for item in stream:
        buf.append(item)
        if len(buf) >= min_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf

# generate response from llm
def gen_response(model, prompt, buffer_size, **kwargs):
    stream = model.generate(prompt, **kwargs)
    buffer = buffer_stream(stream, buffer_size)
    try:
        for batch in buffer:
            text = ''.join(batch)
            print(text, end='', flush=True)
            yield text.encode('utf-8')
    except Exception as e:
        yield f'ERROR: {e}'.encode('utf-8')

# run server
def serve(model, host='127.0.0.1', port=8000, buffer_size=1, **kwargs):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # query templates
    class Query(BaseModel):
        prompt: str
    class Simil(BaseModel):
        prompt: str

    # create api
    app = FastAPI()

    # add in CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:5173'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get('/chunks')
    def chunks():
        return list(model.data.chunks.items())

    @app.post('/simil')
    def simil(request: Simil):
        prompt = request.prompt
        vecs = model.embed.embed(prompt)
        sims = model.data.cindex.simil(vecs)
        sims = uniform_scale(sims)
        sizes = [len(v) for v in model.data.chunks.values()]
        dsims = split_list(sims.tolist(), sizes)
        return dsims

    @app.post('/query')
    def run_query(request: Query):
        prompt = request.prompt
        print(f'\n\nQUERY: {prompt}\nRESPONSE: ', sep='', flush=True)
        return StreamingResponse(
            gen_response(model, prompt, buffer_size, **kwargs), media_type='text/event-stream'
        )

    uvicorn.run(
        app, host=host, port=port, http='h11', log_level='debug'
    )
