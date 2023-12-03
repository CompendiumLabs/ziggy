# streaming llm server

import os
import json

from pydantic import BaseModel

# get this directory
module_dir = os.path.dirname(os.path.realpath(__file__))
static_dir = os.path.relpath(os.path.join(module_dir, 'static'))

# query template
class Query(BaseModel):
    prompt: str

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
def gen_response(model, prompt, buffer_size):
    stream = model.generate(prompt)
    buffer = buffer_stream(stream, buffer_size)
    for batch in buffer:
        text = ''.join(batch)
        print(text, end='', flush=True)
        yield (text+'\0').encode('utf-8')

# run server
def serve(model, host='127.0.0.1', port=8000, buffer_size=1):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, StreamingResponse

    app = FastAPI()
    app.mount('/static', StaticFiles(directory=static_dir), name='static')

    @app.get('/')
    def index():
        index_path = os.path.join(static_dir, 'index.html')
        return FileResponse(index_path)

    @app.post('/query')
    def run_query(query: Query):
        prompt = query.prompt
        print(f'\n\nQUERY: {prompt}\nRESPONSE: ', sep='', flush=True)
        return StreamingResponse(
            gen_response(model, prompt, buffer_size), media_type='text/event-stream'
        )

    uvicorn.run(
        app, host=host, port=port, http='h11', log_level='debug'
    )
