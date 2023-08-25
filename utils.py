# general utilities

from typing import Any
import toml
import operator
import asyncio

##
## decorators
##

def allow_list(func):
    def wrapper(self, keys):
        many = type(keys) is list
        keys = keys if many else [keys]
        rets = func(self, keys)
        if rets is not None:
            return rets if many else rets[0]
    return wrapper

##
## collections
##

class IndexDict(dict):
    @allow_list
    def add(self, keys):
        new = set(keys).difference(self)
        n0, n1 = len(self), len(new)
        ids = range(n0, n0 + n1)
        self.update(zip(new, ids))

    @allow_list
    def idx(self, keys):
        return [self[k] for k in keys]

class Bundle(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for d in args + (kwargs,):
            self.update(d)

    @classmethod
    def from_tree(cls, tree):
        if isinstance(tree, dict):
            return cls([(k, cls.from_tree(v)) for k, v in tree.items()])
        else:
            return tree

    @classmethod
    def from_toml(cls, path):
        return cls.from_tree(toml.load(path))

    def __repr__(self):
        return '\n'.join([f'{k} = {v}' for k, v in self.items()])

    def keys(self):
        return sorted(super().keys())

    def items(self):
        return sorted(super().items(), key=operator.itemgetter(0))

    def values(self):
        return [k for k, _ in self.items()]

    def __getattr__(self, key):
        return self[key]
    
    def __setattr__(self, key, value):
        self[key] = value

##
## async rig
##

# enqueue batch generator
async def loader_func(queue, stream):
    for batch in stream:
        await queue.put(batch)

# process queue items
async def worker_func(queue, func):
    while True:
        data = await queue.get()
        func(data)
        queue.task_done()

# asnychronous indexer
async def process_async(stream, func, maxsize=0):
    # create queue
    queue = asyncio.Queue(maxsize=maxsize)

    # hook up queue
    loader = asyncio.create_task(loader_func(queue, stream))
    worker = asyncio.create_task(worker_func(queue, func))

    # wait for load to finish
    await loader

    # finish remaining tasks
    await queue.join()

def process(stream, func, **kwargs):
    asyncio.run(process_async(stream, func, **kwargs))
