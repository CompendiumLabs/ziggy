# general utilities

from typing import Any
from itertools import islice
import toml
import operator
import asyncio

##
## pure play python
##

# allow list or single item
def allow_list(func):
    def wrapper(self, keys):
        many = type(keys) is list
        keys = keys if many else [keys]
        rets = func(self, keys)
        if rets is not None:
            return rets if many else rets[0]
    return wrapper

# generate (resolved) batches from generator
def batch_generator(gen, batch_size):
    while (batch := list(islice(gen, batch_size))) != []:
        yield batch

##
## collections
##

class IndexDict(dict):
    @classmethod
    def load(cls, data):
        return cls(data)

    def save(self):
        return dict(self)

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
## torch utils
##

def resize_alloc(a, size):
    a.resize_(size, *a.shape[1:])

def l2_mean(a, dim=0):
    return a.square().mean(dim=dim).sqrt()

##
## async rig
##

# enqueue batch generator
async def loader_func(queue, stream):
    for batch in stream:
        await queue.put(batch)

# process queue items
async def worker_func(queue_prev, queue_next, func):
    while True:
        data = await queue_prev.get()
        await queue_next.put(func(data))
        queue_prev.task_done()

async def counter_func(queue, total):
    while True:
        data = await queue.get()
        total[0] += 1
        queue.task_done()

# asnychronous indexer
async def pipeline_func(stream, *funcs, maxsize=0):
    # handle multiple workers
    funcs = [(f, 1) if type(f) is not tuple else f for f in funcs]

    # create queue
    queue_load = asyncio.Queue(maxsize=maxsize)
    queue_work = [asyncio.Queue(maxsize=maxsize) for _ in funcs]

    # initialize loader
    loader = asyncio.create_task(loader_func(queue_load, stream))
    queue_prev = queue_load

    # hook up queue
    workers = []
    for (func, num), queue_next in zip(funcs, queue_work):
        tasks = [
            asyncio.create_task(worker_func(queue_prev, queue_next, func)) for _ in range(num)
        ]
        workers += tasks
        queue_prev = queue_next

    # initialize counter
    total = [0]
    counter = asyncio.create_task(counter_func(queue_next, total))

    # wait for load to finish
    await loader

    # finish remaining tasks
    await asyncio.gather(*[q.join() for q in queue_work])

    # cancel workers
    for worker in workers:
        worker.cancel()
    counter.cancel()

    # get results
    return total[0]

def pipeline_async(stream, *funcs, **kwargs):
    return asyncio.run(pipeline_func(stream, *funcs, **kwargs))
