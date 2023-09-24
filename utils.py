# general utilities

from math import ceil, log2
from itertools import chain, islice, accumulate, groupby
from operator import itemgetter
from queue import Queue
from threading import Thread
import toml
import operator
import asyncio

##
## pure play python
##

# printer for streaming
def sprint(s):
    print(s, end='', flush=True)

def next_power_of_2(x):
    return pow(2, round(ceil(log2(x))))

# allow list or single item
def allow_list(func):
    def wrapper(self, keys):
        many = type(keys) is list
        keys = keys if many else [keys]
        rets = func(self, keys)
        if rets is not None:
            return rets if many else rets[0]
    return wrapper

# group tuples by `idx` element, preserving other orders
def groupby_dict(tups, idx=0):
    getter = itemgetter(idx)
    tups = sorted(tups, key=getter)
    return {
        i: [k for _, k in j] for i, j in groupby(tups, key=getter)
    }

# cumulative sum
def cumsum(lengths):
    return list(chain([0], accumulate(lengths)))

# cumsum generator
def cumul_bounds(seq):
    total = 0
    for item in seq:
        yield total, total+item
        total += item

# generate (resolved) batches from generator
def batch_generator(gen, batch_size):
    while (batch := list(islice(gen, batch_size))) != []:
        yield batch

# get batch indices
def batch_indices(length, batch_size):
    return [(i, min(i+batch_size, length)) for i in range(0, length, batch_size)]

# get cumulative indices
def cumul_indices(lengths):
    sums = cumsum(lengths)
    return [(i, j) for i, j in zip(sums[:-1], sums[1:])]

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

##
## async rig
##

# enqueue batch generator
async def loader_async(queue, stream):
    for batch in stream:
        await queue.put(batch)

# process queue items
async def worker_async(queue_prev, queue_next, func):
    while True:
        data = await queue_prev.get()
        await queue_next.put(func(data))
        queue_prev.task_done()

async def counter_async(queue, total):
    while True:
        data = await queue.get()
        total[0] += 1
        queue.task_done()

# asnychronous indexer
async def pipeline_async_func(stream, *funcs, maxsize=0):
    # handle multiple workers
    funcs = [(f, 1) if type(f) is not tuple else f for f in funcs]

    # create queue
    queue_load = asyncio.Queue(maxsize=maxsize)
    queue_work = [asyncio.Queue(maxsize=maxsize) for _ in funcs]

    # initialize loader
    loader = asyncio.create_task(loader_async(queue_load, stream))
    queue_prev = queue_load

    # hook up queue
    workers = []
    for (func, num), queue_next in zip(funcs, queue_work):
        tasks = [
            asyncio.create_task(worker_async(queue_prev, queue_next, func)) for _ in range(num)
        ]
        workers += tasks
        queue_prev = queue_next

    # initialize counter
    total = [0]
    counter = asyncio.create_task(counter_async(queue_next, total))

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
    return asyncio.run(pipeline_async_func(stream, *funcs, **kwargs))

# process queue items (None terminates)
def worker_thread(func, queue_prev, queue_next):
    for data in iter(queue_prev.get, None):
        queue_next.put(func(data))
        queue_prev.task_done()

def pipeline_threads(load, *funcs, maxsize=0):
    funcs = [(f, 1) if type(f) is not tuple else f for f in funcs]

    # create queues (last should be unbounded)
    queue_load = Queue(maxsize=maxsize)
    queue_work = [Queue(maxsize=maxsize) for _ in funcs[:-1]] + [Queue()]
    queues = [queue_load] + queue_work

    # create num threads for each function
    threads = [
        [Thread(target=worker_thread, args=(f, q0, q1)) for _ in range(n)]
        for (f, n), q0, q1 in zip(funcs, queues[:-1], queues[1:])
    ]

    # start all threads
    for t in chain(*threads):
        t.start()

    # put data in load queue
    for i in load:
        queue_load.put(i)

    # wait for all data to be processed
    for q in queues[:-1]:
        q.join()

    # stop all threads
    for (_, n), t, q in zip(funcs, threads, queues[:-1]):
        for _ in range(n):
            q.put(None)

    # wait for all threads
    for t in chain(*threads):
        t.join()

    # print number processed
    return queue_work[-1].qsize()
