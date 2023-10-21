# general utilities

from math import ceil, log2
from itertools import chain, islice, accumulate, groupby
from operator import itemgetter
from threading import Thread, Event
from queue import Queue, Empty
import toml

##
## pure play python
##

# printer for streaming
def sprint(s):
    print(s, end='', flush=True)

# print iterator while streaming
def tee(iterable):
    for item in iterable:
        sprint(item)
        yield item

# mostly for reallocations
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
def groupby_dict(vals, grps):
    getter = itemgetter(1)
    tups = sorted(zip(vals, grps), key=getter)
    return {
        k: [i for i, _ in v] for k, v in groupby(tups, key=getter)
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
        return sorted(super().items(), key=itemgetter(0))

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
## thread rig
##

# process queue items (None terminates)
def worker_thread(func, queue_prev, queue_next, kill, poll):
    while True:
        try:
            data = queue_prev.get(timeout=poll)
            queue_next.put(func(data))
            queue_prev.task_done()
        except Empty:
            pass
        if kill.is_set():
            break

def pipeline_threads(load, *funcs, maxsize=0, poll=0.01):
    funcs = [(f, 1) if type(f) is not tuple else f for f in funcs]

    # create queues (last should be unbounded)
    queue_load = Queue(maxsize=maxsize)
    queue_work = [Queue(maxsize=maxsize) for _ in funcs[:-1]] + [Queue()]
    queues = [queue_load] + queue_work
    kill = Event()

    # create num threads for each function
    threads = [
        [
            Thread(target=worker_thread, args=(f, q0, q1, kill, poll)) for _ in range(n)
        ]
        for (f, n), q0, q1 in zip(funcs, queues[:-1], queues[1:])
    ]

    # start all threads
    for t in chain(*threads):
        t.start()

    # handle keyboard interrupt gracefully
    try:
        # put data in load queue
        for i in load:
            queue_load.put(i)

        # wait for all data to be processed
        for q in queues[:-1]:
            q.join()
    except KeyboardInterrupt:
        print('Terminating threads...')
    finally:
        # stop all threads
        kill.set()

        # wait for all threads
        for t in chain(*threads):
            t.join()

        # print number processed
        return queue_work[-1].qsize()
