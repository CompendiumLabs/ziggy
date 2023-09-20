import time
from itertools import chain
from queue import Queue
from threading import Thread

def double(x):
    time.sleep(0.1)
    return 2*x

# process queue items (None terminates)
def worker(func, queue_prev, queue_next):
    for data in iter(queue_prev.get, None):
        queue_next.put(func(data))
        print(data)
        queue_prev.task_done()

def pipeline_threads(load, *funcs, maxsize=0):
    funcs = [(f, 1) if type(f) is not tuple else f for f in funcs]

    # create queues (last should be unbounded)
    queue_load = Queue(maxsize=maxsize)
    queue_work = [Queue(maxsize=maxsize) for _ in funcs[:-1]] + [Queue()]
    queues = [queue_load] + queue_work

    # create num threads for each function
    threads = [
        [Thread(target=worker, args=(f, q0, q1)) for _ in range(n)]
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
    print(queue_work[-1].qsize())

if __name__ == '__main__':
    pipeline_threads(range(20), (double, 2), (double, 2), maxsize=10)
