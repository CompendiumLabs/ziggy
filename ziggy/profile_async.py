import time
from itertools import chain
from threading import Thread, Event
from queue import Queue, Empty

def double(x):
    time.sleep(0.1)
    return 2*x

# process queue items (None terminates)
def worker(func, queue_prev, queue_next, kill, poll):
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
        [Thread(target=worker, args=(f, q0, q1, kill, poll)) for _ in range(n)]
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

if __name__ == '__main__':
    # create data
    load = range(10)

    # store data
    data = []
    def store(x):
        data.append(x)

    # run pipeline
    start = time.time()
    n = pipeline_threads(load, (double, 2), (double, 2), store, maxsize=10)
    end = time.time()

    # print results
    print('Processed {} items in {} seconds'.format(n, end - start))
    print(data)
