"""Multi-worker threaded prefetcher — keeps GPU fed while CPU decodes."""

import queue
import threading


_SENTINEL = object()


class PrefetchIterator:
    """Wraps a generator factory with multiple prefetch workers.

    Each worker independently calls ``factory()`` and pushes batches
    to a shared queue.  Designed for infinite training generators.

    Usage:
        def make_gen():
            return load_mkv_batch(paths, bs, shuffle=True)

        for batch in PrefetchIterator(make_gen, n_workers=3, queue_size=6):
            ...
    """

    def __init__(self, factory, n_workers=3, queue_size=6):
        self._queue = queue.Queue(maxsize=queue_size)
        self._exceptions = queue.Queue(maxsize=n_workers)
        self._n_workers = n_workers
        self._sentinel_count = 0

        self._threads = []
        for _ in range(n_workers):
            t = threading.Thread(target=self._worker, args=(factory,), daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self, factory):
        try:
            gen = factory()
            for item in gen:
                self._queue.put(item)
        except Exception as e:
            self._exceptions.put(e)
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self._check_exception()
            item = self._queue.get()
            if item is _SENTINEL:
                self._sentinel_count += 1
                if self._sentinel_count >= self._n_workers:
                    self._check_exception()
                    raise StopIteration
                continue
            return item

    def _check_exception(self):
        try:
            exc = self._exceptions.get_nowait()
        except queue.Empty:
            return
        self._close_threads()
        raise exc

    def _close_threads(self):
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

    def close(self):
        self._close_threads()
