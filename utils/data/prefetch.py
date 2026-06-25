"""Multi-worker threaded prefetcher — shared clip pool, parallel decode.

A single producer shuffles all clips once, then ``n_workers`` threads decode
clips in parallel and push individual frames to a shared frame queue.
The main thread gathers frames and assembles batches of ``batch_size``.
"""

import queue
import threading

import numpy as np

from utils.data.mkv_loader import _decode_clip, discover_clips


_SENTINEL = object()


class PrefetchIterator:
    """Prefetches batches by decoding MKV clips in parallel worker threads.

    Usage::

        loader = PrefetchIterator(dataset_paths, batch_size=8, n_workers=3)
        for batch in loader:
            lr, hr = batch
            ...
        loader.close()
    """

    def __init__(self, paths, batch_size, n_workers=3, frame_queue_size=300):
        self._frame_queue = queue.Queue(maxsize=frame_queue_size)
        self._exceptions = queue.Queue(maxsize=n_workers + 1)
        self._stop_event = threading.Event()
        self._batch_size = batch_size
        self._sentinel_count = 0

        # ── Single global shuffle ──
        clips = discover_clips(paths)
        if not clips:
            raise RuntimeError(f'No MKV clips found in: {paths}')
        np.random.shuffle(clips)

        self._clip_queue = queue.Queue()
        for c in clips:
            self._clip_queue.put(c)
        for _ in range(n_workers):
            self._clip_queue.put(_SENTINEL)

        # ── Start workers ──
        self._threads = []
        for _ in range(n_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self):
        """Pull clips from the shared queue, decode, push frames."""
        try:
            while not self._stop_event.is_set():
                item = self._clip_queue.get()
                if item is _SENTINEL:
                    # Signal end-of-frames for this worker.
                    self._frame_queue.put(_SENTINEL)
                    return
                lr_all, hr_all = _decode_clip(item['lr_path'], item['hr_path'])
                for i in range(len(lr_all)):
                    if self._stop_event.is_set():
                        return
                    self._frame_queue.put((lr_all[i], hr_all[i]))
        except Exception as e:
            try:
                self._exceptions.put(e)
            except queue.Full:
                pass
            self._frame_queue.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        buf_lr = []
        buf_hr = []

        while len(buf_lr) < self._batch_size:
            self._check_exception()
            item = self._frame_queue.get()
            if item is _SENTINEL:
                self._sentinel_count += 1
                if self._sentinel_count >= len(self._threads):
                    self._check_exception()
                    if not buf_lr:
                        raise StopIteration
                    break
                continue
            lr_frame, hr_frame = item
            buf_lr.append(lr_frame[np.newaxis, ...])
            buf_hr.append(hr_frame[np.newaxis, ...])

        batch_lr = np.concatenate(buf_lr, axis=0)
        batch_hr = np.concatenate(buf_hr, axis=0)
        return batch_lr, batch_hr

    def _check_exception(self):
        try:
            exc = self._exceptions.get_nowait()
        except queue.Empty:
            return
        self._close_threads()
        raise exc

    def _close_threads(self):
        self._stop_event.set()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self._clip_queue.empty():
            try:
                self._clip_queue.get_nowait()
            except queue.Empty:
                break
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

    def close(self):
        self._close_threads()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
