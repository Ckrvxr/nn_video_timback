import os
import pickle
import queue
import shutil
import threading
from pathlib import Path

from utils.console import console


_IO_QUEUE = queue.Queue()
_IO_THREAD = None


def _io_worker():
    while True:
        item = _IO_QUEUE.get()
        if item is None:
            break
        fn, args, kwargs = item
        try:
            fn(*args, **kwargs)
        except Exception as e:
            console.error(f'Background IO failed: {e}')


def start_io_worker():
    global _IO_THREAD
    if _IO_THREAD is None:
        _IO_THREAD = threading.Thread(target=_io_worker, daemon=True)
        _IO_THREAD.start()


def stop_io_worker():
    global _IO_THREAD
    if _IO_THREAD is not None:
        _IO_QUEUE.put(None)
        _IO_THREAD.join(timeout=30)
        _IO_THREAD = None


def save_checkpoint(params, opt_state, epoch, run_dir: Path, output_dir: Path, is_best: bool = False):
    run_last = run_dir / 'last.pkl'
    run_best = run_dir / 'best.pkl'
    out_last = output_dir / 'last.pkl'
    out_best = output_dir / 'best.pkl'

    def _write():
        ckpt = {
            'epoch': epoch,
            'params': params,
            'opt_state': opt_state,
        }
        with open(str(run_last), 'wb') as f:
            pickle.dump(ckpt, f)
        if is_best:
            shutil.copy2(str(run_last), str(run_best))
        shutil.copy2(str(run_last), str(out_last))
        if is_best:
            shutil.copy2(str(run_best), str(out_best))

    _IO_QUEUE.put((_write, (), {}))
