import pickle
import queue
import shutil
import threading

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
            console.error(f"Background IO failed: {e}")


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


def _fmt_score(score: float) -> str:
    return f"{score:+.4f}"


def save_epoch_checkpoint(
    epoch, params, opt_state, train_loss, metrics,
    score, lr, loss_weights, run_dir, output_dir,
):
    score_str = _fmt_score(score)
    filename = f"epoch_{epoch + 1:03d}_s{score_str}.pkl"
    epoch_path = run_dir / filename
    run_last = run_dir / "last.pkl"
    out_last = output_dir / "last.pkl"

    ckpt = {
        "epoch": epoch,
        "params": params,
        "opt_state": opt_state,
        "train_loss": train_loss,
        "metrics": metrics,
        "score": score,
        "score_str": score_str,
        "lr": lr,
        "loss_weights": loss_weights,
    }

    def _write():
        with open(str(epoch_path), "wb") as f:
            pickle.dump(ckpt, f)
        shutil.copy2(str(epoch_path), str(run_last))
        shutil.copy2(str(epoch_path), str(out_last))

    _IO_QUEUE.put((_write, (), {}))


def save_checkpoint(params, opt_state, epoch, run_dir, output_dir, **extra):
    score_str = _fmt_score(extra.get("score", 0.0))
    filename = f"epoch_{epoch + 1:03d}_s{score_str}.pkl"
    epoch_path = run_dir / filename
    run_last = run_dir / "last.pkl"
    out_last = output_dir / "last.pkl"

    ckpt = {
        "epoch": epoch,
        "params": params,
        "opt_state": opt_state,
        **extra,
    }

    def _write():
        with open(str(epoch_path), "wb") as f:
            pickle.dump(ckpt, f)
        shutil.copy2(str(epoch_path), str(run_last))
        shutil.copy2(str(epoch_path), str(out_last))

    _IO_QUEUE.put((_write, (), {}))


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
