import argparse
import os
import random
import signal
import sys
import time

import numpy as np


EXIT_FLAG = False
RUN_DIR = None
MAIN_PID = os.getpid()
_LAST_SIGINT = 0.0


def sigint_handler(signum, frame):
    global EXIT_FLAG, _LAST_SIGINT
    if os.getpid() != MAIN_PID:
        return

    now = time.time()
    if now - _LAST_SIGINT < 2.0:
        print("\n\033[31mExiting immediately...\033[0m")
        os._exit(1)
    _LAST_SIGINT = now

    signal.signal(signal.SIGINT, lambda s, f: os._exit(1))

    # Try to restore terminal from tqdm raw mode.
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW,
                          termios.tcgetattr(sys.stdin.fileno()))
    except Exception:
        try:
            os.system('stty sane 2>/dev/null')
        except Exception:
            pass

    print("\n\033[33m═══ Graceful exit requested (Ctrl+C again to force) ═══\033[0m")
    EXIT_FLAG = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    with open('/tmp/jax_seed.txt', 'w') as f:
        f.write(str(seed))
