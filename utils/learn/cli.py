import argparse
import os
import random
import signal
import sys

import numpy as np



EXIT_FLAG = False
RUN_DIR = None
MAIN_PID = os.getpid()


def sigint_handler(signum, frame):
    global EXIT_FLAG, RUN_DIR
    if os.getpid() != MAIN_PID:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # tqdm leaves the terminal in raw mode, which breaks input().
    # Reset to cooked / sane mode before prompting.
    try:
        import termios
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSANOW, termios.tcgetattr(fd))
    except Exception:
        try:
            os.system('stty sane 2>/dev/null')
        except Exception:
            pass
    try:
        from tqdm import tqdm
        tqdm._instances.clear()
    except Exception:
        pass

    sys.stdout.write("\n")
    sys.stdout.write("\033[1;33m═══ Training Paused (Ctrl+C detected) ═══\033[0m\n")
    sys.stdout.write("  \033[1;37m[c]\033[0m Continue training\n")
    if RUN_DIR:
        sys.stdout.write("  \033[1;37m[p]\033[0m Pause (create .pause file, delete to resume)\n")
    sys.stdout.write("  \033[1;37m[s]\033[0m Save checkpoint and exit\n")
    sys.stdout.write("  \033[1;37m[e]\033[0m Exit immediately\n")
    sys.stdout.flush()

    while True:
        try:
            prompt = "\033[1;36mChoice [c/p/s/e]: \033[0m" if RUN_DIR else "\033[1;36mChoice [c/s/e]: \033[0m"
            sys.stdout.write(prompt)
            sys.stdout.flush()
            choice = sys.stdin.readline().strip().lower()
            if not choice:
                continue
            if choice == 'c':
                sys.stdout.write("\033[32mResuming training...\033[0m\n")
                if RUN_DIR:
                    pause_file = RUN_DIR / '.pause'
                    if pause_file.exists():
                        try:
                            pause_file.unlink()
                        except Exception:
                            pass
                signal.signal(signal.SIGINT, sigint_handler)
                return
            elif choice == 'p' and RUN_DIR:
                pause_file = RUN_DIR / '.pause'
                try:
                    pause_file.touch()
                    sys.stdout.write(f"\033[33mCreated '{pause_file}'. Training is now paused.\033[0m\n")
                    sys.stdout.write("To resume: delete the .pause file, or press Ctrl+C again.\n")
                except Exception as e:
                    sys.stdout.write(f"\033[31mError creating pause file: {e}\033[0m\n")
                signal.signal(signal.SIGINT, sigint_handler)
                return
            elif choice == 's':
                sys.stdout.write("\033[33mGraceful exit requested. Will save checkpoint...\033[0m\n")
                EXIT_FLAG = True
                signal.signal(signal.SIGINT, lambda s, f: os._exit(1))
                return
            elif choice == 'e':
                sys.stdout.write("\033[31mExiting immediately...\033[0m\n")
                os._exit(1)
            sys.stdout.flush()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n\033[31mExiting immediately...\033[0m\n")
            os._exit(1)
        except Exception:
            pass


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
    # JAX uses explicit PRNG state — seed doesn't affect jax.random
    # Record it for reproducibility of the random data pipeline
    with open('/tmp/jax_seed.txt', 'w') as f:
        f.write(str(seed))


def check_memory(config, force: bool = False):
    pass
