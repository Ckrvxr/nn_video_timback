import argparse
import os
import random
import signal

import numpy as np
import jax
import jax.numpy as jnp

from utils.console import console


EXIT_FLAG = False
RUN_DIR = None
MAIN_PID = os.getpid()


def sigint_handler(signum, frame):
    global EXIT_FLAG, RUN_DIR
    if os.getpid() != MAIN_PID:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print()
    print("\033[1;33m═══ Training Paused (Ctrl+C detected) ═══\033[0m")
    print("  \033[1;37m[c]\033[0m Continue training")
    if RUN_DIR:
        print("  \033[1;37m[p]\033[0m Pause (create \033[3m.pause\033[0m file, delete to resume)")
    print("  \033[1;37m[s]\033[0m Save checkpoint and exit")
    print("  \033[1;37m[e]\033[0m Exit immediately")

    while True:
        try:
            prompt = "\033[1;36mChoice [c/p/s/e]: \033[0m" if RUN_DIR else "\033[1;36mChoice [c/s/e]: \033[0m"
            choice = input(prompt).strip().lower()
            if choice == 'c':
                print("\033[32mResuming training...\033[0m")
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
                    print(f"\033[33mCreated '{pause_file}'. Training is now paused.\033[0m")
                    print("To resume: delete the .pause file, or press Ctrl+C again.")
                except Exception as e:
                    print(f"\033[31mError creating pause file: {e}\033[0m")
                signal.signal(signal.SIGINT, sigint_handler)
                return
            elif choice == 's':
                print("\033[33mGraceful exit requested. Will save checkpoint...\033[0m")
                EXIT_FLAG = True
                signal.signal(signal.SIGINT, lambda s, f: os._exit(1))
                return
            elif choice == 'e':
                print("\033[31mExiting immediately...\033[0m")
                os._exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\n\033[31mExiting immediately...\033[0m")
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
    mem_cfg = config.get('memory_settings', {})
    max_ram_ratio = mem_cfg.get('max_ram_ratio', 0.3)
    from utils.memory import get_memory_manager
    manager = get_memory_manager(max_ram_ratio, 0.5)
    ram_pressure, vram_pressure = manager.check_memory_pressure(force_gc=force)
    if ram_pressure or vram_pressure or force:
        status = manager.log_memory_status("Memory check: ")
        console.info(status)
        if ram_pressure:
            console.warning(f"RAM pressure detected (limit: {max_ram_ratio*100:.0f}%)")
    return ram_pressure, vram_pressure
