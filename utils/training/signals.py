import time
from pathlib import Path

from utils.console import console


def check_run_signals(run_dir: Path, exit_flag_ref: list) -> bool:
    if run_dir is None:
        return False
    pause_file = run_dir / '.pause'
    exit_file = run_dir / '.exit'

    if exit_file.exists():
        console.warning(f"\nExit signal detected (.exit file found in {run_dir}). Stopping gracefully...")
        exit_flag_ref[0] = True
        try:
            exit_file.unlink()
        except Exception:
            pass

    was_paused = False
    first_pause_msg = True
    while pause_file.exists() and not exit_flag_ref[0]:
        was_paused = True
        if first_pause_msg:
            console.warning(f"\nTraining paused (.pause file found in {run_dir}). Delete the file to resume.")
            first_pause_msg = False
        time.sleep(1.0)
        if exit_file.exists():
            console.warning(f"\nExit signal detected during pause (.exit file found in {run_dir}). Stopping gracefully...")
            exit_flag_ref[0] = True
            try:
                exit_file.unlink()
            except Exception:
                pass

    if was_paused and not exit_flag_ref[0]:
        console.success("Resuming training...")

    return exit_flag_ref[0]
