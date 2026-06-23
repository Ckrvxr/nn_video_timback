import json
import subprocess as sp
import sys
from pathlib import Path


def probe_frames(path: Path) -> int:
    result = sp.run(
        [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def main():
    for data_root in sys.argv[1:]:
        root = Path(data_root)
        if not root.is_dir():
            print(f"Skipping {root}: not a directory")
            continue
        for seg_dir in sorted(root.iterdir()):
            if not seg_dir.is_dir():
                continue
            meta_path = seg_dir / "meta.json"
            if meta_path.exists():
                continue
            vid = seg_dir / "HR.mkv"
            if not vid.exists():
                vid = seg_dir / "LR.mkv"
            if not vid.exists():
                print(f"  SKIP {seg_dir.name}: no HR.mkv or LR.mkv")
                continue
            n = probe_frames(vid)
            meta_path.write_text(json.dumps({"num_frames": n}))
            print(f"  FIX  {seg_dir.name}: {n} frames")


if __name__ == "__main__":
    main()
