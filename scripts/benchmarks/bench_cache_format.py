import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import mmap
import numpy as np
import lz4.block


SAMPLE = Path(__file__).resolve().parent.parent / 'data' / 'newjeans_attention_val' / \
    '00m03s_00m03s_h264_crf30_pveryslow_gop64_yuv420p' / 'lr.npy'
CACHE_FILE = Path(__file__).resolve().parent / 'test_cache.lz4'
WARMUP = 10
BENCH_ROUNDS = 100


def byte_split(data: np.ndarray) -> bytes:
    u16 = data.view(np.uint16)
    hi = ((u16 >> 8) & 0xFF).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)
    return hi.tobytes() + lo.tobytes()


def rebuild(raw: bytes, shape, dtype=np.float16):
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    hi = arr[:len(arr)//2]
    lo = arr[len(arr)//2:]
    u16 = (hi.astype(np.uint16) << 8) | lo.astype(np.uint16)
    return u16.view(dtype).reshape(shape)


def prepare_cache():
    arr = np.load(str(SAMPLE))
    raw = byte_split(arr)
    comp = lz4.block.compress(raw, mode='default', compression=0)
    with open(CACHE_FILE, 'wb') as f:
        f.write(int(len(raw)).to_bytes(4, 'little'))
        f.write(int(len(comp)).to_bytes(4, 'little'))
        f.write(comp)
    print(f"Cache: {len(raw)/1024**2:.1f}MB → {len(comp)/1024**2:.1f}MB  ({len(raw)/len(comp):.2f}x)")
    return arr.shape, arr.dtype


def bench_mmap(shape, dtype):
    with open(CACHE_FILE, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            # Parse header from mmap
            uncomp_size = int.from_bytes(m[:4], 'little')
            comp_size = int.from_bytes(m[4:8], 'little')
            comp_data = m[8:8+comp_size]

            for _ in range(WARMUP):
                raw = lz4.block.decompress(comp_data)
                rebuild(raw, shape, dtype)

            t0 = time.perf_counter()
            for _ in range(BENCH_ROUNDS):
                raw = lz4.block.decompress(comp_data)
                rebuild(raw, shape, dtype)
            t_mmap = (time.perf_counter() - t0) / BENCH_ROUNDS * 1000

            return t_mmap


def bench_np_load(shape, dtype):
    arr = np.load(str(SAMPLE))
    raw = byte_split(arr)

    for _ in range(WARMUP):
        _ = lz4.block.decompress(
            lz4.block.compress(raw, mode='default', compression=0))
        rebuild(raw, shape, dtype)

    comp = lz4.block.compress(raw, mode='default', compression=0)
    t0 = time.perf_counter()
    for _ in range(BENCH_ROUNDS):
        lz4.block.decompress(comp)
        rebuild(raw, shape, dtype)
    t_np = (time.perf_counter() - t0) / BENCH_ROUNDS * 1000

    return t_np, raw


def bench_disk_only():
    """Baseline: just load npy from disk, no compression."""
    for _ in range(WARMUP):
        _ = np.load(str(SAMPLE))
    t0 = time.perf_counter()
    for _ in range(BENCH_ROUNDS):
        _ = np.load(str(SAMPLE))
    t_disk = (time.perf_counter() - t0) / BENCH_ROUNDS * 1000
    return t_disk


def main():
    print("=== Cache Format: LZ4 + mmap Benchmark ===\n")

    # 1. Baseline: disk read
    t_disk = bench_disk_only()
    print(f"1. np.load (from disk, 45MB):        {t_disk:.2f}ms")

    # 2. Prepare LZ4 cache
    shape, dtype = prepare_cache()

    # 3. mmap + decompress
    t_mmap = bench_mmap(shape, dtype)
    print(f"2. mmap + LZ4 decompress (13MB):    {t_mmap:.2f}ms")

    # 4. np.load + LZ4 decompress (for comparison)
    t_np, _ = bench_np_load(shape, dtype)
    print(f"3. np.load + LZ4 decompress (13MB): {t_np:.2f}ms")

    # 5. LZ4 standalone decompress (no I/O at all)
    comp = lz4.block.compress(byte_split(np.load(str(SAMPLE))), mode='default', compression=0)
    for _ in range(WARMUP):
        lz4.block.decompress(comp)
    t0 = time.perf_counter()
    for _ in range(BENCH_ROUNDS):
        lz4.block.decompress(comp)
    t_dec = (time.perf_counter() - t0) / BENCH_ROUNDS * 1000
    print(f"4. LZ4 decompress only (pure):      {t_dec:.2f}ms")

    print(f"\n{'─'*50}")
    print(f"{'Scenario':<35} {'Time':>10}")
    print(f"{'─'*50}")
    print(f"{'np.load (disk 45MB)':<35} {t_disk:>8.2f}ms")
    print(f"{'mmap→LZ4→rebuild':<35} {t_mmap:>8.2f}ms")
    print(f"{'  └─ LZ4 decompress only':<35} {t_dec:>8.2f}ms")
    overhead = t_mmap - t_disk
    print(f"{'─'*50}")
    print(f"{'Overhead vs np.load':<35} {'+' if overhead > 0 else ''}{overhead:>7.2f}ms")

    # Cleanup
    CACHE_FILE.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
