"""MKV video batch loader for JAX training — C++ (librtdecode.so) decoding."""

import ctypes
import json
from pathlib import Path

import numpy as np


_lib = None
_N = ctypes.c_int()
_H = ctypes.c_int()
_W = ctypes.c_int()


def _get_lib():
    global _lib
    if _lib is None:
        so = str(Path(__file__).resolve().parent.parent.parent / 'utils' / 'c_rt_itctp_decoder' / 'build' / 'libc_rt_itctp_decoder.so')
        _lib = ctypes.cdll.LoadLibrary(so)
        _lib.c_rt_itctp_init()
        _lib.c_rt_itctp_decode.restype = ctypes.POINTER(ctypes.c_float)
        _lib.c_rt_itctp_decode.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
    return _lib


def discover_clips(paths):
    """Scan dataset directories for clip dirs containing meta.json / LR.mkv / HR.mkv."""
    clips = []
    for p in paths:
        p = Path(p)
        if not p.is_dir():
            continue
        for entry in sorted(p.iterdir()):
            if not entry.is_dir():
                continue
            lr = entry / 'LR.mkv'
            hr = entry / 'HR.mkv'
            meta = entry / 'meta.json'
            if lr.exists() and hr.exists() and meta.exists():
                with open(meta) as f:
                    info = json.load(f)
                clips.append({
                    'lr_path': lr,
                    'hr_path': hr,
                    'n_frames': info.get('num_frames', 0),
                    'clip_name': entry.name,
                })
    return clips


def _decode_clip(lr_path, hr_path):
    """Decode a single clip via librtdecode.so → (lr, hr) as [N, H, W, 3] float32."""
    lib = _get_lib()
    ptr = lib.c_rt_itctp_decode(str(lr_path).encode(), str(hr_path).encode(),
                                 ctypes.byref(_N), ctypes.byref(_H), ctypes.byref(_W))
    if not ptr:
        raise RuntimeError(f'c_rt_itctp_decode failed for {lr_path}')
    buf = (ctypes.c_float * (_N.value * _H.value * _W.value * 6)).from_address(
        ctypes.addressof(ptr.contents))
    data = np.frombuffer(buf, dtype=np.float32, count=_N.value*_H.value*_W.value*6
                        ).reshape(_N.value, _H.value, _W.value, 6)
    return data[:, :, :, :3].copy(), data[:, :, :, 3:].copy()


def load_mkv_batch(paths, batch_size, shuffle=True, frames=1):
    """Generator that yields (lr_batch, hr_batch) float32 arrays from MKV clips.

    Each clip is decoded via C++ rtdecode pipe → ICtCp float32 arrays.
    When shuffle=True (training), cycles clips indefinitely, reshuffling each epoch.
    When shuffle=False (validation), single pass through all clips, then stops.
    frames=1 only (multi-frame not supported via pipe mode).
    """
    clips = discover_clips(paths)
    if not clips:
        raise RuntimeError(f'No MKV clips found in: {paths}')

    if shuffle:
        np.random.shuffle(clips)

    buf_lr = []
    buf_hr = []

    while True:
        for clip in clips:
            lr_all, hr_all = _decode_clip(clip['lr_path'], clip['hr_path'])

            n = lr_all.shape[0]
            for i in range(0, n, frames):
                i_end = min(i + frames, n)
                if frames > 1:
                    lr_concat = np.concatenate(lr_all[i:i_end], axis=-1)
                    hr_center = hr_all[i + frames // 2]
                else:
                    lr_concat = lr_all[i]
                    hr_center = hr_all[i]

                buf_lr.append(lr_concat[np.newaxis, ...])
                buf_hr.append(hr_center[np.newaxis, ...])

                if len(buf_lr) >= batch_size:
                    yield np.concatenate(buf_lr, axis=0), np.concatenate(buf_hr, axis=0)
                    buf_lr.clear()
                    buf_hr.clear()

        if buf_lr:
            yield np.concatenate(buf_lr, axis=0), np.concatenate(buf_hr, axis=0)
            buf_lr.clear()
            buf_hr.clear()

        if not shuffle:
            break

        np.random.shuffle(clips)
