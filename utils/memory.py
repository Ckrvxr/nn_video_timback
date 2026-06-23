import gc
import os
from typing import Optional, Tuple

import numpy as np
import psutil
import torch


class MemoryManager:
    """Advanced memory manager for 16GB RAM + 7GB VRAM constraints"""
    
    def __init__(self, max_ram_ratio: float = 0.3, max_vram_ratio: float = 0.5):
        self.max_ram_ratio = max_ram_ratio
        self.max_vram_ratio = max_vram_ratio
        self.process = psutil.Process(os.getpid())
        self._ram_warnings = 0
        self._vram_warnings = 0
        self._last_gc_time = 0
        
    def get_memory_stats(self) -> dict:
        """Get current memory statistics"""
        stats = {
            'ram': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent,
                'process_rss': self.process.memory_info().rss,
                'process_rss_gb': self.process.memory_info().rss / (1024**3),
                'process_rss_ratio': self.process.memory_info().rss / psutil.virtual_memory().total,
            },
            'vram': {}
        }
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            stats['vram'] = {
                'total': torch.cuda.get_device_properties(device).total_memory,
                'allocated': torch.cuda.memory_allocated(device),
                'reserved': torch.cuda.memory_reserved(device),
                'free': torch.cuda.mem_get_info(device)[0],
                'allocated_ratio': torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory,
            }
        
        return stats
    
    def check_memory_pressure(self, force_gc: bool = False) -> Tuple[bool, bool]:
        """Check if memory pressure is high, returns (ram_pressure, vram_pressure)"""
        stats = self.get_memory_stats()
        
        ram_pressure = stats['ram']['process_rss_ratio'] > self.max_ram_ratio
        vram_pressure = stats['vram'].get('allocated_ratio', 0) > self.max_vram_ratio if stats['vram'] else False
        
        if ram_pressure or vram_pressure or force_gc:
            self._cleanup_memory(ram_pressure, vram_pressure)
        
        return ram_pressure, vram_pressure
    
    def _cleanup_memory(self, ram_pressure: bool, vram_pressure: bool):
        """Perform memory cleanup based on pressure type"""
        import time
        current_time = time.time()
        
        # Rate limit GC to avoid excessive overhead
        if current_time - self._last_gc_time < 1.0 and not (ram_pressure and self._ram_warnings > 3):
            return
        
        if ram_pressure:
            gc.collect()
            self._ram_warnings += 1
        
        if vram_pressure and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._vram_warnings += 1
        
        self._last_gc_time = current_time
    
    def get_safe_batch_size(self, base_batch_size: int, current_batch_size: int) -> int:
        """Dynamically adjust batch size based on memory pressure"""
        ram_pressure, vram_pressure = self.check_memory_pressure()
        
        if not ram_pressure and not vram_pressure:
            return min(base_batch_size, current_batch_size + 1)
        
        # Reduce batch size if under pressure
        if ram_pressure or vram_pressure:
            return max(1, current_batch_size // 2)
        
        return current_batch_size
    
    def log_memory_status(self, prefix: str = ""):
        """Log current memory status"""
        stats = self.get_memory_stats()
        
        ram_msg = f"{prefix}RAM: {stats['ram']['process_rss_gb']:.2f}GB ({stats['ram']['process_rss_ratio']*100:.1f}% of system)"
        
        if stats['vram']:
            vram_gb = stats['vram']['allocated'] / (1024**3)
            vram_msg = f"VRAM: {vram_gb:.2f}GB ({stats['vram']['allocated_ratio']*100:.1f}% of GPU)"
            return f"{ram_msg} | {vram_msg}"
        
        return ram_msg


# Global memory manager instance
_global_manager: Optional[MemoryManager] = None


def get_memory_manager(max_ram_ratio: float = 0.3, max_vram_ratio: float = 0.5) -> MemoryManager:
    """Get or create global memory manager"""
    global _global_manager
    if _global_manager is None:
        _global_manager = MemoryManager(max_ram_ratio, max_vram_ratio)
    return _global_manager


def cpu_available() -> int:
    return psutil.virtual_memory().available


def gpu_available(device: Optional[torch.device] = None) -> int:
    if device is not None and device.type != 'cuda':
        return 0
    if device is None:
        device = torch.device('cuda')
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info(device)
    return free


def gpu_total(device: Optional[torch.device] = None) -> int:
    if device is not None and device.type != 'cuda':
        return 0
    if device is None:
        device = torch.device('cuda')
    if not torch.cuda.is_available():
        return 0
    _, total = torch.cuda.mem_get_info(device)
    return total


def check_cpu_allocation(size_bytes: int, label: str = '', ratio: float = 0.8) -> bool:
    avail = cpu_available()
    if size_bytes > avail * ratio:
        return False
    return True


def check_gpu_allocation(size_bytes: int, device: Optional[torch.device] = None, ratio: float = 0.85) -> bool:
    avail = gpu_available(device)
    if size_bytes > avail * ratio:
        return False
    return True


def gpu_low(threshold_ratio: float = 0.2, device: Optional[torch.device] = None) -> bool:
    avail = gpu_available(device)
    total = gpu_total(device)
    if total == 0:
        return False
    return avail / total < threshold_ratio


def estimate_frame_size(h: int, w: int, frames: int, dtype: np.dtype = np.dtype(np.uint16)) -> int:
    return h * w * 3 * frames * dtype.itemsize


def safe_max_cached_segments(per_segment_bytes: int, ratio: float = 0.4) -> int:
    avail = cpu_available()
    max_bytes = int(avail * ratio)
    if per_segment_bytes == 0:
        return 0
    return max(1, max_bytes // per_segment_bytes)


def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force Python to release memory
        import ctypes
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
