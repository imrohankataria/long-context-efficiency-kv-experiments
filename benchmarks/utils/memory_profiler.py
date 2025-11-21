"""
Memory Profiling Utilities

Advanced memory profiling for long-context inference.
"""

import torch
import psutil
import gc
from typing import Dict, List, Optional
from contextlib import contextmanager
import time


class MemoryProfiler:
    """Profile memory usage during inference."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize memory profiler.
        
        Args:
            device: Device to profile ('cuda' or 'cpu')
        """
        self.device = device
        self.snapshots = []
        
    def snapshot(self, label: str = ""):
        """
        Take a memory snapshot.
        
        Args:
            label: Optional label for the snapshot
        """
        snapshot = {
            'label': label,
            'timestamp': time.time(),
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            snapshot.update({
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
            })
        
        # CPU memory
        process = psutil.Process()
        snapshot['cpu_memory_mb'] = process.memory_info().rss / 1024**2
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def reset(self):
        """Reset memory statistics."""
        self.snapshots = []
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.snapshots:
            return 0.0
        
        if self.device == 'cuda':
            return max(s.get('max_allocated_mb', 0) for s in self.snapshots)
        else:
            return max(s.get('cpu_memory_mb', 0) for s in self.snapshots)
    
    def get_memory_timeline(self) -> List[Dict]:
        """Get memory usage timeline."""
        return self.snapshots
    
    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            print("No memory snapshots available")
            return
        
        print("\nMemory Usage Summary:")
        print("-" * 60)
        
        if self.device == 'cuda':
            peak_allocated = self.get_peak_memory()
            print(f"Peak GPU Memory Allocated: {peak_allocated:.2f} MB")
            
            if self.snapshots:
                final = self.snapshots[-1]
                print(f"Final GPU Memory Allocated: {final.get('allocated_mb', 0):.2f} MB")
                print(f"Final GPU Memory Reserved: {final.get('reserved_mb', 0):.2f} MB")
        
        # CPU memory
        cpu_memories = [s.get('cpu_memory_mb', 0) for s in self.snapshots]
        print(f"Peak CPU Memory: {max(cpu_memories):.2f} MB")
        print(f"Final CPU Memory: {cpu_memories[-1]:.2f} MB")
        
        print("-" * 60)


@contextmanager
def profile_memory(device: str = "cuda", label: str = "operation"):
    """
    Context manager for profiling memory usage.
    
    Args:
        device: Device to profile
        label: Label for the operation
        
    Yields:
        MemoryProfiler instance
        
    Example:
        >>> with profile_memory(device="cuda", label="inference") as profiler:
        ...     model.generate(**inputs)
        >>> profiler.print_summary()
    """
    profiler = MemoryProfiler(device=device)
    profiler.reset()
    profiler.snapshot(f"{label}_start")
    
    try:
        yield profiler
    finally:
        profiler.snapshot(f"{label}_end")


class GPUMonitor:
    """Monitor GPU utilization and memory during inference."""
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.monitoring = False
        self.stats = []
        
    def start(self):
        """Start monitoring."""
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, monitoring disabled")
            return
        
        self.monitoring = True
        self.stats = []
        
    def record(self):
        """Record current GPU stats."""
        if not self.monitoring or not torch.cuda.is_available():
            return
        
        stats = {
            'timestamp': time.time(),
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        }
        
        # Try to get utilization if py3nvml is available
        try:
            # py3nvml package provides pynvml module
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats['gpu_utilization_percent'] = util.gpu
            stats['memory_utilization_percent'] = util.memory
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            # py3nvml not installed or GPU not available
            pass
        
        self.stats.append(stats)
        
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        
    def get_stats(self) -> List[Dict]:
        """Get recorded stats."""
        return self.stats
    
    def get_average_utilization(self) -> float:
        """Get average GPU utilization."""
        utils = [s.get('gpu_utilization_percent', 0) for s in self.stats]
        return sum(utils) / len(utils) if utils else 0.0
