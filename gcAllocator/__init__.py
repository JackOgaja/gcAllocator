# MIT License
# 
# Copyright (c) 2025 Jack Ogaja
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
gcAllocator - Graceful CUDA Allocator for PyTorch (with backward compatibility)
Advanced Python interface with backward compatibility and proper state management.
"""

import os
import sys
import json
import warnings
import threading
import asyncio
import weakref
import functools
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import IntEnum
from contextlib import contextmanager
import logging
import torch

# Import the C++ extension
try:
    from . import gc_allocator_core
except ImportError as e:
    warnings.warn(
        f"Failed to import gc_allocator_core C++ extension: {e}\n"
        "Please ensure the module is properly built using setup.py"
    )
    raise

__version__ = "0.2.0"

# Configure logging
logger = logging.getLogger("gcAllocator")

# Global state management - CRITICAL for test compatibility
_global_allocator_instance: Optional['GCAllocator'] = None
_is_globally_installed: bool = False


class MemoryPressureLevel(IntEnum):
    """Memory pressure levels for inter-process coordination"""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class RetryConfig:
    """Configuration for retry behavior with validation"""
    
    max_attempts: int = 5
    initial_backoff_ms: int = 100
    backoff_multiplier: float = 2.0
    max_backoff_ms: int = 5000
    enable_cache_flush: bool = True
    enable_checkpointing: bool = True
    checkpoint_threshold: int = 2
    yield_on_wait: bool = True
    memory_pressure_protocol: bool = False
    async_allocation: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_backoff_ms < 0:
            raise ValueError("initial_backoff_ms must be >= 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
            
    @classmethod
    def from_env(cls) -> 'RetryConfig':
        """Load configuration from environment variables"""
        config = {}
        env_mapping = {
            'GC_ALLOCATOR_MAX_RETRY': ('max_attempts', int),
            'GC_ALLOCATOR_INITIAL_BACKOFF_MS': ('initial_backoff_ms', int),
            'GC_ALLOCATOR_BACKOFF_MULTIPLIER': ('backoff_multiplier', float),
            'GC_ALLOCATOR_MAX_BACKOFF_MS': ('max_backoff_ms', int),
            'GC_ALLOCATOR_CACHE_FLUSH': ('enable_cache_flush', lambda x: x.lower() in ('1', 'true')),
            'GC_ALLOCATOR_GRADIENT_CHECKPOINT': ('enable_checkpointing', lambda x: x.lower() in ('1', 'true')),
        }
        
        for env_var, (config_key, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                try:
                    config[config_key] = converter(value)
                except (ValueError, TypeError):
                    pass
        
        return cls(**config)
    
    def to_cpp_config(self) -> Any:
        """Convert to C++ RetryConfig using existing binding helper"""
        # Use the create_retry_config helper that exists in bindings
        return gc_allocator_core.create_retry_config({
            'max_retries': self.max_attempts,
            'initial_delay_ms': self.initial_backoff_ms,
            'backoff_multiplier': self.backoff_multiplier,
            'max_delay_ms': self.max_backoff_ms,
            'enable_cache_flush': self.enable_cache_flush,
            'enable_checkpointing': self.enable_checkpointing
        })


class RetryStats:
    """Wrapper for retry statistics from C++"""

    def __init__(self, cpp_stats=None):
        """
        Initialize RetryStats wrapper.

        Args:
            cpp_stats: C++ RetryStats object or None for empty stats
        """
        if cpp_stats is None:
            # Create empty stats if none provided
            self._stats = self._create_empty_stats()
        else:
            self._stats = cpp_stats

    def _create_empty_stats(self):
        """Create an empty stats object for when allocator isn't installed"""
        # This is a placeholder - actual empty stats should come from C++
        class EmptyStats:
            def get_total_retry_attempts(self): return 0
            def get_cache_flushes(self): return 0
            def get_checkpoint_activations(self): return 0
            def get_successful_recoveries(self): return 0
            def reset(self): pass
        return EmptyStats()

    @property
    def total_retry_attempts(self) -> int:
        return self._stats.get_total_retry_attempts()

    @property
    def cache_flushes(self) -> int:
        return self._stats.get_cache_flushes()

    @property
    def checkpoint_activations(self) -> int:
        return self._stats.get_checkpoint_activations()

    @property
    def successful_recoveries(self) -> int:
        return self._stats.get_successful_recoveries()

    def reset(self):
        """Reset all retry statistics to zero"""
        self._stats.reset()

    def __str__(self) -> str:
        return (f"RetryStats(attempts={self.total_retry_attempts}, "
                f"flushes={self.cache_flushes}, "
                f"checkpoints={self.checkpoint_activations}, "
                f"recoveries={self.successful_recoveries})")

    def __repr__(self) -> str:
        return f"<RetryStats attempts={self.total_retry_attempts} recoveries={self.successful_recoveries}>"

class AllocationStats:
    """Wrapper for allocation statistics from C++ - BACKWARD COMPATIBLE"""
    
    def __init__(self, cpp_stats):
        self._stats = cpp_stats
    
    @property
    def total_allocations(self) -> int:
        return self._stats.get_total_allocations()
    
    @property
    def total_deallocations(self) -> int:
        return self._stats.get_total_deallocations()
    
    @property
    def total_bytes_allocated(self) -> int:
        return self._stats.get_total_bytes_allocated()
    
    @property
    def current_bytes_allocated(self) -> int:
        return self._stats.get_current_bytes_allocated()
    
    @property
    def peak_bytes_allocated(self) -> int:
        return self._stats.get_peak_bytes_allocated()
    
    @property
    def oom_count(self) -> int:
        return self._stats.get_oom_count()
    
    @property
    def active_devices(self) -> list:
        return self._stats.get_active_devices()
    
    def reset(self):
        """Reset all statistics to zero"""
        self._stats.reset()
    
    def __str__(self) -> str:
        return str(self._stats)
    
    def __repr__(self) -> str:
        return f"<AllocationStats allocations={self.total_allocations} current_mb={self.current_bytes_allocated/(1024**2):.2f}>"


class AllocationFuture:
    """Future-like object for asynchronous allocation"""
    
    def __init__(self, size: int, device: int):
        self.size = size
        self.device = device
        self._result = None
        self._exception = None
        self._done = threading.Event()
        self._callbacks = []
        self._lock = threading.Lock()
        
    def result(self, timeout: Optional[float] = None) -> torch.Tensor:
        """Block until allocation completes or timeout"""
        if not self._done.wait(timeout):
            raise TimeoutError(f"Allocation timeout after {timeout}s")
        
        if self._exception:
            raise self._exception
        return self._result
    
    async def async_result(self) -> torch.Tensor:
        """Async/await compatible result retrieval"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.result)
    
    def done(self) -> bool:
        """Check if allocation is complete"""
        return self._done.is_set()
    
    def cancelled(self) -> bool:
        """Check if allocation was cancelled"""
        return isinstance(self._exception, asyncio.CancelledError)
    
    def cancel(self) -> bool:
        """Attempt to cancel the allocation"""
        with self._lock:
            if self._done.is_set():
                return False
            self._exception = asyncio.CancelledError()
            self._done.set()
            self._run_callbacks()
            return True
    
    def add_done_callback(self, fn: Callable) -> None:
        """Add callback to be called when allocation completes"""
        with self._lock:
            if self._done.is_set():
                fn(self)
            else:
                self._callbacks.append(fn)
    
    def _set_result(self, result: torch.Tensor) -> None:
        """Internal: Set successful result"""
        with self._lock:
            self._result = result
            self._done.set()
            self._run_callbacks()
    
    def _set_exception(self, exception: Exception) -> None:
        """Internal: Set failure exception"""
        with self._lock:
            self._exception = exception
            self._done.set()
            self._run_callbacks()
    
    def _run_callbacks(self) -> None:
        """Internal: Run all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception:
                pass

# JO+
class StatsAggregator:
    """Aggregates and formats allocation and retry statistics"""
    
    def __init__(self, allocation_stats: AllocationStats, retry_stats: RetryStats):
        self.allocation_stats = allocation_stats
        self.retry_stats = retry_stats
        self._compute_derived_metrics()
    
    def _compute_derived_metrics(self):
        """Compute derived metrics from raw stats"""
        self.oom_recovery_rate = 0.0
        if self.retry_stats.total_retry_attempts > 0:
            self.oom_recovery_rate = (
                self.retry_stats.successful_recoveries / 
                self.retry_stats.total_retry_attempts
            )
        
        self.memory_efficiency = 0.0
        if self.allocation_stats.total_bytes_allocated > 0:
            self.memory_efficiency = (
                self.allocation_stats.current_bytes_allocated /
                self.allocation_stats.total_bytes_allocated
            )
    
    def to_dict(self) -> dict:
        """Export all stats as dictionary"""
        return {
            'allocation': {
                'total_allocations': self.allocation_stats.total_allocations,
                'current_bytes': self.allocation_stats.current_bytes_allocated,
                'peak_bytes': self.allocation_stats.peak_bytes_allocated,
                'oom_events': self.allocation_stats.oom_count,
            },
            'retry': {
                'attempts': self.retry_stats.total_retry_attempts,
                'recoveries': self.retry_stats.successful_recoveries,
                'cache_flushes': self.retry_stats.cache_flushes,
                'checkpoints': self.retry_stats.checkpoint_activations,
            },
            'derived': {
                'oom_recovery_rate': self.oom_recovery_rate,
                'memory_efficiency': self.memory_efficiency,
            }
        }
    
    def __repr__(self):
        return f"<StatsAggregator OOM_rate={self.oom_recovery_rate:.2%} efficiency={self.memory_efficiency:.2%}>"
# JO-

class GCAllocator:
    """Main interface for graceful CUDA allocator - BACKWARD COMPATIBLE"""
    
    def __init__(self, 
                 config: RetryConfig = None,
                 enable_logging: bool = False,
                 retry_config: RetryConfig = None):
        """
        Initialize GCAllocator with backward compatibility.
        
        Args:
            config: New-style RetryConfig object
            enable_logging: Enable detailed logging
            retry_config: Legacy parameter name for config
        """
        global _global_allocator_instance
        #JO+ self.allocation_tracker = AllocationTracker(self)
        
        # Handle backward compatibility for retry_config parameter
        if retry_config is not None and config is None:
            config = retry_config
            
        # Check environment variable for logging
        env_logging = os.environ.get("GC_ALLOCATOR_LOG", "0")
        self.enable_logging = enable_logging or env_logging.lower() in ("1", "true", "yes", "on")
        
        self.config = config or RetryConfig()
        self._installed = False
        self._manager = None
        self._checkpoint_callbacks = []
        self._lock = threading.Lock()
        
        # Track if this is the global instance
        self._is_global = False
    
    @property  
    def is_installed(self) -> bool:
        """Check if this allocator instance is currently installed"""
        return self._installed
    
    def install(self) -> None:
        """Install the allocator as PyTorch's CUDA allocator"""
        global _global_allocator_instance, _is_globally_installed
       
        # Enable Python-level tracking
        # JO+ self.allocation_tracker.enable_tracking()

        with self._lock:
            if self._installed:
                warnings.warn("GCAllocator is already installed for this instance")
                return
            
            # Handle global state replacement
            if _is_globally_installed and _global_allocator_instance and _global_allocator_instance != self:
                _global_allocator_instance.uninstall()
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. GCAllocator requires CUDA support.")
            
            try:
                # Get manager instance
                self._manager = gc_allocator_core.get_manager()
                
                # Configure retry strategy
                self._manager.configure_retry_strategy(self.config.to_cpp_config())
                
                # Install the allocator
                self._manager.install_allocator()
                
                # Configure logging
                if self.enable_logging:
                    self._manager.enable_logging()
                
                self._installed = True
                self._is_global = True
                _global_allocator_instance = self
                _is_globally_installed = True
                
                if self.enable_logging:
                    print(f"[GCAllocator] v{__version__} installed successfully")
                    
            except Exception as e:
                self._installed = False
                raise RuntimeError(f"Failed to install allocator: {e}")
    
    def uninstall(self) -> None:
        """Uninstall the allocator and restore PyTorch's default allocator"""
        global _global_allocator_instance, _is_globally_installed
       
        # Disable Python-level tracking first
        # JO+ self.allocation_tracker.disable_tracking()

        with self._lock:
            if not self._installed:
                return
            
            try:
                if self._manager:
                    self._manager.uninstall_allocator()
                
                self._installed = False
                
                # Update global state
                if _global_allocator_instance == self:
                    _global_allocator_instance = None
                    _is_globally_installed = False
                
                if self.enable_logging:
                    print("[GCAllocator] Uninstalled successfully")
                    
            except Exception as e:
                warnings.warn(f"Error during uninstall: {e}")
                # Force reset state even if C++ uninstall fails
                self._installed = False
                if _global_allocator_instance == self:
                    _global_allocator_instance = None
                    _is_globally_installed = False
    
    def get_stats(self) -> AllocationStats:
        """Get current allocation statistics"""
        if not self._installed:
            warnings.warn("Getting stats from non-installed allocator")
        
        # JO+++cpp_stats = gc_allocator_core.get_stats()
        return gc_allocator_core.get_stats()
        # JO++return AllocationStats(cpp_stats)
  
    # JO+
    #def get_retry_stats():
    #    mgr = gc_allocator_core.get_manager()
    #    stats = mgr.get_retry_stats()
    #    return {
    #        "retry_attempts": stats.get_total_retry_attempts(),
    #        "cache_flushes": stats.get_cache_flushes(),
    #        "checkpoint_activations": stats.get_checkpoint_activations(),
    #        "successful_recoveries": stats.get_successful_recoveries(),
    #        # Optional new fields if added:
    #        "exhausted_retries": getattr(stats, "get_exhausted_retries", lambda: None)(),
    #        "terminal_failures": getattr(stats, "get_terminal_failures", lambda: None)(),
    #    }
    def get_retry_stats(self) -> RetryStats:
        """Get retry statistics only"""
        if not self._installed:
            return RetryStats()  # Return empty stats

        try:
            manager = gc_allocator_core.get_manager()
            cpp_stats = manager.get_retry_stats()
            return RetryStats(cpp_stats)
        except Exception as e:
            warnings.warn(f"Failed to get retry stats: {e}")
            return RetryStats()
    # JO-

    def reset_stats(self) -> None:
        """Reset allocation statistics"""
        if self._manager:
            gc_allocator_core.reset_stats()
   
    # JO+
    #@staticmethod
    #def get_combined_stats() -> StatsAggregator:
    #    """Get aggregated allocation and retry statistics"""
    #    if not self._installed:
    #        warnings.warn("Getting stats from non-installed allocator")
    #    
    #    try:
    #        # Try to get combined stats if available
    #        combined = gc_allocator_core.get_combined_stats()
    #        return StatsAggregator(
    #            AllocationStats(combined.allocation_stats),
    #            RetryStats(combined.retry_stats)
    #        )
    #    except AttributeError:
    #        # Fallback for older versions
    #        alloc_stats = self.get_stats()
    #        retry_stats = gc_allocator_core.get_retry_stats()
    #        return StatsAggregator(alloc_stats, retry_stats)
    #@staticmethod
    def get_combined_stats(self) -> 'StatsAggregator':
        """Get aggregated allocation and retry statistics"""
        if not self._installed:
            warnings.warn("Getting stats from non-installed allocator")
        
        try:
            # Get allocation stats
            cpp_alloc_stats = gc_allocator_core.get_stats()
            alloc_stats = AllocationStats(cpp_alloc_stats)
            
            # Get retry stats from manager
            manager = gc_allocator_core.get_manager()
            cpp_retry_stats = manager.get_retry_stats()
            retry_stats = RetryStats(cpp_retry_stats)
            
            # Return aggregated stats
            return StatsAggregator(alloc_stats, retry_stats)
            
        except Exception as e:
            warnings.warn(f"Failed to get combined stats: {e}")
            # Return empty stats on failure
            empty_alloc = AllocationStats(gc_allocator_core.get_stats())
            empty_retry = RetryStats()
            return StatsAggregator(empty_alloc, empty_retry)

    # BACKWARD COMPATIBILITY METHODS
    def enable_logging_runtime(self) -> None:
        """Enable logging at runtime - BACKWARD COMPATIBLE"""
        self.enable_logging = True
        if self._installed and self._manager:
            self._manager.enable_logging()
    
    def disable_logging_runtime(self) -> None:
        """Disable logging at runtime - BACKWARD COMPATIBLE"""
        self.enable_logging = False
        if self._installed and self._manager:
            self._manager.disable_logging()
    
    def allocate_async(self, size: int, device: int = 0) -> AllocationFuture:
        """Allocate memory asynchronously (Phase 4 preparation)"""
        if not self._installed:
            raise RuntimeError("Allocator is not installed")
        
        # For now, return a simple future that allocates synchronously
        future = AllocationFuture(size, device)
        
        def allocate():
            try:
                tensor = torch.empty(size // 4, device=f'cuda:{device}', dtype=torch.float32)
                future._set_result(tensor)
            except Exception as e:
                future._set_exception(e)
        
        threading.Thread(target=allocate, daemon=True).start()
        return future
    
    def register_checkpoint(self, callback: Callable[[], bool]) -> None:
        """Register checkpoint callback - BACKWARD COMPATIBLE"""
        self._checkpoint_callbacks.append(callback)
        if self._installed and self._manager:
            self._manager.register_checkpoint_callback(callback)
    
    def __enter__(self):
        """Context manager entry"""
        self.install()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.uninstall()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            if self._installed:
                self.uninstall()
        except:
            pass

def get_combined_stats() -> Optional['StatsAggregator']:
    """Get combined stats from globally installed allocator"""
    global _global_allocator_instance
    if _global_allocator_instance is not None:
        if hasattr(_global_allocator_instance, 'is_installed') and _global_allocator_instance.is_installed:
            # Direct method call on the instance
            stats_method = getattr(_global_allocator_instance, 'get_combined_stats', None)
            if stats_method and callable(stats_method):
                return stats_method()
    return None

def get_allocator() -> Optional[GCAllocator]:
    """Get the global allocator instance"""
    return _global_allocator_instance

# JO+
#def get_manager():
#    """Get the C++ allocator manager instance - BACKWARD COMPATIBLE"""
#    return gc_allocator_core.get_manager()

def register_checkpoint(callback: Callable[[], bool]) -> None:
    """Register checkpoint callback - BACKWARD COMPATIBLE"""
    if _global_allocator_instance:
        _global_allocator_instance.register_checkpoint(callback)

# Global convenience functions 
def install(enable_logging: bool = False, retry_config: RetryConfig = None) -> GCAllocator:
    """Install GCAllocator globally and return the instance"""
    allocator = GCAllocator(enable_logging=enable_logging)
    if retry_config:
        allocator.retry_config = retry_config
    allocator.install()
    return allocator

def uninstall():
    """Uninstall any currently installed GCAllocator"""
    global _global_allocator_instance
    if _global_allocator_instance:
        _global_allocator_instance.uninstall()

def is_installed() -> bool:
    """Check if any GCAllocator is currently installed globally"""
    return _is_globally_installed

def get_stats() -> Optional[AllocationStats]:
    """Get stats from the globally installed allocator, if any"""
    global _global_allocator_instance
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        return _global_allocator_instance.get_stats()
    return None

def reset_stats():
    """Reset stats for the globally installed allocator, if any"""
    global _global_allocator_instance
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        _global_allocator_instance.reset_stats()

def get_combined_stats() -> Optional[StatsAggregator]:
    """Get combined stats from globally installed allocator"""
    global _global_allocator_instance

    # Defensive programming - explicit null check
    if _global_allocator_instance is None:
        return None

    # Check installation status
    if not getattr(_global_allocator_instance, 'is_installed', False):
        return None

    try:
        # Get allocation stats
        alloc_stats = _global_allocator_instance.get_stats()

        # Get retry stats
        retry_stats = _global_allocator_instance.get_retry_stats()

        # Create aggregator
        return StatsAggregator(alloc_stats, retry_stats)

    except Exception as e:
        warnings.warn(f"Failed to get combined stats: {e}")
        return None

def get_retry_stats() -> Optional[RetryStats]:
    """Get retry stats from globally installed allocator"""
    global _global_allocator_instance

    if _global_allocator_instance is None:
        return None

    if not getattr(_global_allocator_instance, 'is_installed', False):
        return None

    try:
        return _global_allocator_instance.get_retry_stats()
    except Exception as e:
        warnings.warn(f"Failed to get retry stats: {e}")
        return None

def get_manager():
    """Get the C++ allocator manager instance"""
    return gc_allocator_core.get_manager()

# Export all public APIs
__all__ = [
    "GCAllocator", "AllocationStats", "RetryStats", "RetryConfig", "AllocationFuture",
    "MemoryPressureLevel", "install", "uninstall", "get_stats", 
    "is_installed", "reset_stats", "get_allocator", "get_manager",
    "register_checkpoint"
]
