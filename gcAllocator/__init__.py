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
gcAllocator - Graceful CUDA Allocator for PyTorch with Proxy Pattern
Complete Python interface with full proxy pattern support and statistics tracking.
"""

import os
import sys
import json
import warnings
import threading
import asyncio
import weakref
import functools
import logging
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import IntEnum
from contextlib import contextmanager
import torch

### JO====== TEST===
import pprint
### JO -------------

# Import the C++ extension
try:
    from . import gc_allocator_core
except ImportError as e:
    warnings.warn(
        f"Failed to import gc_allocator_core C++ extension: {e}\n"
        "Please ensure the module is properly built using setup.py"
    )
    raise

__version__ = "0.3.0"  # Updated for proxy pattern

# Configure logging
logger = logging.getLogger("gcAllocator")
logger.setLevel(logging.INFO)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Global state management
_global_allocator_instance: Optional['GCAllocator'] = None
_is_globally_installed: bool = False
_allocator_lock = threading.Lock()


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
        if self.max_backoff_ms < self.initial_backoff_ms:
            raise ValueError("max_backoff_ms must be >= initial_backoff_ms")
            
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
                    logger.warning(f"Invalid value for {env_var}: {value}")
        
        return cls(**config)
    
    def to_cpp_config(self) -> Any:
        """Convert to C++ RetryConfig"""
        return gc_allocator_core.create_retry_config({
            'max_retries': self.max_attempts,
            'initial_delay_ms': self.initial_backoff_ms,
            'backoff_multiplier': self.backoff_multiplier,
            'max_delay_ms': self.max_backoff_ms,
            'enable_cache_flush': self.enable_cache_flush,
            'enable_checkpointing': self.enable_checkpointing
        })
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return asdict(self)


class RetryStats:
    """Enhanced wrapper for retry statistics with proxy pattern support"""
    
    def __init__(self, cpp_stats=None):
        """Initialize RetryStats wrapper"""
        if cpp_stats is None:
            self._stats = self._create_empty_stats()
        else:
            self._stats = cpp_stats
    
    def _create_empty_stats(self):
        """Create empty stats object"""
        class EmptyStats:
            def get_total_retry_attempts(self): return 0
            def get_cache_flushes(self): return 0
            def get_checkpoint_activations(self): return 0
            def get_successful_recoveries(self): return 0
            def get_exhausted_retries(self): return 0
            def get_terminal_failures(self): return 0
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
    
    @property
    def exhausted_retries(self) -> int:
        try:
            return self._stats.get_exhausted_retries()
        except AttributeError:
            return 0
    
    @property
    def terminal_failures(self) -> int:
        try:
            return self._stats.get_terminal_failures()
        except AttributeError:
            return 0
    
    def reset(self):
        """Reset all retry statistics"""
        self._stats.reset()
    
    def to_dict(self) -> dict:
        """Export statistics as dictionary"""
        return {
            'total_retry_attempts': self.total_retry_attempts,
            'cache_flushes': self.cache_flushes,
            'checkpoint_activations': self.checkpoint_activations,
            'successful_recoveries': self.successful_recoveries,
            'exhausted_retries': self.exhausted_retries,
            'terminal_failures': self.terminal_failures,
        }
    
    def __str__(self) -> str:
        return (f"RetryStats(attempts={self.total_retry_attempts}, "
                f"flushes={self.cache_flushes}, "
                f"checkpoints={self.checkpoint_activations}, "
                f"recoveries={self.successful_recoveries})")
    
    def __repr__(self) -> str:
        return f"<RetryStats attempts={self.total_retry_attempts} recoveries={self.successful_recoveries}>"


class AllocationStats:
    """Enhanced allocation statistics wrapper with proxy pattern support"""
    
    def __init__(self, cpp_stats):
        self._stats = cpp_stats
    
    # Core statistics properties
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
    
    # Proxy pattern statistics
    @property
    def total_requests(self) -> int:
        return self._stats.get_total_requests()
    
    @property
    def cache_hits(self) -> int:
        return self._stats.get_cache_hits()
    
    @property
    def cache_flushes(self) -> int:
        return self._stats.get_cache_flushes()
    
    @property
    def stream_events(self) -> int:
        return self._stats.get_stream_events()
    
    @property
    def cache_hit_rate(self) -> float:
        return self._stats.get_cache_hit_rate()
    
    @property
    def active_devices(self) -> list:
        return self._stats.get_active_devices()
    
    # Computed properties
    @property
    def actual_allocations(self) -> int:
        """Number of actual CUDA allocations (excluding cache hits)"""
        return self.total_requests - self.cache_hits
    
    @property
    def memory_efficiency(self) -> float:
        """Ratio of current to total allocated memory"""
        if self.total_bytes_allocated == 0:
            return 0.0
        return self.current_bytes_allocated / self.total_bytes_allocated
    
    @property
    def allocation_overhead(self) -> float:
        """Average bytes per allocation"""
        if self.total_allocations == 0:
            return 0.0
        return self.total_bytes_allocated / self.total_allocations
    
    def reset(self):
        """Reset all statistics"""
        self._stats.reset()
    
    def get_device_stats(self, device: int) -> dict:
        """Get statistics for specific device"""
        try:
            # Assuming C++ returns device stats
            return self._stats.get_device_stats(device)
        except AttributeError:
            return {}
    
    def to_dict(self) -> dict:
        """Export statistics as dictionary"""
        return {
            'total_allocations': self.total_allocations,
            'total_deallocations': self.total_deallocations,
            'total_bytes_allocated': self.total_bytes_allocated,
            'current_bytes_allocated': self.current_bytes_allocated,
            'peak_bytes_allocated': self.peak_bytes_allocated,
            'oom_count': self.oom_count,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_flushes': self.cache_flushes,
            'stream_events': self.stream_events,
            'cache_hit_rate': self.cache_hit_rate,
            'actual_allocations': self.actual_allocations,
            'memory_efficiency': self.memory_efficiency,
            'active_devices': self.active_devices,
        }
    
    def __str__(self) -> str:
        return str(self._stats)
    
    def __repr__(self) -> str:
        return (f"<AllocationStats requests={self.total_requests} "
                f"cache_hits={self.cache_hits} "
                f"current_mb={self.current_bytes_allocated/(1024**2):.2f}>")


class StatsAggregator:
    """Aggregates allocation and retry statistics with proxy insights"""
    
    def __init__(self, allocation_stats: AllocationStats, retry_stats: RetryStats):
        self.allocation_stats = allocation_stats
        self.retry_stats = retry_stats
        self._compute_derived_metrics()
    
    def _compute_derived_metrics(self):
        """Compute derived metrics from raw stats"""
        # OOM recovery rate
        self.oom_recovery_rate = 0.0
        if self.retry_stats.total_retry_attempts > 0:
            self.oom_recovery_rate = (
                self.retry_stats.successful_recoveries / 
                self.retry_stats.total_retry_attempts
            )
        
        # Memory efficiency
        self.memory_efficiency = self.allocation_stats.memory_efficiency
        
        # Cache effectiveness
        self.cache_effectiveness = self.allocation_stats.cache_hit_rate
        
        # Allocation success rate
        total_attempts = (self.allocation_stats.total_requests + 
                         self.allocation_stats.oom_count)
        self.allocation_success_rate = 0.0
        if total_attempts > 0:
            self.allocation_success_rate = (
                self.allocation_stats.total_requests / total_attempts
            )
    
    def to_dict(self) -> dict:
        """Export all stats as dictionary"""
        return {
            'allocation': self.allocation_stats.to_dict(),
            'retry': self.retry_stats.to_dict(),
            'derived': {
                'oom_recovery_rate': self.oom_recovery_rate,
                'memory_efficiency': self.memory_efficiency,
                'cache_effectiveness': self.cache_effectiveness,
                'allocation_success_rate': self.allocation_success_rate,
            }
        }
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        return (
            f"=== Allocation Summary ===\n"
            f"Total Requests: {self.allocation_stats.total_requests:,}\n"
            f"Cache Hit Rate: {self.cache_effectiveness:.1%}\n"
            f"Current Memory: {self.allocation_stats.current_bytes_allocated/(1024**3):.2f} GB\n"
            f"Peak Memory: {self.allocation_stats.peak_bytes_allocated/(1024**3):.2f} GB\n"
            f"OOM Recovery Rate: {self.oom_recovery_rate:.1%}\n"
            f"Allocation Success Rate: {self.allocation_success_rate:.1%}\n"
        )
    
    def __repr__(self):
        return (f"<StatsAggregator cache_rate={self.cache_effectiveness:.1%} "
                f"oom_recovery={self.oom_recovery_rate:.1%}>")


class AllocationFuture:
    """Future for asynchronous allocations (preparation for async support)"""
    
    def __init__(self, size: int, device: int):
        self.size = size
        self.device = device
        self._result = None
        self._exception = None
        self._done = threading.Event()
        self._callbacks = []
        self._lock = threading.Lock()
    
    def result(self, timeout: Optional[float] = None) -> torch.Tensor:
        """Block until allocation completes"""
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
        """Attempt to cancel allocation"""
        with self._lock:
            if self._done.is_set():
                return False
            self._exception = asyncio.CancelledError()
            self._done.set()
            self._run_callbacks()
            return True
    
    def add_done_callback(self, fn: Callable) -> None:
        """Add callback for completion"""
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
        """Internal: Run registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.exception(f"Callback error: {e}")


class GCAllocator:
    """Main interface for graceful CUDA allocator with Proxy Pattern"""
    
    def __init__(self, 
                 config: RetryConfig = None,
                 enable_logging: bool = False,
                 retry_config: RetryConfig = None):
        """
        Initialize GCAllocator with proxy pattern support
        
        Args:
            config: RetryConfig object for retry behavior
            enable_logging: Enable detailed logging
            retry_config: Legacy parameter (backward compatibility)
        """
        global _global_allocator_instance
        
        # Handle backward compatibility
        if retry_config is not None and config is None:
            config = retry_config
        
        # Configuration
        self.config = config or RetryConfig()
        
        # Logging configuration
        env_logging = os.environ.get("GC_ALLOCATOR_LOG", "0")
        self.enable_logging = enable_logging or env_logging.lower() in ("1", "true")
        
        # State tracking
        self._installed = False
        self._manager = None
        self._checkpoint_callbacks = []
        self._lock = threading.Lock()
        self._is_global = False
        
        # Statistics cache for performance
        self._stats_cache = None
        self._stats_cache_time = 0
        self._stats_cache_duration = 0.1  # Cache for 100ms
        
        if self.enable_logging:
            logger.info(f"GCAllocator initialized with config: {self.config}")
    
    @property  
    def is_installed(self) -> bool:
        """Check if allocator is installed"""
        return self._installed
    
    def install(self) -> None:
        """Install allocator as PyTorch's CUDA allocator using proxy pattern"""
        global _global_allocator_instance, _is_globally_installed
        
        with self._lock:
            if self._installed:
                warnings.warn("GCAllocator already installed for this instance")
                return
            
            # Handle global state replacement
            if _is_globally_installed and _global_allocator_instance and _global_allocator_instance != self:
                logger.info("Replacing existing global allocator")
                _global_allocator_instance.uninstall()
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            try:
                # Get manager instance
                self._manager = gc_allocator_core.get_manager()
                
                # Configure retry strategy
                self._manager.configure_retry_strategy(self.config.to_cpp_config())
                
                # Install allocator with proxy pattern
                self._manager.install_allocator()
                
                # Configure logging
                if self.enable_logging:
                    self._manager.enable_logging()
                
                # Set installation state
                self._installed = True
                
                # Update global state
                _global_allocator_instance = self
                _is_globally_installed = True
                
                if self.enable_logging:
                    logger.info(f"GCAllocator v{__version__} installed (proxy mode)")
                    
            except Exception as e:
                self._installed = False
                _global_allocator_instance = None
                _is_globally_installed = False
                raise RuntimeError(f"Failed to install allocator: {e}")
    
    def uninstall(self) -> None:
        """Uninstall allocator and restore default"""
        global _global_allocator_instance, _is_globally_installed
        
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
                    logger.info("GCAllocator uninstalled")
                    
            except Exception as e:
                warnings.warn(f"Error during uninstall: {e}")
                self._installed = False
                if _global_allocator_instance == self:
                    _global_allocator_instance = None
                    _is_globally_installed = False
    
    def get_stats(self) -> AllocationStats:
        """Get current allocation statistics"""
        if not self._installed:
            warnings.warn("Getting stats from non-installed allocator")
        
        try:
            cpp_stats = gc_allocator_core.get_stats()
            ## JO++ TEST
            ##pprint.pprint(cpp_stats)
            return AllocationStats(cpp_stats)
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
    
    def get_retry_stats(self) -> RetryStats:
        """Get retry statistics"""
        if not self._installed:
            return RetryStats()
        
        try:
            manager = gc_allocator_core.get_manager()
            cpp_stats = manager.get_retry_stats()
            return RetryStats(cpp_stats)
        except Exception as e:
            logger.warning(f"Failed to get retry stats: {e}")
            return RetryStats()
    
    def get_combined_stats(self) -> StatsAggregator:
        """Get combined allocation and retry statistics"""
        try:
            alloc_stats = self.get_stats()
            retry_stats = self.get_retry_stats()
            return StatsAggregator(alloc_stats, retry_stats)
        except Exception as e:
            logger.warning(f"Failed to get combined stats: {e}")
            return StatsAggregator(
                AllocationStats(gc_allocator_core.get_stats()),
                RetryStats()
            )
    
    def reset_stats(self) -> None:
        """Reset all statistics"""
        if self._manager:
            gc_allocator_core.reset_stats()
            self._manager.reset_retry_stats()
            logger.info("Statistics reset")
    
    def configure_retry(self, config: RetryConfig) -> None:
        """Update retry configuration"""
        self.config = config
        if self._manager:
            self._manager.configure_retry_strategy(config.to_cpp_config())
            logger.info(f"Retry configuration updated: {config}")
    
    def register_checkpoint(self, callback: Callable[[], bool]) -> None:
        """Register checkpoint callback for OOM recovery"""
        self._checkpoint_callbacks.append(callback)
        if self._installed and self._manager:
            self._manager.register_checkpoint_callback(callback)
            logger.info("Checkpoint callback registered")
    
    def enable_logging_runtime(self) -> None:
        """Enable logging at runtime"""
        self.enable_logging = True
        if self._installed and self._manager:
            self._manager.enable_logging()
            logger.info("Logging enabled")
    
    def disable_logging_runtime(self) -> None:
        """Disable logging at runtime"""
        self.enable_logging = False
        if self._installed and self._manager:
            self._manager.disable_logging()
            logger.info("Logging disabled")
    
    def empty_cache(self) -> None:
        """Empty CUDA cache"""
        if self._installed:
            torch.cuda.empty_cache()
            logger.info("CUDA cache emptied")
    
    def get_memory_info(self, device: Optional[int] = None) -> dict:
        """Get detailed memory information"""
        if device is None:
            device = torch.cuda.current_device()
        
        stats = self.get_stats()
        
        # Get CUDA memory info
        free, total = torch.cuda.mem_get_info(device)
        
        return {
            'device': device,
            'total_memory': total,
            'free_memory': free,
            'allocated_by_pytorch': stats.current_bytes_allocated,
            'cached_by_pytorch': torch.cuda.memory_reserved(device),
            'peak_allocated': stats.peak_bytes_allocated,
            'cache_hit_rate': stats.cache_hit_rate,
            'oom_events': stats.oom_count,
        }
    
    def allocate_async(self, size: int, device: int = 0) -> AllocationFuture:
        """Allocate memory asynchronously (future support)"""
        if not self._installed:
            raise RuntimeError("Allocator not installed")
        
        future = AllocationFuture(size, device)
        
        def allocate():
            try:
                tensor = torch.empty(size // 4, device=f'cuda:{device}', dtype=torch.float32)
                future._set_result(tensor)
            except Exception as e:
                future._set_exception(e)
        
        threading.Thread(target=allocate, daemon=True).start()
        return future
    
    def __enter__(self):
        """Context manager entry"""
        self.install()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.uninstall()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self._installed:
                self.uninstall()
        except:
            pass
    
    def __repr__(self):
        return f"<GCAllocator installed={self._installed} logging={self.enable_logging}>"


# Global convenience functions
def install(enable_logging: bool = False, 
           config: RetryConfig = None,
           retry_config: RetryConfig = None) -> GCAllocator:
    """Install GCAllocator globally"""
    global _global_allocator_instance
    
    # Handle backward compatibility
    if retry_config and not config:
        config = retry_config
    
    with _allocator_lock:
        if _global_allocator_instance and _global_allocator_instance.is_installed:
            logger.warning("Replacing existing global allocator")
            _global_allocator_instance.uninstall()
        
        allocator = GCAllocator(config=config, enable_logging=enable_logging)
        allocator.install()
        return allocator


def uninstall():
    """Uninstall global GCAllocator"""
    global _global_allocator_instance
    
    with _allocator_lock:
        if _global_allocator_instance:
            _global_allocator_instance.uninstall()
            _global_allocator_instance = None


def is_installed() -> bool:
    """Check if GCAllocator is installed"""
    global _is_globally_installed, _global_allocator_instance
    
    if _is_globally_installed and _global_allocator_instance is not None:
        try:
            return _global_allocator_instance.is_installed
        except:
            return False
    return False


def get_allocator() -> Optional[GCAllocator]:
    """Get global allocator instance"""
    return _global_allocator_instance


def get_stats() -> Optional[AllocationStats]:
    """Get stats from global allocator"""
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        return _global_allocator_instance.get_stats()
    return None


def get_retry_stats() -> Optional[RetryStats]:
    """Get retry stats from global allocator"""
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        return _global_allocator_instance.get_retry_stats()
    return None


def get_combined_stats() -> Optional[StatsAggregator]:
    """Get combined stats from global allocator"""
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        return _global_allocator_instance.get_combined_stats()
    return None


def reset_stats():
    """Reset stats for global allocator"""
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        _global_allocator_instance.reset_stats()


def get_memory_info(device: Optional[int] = None) -> Optional[dict]:
    """Get memory info from global allocator"""
    if _global_allocator_instance and _global_allocator_instance.is_installed:
        return _global_allocator_instance.get_memory_info(device)
    return None


def empty_cache():
    """Empty CUDA cache"""
    torch.cuda.empty_cache()
    if _global_allocator_instance:
        logger.info("CUDA cache emptied")


def register_checkpoint(callback: Callable[[], bool]) -> None:
    """Register checkpoint callback"""
    if _global_allocator_instance:
        _global_allocator_instance.register_checkpoint(callback)


def get_manager():
    """Get C++ allocator manager"""
    return gc_allocator_core.get_manager()


# Context manager for temporary installation
@contextmanager
def allocator_context(config: RetryConfig = None, enable_logging: bool = False):
    """Context manager for temporary allocator installation"""
    allocator = GCAllocator(config=config, enable_logging=enable_logging)
    allocator.install()
    try:
        yield allocator
    finally:
        allocator.uninstall()


# Export public API
__all__ = [
    # Classes
    "GCAllocator",
    "AllocationStats",
    "RetryStats",
    "StatsAggregator",
    "RetryConfig",
    "AllocationFuture",
    "MemoryPressureLevel",
    
    # Functions
    "install",
    "uninstall",
    "is_installed",
    "get_allocator",
    "get_stats",
    "get_retry_stats",
    "get_combined_stats",
    "reset_stats",
    "get_memory_info",
    "empty_cache",
    "register_checkpoint",
    "get_manager",
    "allocator_context",
    
    # Version
    "__version__",
]
