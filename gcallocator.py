import gcAllocator
import torch
import gc
import warnings
import threading
import time
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

class GCAllocatorManager:
    """Enhanced Python interface for the GC Allocator."""
    
    def __init__(self):
        try:
            self.manager = gcAllocator.gc_allocator_core.get_manager()
            if self.manager is None:
                raise RuntimeError("Failed to get manager instance")
        except Exception as e:
            warnings.warn(f"Failed to initialize GCAllocator manager: {e}")
            self.manager = None
        
        self.installed = False
        self._lock = threading.Lock()
        self._checkpoint_callbacks = []
    
    def install(self, enable_logging: bool = False, retry_config: Optional[Dict[str, Any]] = None) -> None:
        """Install the GC allocator with enhanced OOM handling.
        
        Args:
            enable_logging: Enable detailed logging
            retry_config: Dictionary with retry strategy configuration
        """
        if self.manager is None:
            raise RuntimeError("GCAllocator manager is not available")
            
        with self._lock:
            if self.installed:
                warnings.warn("GCAllocator is already installed")
                return
            
            try:
                # Configure retry strategy before installation
                if retry_config:
                    self._configure_retry_strategy_from_dict(retry_config)
                
                # Install the allocator
                self.manager.install_allocator()
                
                if enable_logging:
                    self.manager.enable_logging()
                
                # Force PyTorch to reinitialize CUDA context with our allocator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Warm up the allocator with a small allocation
                    try:
                        test_tensor = torch.zeros(1, device='cuda')
                        del test_tensor
                        torch.cuda.empty_cache()
                    except:
                        pass  # Ignore warm-up failures
                
                self.installed = True
                print("[GCAllocator] Successfully installed with enhanced OOM handling")
                
            except Exception as e:
                warnings.warn(f"Failed to install GCAllocator: {e}")
                raise
    
    def uninstall(self) -> None:
        """Uninstall the GC allocator."""
        if self.manager is None:
            return
            
        with self._lock:
            if not self.installed:
                return
            
            try:
                # Clear all GPU memory before uninstalling
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                
                self.manager.uninstall_allocator()
                self.installed = False
                print("[GCAllocator] Successfully uninstalled")
            except Exception as e:
                warnings.warn(f"Error during uninstall: {e}")
    
    def configure_retry_strategy(self, max_retries: int = 5, initial_delay_ms: int = 50, 
                               backoff_multiplier: float = 1.5, max_delay_ms: int = 2000,
                               enable_cache_flush: bool = True, 
                               enable_gradient_checkpointing: bool = True) -> None:
        """Configure the retry strategy for OOM handling."""
        if self.manager is None:
            warnings.warn("Cannot configure retry strategy: manager not available")
            return
            
        config = gcAllocator.gc_allocator_core.RetryConfig()
        config.max_retries = max_retries
        config.initial_delay = gcAllocator.gc_allocator_core.milliseconds(initial_delay_ms)
        config.backoff_multiplier = backoff_multiplier
        config.max_delay = gcAllocator.gc_allocator_core.milliseconds(max_delay_ms)
        config.enable_cache_flush = enable_cache_flush
        config.enable_gradient_checkpointing = enable_gradient_checkpointing
        
        self.manager.configure_retry_strategy(config)
        print(f"[GCAllocator] Configured retry strategy: max_retries={max_retries}, "
              f"initial_delay={initial_delay_ms}ms, cache_flush={enable_cache_flush}")
    
    def _configure_retry_strategy_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Helper to configure retry strategy from dictionary."""
        self.configure_retry_strategy(**config_dict)
    
    def register_checkpoint_callback(self, callback: Callable[[], bool]) -> None:
        """Register a gradient checkpointing callback for memory recovery."""
        if self.manager is None:
            warnings.warn("Cannot register callback: manager not available")
            return
            
        self._checkpoint_callbacks.append(callback)
        self.manager.register_checkpoint_callback(callback)
    
    def get_retry_stats(self) -> Dict[str, int]:
        """Get retry statistics as a dictionary."""
        if self.manager is None:
            return {'total_retry_attempts': 0, 'cache_flushes': 0, 
                   'checkpoint_activations': 0, 'successful_recoveries': 0}
            
        try:
            stats = self.manager.get_retry_stats()
            return {
                'total_retry_attempts': stats.get_total_retry_attempts(),
                'cache_flushes': stats.get_cache_flushes(),
                'checkpoint_activations': stats.get_checkpoint_activations(),
                'successful_recoveries': stats.get_successful_recoveries()
            }
        except Exception as e:
            warnings.warn(f"Failed to get retry stats: {e}")
            return {'total_retry_attempts': 0, 'cache_flushes': 0, 
                   'checkpoint_activations': 0, 'successful_recoveries': 0}
    
    def reset_retry_stats(self) -> None:
        """Reset retry statistics."""
        if self.manager is not None:
            self.manager.reset_retry_stats()
    
    def print_stats(self) -> None:
        """Print current retry statistics."""
        stats = self.get_retry_stats()
        print("[GCAllocator] Retry Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    @contextmanager
    def temporary_install(self, **kwargs):
        """Context manager for temporary allocator installation."""
        self.install(**kwargs)
        try:
            yield self
        finally:
            self.uninstall()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstall()

# Global manager instance
_allocator_manager = None

def get_allocator_manager() -> GCAllocatorManager:
    """Get the global allocator manager instance."""
    global _allocator_manager
    if _allocator_manager is None:
        _allocator_manager = GCAllocatorManager()
    return _allocator_manager

def install_allocator(enable_logging: bool = False, **retry_config) -> GCAllocatorManager:
    """Install the GC allocator with enhanced configuration."""
    manager = get_allocator_manager()
    if retry_config:
        manager.configure_retry_strategy(**retry_config)
    manager.install(enable_logging=enable_logging)
    return manager

def uninstall_allocator() -> None:
    """Uninstall the GC allocator."""
    global _allocator_manager
    if _allocator_manager:
        _allocator_manager.uninstall()

@contextmanager
def gc_allocator(enable_logging: bool = False, **retry_config):
    """Context manager for using GC allocator."""
    manager = install_allocator(enable_logging=enable_logging, **retry_config)
    try:
        yield manager
    finally:
        uninstall_allocator()

# Utility functions for common use cases
def create_checkpoint_callback(model: torch.nn.Module) -> Callable[[], bool]:
    """Create a checkpoint callback for a PyTorch model."""
    def checkpoint_callback():
        try:
            # Clear gradients
            model.zero_grad()
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
        except Exception:
            return False
    
    return checkpoint_callback

def aggressive_memory_cleanup():
    """Perform aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
