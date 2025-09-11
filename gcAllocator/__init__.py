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
gcAllocator - Graceful CUDA Allocator for PyTorch
A custom memory allocator that handles OOM conditions gracefully.
"""

import os
import sys
import warnings
from typing import Optional
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

__author__ = "Jack Ogaja"
__version__ = "0.1.0"
__all__ = ["GCAllocator", "AllocationStats", "install", "uninstall", "get_stats", "is_installed", "reset_stats"]

# Global state management
_global_allocator_instance = None
_is_globally_installed = False


class AllocationStats:
    """Wrapper for allocation statistics from C++"""
    
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


class GCAllocator:
    """Main interface for the graceful CUDA allocator"""
    
    def __init__(self, enable_logging: bool = False):
        """
        Initialize the GCAllocator.
        
        Args:
            enable_logging: Whether to enable detailed logging of allocations
        """
        # Check environment variable for logging
        env_logging = os.environ.get("GC_ALLOCATOR_LOG", "0")
        self.enable_logging = enable_logging or env_logging.lower() in ("1", "true", "yes", "on")
        self._installed = False
        
        # Check what logging functions are available in the C++ extension
        self._has_enable_logging = hasattr(gc_allocator_core, 'enable_logging')
        self._has_disable_logging = hasattr(gc_allocator_core, 'disable_logging')
    
    def install(self):
        """Install the GCAllocator as the default CUDA allocator"""
        global _global_allocator_instance, _is_globally_installed
        
        if self._installed:
            warnings.warn("GCAllocator is already installed for this instance")
            return
        
        if _is_globally_installed:
            # Properly handle reinstallation by uninstalling first
            if _global_allocator_instance and _global_allocator_instance != self:
                _global_allocator_instance.uninstall()
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GCAllocator requires CUDA support.")

        try:
            # Install the allocator
            gc_allocator_core.install_allocator()
            
            # Configure logging if available
            self._configure_logging()
            
            self._installed = True
            _global_allocator_instance = self
            _is_globally_installed = True
            
            if self.enable_logging:
                print("GCAllocator v{} installed successfully".format(__version__))
                
        except RuntimeError as e:
            if "already installed" in str(e).lower():
                # Force reinstallation by first uninstalling
                try:
                    gc_allocator_core.uninstall_allocator()
                    gc_allocator_core.install_allocator()
                    
                    # Configure logging if available
                    self._configure_logging()
                    
                    self._installed = True
                    _global_allocator_instance = self
                    _is_globally_installed = True
                    
                    if self.enable_logging:
                        print("GCAllocator v{} reinstalled successfully".format(__version__))
                except Exception as reinstall_error:
                    raise RuntimeError(f"Failed to reinstall allocator: {reinstall_error}")
            else:
                raise
    
    def _configure_logging(self):
        """Configure logging if the C++ extension supports it"""
        try:
            if self.enable_logging and self._has_enable_logging:
                gc_allocator_core.enable_logging()
            elif not self.enable_logging and self._has_disable_logging:
                gc_allocator_core.disable_logging()
        except Exception as e:
            warnings.warn(f"Failed to configure logging: {e}")
    
    def uninstall(self):
        """Uninstall the GCAllocator and restore PyTorch's default allocator"""
        global _global_allocator_instance, _is_globally_installed
        
        if not self._installed:
            return
        
        try:
            gc_allocator_core.uninstall_allocator()
            self._installed = False
            
            if _global_allocator_instance == self:
                _global_allocator_instance = None
                _is_globally_installed = False
                
            if self.enable_logging:
                print("GCAllocator uninstalled successfully")
                
        except Exception as e:
            warnings.warn(f"Error during uninstall: {e}")
            # Force reset global state even if C++ uninstall fails
            self._installed = False
            if _global_allocator_instance == self:
                _global_allocator_instance = None
                _is_globally_installed = False
    
    @property
    def is_installed(self) -> bool:
        """Check if this allocator instance is currently installed"""
        return self._installed
    
    def get_stats(self) -> AllocationStats:
        """Get current allocation statistics"""
        if not self._installed:
            warnings.warn("Getting stats from non-installed allocator")
        
        cpp_stats = gc_allocator_core.get_stats()
        return AllocationStats(cpp_stats)
    
    def reset_stats(self):
        """Reset allocation statistics"""
        if hasattr(gc_allocator_core, 'reset_stats'):
            gc_allocator_core.reset_stats()
        else:
            warnings.warn("reset_stats not available in C++ extension")
    
    def enable_logging_runtime(self):
        """Enable logging at runtime"""
        self.enable_logging = True
        if self._installed:
            self._configure_logging()
    
    def disable_logging_runtime(self):
        """Disable logging at runtime"""
        self.enable_logging = False
        if self._installed:
            self._configure_logging()
    
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
            pass  # Ignore errors during destruction


# Global convenience functions
def install(enable_logging: bool = False) -> GCAllocator:
    """Install GCAllocator globally and return the instance"""
    allocator = GCAllocator(enable_logging=enable_logging)
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
