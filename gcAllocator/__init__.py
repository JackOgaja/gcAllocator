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
__all__ = ["GCAllocator", "AllocationStats", "install", "uninstall", "get_stats"]


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
        self.enable_logging = enable_logging
        self._installed = False
    
    def install(self):
        """Install the GCAllocator as the default CUDA allocator"""
        if self._installed:
            warnings.warn("GCAllocator is already installed")
            return
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GCAllocator requires CUDA support.")
        
        # Install the allocator
        gc_allocator_core.install_allocator()
        
        # Set logging if enabled
        if self.enable_logging:
            gc_allocator_core.set_logging_enabled(True)
        
        self._installed = True
        print(f"GCAllocator v{__version__} installed successfully")
    
    def uninstall(self):
        """Uninstall the GCAllocator and restore the original allocator"""
        if not self._installed:
            return
        
        gc_allocator_core.uninstall_allocator()
        self._installed = False
        print("GCAllocator uninstalled")
    
    def get_stats(self) -> AllocationStats:
        """Get current allocation statistics"""
        if not self._installed:
            raise RuntimeError("GCAllocator is not installed")
        
        cpp_stats = gc_allocator_core.get_stats()
        return AllocationStats(cpp_stats)
    
    def reset_stats(self):
        """Reset allocation statistics"""
        if not self._installed:
            raise RuntimeError("GCAllocator is not installed")
        
        gc_allocator_core.reset_stats()
    
    def set_logging(self, enabled: bool):
        """Enable or disable logging"""
        if self._installed:
            gc_allocator_core.set_logging_enabled(enabled)
        self.enable_logging = enabled
    
    def __enter__(self):
        """Context manager entry"""
        self.install()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.uninstall()
        return False
    
    @property
    def is_installed(self) -> bool:
        """Check if the allocator is currently installed"""
        return self._installed and gc_allocator_core.is_installed()


# Convenience functions
_default_allocator: Optional[GCAllocator] = None


def install(enable_logging: bool = False):
    """
    Install the GCAllocator globally.
    
    Args:
        enable_logging: Whether to enable detailed logging
    """
    global _default_allocator
    if _default_allocator is None:
        _default_allocator = GCAllocator(enable_logging=enable_logging)
    _default_allocator.install()


def uninstall():
    """Uninstall the GCAllocator"""
    global _default_allocator
    if _default_allocator is not None:
        _default_allocator.uninstall()


def get_stats() -> Optional[AllocationStats]:
    """Get allocation statistics from the default allocator"""
    global _default_allocator
    if _default_allocator is not None and _default_allocator.is_installed:
        return _default_allocator.get_stats()
    return None


def is_installed() -> bool:
    """Check if GCAllocator is installed"""
    return gc_allocator_core.is_installed()
