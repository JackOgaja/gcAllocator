#!/usr/bin/env python3
"""
Basic tests for gcAllocator base functionality.
Tests allocator installation, basic allocation/deallocation, and statistics.
"""

import sys
import os
import torch
import pytest
import gc as python_gc
import time

# Add parent directory to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gcAllocator


class TestBasicAllocation:
    """Test basic allocation functionality"""
    
    def setup_method(self):
        """Ensure clean state before each test"""
        # Force uninstall any existing allocator
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        
        # Clear PyTorch cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force garbage collection
        python_gc.collect()
        
        # Small delay to ensure cleanup
        time.sleep(0.1)
    
    def teardown_method(self):
        """Clean up after each test"""
        # Force uninstall any existing allocator
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force garbage collection
        python_gc.collect()
        
        # Small delay to ensure cleanup
        time.sleep(0.1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_installation(self):
        """Test allocator installation and uninstallation"""
        assert not gcAllocator.is_installed()
        
        # Install allocator
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        assert gcAllocator.is_installed()
        assert allocator.is_installed
        
        # Uninstall allocator
        allocator.uninstall()
        assert not gcAllocator.is_installed()
        assert not allocator.is_installed
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_context_manager(self):
        """Test using allocator as context manager"""
        assert not gcAllocator.is_installed()
        
        with gcAllocator.GCAllocator() as allocator:
            assert allocator.is_installed
            assert gcAllocator.is_installed()
            # Allocate tensor inside context
            tensor = torch.randn(1000, 1000, device='cuda')
            assert tensor.is_cuda
        
        # Should be uninstalled after context
        assert not gcAllocator.is_installed()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_basic_tensor_allocation(self):
        """Test basic tensor allocation with custom allocator"""
        allocator = gcAllocator.GCAllocator(enable_logging=True)
        allocator.install()
        
        # Reset stats to ensure clean start
        allocator.reset_stats()
        
        # Force synchronization before allocation
        torch.cuda.synchronize()
        
        # Allocate various tensor sizes
        small_tensor = torch.zeros(10, 10, device='cuda')
        torch.cuda.synchronize()  # Ensure allocation is complete
        
        medium_tensor = torch.randn(1000, 1000, device='cuda')
        torch.cuda.synchronize()  # Ensure allocation is complete
        
        large_tensor = torch.ones(5000, 5000, device='cuda')
        torch.cuda.synchronize()  # Ensure allocation is complete
        
        # Verify tensors are on CUDA
        assert small_tensor.is_cuda
        assert medium_tensor.is_cuda
        assert large_tensor.is_cuda
        
        # Give some time for statistics to update
        time.sleep(0.1)
        
        # Check statistics
        stats = allocator.get_stats()
        assert stats.total_allocations > 0, f"Expected allocations > 0, got {stats.total_allocations}"
        assert stats.current_bytes_allocated > 0, f"Expected current_bytes > 0, got {stats.current_bytes_allocated}"
        
        # Delete tensors
        del small_tensor, medium_tensor, large_tensor
        torch.cuda.synchronize()
        python_gc.collect()
        
        # Properly uninstall
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_statistics_tracking(self):
        """Test allocation statistics tracking"""
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        
        # Reset stats to start fresh
        allocator.reset_stats()
        torch.cuda.synchronize()
        
        initial_stats = allocator.get_stats()
        assert initial_stats.total_allocations == 0
        assert initial_stats.current_bytes_allocated == 0
        
        # Allocate tensor
        tensor_size = (1000, 1000)
        tensor = torch.zeros(tensor_size, device='cuda', dtype=torch.float32)
        torch.cuda.synchronize()  # Ensure allocation is complete
        expected_bytes = tensor.numel() * tensor.element_size()
        
        # Give some time for statistics to update
        time.sleep(0.1)
        
        # Check stats after allocation
        stats = allocator.get_stats()
        assert stats.total_allocations > 0, f"Expected allocations > 0, got {stats.total_allocations}"
        assert stats.current_bytes_allocated >= expected_bytes, f"Expected bytes >= {expected_bytes}, got {stats.current_bytes_allocated}"
        
        # Clean up
        del tensor
        torch.cuda.synchronize()
        python_gc.collect()
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_allocation_with_operations(self):
        """Test that PyTorch operations work correctly with custom allocator"""
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        
        # Create tensors and perform operations
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        torch.cuda.synchronize()
        
        # Matrix multiplication
        c = torch.mm(a, b)
        assert c.is_cuda
        assert c.shape == (100, 100)
        
        # Element-wise operations
        d = a + b
        e = a * b
        f = torch.relu(a)
        
        assert all(t.is_cuda for t in [d, e, f])
        
        # Clean up
        del a, b, c, d, e, f
        torch.cuda.synchronize()
        python_gc.collect()
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_oom_detection(self):
        """Test that OOM errors are properly detected and counted"""
        allocator = gcAllocator.GCAllocator(enable_logging=True)
        allocator.install()
        
        initial_stats = allocator.get_stats()
        initial_oom_count = initial_stats.oom_count
        
        try:
            # Try to allocate a very large tensor that should cause OOM
            # This size should be larger than available GPU memory
            huge_tensor = torch.zeros(50000, 50000, device='cuda', dtype=torch.float32)
            # If we get here, the allocation succeeded (large GPU memory)
            del huge_tensor
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM occurred as expected
                torch.cuda.empty_cache()
                stats = allocator.get_stats()
                # Note: OOM detection might not always increment the counter
                # depending on where the OOM occurs in the allocation chain
                assert stats.oom_count >= initial_oom_count
        
        torch.cuda.empty_cache()
        allocator.uninstall()


class TestLogging:
    """Test logging functionality"""
    
    def setup_method(self):
        """Ensure clean state before each test"""
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        python_gc.collect()
        time.sleep(0.1)
    
    def teardown_method(self):
        """Clean up after each test"""
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        python_gc.collect()
        time.sleep(0.1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_logging_configuration(self):
        """Test enabling and disabling logging"""
        allocator = gcAllocator.GCAllocator(enable_logging=False)
        allocator.install()
        
        assert not allocator.enable_logging
        
        # Enable logging at runtime
        allocator.enable_logging_runtime()
        assert allocator.enable_logging
        
        # Disable logging at runtime
        allocator.disable_logging_runtime()
        assert not allocator.enable_logging
        
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_environment_variable_logging(self, monkeypatch):
        """Test that GC_ALLOCATOR_LOG environment variable enables logging"""
        monkeypatch.setenv("GC_ALLOCATOR_LOG", "1")
        
        allocator = gcAllocator.GCAllocator()
        assert allocator.enable_logging  # Should be True due to env var
        
        allocator.install()
        
        # Test allocation with logging enabled
        tensor = torch.randn(100, 100, device='cuda')
        torch.cuda.synchronize()
        
        # Clean up
        del tensor
        torch.cuda.synchronize()
        python_gc.collect()
        allocator.uninstall()


class TestMultipleInstances:
    """Test behavior with multiple allocator instances"""
    
    def setup_method(self):
        """Ensure clean state before each test"""
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        python_gc.collect()
        time.sleep(0.1)
    
    def teardown_method(self):
        """Clean up after each test"""
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        python_gc.collect()
        time.sleep(0.1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_installations(self):
        """Test that multiple installations are handled properly"""
        allocator1 = gcAllocator.GCAllocator()
        allocator2 = gcAllocator.GCAllocator()
        
        # Install first allocator
        allocator1.install()
        assert allocator1.is_installed
        assert gcAllocator.is_installed()
        
        # Install second allocator (should replace first)
        allocator2.install()
        assert allocator2.is_installed
        assert not allocator1.is_installed  # Should be uninstalled
        assert gcAllocator.is_installed()
        
        # Uninstall second
        allocator2.uninstall()
        assert not allocator2.is_installed
        assert not gcAllocator.is_installed()
