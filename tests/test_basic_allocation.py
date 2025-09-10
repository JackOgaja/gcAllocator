#!/usr/bin/env python3
"""
Basic tests for gcAllocator Phase 1 functionality.
Tests allocator installation, basic allocation/deallocation, and statistics.
"""

import sys
import os
import torch
import pytest
import gc as python_gc

# Add parent directory to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gcAllocator


class TestBasicAllocation:
    """Test basic allocation functionality"""
    
    def setup_method(self):
        """Ensure clean state before each test"""
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def teardown_method(self):
        """Clean up after each test"""
        if gcAllocator.is_installed():
            gcAllocator.uninstall()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_installation(self):
        """Test allocator installation and uninstallation"""
        assert not gcAllocator.is_installed()
        
        # Install allocator
        gcAllocator.install()
        assert gcAllocator.is_installed()
        
        # Uninstall allocator
        gcAllocator.uninstall()
        assert not gcAllocator.is_installed()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_context_manager(self):
        """Test using allocator as context manager"""
        assert not gcAllocator.is_installed()
        
        with gcAllocator.GCAllocator() as allocator:
            assert allocator.is_installed
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
        
        # Allocate various tensor sizes
        small_tensor = torch.zeros(10, 10, device='cuda')
        medium_tensor = torch.randn(1000, 1000, device='cuda')
        large_tensor = torch.ones(5000, 5000, device='cuda')
        
        # Verify tensors are on CUDA
        assert small_tensor.is_cuda
        assert medium_tensor.is_cuda
        assert large_tensor.is_cuda
        
        # Check statistics
        stats = allocator.get_stats()
        assert stats.total_allocations > 0
        assert stats.current_bytes_allocated > 0
        
        # Delete tensors
        del small_tensor, medium_tensor, large_tensor
        torch.cuda.synchronize()
        python_gc.collect()
        
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_statistics_tracking(self):
        """Test allocation statistics tracking"""
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        
        # Reset stats to start fresh
        allocator.reset_stats()
        
        initial_stats = allocator.get_stats()
        assert initial_stats.total_allocations == 0
        assert initial_stats.current_bytes_allocated == 0
        
        # Allocate tensor
        tensor_size = (1000, 1000)
        tensor = torch.zeros(tensor_size, device='cuda', dtype=torch.float32)
        expected_bytes = tensor.numel() * tensor.element_size()
        
        # Check stats after allocation
        stats = allocator.get_stats()
        assert stats.total_allocations > 0
        assert stats.current_bytes_allocated >= expected_bytes
        assert stats.peak_bytes_allocated >= expected_bytes
        
        # Delete tensor
        del tensor
        torch.cuda.synchronize()
        python_gc.collect()
        
        # Check stats after deallocation
        final_stats = allocator.get_stats()
        assert final_stats.total_deallocations > 0
        
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_devices(self):
        """Test allocation on multiple CUDA devices if available"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple CUDA devices not available")
        
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        
        tensors = []
        for device_id in range(min(2, torch.cuda.device_count())):
            tensor = torch.randn(1000, 1000, device=f'cuda:{device_id}')
            tensors.append(tensor)
        
        # Check that we have allocations on multiple devices
        stats = allocator.get_stats()
        active_devices = stats.active_devices
        assert len(active_devices) >= 2
        
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_allocation_with_operations(self):
        """Test that PyTorch operations work correctly with custom allocator"""
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        
        # Perform various operations
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        # Matrix multiplication
        c = torch.matmul(a, b)
        assert c.shape == (1000, 1000)
        
        # In-place operations
        a.add_(1.0)
        
        # Tensor creation from operations
        d = a + b
        e = torch.relu(d)
        
        # Verify operations completed
        torch.cuda.synchronize()
        
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_oom_detection(self):
        """Test that OOM errors are properly detected and counted"""
        allocator = gcAllocator.GCAllocator(enable_logging=True)
        allocator.install()
        
        initial_oom_count = allocator.get_stats().oom_count
        
        try:
            # Try to allocate more memory than available
            # This should trigger an OOM error
            huge_tensor = torch.zeros(
                100000, 100000, 
                device='cuda', 
                dtype=torch.float32
            )
        except torch.cuda.OutOfMemoryError:
            # Expected behavior for Phase 1 - OOM should still raise
            pass
        
        # Check that OOM was counted
        stats = allocator.get_stats()
        assert stats.oom_count > initial_oom_count
        
        allocator.uninstall()


class TestLogging:
    """Test logging functionality"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_logging_configuration(self):
        """Test enabling and disabling logging"""
        allocator = gcAllocator.GCAllocator(enable_logging=False)
        allocator.install()
        
        # Initially disabled
        assert not allocator.enable_logging
        
        # Enable logging
        allocator.set_logging(True)
        assert allocator.enable_logging
        
        # Disable logging
        allocator.set_logging(False)
        assert not allocator.enable_logging
        
        allocator.uninstall()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_environment_variable_logging(self, monkeypatch):
        """Test that GC_ALLOCATOR_LOG environment variable enables logging"""
        monkeypatch.setenv("GC_ALLOCATOR_LOG", "1")
        
        allocator = gcAllocator.GCAllocator()
        allocator.install()
        
        # Should detect environment variable
        # Note: This is set in C++ constructor, so we can't directly test
        # but we can verify it doesn't crash
        tensor = torch.zeros(100, 100, device='cuda')
        del tensor
        
        allocator.uninstall()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
