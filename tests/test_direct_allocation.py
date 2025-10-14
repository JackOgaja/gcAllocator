#!/usr/bin/env python3
"""
Direct test of allocator methods bypassing PyTorch tensor creation
"""

import os
import sys
import ctypes
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable instrumentation
os.environ['GC_ALLOCATOR_INSTRUMENT'] = '1'
os.environ['GC_ALLOCATOR_LOG'] = '1'

import gcAllocator
import gc_allocator_core

def test_direct_cpp_allocation():
    """Test allocation directly through C++ bindings"""
    print("\n" + "="*60)
    print("DIRECT C++ ALLOCATION TEST")
    print("="*60)
    
    # Get the manager
    manager = gc_allocator_core.get_manager()
    print(f"Manager instance: {manager}")
    
    # Install allocator
    print("\n1. Installing allocator...")
    manager.install_allocator()
    print(f"Is installed: {manager.is_installed()}")
    
    # Get the allocator instance
    allocator = manager.get_allocator()
    print(f"Allocator instance: {allocator}")
    
    if allocator is None:
        print("ERROR: Allocator is None after installation!")
        return
    
    # Check wrapped allocator status
    print("\n2. Checking wrapped allocator...")
    # Note: We need to expose getWrappedAllocator in bindings
    
    # Get stats before
    print("\n3. Initial statistics:")
    stats = gc_allocator_core.get_stats()
    print(f"  Total allocations: {stats.get_total_allocations()}")
    print(f"  Total requests: {stats.get_total_requests()}")
    print(f"  Current bytes: {stats.get_current_bytes_allocated()}")
    
    # Try to trigger allocation through tensor
    print("\n4. Creating tensor to trigger allocation...")
    try:
        tensor = torch.zeros(1000, 1000, device='cuda')
        print(f"  Created tensor: {tensor.shape}")
        print(f"  Data pointer: {tensor.data_ptr()}")
        
        # Manual tracking (if allocate wasn't called)
        if stats.get_total_allocations() == 0:
            print("\n  WARNING: Allocation not tracked automatically")
            print("  Attempting manual tracking...")
            gc_allocator_core.track_allocation(
                tensor.data_ptr(), 
                tensor.numel() * tensor.element_size(), 
                tensor.device.index or 0
            )
    except Exception as e:
        print(f"  Error creating tensor: {e}")
    
    # Get stats after
    print("\n5. Statistics after allocation:")
    stats = gc_allocator_core.get_stats()
    print(f"  Total allocations: {stats.get_total_allocations()}")
    print(f"  Total requests: {stats.get_total_requests()}")
    print(f"  Current bytes: {stats.get_current_bytes_allocated()}")
    
    # Uninstall
    print("\n6. Uninstalling allocator...")
    manager.uninstall_allocator()
    print("Test complete")

def test_allocator_configuration():
    """Test allocator configuration and retry settings"""
    print("\n" + "="*60)
    print("ALLOCATOR CONFIGURATION TEST")
    print("="*60)
    
    # Create custom configuration
    config = gcAllocator.RetryConfig(
        max_attempts=10,
        initial_backoff_ms=50,
        backoff_multiplier=1.5,
        max_backoff_ms=2000,
        enable_cache_flush=True,
        enable_checkpointing=False
    )
    
    print(f"Configuration: {config}")
    
    # Create allocator with config
    allocator = gcAllocator.GCAllocator(config=config, enable_logging=True)
    
    print("\n1. Installing with custom config...")
    allocator.install()
    
    # Verify installation
    assert allocator.is_installed
    print("✓ Allocator installed")
    
    # Get initial stats
    stats = allocator.get_stats()
    print(f"\n2. Initial stats:")
    print(f"  Requests: {stats.total_requests}")
    print(f"  Allocations: {stats.total_allocations}")
    
    # Force an allocation
    print("\n3. Forcing allocation...")
    small_tensor = torch.randn(10, 10, device='cuda')
    
    # Check stats
    import time
    time.sleep(0.1)  # Allow propagation
    
    stats = allocator.get_stats()
    print(f"\n4. Stats after allocation:")
    print(f"  Requests: {stats.total_requests}")
    print(f"  Allocations: {stats.total_allocations}")
    
    if stats.total_requests == 0:
        print("\n⚠ WARNING: Allocation not going through proxy!")
        print("  This confirms the proxy pattern is not intercepting allocations")
    else:
        print("\n✓ Allocation successfully tracked")
    
    # Check retry stats
    retry_stats = allocator.get_retry_stats()
    print(f"\n5. Retry stats:")
    print(f"  Attempts: {retry_stats.total_retry_attempts}")
    print(f"  Recoveries: {retry_stats.successful_recoveries}")
    
    allocator.uninstall()
    print("\nTest complete")

def test_allocation_lifecycle():
    """Test the complete allocation lifecycle"""
    print("\n" + "="*60)
    print("ALLOCATION LIFECYCLE TEST")
    print("="*60)
    
    with gcAllocator.allocator_context(enable_logging=True) as allocator:
        print("1. Allocator installed via context manager")
        
        # Reset stats for clean test
        allocator.reset_stats()
        print("2. Stats reset")
        
        # Multiple allocations
        tensors = []
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            print(f"\n3. Allocating tensor of size {size}x{size}")
            t = torch.zeros(size, size, device='cuda')
            tensors.append(t)
            
            stats = allocator.get_stats()
            print(f"   Current allocations: {stats.total_allocations}")
            print(f"   Current bytes: {stats.current_bytes_allocated / (1024**2):.2f} MB")
        
        # Peak memory check
        stats = allocator.get_stats()
        print(f"\n4. Peak memory: {stats.peak_bytes_allocated / (1024**2):.2f} MB")
        
        # Deallocate
        print("\n5. Deallocating tensors...")
        tensors.clear()
        torch.cuda.empty_cache()
        
        stats = allocator.get_stats()
        print(f"   Deallocations: {stats.total_deallocations}")
        print(f"   Current bytes: {stats.current_bytes_allocated / (1024**2):.2f} MB")
    
    print("\n6. Allocator uninstalled via context manager")
    print("Test complete")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available - cannot test")
        sys.exit(1)
    
    print("="*60)
    print("GCALLOCATOR DIRECT TESTING SUITE")
    print("="*60)
    
    test_direct_cpp_allocation()
    test_allocator_configuration()
    test_allocation_lifecycle()
    
    print("\n" + "="*60)
    print("ANALYSIS OF TEST RESULTS")
    print("="*60)
    print("""
    Expected Behavior Pattern:
    - Allocator should intercept all CUDA allocations
    - Statistics should increment with each allocation
    - Wrapped allocator should be non-null
    
    Actual Behavior (if proxy fails):
    - Statistics remain at zero
    - Allocations bypass the proxy
    - Wrapped allocator is null
    
    Root Cause Indicators:
    - Check if wrapped_allocator_ is properly set
    - Verify c10::SetAllocator actually updates PyTorch
    - Confirm allocate() method is being called
    """)
