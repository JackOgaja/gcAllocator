#!/usr/bin/env python3
"""
Test script for Proxy Pattern implementation of gcAllocator
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gcAllocator

def test_proxy_statistics():
    """Test that proxy pattern correctly tracks statistics"""
    print("=== Testing Proxy Pattern Statistics Tracking ===\n")
    
    # Install allocator with logging
    allocator = gcAllocator.GCAllocator(enable_logging=True)
    allocator.install()
    
    print("Allocator installed in proxy mode\n")
    
    # Reset stats for clean test
    allocator.reset_stats()
    
    # Test 1: Small allocation (likely cache hit after first)
    print("Test 1: Small repeated allocations")
    for i in range(3):
        tensor = torch.zeros(100, 100, device='cuda')
        time.sleep(0.1)  # Allow stats to update
        #time.sleep(1.0)  # Allow stats to update
        stats = allocator.get_stats()
        print(f"  Iteration {i+1}:")
        #print(f"    STATS: {stats}")
        print(f"    Total requests: {stats.total_requests}")
        print(f"    Cache hits: {stats.cache_hits}")
        print(f"    Actual allocations: {stats.total_allocations}")
        del tensor
        torch.cuda.empty_cache()
    
    print("\nTest 2: Large allocation (likely new allocation)")
    large_tensor = torch.randn(10000, 10000, device='cuda')
    time.sleep(0.1)
    stats = allocator.get_stats()
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Current memory: {stats.current_bytes_allocated / (1024**2):.2f} MB")
    print(f"  Peak memory: {stats.peak_bytes_allocated / (1024**2):.2f} MB")
    
    print("\nTest 3: Multiple tensors")
    tensors = []
    for i in range(5):
        tensors.append(torch.randn(1000, 1000, device='cuda'))
    
    time.sleep(0.1)
    stats = allocator.get_stats()
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Deallocations: {stats.total_deallocations}")
    
    # Clean up
    del tensors
    del large_tensor
    torch.cuda.empty_cache()
    
    print("\nTest 4: Final statistics")
    time.sleep(0.1)
    stats = allocator.get_stats()
    print(stats)
    
    # Get combined stats
    combined = allocator.get_combined_stats()
    print(f"\nCache hit rate: {(stats.cache_hits / stats.total_requests * 100):.1f}%")
    
    allocator.uninstall()
    print("\nProxy allocator test complete!")

def test_oom_handling():
    """Test OOM handling with proxy pattern"""
    print("\n=== Testing OOM Handling with Proxy Pattern ===\n")
    
    allocator = gcAllocator.GCAllocator(enable_logging=True)
    allocator.install()
    
    try:
        # Try to allocate more than available
        total_memory = torch.cuda.get_device_properties(0).total_memory
        oversized = int(total_memory * 2 / 4)  # Request 2x available memory
        
        print(f"Attempting to allocate {oversized / (1024**3):.2f} GB")
        try:
            tensor = torch.zeros(oversized, device='cuda', dtype=torch.float32)
            print("Unexpected: Allocation succeeded")
        except RuntimeError as e:
            print(f"Expected OOM: {e}")
            stats = allocator.get_stats()
            print(f"OOM count: {stats.oom_count}")
            
            retry_stats = allocator.get_retry_stats()
            print(f"Retry attempts: {retry_stats.total_retry_attempts}")
            print(f"Cache flushes: {retry_stats.cache_flushes}")
    
    finally:
        allocator.uninstall()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available - cannot test allocator")
        sys.exit(1)
    
    test_proxy_statistics()
    test_oom_handling()
