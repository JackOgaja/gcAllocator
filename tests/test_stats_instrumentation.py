#!/usr/bin/env python3
# tests/test_stats_instrumentation.py
"""
Test script to verify statistics tracking with instrumentation enabled
"""

import os
import sys
import torch
import time

# Enable instrumentation via environment variable
os.environ['GC_ALLOCATOR_INSTRUMENT'] = '1'
os.environ['GC_ALLOCATOR_LOG'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gcAllocator

def test_stats_instrumentation():
    """Test with instrumentation to trace statistics updates"""
    print("=" * 60)
    print("STATISTICS INSTRUMENTATION TEST")
    print("=" * 60)
    
    # Install allocator
    print("\n1. Installing allocator...")
    allocator = gcAllocator.GCAllocator(enable_logging=True)
    allocator.install()
    
    # Get initial stats
    print("\n2. Getting initial stats...")
    stats = allocator.get_stats()
    print(f"Initial stats - requests: {stats.total_requests}, allocations: {stats.total_allocations}")
    
    # Perform allocation
    print("\n3. Allocating tensor...")
    tensor1 = torch.zeros(1000, 1000, device='cuda')
    
    # Wait for stats to propagate
    time.sleep(0.1)
    
    # Get stats after allocation
    print("\n4. Getting stats after allocation...")
    stats = allocator.get_stats()
    print(f"After allocation - requests: {stats.total_requests}, allocations: {stats.total_allocations}")
    
    # Perform another allocation
    print("\n5. Allocating another tensor...")
    tensor2 = torch.randn(500, 500, device='cuda')
    
    time.sleep(0.1)
    
    # Get final stats
    print("\n6. Getting final stats...")
    stats = allocator.get_stats()
    print(f"Final stats - requests: {stats.total_requests}, allocations: {stats.total_allocations}")
    
    # Cleanup
    del tensor1, tensor2
    torch.cuda.empty_cache()
    
    print("\n7. After cleanup...")
    stats = allocator.get_stats()
    print(f"After cleanup - deallocations: {stats.total_deallocations}")
    
    allocator.uninstall()
    
    print("\n" + "=" * 60)
    print("INSTRUMENTATION ANALYSIS:")
    print("Check the output above to verify:")
    print("1. Same instance ID throughout the test")
    print("2. Recording methods called on correct instance")
    print("3. Stats values actually incrementing")
    print("=" * 60)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available - cannot test")
        sys.exit(1)
    
    test_stats_instrumentation()
