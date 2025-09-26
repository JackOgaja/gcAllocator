# gcAllocator - Graceful CUDA Allocator for PyTorch

A custom GPU memory allocator for PyTorch that handles out-of-memory (OOM) conditions gracefully through intelligent retry mechanisms and inter-process coordination.

## Core Infrastructure

This initial phase establishes the foundation for graceful OOM handling by implementing a custom allocator that intercepts PyTorch's CUDA memory allocation requests.

### Base Features

-  Custom CUDA allocator that integrates with PyTorch
-  Transparent pass-through to original PyTorch allocator
-  Comprehensive allocation statistics tracking
-  Per-device memory usage monitoring
-  OOM event detection and counting
-  Optional detailed logging of allocation events
-  Python and C++ interfaces

### Data Flow

```uml
@startuml
!theme plain
title "Solution 2: CORRECTED CUDACachingAllocator Hook - Complete Data Flow"

actor User
participant "Python Layer" as Python
participant "GCAllocatorManager" as Manager
participant "GCAllocator" as Allocator
participant "RetryStrategy" as Retry
participant "AllocationStats" as AllocStats
participant "RetryStats" as RetryStats
participant "Original\nAllocator" as Original

== Allocation Phase with Stats Updates ==
User -> Python: torch.zeros(1000,1000,'cuda')
Python -> Allocator: allocate(4MB)
Allocator -> AllocStats: recordAllocationRequest(4MB)
note right: Stats Updated:\ntotal_allocations++\nrequested_bytes += 4MB

Allocator -> Retry: executeWithRetry(lambda)
activate Retry
    Retry -> Original: allocate(4MB)
    alt OOM Occurs
        Original -->> Retry: throw OutOfMemoryError
        Retry -> RetryStats: total_retry_attempts++
        RetryStats -->> Retry: void (stats updated internally)
        Retry -> RetryStats: cache_flushes++
        RetryStats -->> Retry: void
        Retry -> Original: emptyCache()
        Retry -> Original: allocate(4MB) //retry
    end
    Original -->> Retry: DataPtr(0x7ff2f3200000)
    Retry -> RetryStats: successful_recoveries++
    RetryStats -->> Retry: void
deactivate Retry

Retry -->> Allocator: DataPtr(0x7ff2f3200000)
Allocator -> AllocStats: recordSuccessfulAllocation(4MB)
AllocStats -> AllocStats: current_bytes += 4MB\npeak_bytes = max(peak, current)
AllocStats -->> Allocator: void
Allocator -->> Python: DataPtr
Python -->> User: Tensor

== Stats Retrieval with Return Paths ==
User -> Python: get_combined_stats()
activate Python
    Python -> Manager: getAllocator()
    Manager -->> Python: GCAllocator* (0x1234)
    
    Python -> Allocator: getStats()
    activate Allocator
        Allocator -> AllocStats: getTotalAllocations()
        AllocStats -->> Allocator: 5
        Allocator -> AllocStats: getCurrentBytesAllocated()
        AllocStats -->> Allocator: 4194304
        Allocator -> AllocStats: getPeakBytesAllocated()
        AllocStats -->> Allocator: 8388608
        Allocator -> AllocStats: getOOMCount()
        AllocStats -->> Allocator: 1
        Allocator -> Allocator: Create AllocationStats copy
    deactivate Allocator
    Allocator -->> Python: AllocationStats{allocations=5,current=4MB,peak=8MB,oom=1}
    
    Python -> Allocator: getRetryStats()
    activate Allocator
        Allocator -> Retry: getStats()
        activate Retry
            Retry -> RetryStats: getTotalRetryAttempts()
            RetryStats -->> Retry: 3
            Retry -> RetryStats: getSuccessfulRecoveries()
            RetryStats -->> Retry: 2
            Retry -> RetryStats: getCacheFlushes()
            RetryStats -->> Retry: 3
            Retry -> RetryStats: getCheckpointActivations()
            RetryStats -->> Retry: 1
        deactivate Retry
        Retry -->> Allocator: RetryStats{attempts=3,recoveries=2,flushes=3,checkpoints=1}
    deactivate Allocator
    Allocator -->> Python: RetryStats (by reference)
    
    Python -> Python: Create StatsAggregator
    Python -> Python: compute_derived_metrics()
    note right: oom_recovery_rate = 2/3 = 66.7%\nmemory_efficiency = 4MB/8MB = 50%
    
deactivate Python
Python -->> User: StatsAggregator{\n  allocations: 5\n  current: 4MB\n  peak: 8MB\n  oom_events: 1\n  retry_attempts: 3\n  recoveries: 2\n  oom_recovery_rate: 66.7%\n  memory_efficiency: 50%\n}

@enduml
```

### Installation

#### Prerequisites

- PyTorch >= 1.9.0 with CUDA support
- CUDA Toolkit (matching PyTorch's CUDA version)
- C++ compiler with C++17 support
- Python >= 3.7

#### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/gcAllocator.git
cd gcAllocator

# Build and install
pip install -e .

# Or for development with debug symbols
DEBUG_BUILD=1 pip install -e .
```

### Usage

#### Basic Usage

```python
#!/usr/bin/env python3

import gcAllocator
import torch

def test_allocation_tracking():
    """Test allocation tracking"""
    print("=== Testing Allocation Tracking ===")

    # Install allocator
    allocator = gcAllocator.install(enable_logging=True)

    # Create a tensor and manually track it
    tensor = torch.randn(100000, 100000, device='cuda')

    # Get tensor info
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    device = tensor.device.index or 0

    print(f"Created tensor: ptr={hex(ptr)}, size={size}, device={device}")

    # Manually track the allocation
    gcAllocator.gc_allocator_core.track_allocation(ptr, size, device)

    # Check stats
    stats = gcAllocator.get_stats()
    print(f"After manual tracking: {stats}")

    # Delete tensor and track deallocation
    del tensor
    torch.cuda.empty_cache()

    # Manually track deallocation
    gcAllocator.gc_allocator_core.track_deallocation(ptr)

    # Check final stats
    stats = gcAllocator.get_stats()
    print(f"After deallocation: {stats}")

    # Uninstall
    gcAllocator.uninstall()

if __name__ == "__main__":
    test_allocation_tracking()
```

#### Context Manager

```python
import torch
from gcAllocator import GCAllocator

# Use as a context manager for automatic cleanup
with GCAllocator(enable_logging=True) as allocator:
    tensor = torch.randn(5000, 5000, device='cuda')
    stats = allocator.get_stats()
    print(f"Peak memory: {stats.peak_bytes_allocated / 1024**3:.2f} GB")
```

#### Environment Variables

- `GC_ALLOCATOR_LOG=1` - Enable detailed logging of allocation events

### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v -s

# Run specific test file
python tests/test_basic_allocation.py
```

### Project Structure

```
gcAllocator/
├── gcAllocator/
│   ├── __init__.py           # Python interface
│   └── src/
│       ├── gc_allocator_core.h     # Main allocator header
│       ├── gc_allocator_core.cpp   # Main allocator implementation
│       ├── allocator_stats.h       # Statistics tracking
│       ├── allocator_stats.cpp     # Statistics implementation
|       |-- retry_strategy.h
|       |-- retry_strategy.cpp
│       └── python_bindings.cpp     # Python bindings
├── tests/
│   |-- test_basic_allocation.py    # Basic functionality tests
|   |__ advanced_pytorch_example.py  
├── setup.py                         # Build configuration
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

### Current Features:
- Retry mechanism with exponential backoff
- Checkpointing with simple features
- Retry statistical profile implemented

### Not thoroughly tested
- OOM errors graceful handling to be fully tested
  
### Current Limitations:

- No inter-process coordination 
- Synchronous allocation only (async support to be added)

### Upcoming Features:

- Enhanced Python bindings and configuration
- Asynchronous allocation support
- Inter-process memory pressure protocol
- Checkpointing and advanced features
- Production hardening

### Performance

Current state introduces minimal overhead (<1%) during normal operation since it primarily passes through to the original allocator while collecting statistics.

### License

MIT License - See LICENSE file for details

```
