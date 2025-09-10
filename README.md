# gcAllocator - Graceful CUDA Allocator for PyTorch

A custom GPU memory allocator for PyTorch that handles out-of-memory (OOM) conditions gracefully through intelligent retry mechanisms and inter-process coordination.

## Phase 1 - Core Infrastructure (Current)

This initial phase establishes the foundation for graceful OOM handling by implementing a custom allocator that intercepts PyTorch's CUDA memory allocation requests.

### Features (Phase 1)

-  Custom CUDA allocator that integrates with PyTorch
-  Transparent pass-through to original PyTorch allocator
-  Comprehensive allocation statistics tracking
-  Per-device memory usage monitoring
-  OOM event detection and counting
-  Optional detailed logging of allocation events
-  Python and C++ interfaces

### Architecture

```mermaid
graph TD
    A[PyTorch Tensor Operation] --> B[GCAllocator::allocate]
    B --> C{Allocation Request}
    C --> D[Statistics Recording]
    D --> E[Original PyTorch Allocator]
    E --> F{Allocation Result}
    F -->|Success| G[Record Success & Return]
    F -->|OOM| H[Record OOM Event]
    H --> I[Throw Exception]
    
    J[Memory Deallocation] --> K[GCAllocator::deallocate]
    K --> L[Update Statistics]
    L --> M[Original Deallocator]
