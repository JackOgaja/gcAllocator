/*
 * MIT License
 * 
 * Copyright (c) 2025 Jack Ogaja
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef GC_ALLOCATOR_CORE_H
#define GC_ALLOCATOR_CORE_H

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>

namespace gc_allocator {

// Forward declarations
class AllocationStats;

class GCAllocator : public c10::Allocator {
public:
    GCAllocator();
    ~GCAllocator() override;

    // Core allocator interface - required by c10::Allocator
    c10::DataPtr allocate(size_t n) override;
    c10::DeleterFnPtr raw_deleter() const override;

    // Get the original allocator for passthrough
    c10::Allocator* getOriginalAllocator() const { return original_allocator_; }
    
    // Statistics interface
    AllocationStats getStats() const;
    void resetStats();
    
    // Enable/disable logging
    void setLoggingEnabled(bool enabled) { logging_enabled_.store(enabled); }
    bool isLoggingEnabled() const { return logging_enabled_.load(); }

protected:
    // Internal allocation tracking
    void recordAllocation(void* ptr, size_t size, int device);
    void recordDeallocation(void* ptr);
    
private:
    // Original allocator that we're wrapping
    c10::Allocator* original_allocator_;
    
    // Thread-safe statistics tracking
    mutable std::mutex stats_mutex_;
    std::unique_ptr<AllocationStats> stats_;
    
    // Configuration
    std::atomic<bool> logging_enabled_{false};
    
    // Track allocations for proper cleanup
    struct AllocationInfo {
        size_t size;
        int device;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::unordered_map<void*, AllocationInfo> active_allocations_;
    mutable std::mutex allocations_mutex_;
    
    // Static deleter function that routes back to instance
    static void deleteFunction(void* ptr);
    
    // Thread-local storage for current allocator instance
    static thread_local GCAllocator* current_allocator_;
};

// Global allocator instance management
class GCAllocatorManager {
public:
    static GCAllocatorManager& getInstance();
    
    // Install/uninstall the custom allocator
    void installAllocator();
    void uninstallAllocator();
    
    // Check if custom allocator is installed
    bool isInstalled() const { return installed_.load(); }
    
    // Get the current allocator instance
    GCAllocator* getAllocator() { return allocator_.get(); }
    
private:
    GCAllocatorManager() = default;
    ~GCAllocatorManager() = default;
    
    // Prevent copying
    GCAllocatorManager(const GCAllocatorManager&) = delete;
    GCAllocatorManager& operator=(const GCAllocatorManager&) = delete;
    
    std::unique_ptr<GCAllocator> allocator_;
    c10::Allocator* original_cuda_allocator_{nullptr};
    std::atomic<bool> installed_{false};
    mutable std::mutex install_mutex_;
};

} // namespace gc_allocator

#endif // GC_ALLOCATOR_CORE_H
