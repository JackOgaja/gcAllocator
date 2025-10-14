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
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>
#include "retry_strategy.h"
#include "allocator_stats.h"

namespace gc_allocator {

class GCAllocatorManager;

// Enhanced GCAllocator with Proxy Pattern
class GCAllocator : public c10::Allocator {
public:
    GCAllocator();
    ~GCAllocator() override;

    // Core allocator interface
    c10::DataPtr allocate(size_t n) override;
    c10::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const override;

    // Additional CachingAllocator interface methods for complete proxying
    void recordStream(const c10::DataPtr& ptr, c10::cuda::CUDAStream stream);
    void emptyCache();
    size_t getCachedBytes(int device);

    // Proxy pattern setup
    void setWrappedAllocator(c10::Allocator* wrapped);
    c10::Allocator* getWrappedAllocator() const { return wrapped_allocator_; }

    // Statistics interface
    const AllocationStats& getStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

	// JO + INSTRUMENTATION
	// In the getStats method, add instrumentation:
    	if (stats_->isInstrumentationEnabled()) {
        	std::cout << "[GC_INSTRUMENT] getStats() returning instance #"
                	  << stats_->getInstanceId() << " at " << stats_.get()
                  	  << " with total_allocations=" << stats_->getTotalAllocations()
                  	  << " total_requests=" << stats_->getTotalRequests()
                  	  << std::endl;
    	}
    	// JO---
	
        return *stats_;
    }
    void resetStats();

    // Configuration
    void setLoggingEnabled(bool enabled) { logging_enabled_.store(enabled); }
    bool isLoggingEnabled() const { return logging_enabled_.load(); }
    void configureRetryStrategy(const RetryConfig& config);
    void registerCheckpointCallback(std::function<bool()> callback);
    const RetryStats& getRetryStats() const;
    void resetRetryStats();

    // Enhanced tracking with allocation map
    void recordAllocation(void* ptr, size_t size, int device);
    void recordDeallocation(void* ptr);
    void recordAllocationRequest(size_t size, int device);
    void recordOOMEvent(size_t size, int device);

    struct CombinedStats {
        AllocationStats allocation_stats;
        RetryStats retry_stats;
        std::string toString() const;
    };

    CombinedStats getCombinedStats() const {
        CombinedStats combined;
        combined.allocation_stats = getStats();
        combined.retry_stats = getRetryStats();
        return combined;
    }

private:
    // PROXY PATTERN: Wrapped allocator instead of original
    c10::Allocator* wrapped_allocator_{nullptr};

    // Thread-safe statistics tracking
    mutable std::mutex stats_mutex_;
    std::unique_ptr<AllocationStats> stats_;

    // Retry strategy
    std::unique_ptr<RetryStrategy> retry_strategy_;

    // Configuration
    std::atomic<bool> logging_enabled_{false};

    // ENHANCED: Complete allocation tracking with size info
    struct AllocationInfo {
        size_t size;
        int device;
        std::chrono::steady_clock::time_point timestamp;
    };

    // CRITICAL: Use concurrent map for thread-safe access
    std::unordered_map<void*, AllocationInfo> allocation_map_;
    mutable std::mutex allocation_map_mutex_;

    // Track CUDA memory for cache detection
    std::atomic<size_t> last_cuda_bytes_allocated_{0};

    // Delete function routing
    static void deleteFunction(void* ptr);

    // Thread-local storage for current allocator
    static thread_local GCAllocator* current_allocator_;

    friend class GCAllocatorManager;
};

class GCAllocatorManager {
public:
    static GCAllocatorManager& getInstance();

    void installAllocator();
    void uninstallAllocator();
    bool isInstalled() const { return installed_.load(); }
    GCAllocator* getAllocator() { return allocator_.get(); }

    void enableLogging();
    void disableLogging();
    void configureRetryStrategy(const RetryConfig& config);
    void registerCheckpointCallback(std::function<bool()> callback);
    const RetryStats& getRetryStats() const;
    void resetRetryStats();

    ~GCAllocatorManager();

private:
    GCAllocatorManager() = default;
    GCAllocatorManager(const GCAllocatorManager&) = delete;
    GCAllocatorManager& operator=(const GCAllocatorManager&) = delete;

    std::unique_ptr<GCAllocator> allocator_;
    c10::Allocator* original_cuda_allocator_{nullptr};
    std::atomic<bool> installed_{false};
    mutable std::mutex install_mutex_;
};

} // namespace gc_allocator

#endif // GC_ALLOCATOR_CORE_H
