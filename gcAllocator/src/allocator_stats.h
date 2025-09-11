/*
 * MIT License
 * 
 * Copyright (c) 2024 Jack Ogaja
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

#ifndef ALLOCATOR_STATS_H
#define ALLOCATOR_STATS_H

#include <atomic>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <mutex>

namespace gc_allocator {

class AllocationStats {
public:
    AllocationStats();
    
    // Record events
    void recordAllocationRequest(size_t size, int device);
    void recordSuccessfulAllocation(size_t size, int device);
    void recordDeallocation(size_t size, int device);
    void recordOOMEvent(size_t size, int device);
    
    // Get statistics
    size_t getTotalAllocations() const { return total_allocations_.load(); }
    size_t getTotalDeallocations() const { return total_deallocations_.load(); }
    size_t getTotalBytesAllocated() const { return total_bytes_allocated_.load(); }
    size_t getCurrentBytesAllocated() const { return current_bytes_allocated_.load(); }
    size_t getPeakBytesAllocated() const { return peak_bytes_allocated_.load(); }
    size_t getOOMCount() const { return oom_count_.load(); }
    
    // Per-device statistics - Non-atomic version for external use
    struct DeviceStats {
        size_t allocations;
        size_t deallocations;
        size_t bytes_allocated;
        size_t current_bytes;
        size_t peak_bytes;
        size_t oom_events;
        
        DeviceStats() : allocations(0), deallocations(0), bytes_allocated(0),
                       current_bytes(0), peak_bytes(0), oom_events(0) {}
        
        DeviceStats(size_t alloc, size_t dealloc, size_t bytes_alloc,
                   size_t curr_bytes, size_t peak, size_t oom)
            : allocations(alloc), deallocations(dealloc), bytes_allocated(bytes_alloc),
              current_bytes(curr_bytes), peak_bytes(peak), oom_events(oom) {}
    };
    
    DeviceStats getDeviceStats(int device) const;
    std::vector<int> getActiveDevices() const;
    
    // Reset all statistics
    void reset();
    
    // Export stats as string for logging
    std::string toString() const;
    
private:
    // Internal atomic version for thread-safe operations
    struct AtomicDeviceStats {
        std::atomic<size_t> allocations{0};
        std::atomic<size_t> deallocations{0};
        std::atomic<size_t> bytes_allocated{0};
        std::atomic<size_t> current_bytes{0};
        std::atomic<size_t> peak_bytes{0};
        std::atomic<size_t> oom_events{0};
    };
    
    // Global statistics
    std::atomic<size_t> total_allocations_{0};
    std::atomic<size_t> total_deallocations_{0};
    std::atomic<size_t> total_bytes_allocated_{0};
    std::atomic<size_t> current_bytes_allocated_{0};
    std::atomic<size_t> peak_bytes_allocated_{0};
    std::atomic<size_t> oom_count_{0};
    
    // Per-device statistics - using internal atomic version
    mutable std::unordered_map<int, AtomicDeviceStats> device_stats_;
    mutable std::mutex device_stats_mutex_;
    
    // Timing statistics
    std::chrono::steady_clock::time_point start_time_;
    
    // Update peak memory if needed
    void updatePeakMemory(size_t current);
};

} // namespace gc_allocator

#endif // ALLOCATOR_STATS_H
