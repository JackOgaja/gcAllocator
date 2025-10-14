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

#ifndef ALLOCATOR_STATS_H
#define ALLOCATOR_STATS_H

#include <atomic>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <mutex>

#include <iostream>  // Add for instrumentation
#include <thread>    // Add for thread ID tracking

namespace gc_allocator {

// Add instrumentation control flag
#ifdef GC_ALLOCATOR_INSTRUMENT_STATS
    #define STATS_INSTRUMENT 1
#else
    #define STATS_INSTRUMENT 0
#endif


class AllocationStats {
public:
    AllocationStats();
    AllocationStats(const AllocationStats& other);
    AllocationStats& operator=(const AllocationStats& other);

    // Core tracking methods
    void recordAllocationRequest(size_t size, int device);
    void recordSuccessfulAllocation(size_t size, int device);
    void recordDeallocation(size_t size, int device);
    void recordOOMEvent(size_t size, int device);

    // NEW: Cache tracking methods
    void recordCacheHit(size_t size, int device);
    void recordCacheFlush();
    void recordStreamEvent();

    // Statistics getters
    size_t getTotalAllocations() const { return total_allocations_.load(); }
    size_t getTotalDeallocations() const { return total_deallocations_.load(); }
    size_t getTotalBytesAllocated() const { return total_bytes_allocated_.load(); }
    size_t getCurrentBytesAllocated() const { return current_bytes_allocated_.load(); }
    size_t getPeakBytesAllocated() const { return peak_bytes_allocated_.load(); }
    size_t getOOMCount() const { return oom_count_.load(); }

    // NEW: Cache statistics getters
    size_t getTotalRequests() const { return total_requests_.load(); }
    size_t getCacheHits() const { return cache_hits_.load(); }
    size_t getCacheFlushes() const { return cache_flushes_.load(); }
    size_t getStreamEvents() const { return stream_events_.load(); }

    double getCacheHitRate() const {
        size_t requests = total_requests_.load();
        if (requests == 0) return 0.0;
        return static_cast<double>(cache_hits_.load()) / requests;
    }

    struct DeviceStats {
        size_t allocations;
        size_t deallocations;
        size_t bytes_allocated;
        size_t current_bytes;
        size_t peak_bytes;
        size_t oom_events;
        size_t cache_hits;  // NEW

        DeviceStats() : allocations(0), deallocations(0), bytes_allocated(0),
                       current_bytes(0), peak_bytes(0), oom_events(0), cache_hits(0) {}

        DeviceStats(size_t alloc, size_t dealloc, size_t bytes_alloc,
                   size_t curr_bytes, size_t peak, size_t oom, size_t hits)
            : allocations(alloc), deallocations(dealloc), bytes_allocated(bytes_alloc),
              current_bytes(curr_bytes), peak_bytes(peak), oom_events(oom), cache_hits(hits) {}
    };

    DeviceStats getDeviceStats(int device) const;
    std::vector<int> getActiveDevices() const;

    void reset();
    std::string toString() const;

    // Public atomics for direct access
    std::atomic<size_t> total_allocations_{0};
    std::atomic<size_t> total_deallocations_{0};
    std::atomic<size_t> total_bytes_allocated_{0};
    std::atomic<size_t> current_bytes_allocated_{0};
    std::atomic<size_t> peak_bytes_allocated_{0};
    std::atomic<size_t> oom_count_{0};

    // NEW: Cache tracking atomics
    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> cache_hits_{0};
    std::atomic<size_t> cache_flushes_{0};
    std::atomic<size_t> stream_events_{0};

    // JO + INSTRUMENTATION
    // Add instrumentation methods (minimal addition)
    void enableInstrumentation(bool enable = true) {
        instrumentation_enabled_.store(enable);
    }

    bool isInstrumentationEnabled() const {
        return instrumentation_enabled_.load();
    }

    // Instance identifier for tracking
    size_t getInstanceId() const { return instance_id_; }

private:
    struct AtomicDeviceStats {
        std::atomic<size_t> allocations{0};
        std::atomic<size_t> deallocations{0};
        std::atomic<size_t> bytes_allocated{0};
        std::atomic<size_t> current_bytes{0};
        std::atomic<size_t> peak_bytes{0};
        std::atomic<size_t> oom_events{0};
        std::atomic<size_t> cache_hits{0};  // NEW
    };

    mutable std::unordered_map<int, AtomicDeviceStats> device_stats_;
    mutable std::mutex device_stats_mutex_;
    std::chrono::steady_clock::time_point start_time_;

    void updatePeakMemory(size_t current);

    // Add instrumentation members at the end of class
    static std::atomic<size_t> next_instance_id_;
    size_t instance_id_;
    std::atomic<bool> instrumentation_enabled_{STATS_INSTRUMENT};

    // Instrumentation helper
    void logMethodCall(const char* method_name, size_t size = 0, int device = -1) const;

};

} // namespace gc_allocator

#endif // ALLOCATOR_STATS_H
