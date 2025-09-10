#ifndef ALLOCATOR_STATS_H
#define ALLOCATOR_STATS_H

#include <atomic>
#include <vector>
#include <chrono>
#include <unordered_map>

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
    
    // Per-device statistics
    struct DeviceStats {
        std::atomic<size_t> allocations{0};
        std::atomic<size_t> deallocations{0};
        std::atomic<size_t> bytes_allocated{0};
        std::atomic<size_t> current_bytes{0};
        std::atomic<size_t> peak_bytes{0};
        std::atomic<size_t> oom_events{0};
    };
    
    DeviceStats getDeviceStats(int device) const;
    std::vector<int> getActiveDevices() const;
    
    // Reset all statistics
    void reset();
    
    // Export stats as string for logging
    std::string toString() const;
    
private:
    // Global statistics
    std::atomic<size_t> total_allocations_{0};
    std::atomic<size_t> total_deallocations_{0};
    std::atomic<size_t> total_bytes_allocated_{0};
    std::atomic<size_t> current_bytes_allocated_{0};
    std::atomic<size_t> peak_bytes_allocated_{0};
    std::atomic<size_t> oom_count_{0};
    
    // Per-device statistics
    mutable std::unordered_map<int, DeviceStats> device_stats_;
    mutable std::mutex device_stats_mutex_;
    
    // Timing statistics
    std::chrono::steady_clock::time_point start_time_;
    
    // Update peak memory if needed
    void updatePeakMemory(size_t current);
};

} // namespace gc_allocator

#endif // ALLOCATOR_STATS_H
