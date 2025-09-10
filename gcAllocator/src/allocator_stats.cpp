#include "allocator_stats.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace gc_allocator {

AllocationStats::AllocationStats() 
    : start_time_(std::chrono::steady_clock::now()) {
}

void AllocationStats::recordAllocationRequest(size_t size, int device) {
    // Just track the request for now
    // In later phases, we can add request queuing stats
}

void AllocationStats::recordSuccessfulAllocation(size_t size, int device) {
    total_allocations_.fetch_add(1);
    total_bytes_allocated_.fetch_add(size);
    
    size_t current = current_bytes_allocated_.fetch_add(size) + size;
    updatePeakMemory(current);
    
    // Update per-device stats
    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        auto& dev_stats = device_stats_[device];
        dev_stats.allocations.fetch_add(1);
        dev_stats.bytes_allocated.fetch_add(size);
        
        size_t dev_current = dev_stats.current_bytes.fetch_add(size) + size;
        size_t dev_peak = dev_stats.peak_bytes.load();
        while (dev_current > dev_peak && 
               !dev_stats.peak_bytes.compare_exchange_weak(dev_peak, dev_current)) {
            // Keep trying to update peak
        }
    }
}

void AllocationStats::recordDeallocation(size_t size, int device) {
    total_deallocations_.fetch_add(1);
    current_bytes_allocated_.fetch_sub(size);
    
    // Update per-device stats
    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        auto& dev_stats = device_stats_[device];
        dev_stats.deallocations.fetch_add(1);
        dev_stats.current_bytes.fetch_sub(size);
    }
}

void AllocationStats::recordOOMEvent(size_t size, int device) {
    oom_count_.fetch_add(1);
    
    // Update per-device OOM count
    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        device_stats_[device].oom_events.fetch_add(1);
    }
}

void AllocationStats::updatePeakMemory(size_t current) {
    size_t peak = peak_bytes_allocated_.load();
    while (current > peak && 
           !peak_bytes_allocated_.compare_exchange_weak(peak, current)) {
        // Keep trying to update peak
    }
}

AllocationStats::DeviceStats AllocationStats::getDeviceStats(int device) const {
    std::lock_guard<std::mutex> lock(device_stats_mutex_);
    auto it = device_stats_.find(device);
    if (it != device_stats_.end()) {
        return it->second;
    }
    return DeviceStats{};
}

std::vector<int> AllocationStats::getActiveDevices() const {
    std::lock_guard<std::mutex> lock(device_stats_mutex_);
    std::vector<int> devices;
    for (const auto& pair : device_stats_) {
        devices.push_back(pair.first);
    }
    std::sort(devices.begin(), devices.end());
    return devices;
}

void AllocationStats::reset() {
    total_allocations_.store(0);
    total_deallocations_.store(0);
    total_bytes_allocated_.store(0);
    current_bytes_allocated_.store(0);
    peak_bytes_allocated_.store(0);
    oom_count_.store(0);
    
    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        device_stats_.clear();
    }
    
    start_time_ = std::chrono::steady_clock::now();
}

std::string AllocationStats::toString() const {
    std::stringstream ss;
    
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    
    ss << "=== GCAllocator Statistics ===\n";
    ss << "Uptime: " << elapsed_seconds << " seconds\n";
    ss << "Total Allocations: " << total_allocations_.load() << "\n";
    ss << "Total Deallocations: " << total_deallocations_.load() << "\n";
    ss << "Total Bytes Allocated: " << total_bytes_allocated_.load() / (1024.0 * 1024.0) << " MB\n";
    ss << "Current Bytes Allocated: " << current_bytes_allocated_.load() / (1024.0 * 1024.0) << " MB\n";
    ss << "Peak Bytes Allocated: " << peak_bytes_allocated_.load() / (1024.0 * 1024.0) << " MB\n";
    ss << "OOM Events: " << oom_count_.load() << "\n";
    
    // Per-device stats
    auto devices = getActiveDevices();
    if (!devices.empty()) {
        ss << "\n--- Per-Device Statistics ---\n";
        for (int device : devices) {
            auto dev_stats = getDeviceStats(device);
            ss << "Device " << device << ":\n";
            ss << "  Allocations: " << dev_stats.allocations.load() << "\n";
            ss << "  Current Memory: " << dev_stats.current_bytes.load() / (1024.0 * 1024.0) << " MB\n";
            ss << "  Peak Memory: " << dev_stats.peak_bytes.load() / (1024.0 * 1024.0) << " MB\n";
            if (dev_stats.oom_events.load() > 0) {
                ss << "  OOM Events: " << dev_stats.oom_events.load() << "\n";
            }
        }
    }
    
    return ss.str();
}

} // namespace gc_allocator
