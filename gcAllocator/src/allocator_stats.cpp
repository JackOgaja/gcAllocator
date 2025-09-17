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

#include "allocator_stats.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace gc_allocator {

AllocationStats::AllocationStats() : start_time_(std::chrono::steady_clock::now()) {
}

// Copy constructor to handle atomic members
AllocationStats::AllocationStats(const AllocationStats& other) 
    : total_allocations_(other.total_allocations_.load()),
      total_deallocations_(other.total_deallocations_.load()),
      total_bytes_allocated_(other.total_bytes_allocated_.load()),
      current_bytes_allocated_(other.current_bytes_allocated_.load()),
      peak_bytes_allocated_(other.peak_bytes_allocated_.load()),
      oom_count_(other.oom_count_.load()),
      start_time_(other.start_time_) {
    
    // Copy device stats
    std::lock_guard<std::mutex> lock(other.device_stats_mutex_);
    for (const auto& pair : other.device_stats_) {
        int device = pair.first;
        const auto& other_stats = pair.second;
        auto& my_stats = device_stats_[device];
        my_stats.allocations.store(other_stats.allocations.load());
        my_stats.deallocations.store(other_stats.deallocations.load());
        my_stats.bytes_allocated.store(other_stats.bytes_allocated.load());
        my_stats.current_bytes.store(other_stats.current_bytes.load());
        my_stats.peak_bytes.store(other_stats.peak_bytes.load());
        my_stats.oom_events.store(other_stats.oom_events.load());
    }
}

// Assignment operator to handle atomic members
AllocationStats& AllocationStats::operator=(const AllocationStats& other) {
    if (this != &other) {
        total_allocations_.store(other.total_allocations_.load());
        total_deallocations_.store(other.total_deallocations_.load());
        total_bytes_allocated_.store(other.total_bytes_allocated_.load());
        current_bytes_allocated_.store(other.current_bytes_allocated_.load());
        peak_bytes_allocated_.store(other.peak_bytes_allocated_.load());
        oom_count_.store(other.oom_count_.load());
        start_time_ = other.start_time_;
        
        // Copy device stats
        std::lock_guard<std::mutex> lock1(device_stats_mutex_);
        std::lock_guard<std::mutex> lock2(other.device_stats_mutex_);
        device_stats_.clear();
        for (const auto& pair : other.device_stats_) {
            int device = pair.first;
            const auto& other_stats = pair.second;
            auto& my_stats = device_stats_[device];
            my_stats.allocations.store(other_stats.allocations.load());
            my_stats.deallocations.store(other_stats.deallocations.load());
            my_stats.bytes_allocated.store(other_stats.bytes_allocated.load());
            my_stats.current_bytes.store(other_stats.current_bytes.load());
            my_stats.peak_bytes.store(other_stats.peak_bytes.load());
            my_stats.oom_events.store(other_stats.oom_events.load());
        }
    }
    return *this;
}

// ... (other methods remain the same) ...

void AllocationStats::recordSuccessfulAllocation(size_t size, int device) {
    total_allocations_.fetch_add(1);
    total_bytes_allocated_.fetch_add(size);
    
    size_t current = current_bytes_allocated_.fetch_add(size) + size;
    updatePeakMemory(current);
    
    // Update per-device stats - now using AtomicDeviceStats
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
    
    // Update per-device stats - now using AtomicDeviceStats
    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        auto& dev_stats = device_stats_[device];
        dev_stats.deallocations.fetch_add(1);
        dev_stats.current_bytes.fetch_sub(size);
    }
}

void AllocationStats::recordOOMEvent(size_t size, int device) {
    oom_count_.fetch_add(1);
    
    // Update per-device OOM count - now using AtomicDeviceStats
    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        device_stats_[device].oom_events.fetch_add(1);
    }
}

// CORRECTED METHOD - Properly handles atomic members
AllocationStats::DeviceStats AllocationStats::getDeviceStats(int device) const {
    std::lock_guard<std::mutex> lock(device_stats_mutex_);
    auto it = device_stats_.find(device);
    if (it != device_stats_.end()) {
        // Extract values from atomic members and construct new DeviceStats
        const auto& atomic_stats = it->second;
        return DeviceStats(
            atomic_stats.allocations.load(),
            atomic_stats.deallocations.load(),
            atomic_stats.bytes_allocated.load(),
            atomic_stats.current_bytes.load(),
            atomic_stats.peak_bytes.load(),
            atomic_stats.oom_events.load()
        );
    }
    return DeviceStats{};  // Return default-constructed stats if device not found
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
    
    // Per-device stats - now using the corrected getDeviceStats
    auto devices = getActiveDevices();
    if (!devices.empty()) {
        ss << "\n--- Per-Device Statistics ---\n";
        for (int device : devices) {
            auto dev_stats = getDeviceStats(device);  // Now returns copyable DeviceStats
            ss << "Device " << device << ":\n";
            ss << "  Allocations: " << dev_stats.allocations << "\n";
            ss << "  Current Memory: " << dev_stats.current_bytes / (1024.0 * 1024.0) << " MB\n";
            ss << "  Peak Memory: " << dev_stats.peak_bytes / (1024.0 * 1024.0) << " MB\n";
            if (dev_stats.oom_events > 0) {
                ss << "  OOM Events: " << dev_stats.oom_events << "\n";
            }
        }
    }
    
    return ss.str();
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

void AllocationStats::recordAllocationRequest(size_t size, int device) {
    // Currently just a placeholder - could track failed allocation attempts
}

std::vector<int> AllocationStats::getActiveDevices() const {
    std::lock_guard<std::mutex> lock(device_stats_mutex_);
    std::vector<int> devices;
    for (const auto& pair : device_stats_) {
        devices.push_back(pair.first);
    }
    return devices;
}

void AllocationStats::updatePeakMemory(size_t current) {
    size_t peak = peak_bytes_allocated_.load();
    while (current > peak && 
           !peak_bytes_allocated_.compare_exchange_weak(peak, current)) {
        // Keep trying to update peak
    }
}

} // namespace gc_allocator
