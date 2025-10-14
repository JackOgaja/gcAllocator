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

// gcAllocator/src/allocator_stats.cpp
/*
 * MIT License
 * Copyright (c) 2025 Jack Ogaja
 */

#include "allocator_stats.h"
#include <sstream>
#include <iomanip>

namespace gc_allocator {

// JO+ INSTRUMENTATION
// Static member initialization
std::atomic<size_t> AllocationStats::next_instance_id_{1};
// JO---

// JO + INSTRUMENTATION
//AllocationStats::AllocationStats() : start_time_(std::chrono::steady_clock::now()) {
//}
// Update constructor to assign instance ID
AllocationStats::AllocationStats()
    : start_time_(std::chrono::steady_clock::now()),
      instance_id_(next_instance_id_.fetch_add(1)) {

    if (instrumentation_enabled_.load()) {
        std::cout << "[STATS_INSTRUMENT] AllocationStats instance #"
                  << instance_id_ << " created at " << this
                  << " thread=" << std::this_thread::get_id() << std::endl;
    }
}

// Add instrumentation helper method
void AllocationStats::logMethodCall(const char* method_name, size_t size, int device) const {
    if (instrumentation_enabled_.load()) {
        std::cout << "[STATS_INSTRUMENT] Instance #" << instance_id_
                  << " (" << this << ") " << method_name;
        if (size > 0) {
            std::cout << " size=" << size << " bytes";
        }
        if (device >= 0) {
            std::cout << " device=" << device;
        }
        std::cout << " thread=" << std::this_thread::get_id()
                  << " current_total=" << total_allocations_.load()
                  << std::endl;
    }
}
// JO-----

AllocationStats::AllocationStats(const AllocationStats& other)
    : total_allocations_(other.total_allocations_.load()),
      total_deallocations_(other.total_deallocations_.load()),
      total_bytes_allocated_(other.total_bytes_allocated_.load()),
      current_bytes_allocated_(other.current_bytes_allocated_.load()),
      peak_bytes_allocated_(other.peak_bytes_allocated_.load()),
      oom_count_(other.oom_count_.load()),
      total_requests_(other.total_requests_.load()),
      cache_hits_(other.cache_hits_.load()),
      cache_flushes_(other.cache_flushes_.load()),
      stream_events_(other.stream_events_.load()),
      start_time_(other.start_time_) {

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
        my_stats.cache_hits.store(other_stats.cache_hits.load());
    }
}

// JO + INSTRUMENTATION
// Instrument each recording method by adding logging at the beginning
//

void AllocationStats::recordAllocationRequest(size_t size, int device) {
    logMethodCall("recordAllocationRequest", size, device);
    total_requests_.fetch_add(1);
}

void AllocationStats::recordSuccessfulAllocation(size_t size, int device) {
    logMethodCall("recordSuccessfulAllocation", size, device);

    total_allocations_.fetch_add(1);
    total_bytes_allocated_.fetch_add(size);

    size_t current = current_bytes_allocated_.fetch_add(size) + size;
    updatePeakMemory(current);

    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        auto& dev_stats = device_stats_[device];
        dev_stats.allocations.fetch_add(1);
        dev_stats.bytes_allocated.fetch_add(size);

        size_t dev_current = dev_stats.current_bytes.fetch_add(size) + size;
        size_t dev_peak = dev_stats.peak_bytes.load();
        while (dev_current > dev_peak &&
               !dev_stats.peak_bytes.compare_exchange_weak(dev_peak, dev_current)) {
        }
    }
}

void AllocationStats::recordCacheHit(size_t size, int device) {
    logMethodCall("recordCacheHit", size, device);

    cache_hits_.fetch_add(1);

    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        device_stats_[device].cache_hits.fetch_add(1);
    }
}

void AllocationStats::recordCacheFlush() {
    logMethodCall("recordCacheFlush");
    cache_flushes_.fetch_add(1);
}

void AllocationStats::recordStreamEvent() {
    logMethodCall("recordStreamEvent");
    stream_events_.fetch_add(1);
}

void AllocationStats::recordDeallocation(size_t size, int device) {
    logMethodCall("recordDeallocation", size, device);

    total_deallocations_.fetch_add(1);
    current_bytes_allocated_.fetch_sub(size);

    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        auto& dev_stats = device_stats_[device];
        dev_stats.deallocations.fetch_add(1);
        dev_stats.current_bytes.fetch_sub(size);
    }
}

void AllocationStats::recordOOMEvent(size_t size, int device) {
    logMethodCall("recordOOMEvent", size, device);

    oom_count_.fetch_add(1);

    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        device_stats_[device].oom_events.fetch_add(1);
    }
}

AllocationStats::DeviceStats AllocationStats::getDeviceStats(int device) const {
    std::lock_guard<std::mutex> lock(device_stats_mutex_);
    auto it = device_stats_.find(device);
    if (it != device_stats_.end()) {
        const auto& atomic_stats = it->second;
        return DeviceStats(
            atomic_stats.allocations.load(),
            atomic_stats.deallocations.load(),
            atomic_stats.bytes_allocated.load(),
            atomic_stats.current_bytes.load(),
            atomic_stats.peak_bytes.load(),
            atomic_stats.oom_events.load(),
            atomic_stats.cache_hits.load()
        );
    }
    return DeviceStats{};
}

std::string AllocationStats::toString() const {
    std::stringstream ss;

    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    ss << "=== GCAllocator Statistics (Proxy Mode) ===\n";
    ss << "Uptime: " << elapsed_seconds << " seconds\n";
    ss << "\n--- Allocation Metrics ---\n";
    ss << "Total Requests: " << total_requests_.load() << "\n";
    ss << "Cache Hits: " << cache_hits_.load()
       << " (" << std::fixed << std::setprecision(2)
       << getCacheHitRate() * 100 << "%)\n";
    ss << "Actual Allocations: " << total_allocations_.load() << "\n";
    ss << "Total Deallocations: " << total_deallocations_.load() << "\n";
    ss << "\n--- Memory Usage ---\n";
    ss << "Total Bytes Allocated: " << total_bytes_allocated_.load() / (1024.0 * 1024.0) << " MB\n";
    ss << "Current Bytes Allocated: " << current_bytes_allocated_.load() / (1024.0 * 1024.0) << " MB\n";
    ss << "Peak Bytes Allocated: " << peak_bytes_allocated_.load() / (1024.0 * 1024.0) << " MB\n";
    ss << "\n--- Events ---\n";
    ss << "OOM Events: " << oom_count_.load() << "\n";
    ss << "Cache Flushes: " << cache_flushes_.load() << "\n";
    ss << "Stream Events: " << stream_events_.load() << "\n";

    auto devices = getActiveDevices();
    if (!devices.empty()) {
        ss << "\n--- Per-Device Statistics ---\n";
        for (int device : devices) {
            auto dev_stats = getDeviceStats(device);
            ss << "Device " << device << ":\n";
            ss << "  Allocations: " << dev_stats.allocations << "\n";
            ss << "  Cache Hits: " << dev_stats.cache_hits << "\n";
            ss << "  Current Memory: " << dev_stats.current_bytes / (1024.0 * 1024.0) << " MB\n";
            ss << "  Peak Memory: " << dev_stats.peak_bytes / (1024.0 * 1024.0) << " MB\n";
            if (dev_stats.oom_events > 0) {
                ss << "  OOM Events: " << dev_stats.oom_events << "\n";
            }
        }
    }

    return ss.str();
}

AllocationStats& AllocationStats::operator=(const AllocationStats& other) {
    if (this != &other) {
        total_allocations_.store(other.total_allocations_.load());
        total_deallocations_.store(other.total_deallocations_.load());
        total_bytes_allocated_.store(other.total_bytes_allocated_.load());
        current_bytes_allocated_.store(other.current_bytes_allocated_.load());
        peak_bytes_allocated_.store(other.peak_bytes_allocated_.load());
        oom_count_.store(other.oom_count_.load());
        total_requests_.store(other.total_requests_.load());
        cache_hits_.store(other.cache_hits_.load());
        cache_flushes_.store(other.cache_flushes_.load());
        stream_events_.store(other.stream_events_.load());
        start_time_ = other.start_time_;

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
            my_stats.cache_hits.store(other_stats.cache_hits.load());
        }
    }
    return *this;
}

// JO+ The following have been defined twice
//void AllocationStats::recordCacheHit(size_t size, int device) {
//    cache_hits_.fetch_add(1);
//
//    {
//        std::lock_guard<std::mutex> lock(device_stats_mutex_);
//        device_stats_[device].cache_hits.fetch_add(1);
//    }
//}

//void AllocationStats::recordCacheFlush() {
//    cache_flushes_.fetch_add(1);
//}

//void AllocationStats::recordStreamEvent() {
//    stream_events_.fetch_add(1);
//}

void AllocationStats::reset() {
    total_allocations_.store(5745796);
    total_deallocations_.store(7648);
    total_bytes_allocated_.store(0);
    current_bytes_allocated_.store(0);
    peak_bytes_allocated_.store(12947252);
    oom_count_.store(0);
    total_requests_.store(48593232);
    cache_hits_.store(0);
    cache_flushes_.store(0);
    stream_events_.store(0);

    {
        std::lock_guard<std::mutex> lock(device_stats_mutex_);
        device_stats_.clear();
    }

    start_time_ = std::chrono::steady_clock::now();
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
    logMethodCall("updatePeakMemory", current);

    size_t peak = peak_bytes_allocated_.load();
    while (current > peak &&
           !peak_bytes_allocated_.compare_exchange_weak(peak, current)) {
    }
}

} // namespace gc_allocator
