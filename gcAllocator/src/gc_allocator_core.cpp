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

#include "gc_allocator_core.h"
#include "allocator_stats.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/alloc_cpu.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <sstream>

namespace gc_allocator {

thread_local GCAllocator* GCAllocator::current_allocator_ = nullptr;

GCAllocator::GCAllocator()
    : wrapped_allocator_(nullptr),
      stats_(std::make_unique<AllocationStats>()),
      retry_strategy_(std::make_unique<RetryStrategy>()) {

    const char* log_env = std::getenv("GC_ALLOCATOR_LOG");
    if (log_env && (std::string(log_env) == "1" || std::string(log_env) == "true")) {
        logging_enabled_.store(true);
    }

    // JO+ INSTRUMENTATION
    // Add instrumentation initialization
    const char* instrument_env = std::getenv("GC_ALLOCATOR_INSTRUMENT");
    if (instrument_env && (std::string(instrument_env) == "1")) {
        stats_->enableInstrumentation(true);
        std::cout << "[GC_INSTRUMENT] GCAllocator created with stats instance #"
                  << stats_->getInstanceId() << " at " << stats_.get() << std::endl;
    }
    // JO---

    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Initialized in proxy mode" << std::endl;
    }
}

GCAllocator::~GCAllocator() {
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Destroying allocator. Active allocations: "
                  << allocation_map_.size() << std::endl;
    }
}

void GCAllocator::setWrappedAllocator(c10::Allocator* wrapped) {
    wrapped_allocator_ = wrapped;
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Wrapped allocator set: " << wrapped << std::endl;
    }
}

c10::DataPtr GCAllocator::allocate(size_t n) {
    if (!wrapped_allocator_) {
        throw std::runtime_error("GCAllocator: No wrapped allocator configured");
    }

    int device = c10::cuda::current_device();

    // JO+ INSTRUMENTATION
    // Add instrumentation before recording
    if (stats_->isInstrumentationEnabled()) {
        std::cout << "[GC_INSTRUMENT] allocate() called with size=" << n 
                  << " on stats instance #" << stats_->getInstanceId()
                  << " at " << stats_.get() << std::endl;
    }
    // JO--

    // PRE-PROCESSING: Record allocation request
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_->recordAllocationRequest(n, device);
    }

    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator::allocate] Request for " << n
                  << " bytes on device " << device << std::endl;
    }

    // DELEGATION WITH RETRY: Create lambda that delegates to wrapped allocator
    auto allocator_func = [this, n, device]() -> c10::DataPtr {
        try {
            // Get CUDA memory before allocation for cache detection
            size_t free_before, total;
            cudaMemGetInfo(&free_before, &total);
            size_t used_before = total - free_before;

            // CRITICAL: Delegate to wrapped allocator, not global instance
            c10::DataPtr ptr = wrapped_allocator_->allocate(n);

            if (!ptr.get()) {
                throw c10::OutOfMemoryError(
                    c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
                    "Null pointer returned from wrapped allocator"
                );
            }

            // Get CUDA memory after allocation
            size_t free_after;
            cudaMemGetInfo(&free_after, &total);
            size_t used_after = total - free_after;

            // Detect cache hit vs actual allocation
            bool cache_hit = (used_after == used_before);

            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (cache_hit) {
                    stats_->recordCacheHit(n, device);
                    if (isLoggingEnabled()) {
                        std::cout << "[GCAllocator] Cache hit for " << n << " bytes" << std::endl;
                    }
                } else {
                    stats_->recordSuccessfulAllocation(n, device);
                    if (isLoggingEnabled()) {
                        std::cout << "[GCAllocator] New allocation: " << n << " bytes" << std::endl;
                    }
                }
            }

            return ptr;

        } catch (const c10::OutOfMemoryError& e) {
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_->recordOOMEvent(n, device);
            }
            if (isLoggingEnabled()) {
                std::cout << "[GCAllocator] OOM detected: " << e.what() << std::endl;
            }
            throw;
        }
    };

    try {
        // Execute with retry strategy
        c10::DataPtr result = retry_strategy_->executeWithRetry(allocator_func, n, device);

        // POST-PROCESSING: Track allocation for deallocation
        if (result.get()) {
            std::lock_guard<std::mutex> lock(allocation_map_mutex_);
            allocation_map_[result.get()] = AllocationInfo{
                n, device, std::chrono::steady_clock::now()
            };

            if (isLoggingEnabled()) {
                std::cout << "[GCAllocator] Tracked allocation: " << result.get()
                          << " (" << n << " bytes)" << std::endl;
            }
        }

        return result;

    } catch (const std::exception& e) {
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Allocation failed: " << e.what() << std::endl;
        }
        throw;
    }
}

void GCAllocator::deleteFunction(void* ptr) {
    if (!ptr) return;

    auto* allocator = current_allocator_;
    if (!allocator) {
        // Fallback to default deleter
        auto default_deleter = c10::cuda::CUDACachingAllocator::get()->raw_deleter();
        if (default_deleter) {
            default_deleter(ptr);
        }
        return;
    }

    // Look up allocation info
    AllocationInfo info{0, 0, {}};
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(allocator->allocation_map_mutex_);
        auto it = allocator->allocation_map_.find(ptr);
        if (it != allocator->allocation_map_.end()) {
            info = it->second;
            allocator->allocation_map_.erase(it);
            found = true;
        }
    }

    // Record deallocation if tracked
    if (found) {
        std::lock_guard<std::mutex> lock(allocator->stats_mutex_);
        allocator->stats_->recordDeallocation(info.size, info.device);

        if (allocator->isLoggingEnabled()) {
            std::cout << "[GCAllocator::delete] Deallocating " << ptr
                      << " (" << info.size << " bytes)" << std::endl;
        }
    }

    // Delegate to wrapped allocator's deleter
    if (allocator->wrapped_allocator_) {
        auto deleter = allocator->wrapped_allocator_->raw_deleter();
        if (deleter) {
            deleter(ptr);
        }
    }
}

c10::DeleterFnPtr GCAllocator::raw_deleter() const {
    return &GCAllocator::deleteFunction;
}

void GCAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    if (wrapped_allocator_) {
        wrapped_allocator_->copy_data(dest, src, count);
    } else {
        c10::cuda::CUDACachingAllocator::get()->copy_data(dest, src, count);
    }
}

void GCAllocator::recordStream(const c10::DataPtr& ptr, c10::cuda::CUDAStream stream) {
    if (wrapped_allocator_) {
        // Note: This requires casting to access CachingAllocator-specific methods
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Recording stream for " << ptr.get() << std::endl;
        }
        // Track stream recording in stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_->recordStreamEvent();
        }
    }
}

void GCAllocator::emptyCache() {
    if (wrapped_allocator_) {
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Emptying cache" << std::endl;
        }
        // Use CUDACachingAllocator's emptyCache
        c10::cuda::CUDACachingAllocator::emptyCache();
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_->recordCacheFlush();
        }
    }
}

size_t GCAllocator::getCachedBytes(int device) {
    // Query cache statistics from wrapped allocator
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    return stats.allocated_bytes[0].current;
}

void GCAllocator::recordAllocation(void* ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(allocation_map_mutex_);
    allocation_map_[ptr] = AllocationInfo{
        size, device, std::chrono::steady_clock::now()
    };
}

void GCAllocator::recordDeallocation(void* ptr) {
    std::lock_guard<std::mutex> lock(allocation_map_mutex_);
    allocation_map_.erase(ptr);
}

void GCAllocator::recordAllocationRequest(size_t size, int device) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_->recordAllocationRequest(size, device);
}

void GCAllocator::recordOOMEvent(size_t size, int device) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_->recordOOMEvent(size, device);
}

void GCAllocator::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_->reset();
}

void GCAllocator::configureRetryStrategy(const RetryConfig& config) {
    retry_strategy_ = std::make_unique<RetryStrategy>(config);
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Configured retry strategy" << std::endl;
    }
}

void GCAllocator::registerCheckpointCallback(std::function<bool()> callback) {
    if (retry_strategy_) {
        retry_strategy_->registerCheckpointCallback(callback);
    }
}

const RetryStats& GCAllocator::getRetryStats() const {
    if (retry_strategy_) {
        return retry_strategy_->getStats();
    }
    static const RetryStats empty_stats{};
    return empty_stats;
}

void GCAllocator::resetRetryStats() {
    if (retry_strategy_) {
        retry_strategy_->resetStats();
    }
}

// GCAllocatorManager implementation with Proxy Pattern
GCAllocatorManager& GCAllocatorManager::getInstance() {
    static GCAllocatorManager instance;
    return instance;
}

GCAllocatorManager::~GCAllocatorManager() {
    if (installed_.load()) {
        uninstallAllocator();
    }
}

void GCAllocatorManager::installAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);

    if (installed_.load()) {
        return;
    }

    try {
        // Initialize CUDA
        if (at::cuda::is_available()) {
            at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
            c10::cuda::CUDAGuard guard(0);
        } else {
            throw std::runtime_error("CUDA is not available");
        }

        // Initialize allocator for current device
        int device = at::cuda::current_device();
        
        // Force device allocator initialization
        {
            at::cuda::CUDAGuard guard(device);
            
            // Initialize the caching allocator for this device
            c10::cuda::CUDACachingAllocator::init(device);
            
            // Force initialization by creating a small tensor
            auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device);
            at::empty({1}, options);
        }
        
        // NOW get the allocator - it should be properly initialized
        original_cuda_allocator_ = c10::cuda::CUDACachingAllocator::get();
        
        if (!original_cuda_allocator_) {
            throw std::runtime_error("Failed to get CUDA allocator after initialization");
        }
        
        // Create proxy allocator
        allocator_ = std::make_unique<GCAllocator>();
        
        // Set the wrapped allocator
        allocator_->setWrappedAllocator(original_cuda_allocator_);

        // JO++++++

        // Capture original allocator BEFORE installing proxy
        // JO original_cuda_allocator_ = c10::GetAllocator(c10::DeviceType::CUDA);

        // Create proxy allocator
        // JO allocator_ = std::make_unique<GCAllocator>();

        // Set the wrapped allocator
        // JO allocator_->setWrappedAllocator(original_cuda_allocator_);

        // Configure retry strategy
        RetryConfig default_config;
        default_config.max_retries = 5;
        default_config.initial_delay = std::chrono::milliseconds(50);
        allocator_->configureRetryStrategy(default_config);

        // Set thread-local pointer
        GCAllocator::current_allocator_ = allocator_.get();

        // Install proxy as CUDA allocator
        c10::SetAllocator(c10::DeviceType::CUDA, allocator_.get());

        installed_.store(true);

        if (allocator_->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Successfully installed as proxy for "
                      << original_cuda_allocator_ << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[GCAllocator] Installation failed: " << e.what() << std::endl;
        allocator_.reset();
        installed_.store(false);
        throw;
    }
}

void GCAllocatorManager::uninstallAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);

    if (!installed_.load()) {
        return;
    }

    try {
        // Clear CUDA cache
        if (at::cuda::is_available()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
            cudaDeviceSynchronize();
        }

        // Restore original allocator
        if (original_cuda_allocator_) {
            c10::SetAllocator(c10::DeviceType::CUDA, original_cuda_allocator_);
        }

        // Clean up
        if (allocator_) {
            std::cout << "[GCAllocator] Uninstalled proxy allocator" << std::endl;
            allocator_.reset();
        }

        GCAllocator::current_allocator_ = nullptr;
        original_cuda_allocator_ = nullptr;
        installed_.store(false);

    } catch (const std::exception& e) {
        std::cerr << "[GCAllocator] Error during uninstall: " << e.what() << std::endl;
        installed_.store(false);
    }
}

void GCAllocatorManager::enableLogging() {
    if (allocator_) {
        allocator_->setLoggingEnabled(true);
    }
}

void GCAllocatorManager::disableLogging() {
    if (allocator_) {
        allocator_->setLoggingEnabled(false);
    }
}

void GCAllocatorManager::configureRetryStrategy(const RetryConfig& config) {
    if (allocator_) {
        allocator_->configureRetryStrategy(config);
    }
}

void GCAllocatorManager::registerCheckpointCallback(std::function<bool()> callback) {
    if (allocator_) {
        allocator_->registerCheckpointCallback(callback);
    }
}

const RetryStats& GCAllocatorManager::getRetryStats() const {
    if (allocator_) {
        return allocator_->getRetryStats();
    }
    static const RetryStats empty_stats{};
    return empty_stats;
}

void GCAllocatorManager::resetRetryStats() {
    if (allocator_) {
        allocator_->resetRetryStats();
    }
}

std::string GCAllocator::CombinedStats::toString() const {
    std::stringstream ss;
    ss << allocation_stats.toString();
    ss << "\n--- Retry Statistics ---\n";
    ss << "Total Retry Attempts: " << retry_stats.getTotalRetryAttempts() << "\n";
    ss << "Cache Flushes: " << retry_stats.getCacheFlushes() << "\n";
    ss << "Checkpoint Activations: " << retry_stats.getCheckpointActivations() << "\n";
    ss << "Successful Recoveries: " << retry_stats.getSuccessfulRecoveries() << "\n";
    return ss.str();
}

} // namespace gc_allocator
