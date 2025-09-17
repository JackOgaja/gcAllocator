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
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>  // For dlsym

namespace gc_allocator {

// Static member definitions
thread_local GCAllocator* GCAllocator::current_allocator_ = nullptr;
std::unordered_map<void*, size_t> GCAllocatorManager::global_allocations_;
std::mutex GCAllocatorManager::global_allocations_mutex_;

GCAllocator::GCAllocator() 
    : original_allocator_(nullptr),
      stats_(std::make_unique<AllocationStats>()),
      retry_strategy_(std::make_unique<RetryStrategy>()) {
    
    // Get the original allocator
    original_allocator_ = c10::cuda::CUDACachingAllocator::get();
    
    // Check for environment variable to enable logging
    const char* log_env = std::getenv("GC_ALLOCATOR_LOG");
    if (log_env && (std::string(log_env) == "1" || std::string(log_env) == "true")) {
        logging_enabled_.store(true);
    }
    
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Initialized with original allocator: " 
                  << original_allocator_ << std::endl;
    }
}

GCAllocator::~GCAllocator() {
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Destroying allocator. Active allocations: " 
                  << active_allocations_.size() << std::endl;
    }
}

c10::DataPtr GCAllocator::allocate(size_t n) {
    // Debug output to confirm our allocator is being called
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator::allocate] Called with size=" << n 
                  << " bytes (" << (n / (1024.0 * 1024.0)) << " MB)" << std::endl;
    }
    
    // Update statistics - Use correct member names with underscores
    stats_->total_allocations_.fetch_add(1);
    stats_->total_bytes_allocated_.fetch_add(n);
    stats_->current_bytes_allocated_.fetch_add(n);
    
    // Update peak if necessary
    size_t current_total = stats_->current_bytes_allocated_.load();
    size_t current_peak = stats_->peak_bytes_allocated_.load();
    while (current_total > current_peak && 
           !stats_->peak_bytes_allocated_.compare_exchange_weak(current_peak, current_total)) {
        current_peak = stats_->peak_bytes_allocated_.load();
    }
    
    int device = c10::cuda::current_device();
    
    // Create allocation function for retry strategy
    auto allocator_func = [this, n, device]() -> c10::DataPtr {
        try {
            // Use the default CUDA caching allocator as underlying allocator
            auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
            c10::DataPtr ptr = cuda_allocator->allocate(n);
            
            if (isLoggingEnabled() && ptr.get()) {
                std::cout << "[GCAllocator] Underlying allocation successful: " 
                          << ptr.get() << std::endl;
            }
            
            // Record successful allocation
            stats_->recordSuccessfulAllocation(n, device);
            
            return ptr;
            
        } catch (const c10::OutOfMemoryError& e) {
            if (isLoggingEnabled()) {
                std::cout << "[GCAllocator] CUDA OOM detected in underlying allocator: " 
                          << e.what() << std::endl;
            }
             
            // Use correct member name with underscore
            stats_->oom_count_.fetch_add(1);
            // Also record OOM event with device info
            stats_->recordOOMEvent(n, device);
            throw;  // Re-throw for retry strategy
            
        } catch (const std::exception& e) {
            // Convert CUDA memory errors to OOM for retry handling
            std::string error_msg = e.what();
            if (error_msg.find("out of memory") != std::string::npos || 
                error_msg.find("CUDA_ERROR_OUT_OF_MEMORY") != std::string::npos ||
                error_msg.find("CUDA out of memory") != std::string::npos) {
                
                if (isLoggingEnabled()) {
                    std::cout << "[GCAllocator] Converting CUDA error to OOM: " 
                              << error_msg << std::endl;
                }
                // FIX: Use correct member name with underscore
                stats_->oom_count_.fetch_add(1);
                stats_->recordOOMEvent(n, device);
                
                throw c10::OutOfMemoryError(
                    c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
                    error_msg
                );
            }
            throw;  // Re-throw other errors
        }
    };
    
    try {
        // Record allocation request
        stats_->recordAllocationRequest(n, device);
        
        // Execute allocation with retry strategy
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Starting allocation with retry strategy..." << std::endl;
        }
        
        // FIX: Use -> for unique_ptr and correct method name
        c10::DataPtr result = retry_strategy_->executeWithRetry(allocator_func, n, device);
        
        if (result.get() && isLoggingEnabled()) {
            std::cout << "[GCAllocator] Allocation completed successfully" << std::endl;
        }
        
        return result;
        
    } catch (const c10::OutOfMemoryError& e) {
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Final allocation failed after all retries: " 
                      << e.what() << std::endl;
        }
        throw;
    } catch (const std::exception& e) {
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Unexpected error during allocation: " 
                      << e.what() << std::endl;
        }
        throw;
    }
}

void GCAllocator::deleteFunction(void* ptr) {
    if (!ptr) return;
    
    auto* allocator = current_allocator_;
    if (allocator) {
        if (allocator->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Deallocation: ptr=" << ptr << std::endl;
        }
        
        {
            std::lock_guard<std::mutex> lock(GCAllocatorManager::global_allocations_mutex_);
            GCAllocatorManager::global_allocations_.erase(ptr);
        }
        
        // Record deallocation in our stats
        allocator->recordDeallocation(ptr);
    }
    
    // Use the original allocator's raw deleter to properly free memory
    // This ensures we use PyTorch's optimized deallocation
    auto* manager = &GCAllocatorManager::getInstance();
    if (manager->isInstalled() && allocator && allocator->getOriginalAllocator()) {
        auto deleter = allocator->getOriginalAllocator()->raw_deleter();
        if (deleter) {
            deleter(ptr);
            return;
        }
    }
    
    // Fallback to default CUDA caching allocator deleter
    auto default_deleter = c10::cuda::CUDACachingAllocator::get()->raw_deleter();
    if (default_deleter) {
        default_deleter(ptr);
    }
}

c10::DeleterFnPtr GCAllocator::raw_deleter() const {
    return &GCAllocator::deleteFunction;
}

void GCAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    if (original_allocator_) {
        original_allocator_->copy_data(dest, src, count);
    } else {
        // Fallback to default implementation
        c10::cuda::CUDACachingAllocator::get()->copy_data(dest, src, count);
    }
}

void GCAllocator::recordAllocationRequest(size_t size, int device) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_->recordAllocationRequest(size, device);
}

void GCAllocator::recordAllocation(void* ptr, size_t size, int device) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    active_allocations_[ptr] = AllocationInfo{
        size, 
        device, 
        std::chrono::steady_clock::now()
    };
    
    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_->recordSuccessfulAllocation(size, device);
    }
}

void GCAllocator::recordDeallocation(void* ptr) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    auto it = active_allocations_.find(ptr);
    if (it != active_allocations_.end()) {
        // Update statistics
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_->recordDeallocation(it->second.size, it->second.device);
        }
        active_allocations_.erase(it);
    }
}

void GCAllocator::recordOOMEvent(size_t size, int device) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_->recordOOMEvent(size, device);
}

AllocationStats GCAllocator::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return *stats_;
}

void GCAllocator::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_->reset();
}

// Retry strategy configuration methods
void GCAllocator::configureRetryStrategy(const RetryConfig& config) {
    retry_strategy_ = std::make_unique<RetryStrategy>(config);
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Configured retry strategy: max_retries=" 
                  << config.max_retries << ", initial_delay=" << config.initial_delay.count() 
                  << "ms, cache_flush=" << config.enable_cache_flush << std::endl;
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
    // Return a static empty stats object if no retry strategy exists
    static const RetryStats empty_stats{};
    return empty_stats;
}

void GCAllocator::resetRetryStats() {
    if (retry_strategy_) {
        retry_strategy_->resetStats();
    }
}

// GCAllocatorManager implementation
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
        // Force CUDA initialization first
        if (at::cuda::is_available()) {
            at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
            
            // Ensure CUDA context is created
            c10::cuda::CUDAGuard guard(0);
            cudaDeviceSynchronize();
        }
        
        // Create our allocator instance
        allocator_ = std::make_unique<GCAllocator>();
        
        // Configure retry strategy with defaults
        RetryConfig default_config;
        default_config.max_retries = 5;
        default_config.initial_delay = std::chrono::milliseconds(50);
        default_config.backoff_multiplier = 1.5;
        default_config.max_delay = std::chrono::milliseconds(2000);
        default_config.enable_cache_flush = true;
        default_config.enable_gradient_checkpointing = true;
        
        allocator_->configureRetryStrategy(default_config);
        
        // Store the original allocator for restoration
        original_cuda_allocator_ = c10::GetAllocator(c10::DeviceType::CUDA);
        
        // Set our allocator as the CUDA allocator - CORRECT API
        c10::SetAllocator(c10::DeviceType::CUDA, allocator_.get());
        
        installed_.store(true);
        
        if (allocator_->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Successfully installed custom allocator" << std::endl;
            
            // Test allocation to verify it works
            try {
                auto test_ptr = allocator_->allocate(1024);
                if (test_ptr.get()) {
                    std::cout << "[GCAllocator] Allocator test successful" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "[GCAllocator] Allocator test failed: " << e.what() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[GCAllocator] Failed to install allocator: " << e.what() << std::endl;
        allocator_.reset();
        installed_.store(false);
        throw;
    }
}

//void GCAllocatorManager::uninstallAllocator() {
//    std::lock_guard<std::mutex> lock(install_mutex_);
//    
//    if (!installed_.load()) {
//        return;
//    }
//    
//    try {
//        // Restore original allocator safely
//        if (original_cuda_allocator_) {
//            c10::SetAllocator(c10::DeviceType::CUDA, original_cuda_allocator_);
//        }
//        
//        if (allocator_ && allocator_->isLoggingEnabled()) {
//            std::cout << "[GCAllocator] Uninstalled custom allocator safely" << std::endl;
//        }
//    } catch (const std::exception& e) {
//        std::cerr << "[GCAllocator] Error during uninstall: " << e.what() << std::endl;
//    }
//    
//    // Clear global allocations tracking
//    {
//        std::lock_guard<std::mutex> alloc_lock(global_allocations_mutex_);
//        global_allocations_.clear();
//    }
//    
//    // Clean up our allocator
//    allocator_.reset();
//    original_cuda_allocator_ = nullptr;
//    installed_.store(false);
//}
void GCAllocatorManager::uninstallAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);
    
    if (!installed_.load()) {
        return;
    }
    
    try {
        // Clear any remaining CUDA memory - FIX: Use correct namespace
        if (at::cuda::is_available()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
            cudaDeviceSynchronize();
        }
        
        // Restore the original allocator
        if (original_cuda_allocator_) {
            c10::SetAllocator(c10::DeviceType::CUDA, original_cuda_allocator_);
        }
        
        // Reset our allocator
        if (allocator_) {
            std::cout << "[GCAllocator] Uninstalled custom allocator safely" << std::endl;
            allocator_.reset();
        }
        
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
    // Return a static empty stats object if no allocator exists
    static const RetryStats empty_stats{};
    return empty_stats;
}

void GCAllocatorManager::resetRetryStats() {
    if (allocator_) {
        allocator_->resetRetryStats();
    }
}

} // namespace gc_allocator
