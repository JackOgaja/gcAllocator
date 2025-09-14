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
      stats_(std::make_unique<AllocationStats>()) {
    
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
    // Set thread-local pointer for deleter callback
    current_allocator_ = this;
    
    // Get current CUDA device
    int device = 0;
    if (at::cuda::is_available()) {
        cudaGetDevice(&device);
    }
    
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Allocation request: size=" << n 
                  << " bytes, device=" << device << std::endl;
    }
    
    // Record allocation request
    recordAllocationRequest(n, device);
    
    // Use PyTorch's caching allocator instead of direct CUDA allocation
    c10::DataPtr result;
    try {
        if (original_allocator_) {
            result = original_allocator_->allocate(n);
        } else {
            // Fallback to default CUDA caching allocator
            result = c10::cuda::CUDACachingAllocator::get()->allocate(n);
        }
        
        if (result.get()) {
            // Track the allocation
            recordAllocation(result.get(), n, device);
            
            // Track globally for interception
            {
                std::lock_guard<std::mutex> lock(GCAllocatorManager::global_allocations_mutex_);
                GCAllocatorManager::global_allocations_[result.get()] = n;
            }
            
            if (isLoggingEnabled()) {
                std::cout << "[GCAllocator] Allocation successful: ptr=" << result.get() 
                          << ", size=" << n << std::endl;
            }
            
            // Wrap the DataPtr to intercept deallocation
            return c10::DataPtr(
                result.get(),
                result.get_context(),
                &deleteFunction,
                result.device()
            );
        } else {
            // Record OOM
            recordOOMEvent(n, device);
        }
    } catch (const std::exception& e) {
        // Record OOM on exception
        recordOOMEvent(n, device);
        throw;
    }
    
    return result;
}

void GCAllocator::deleteFunction(void* ptr) {
    if (!ptr) return;
    
    auto* allocator = current_allocator_;
    if (allocator) {
        if (allocator->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Deallocation: ptr=" << ptr << std::endl;
        }
        
        // Remove from global tracking
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
    
    // Create our allocator instance
    allocator_ = std::make_unique<GCAllocator>();
    
    // Store the original CUDA allocator for restoration later
    original_cuda_allocator_ = c10::cuda::CUDACachingAllocator::get();
    
    try {
        // Set our allocator as the CUDA allocator using the proper API
        c10::SetAllocator(c10::DeviceType::CUDA, allocator_.get());
        
        installed_.store(true);
        
        if (allocator_->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Successfully installed custom allocator with PyTorch caching" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[GCAllocator] Failed to install allocator: " << e.what() << std::endl;
        allocator_.reset();
        throw;
    }
}

void GCAllocatorManager::uninstallAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);
    
    if (!installed_.load()) {
        return;
    }
    
    try {
        // Restore original allocator safely
        if (original_cuda_allocator_) {
            c10::SetAllocator(c10::DeviceType::CUDA, original_cuda_allocator_);
        }
        
        if (allocator_ && allocator_->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Uninstalled custom allocator safely" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[GCAllocator] Error during uninstall: " << e.what() << std::endl;
    }
    
    // Clear global allocations tracking
    {
        std::lock_guard<std::mutex> alloc_lock(global_allocations_mutex_);
        global_allocations_.clear();
    }
    
    // Clean up our allocator
    allocator_.reset();
    original_cuda_allocator_ = nullptr;
    installed_.store(false);
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

} // namespace gc_allocator
