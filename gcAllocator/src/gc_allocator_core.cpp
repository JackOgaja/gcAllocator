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
#include <c10/core/Allocator.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace gc_allocator {

thread_local GCAllocator* GCAllocator::current_allocator_ = nullptr;

GCAllocator::GCAllocator() 
    : original_allocator_(c10::cuda::CUDACachingAllocator::get()),
      stats_(std::make_unique<AllocationStats>()) {
    
    // Check for environment variable to enable logging
    const char* log_env = std::getenv("GC_ALLOCATOR_LOG");
    if (log_env && std::string(log_env) == "1") {
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
    if (at::hasCUDA()) {
        cudaGetDevice(&device);
    }
    
    if (isLoggingEnabled()) {
        std::cout << "[GCAllocator] Allocation request: size=" << n 
                  << " bytes, device=" << device << std::endl;
    }
    
    // Record stats before allocation
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_->recordAllocationRequest(n, device);
    }
    
    // Pass through to original allocator
    c10::DataPtr data_ptr;
    try {
        data_ptr = original_allocator_->allocate(n);
        
        // Record successful allocation
        if (data_ptr.get()) {
            recordAllocation(data_ptr.get(), n, device);
            
            // Wrap with our custom deleter
            auto* raw_ptr = data_ptr.get();
            auto ctx = data_ptr.get_context();
            
            // Create new DataPtr with our deleter
            return c10::DataPtr(
                raw_ptr,
                ctx,
                &GCAllocator::deleteFunction,
                data_ptr.device()
            );
        }
    } catch (const c10::OutOfMemoryError& e) {
        // Record OOM event
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_->recordOOMEvent(n, device);
        }
        
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] OOM detected: " << e.what() << std::endl;
        }
        
        // For Phase 1, just rethrow - retry logic comes in Phase 2
        throw;
    } catch (const std::exception& e) {
        if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Allocation failed: " << e.what() << std::endl;
        }
        throw;
    }
    
    return data_ptr;
}

void GCAllocator::deleteFunction(void* ptr) {
    if (!ptr) return;
    
    // Get the allocator instance from thread-local storage
    GCAllocator* allocator = current_allocator_;
    if (!allocator) {
        // Fallback: get from manager
        allocator = GCAllocatorManager::getInstance().getAllocator();
    }
    
    if (allocator) {
        allocator->recordDeallocation(ptr);
        
        if (allocator->isLoggingEnabled()) {
            std::cout << "[GCAllocator] Deallocation: ptr=" << ptr << std::endl;
        }
        
        // Use original allocator's deleter
        auto deleter = allocator->getOriginalAllocator()->raw_deleter();
        if (deleter) {
            deleter(ptr);
        }
    }
}

c10::DeleterFnPtr GCAllocator::raw_deleter() const {
    return &GCAllocator::deleteFunction;
}

void GCAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    // Delegate to the original allocator's copy_data method if available,
    // otherwise use standard memcpy
    if (original_allocator_) {
        original_allocator_->copy_data(dest, src, count);
    } else {
        std::memcpy(dest, src, count);
    }
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

void GCAllocatorManager::installAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);
    
    if (installed_.load()) {
        throw std::runtime_error("GCAllocator is already installed");
    }
    
    // Create our allocator instance
    allocator_ = std::make_unique<GCAllocator>();
    
    // Store the original CUDA allocator
    original_cuda_allocator_ = c10::cuda::CUDACachingAllocator::get();
    
    // Replace the CUDA allocator with ours
    c10::SetAllocator(c10::DeviceType::CUDA, allocator_.get());
    
    installed_.store(true);
    
    if (allocator_->isLoggingEnabled()) {
        std::cout << "[GCAllocator] Successfully installed custom allocator" << std::endl;
    }
}

void GCAllocatorManager::uninstallAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);
    
    if (!installed_.load()) {
        return;
    }
    
    // Restore original allocator
    if (original_cuda_allocator_) {
        c10::SetAllocator(c10::DeviceType::CUDA, original_cuda_allocator_);
    }
    
    if (allocator_ && allocator_->isLoggingEnabled()) {
        std::cout << "[GCAllocator] Uninstalled custom allocator" << std::endl;
    }
    
    // Clean up our allocator
    allocator_.reset();
    installed_.store(false);
}

} // namespace gc_allocator
