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
#include <ATen/Context.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

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
    
    // Record stats before allocation
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_->recordAllocationRequest(n, device);
    }
    
    // Pass through to original allocator
    c10::DataPtr data_ptr;
    try {
        if (original_allocator_) {
            data_ptr = original_allocator_->allocate(n);
        } else {
            // Fallback to direct CUDA allocation
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, n);
            if (err != cudaSuccess) {
                C10_CUDA_CHECK(err);
            }
            data_ptr = c10::DataPtr(ptr, ptr, &deleteFunction, c10::Device(c10::DeviceType::CUDA, device));
        }
        
        // Record successful allocation
        if (data_ptr.get()) {
            void* raw_ptr = data_ptr.get();
            recordAllocation(raw_ptr, n, device);
            
            // Track globally for interception
            {
                std::lock_guard<std::mutex> lock(GCAllocatorManager::global_allocations_mutex_);
                GCAllocatorManager::global_allocations_[raw_ptr] = n;
            }
            
            // Wrap with our custom deleter
            return c10::DataPtr(
                raw_ptr,
                data_ptr.get_context(),
                &deleteFunction,
                data_ptr.device()
            );
        }
    } catch (const c10::Error& e) {
        // Check if it's an OOM error
        if (e.what_without_backtrace().find("out of memory") != std::string::npos ||
            e.what_without_backtrace().find("CUDA out of memory") != std::string::npos) {
            // Record OOM event
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_->recordOOMEvent(n, device);
            }
            
            if (isLoggingEnabled()) {
                std::cout << "[GCAllocator] OOM detected: " << e.what() << std::endl;
            }
        } else if (isLoggingEnabled()) {
            std::cout << "[GCAllocator] Allocation failed: " << e.what() << std::endl;
        }
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
        
        // Remove from global tracking
        {
            std::lock_guard<std::mutex> lock(GCAllocatorManager::global_allocations_mutex_);
            GCAllocatorManager::global_allocations_.erase(ptr);
        }
        
        // Use original allocator's deleter if available
        if (allocator->getOriginalAllocator()) {
            auto deleter = allocator->getOriginalAllocator()->raw_deleter();
            if (deleter) {
                deleter(ptr);
                return;
            }
        }
        
        // Fallback to cudaFree
        cudaFree(ptr);
    } else {
        // Direct cudaFree if no allocator
        cudaFree(ptr);
    }
}

c10::DeleterFnPtr GCAllocator::raw_deleter() const {
    return &GCAllocator::deleteFunction;
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

GCAllocatorManager::~GCAllocatorManager() {
    if (installed_.load()) {
        uninstallAllocator();
    }
}

void GCAllocatorManager::installAllocator() {
    std::lock_guard<std::mutex> lock(install_mutex_);
    
    if (installed_.load()) {
        // Already installed, just return
        return;
    }
    
    // Create our allocator instance
    allocator_ = std::make_unique<GCAllocator>();
    
    // Store the original CUDA allocator
    original_cuda_allocator_ = c10::cuda::CUDACachingAllocator::get();
    
    // Replace the allocator for CUDA device type
    // This is the key to intercepting allocations
    c10::SetAllocator(c10::DeviceType::CUDA, allocator_.get());
    
    // Also try to set it directly on the caching allocator if the API exists
    // Note: This might not exist in all PyTorch versions
    try {
        c10::cuda::CUDACachingAllocator::set(allocator_.get());
    } catch (...) {
        // If this API doesn't exist, that's okay, we've already set it via SetAllocator
    }
    
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
        
        // Also try to restore it directly on the caching allocator
        try {
            c10::cuda::CUDACachingAllocator::set(original_cuda_allocator_);
        } catch (...) {
            // If this API doesn't exist, that's okay
        }
    }
    
    if (allocator_ && allocator_->isLoggingEnabled()) {
        std::cout << "[GCAllocator] Uninstalled custom allocator" << std::endl;
    }
    
    // Clear global allocations tracking
    {
        std::lock_guard<std::mutex> alloc_lock(global_allocations_mutex_);
        global_allocations_.clear();
    }
    
    // Clean up our allocator
    allocator_.reset();
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
