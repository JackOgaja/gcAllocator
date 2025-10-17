// tests/test_allocator_direct.cpp
// FULLY CORRECTED VERSION with proper PyTorch namespaces

#include <iostream>
#include <cassert>
#include <memory>
#include <vector>
#include "../../gcAllocator/src/gc_allocator_core.h"
#include "../../gcAllocator/src/allocator_stats.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>
#include <ATen/cuda/CUDAContext.h>  // For at::cuda functions
#include <cuda_runtime.h>

using namespace gc_allocator;

class CUDAInitializer {
public:
    static bool initializeCUDA() {
        // Use at::cuda namespace for availability check
        if (!at::cuda::is_available()) {
            return false;
        }
        
        // Direct CUDA runtime initialization
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        
        // Force CUDA context creation
        void* dummy;
        err = cudaMalloc(&dummy, 1);
        if (err == cudaSuccess) {
            cudaFree(dummy);
        }
        
        // Use c10::cuda for device setting
        c10::cuda::set_device(0);
        
        // Initialize caching allocator
        c10::cuda::CUDACachingAllocator::init(0);
        
        return true;
    }
    
    static bool ensureCUDAContext() {
        try {
            // Use at::cuda for device count
            if (at::cuda::device_count() > 0) {
                c10::cuda::set_device(0);
                return true;
            }
        } catch (const std::exception& e) {
            std::cerr << "CUDA context creation failed: " << e.what() << "\n";
        }
        return false;
    }
};

class AllocatorDirectTest {
public:
    static void runAllTests() {
        std::cout << "\n========================================\n";
        std::cout << "DIRECT ALLOCATOR UNIT TESTS\n";
        std::cout << "========================================\n\n";
        
        // Initialize CUDA once at the beginning
        if (at::cuda::is_available()) {  // CORRECTED: at::cuda
            std::cout << "Initializing CUDA context...\n";
            if (CUDAInitializer::initializeCUDA()) {
                std::cout << "✓ CUDA initialized successfully\n\n";
            } else {
                std::cout << "⚠ CUDA initialization failed\n\n";
            }
        }
        
        testAllocatorCreation();
        testWrappedAllocatorSetting();
        testDirectAllocation();
        testNullWrappedAllocator();
        testStatisticsUpdates();
        testAllocationWithRetry();
        testDeallocation();
        
        std::cout << "\n========================================\n";
        std::cout << "ALL DIRECT TESTS COMPLETED\n";
        std::cout << "========================================\n";
    }

private:
    static void testAllocatorCreation() {
        std::cout << "TEST 1: Allocator Creation\n";
        std::cout << "--------------------------\n";
        
        auto allocator = std::make_unique<GCAllocator>();
        assert(allocator != nullptr);
        std::cout << "✓ Allocator created successfully at " << allocator.get() << "\n";
        
        auto stats = allocator->getStats();
        assert(stats.getTotalAllocations() == 0);
        std::cout << "✓ Stats initialized with zero allocations\n";
        
        auto wrapped = allocator->getWrappedAllocator();
        std::cout << "✓ Initial wrapped allocator: " << wrapped 
                  << " (expected null)\n\n";
    }
    
    static void testWrappedAllocatorSetting() {
        std::cout << "TEST 2: Setting Wrapped Allocator\n";
        std::cout << "-----------------------------------\n";
        
        auto allocator = std::make_unique<GCAllocator>();
        
        c10::Allocator* cuda_allocator = nullptr;
        
        cuda_allocator = c10::cuda::CUDACachingAllocator::get();
        std::cout << "CUDACachingAllocator::get() returned: " << cuda_allocator << "\n";
        
        if (cuda_allocator == nullptr && at::cuda::is_available()) {  // CORRECTED
            std::cout << "Attempting alternative initialization methods...\n";
            
            CUDAInitializer::ensureCUDAContext();
            c10::cuda::CUDACachingAllocator::init(0);
            cuda_allocator = c10::cuda::CUDACachingAllocator::get();
            std::cout << "After init, allocator: " << cuda_allocator << "\n";
        }
        
        if (cuda_allocator == nullptr) {
            std::cout << "⚠ WARNING: Could not get CUDA allocator\n";
            std::cout << "  This is the ROOT CAUSE of the proxy pattern failure!\n\n";
            return;
        }
        
        allocator->setWrappedAllocator(cuda_allocator);
        
        auto wrapped = allocator->getWrappedAllocator();
        assert(wrapped == cuda_allocator);
        std::cout << "✓ Wrapped allocator set to: " << wrapped << "\n";
        std::cout << "✓ Verification: wrapped == original: " 
                  << (wrapped == cuda_allocator ? "YES" : "NO") << "\n\n";
    }
    
    static void testDirectAllocation() {
        std::cout << "TEST 3: Direct Allocation Call\n";
        std::cout << "-------------------------------\n";
        
        if (!at::cuda::is_available()) {  // CORRECTED
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        CUDAInitializer::ensureCUDAContext();
        
        auto allocator = std::make_unique<GCAllocator>();
        
        c10::cuda::CUDACachingAllocator::init(0);
        auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
        
        if (!cuda_allocator) {
            std::cout << "✗ ERROR: Cannot get CUDA allocator\n";
            std::cout << "  This confirms the ROOT CAUSE!\n\n";
            return;
        }
        
        std::cout << "Setting wrapped allocator to: " << cuda_allocator << "\n";
        allocator->setWrappedAllocator(cuda_allocator);
        allocator->setLoggingEnabled(true);
        
        auto initial_stats = allocator->getStats();
        std::cout << "Initial allocations: " << initial_stats.getTotalAllocations() << "\n";
        std::cout << "Initial requests: " << initial_stats.getTotalRequests() << "\n";
        
        size_t alloc_size = 1024 * 1024;  // 1 MB
        std::cout << "\nAttempting to allocate " << alloc_size << " bytes...\n";
        
        try {
            c10::cuda::set_device(0);
            
            c10::DataPtr result = allocator->allocate(alloc_size);
            
            if (result.get() != nullptr) {
                std::cout << "✓ Allocation successful! Pointer: " << result.get() << "\n";
                
                auto after_stats = allocator->getStats();
                std::cout << "After allocations: " << after_stats.getTotalAllocations() << "\n";
                std::cout << "After requests: " << after_stats.getTotalRequests() << "\n";
                
                if (after_stats.getTotalRequests() > initial_stats.getTotalRequests()) {
                    std::cout << "✓ Statistics were updated correctly\n";
                } else {
                    std::cout << "✗ ERROR: Statistics not updated!\n";
                }
                
            } else {
                std::cout << "✗ Allocation returned null pointer\n";
            }
        } catch (const std::exception& e) {
            std::cout << "✗ Allocation threw exception: " << e.what() << "\n";
        }
        
        std::cout << "\n";
    }
    
    static void testNullWrappedAllocator() {
        std::cout << "TEST 4: Null Wrapped Allocator Behavior\n";
        std::cout << "----------------------------------------\n";
        
        auto allocator = std::make_unique<GCAllocator>();
        
        std::cout << "Wrapped allocator: " << allocator->getWrappedAllocator() 
                  << " (null)\n";
        std::cout << "Attempting allocation with null wrapped allocator...\n";
        
        try {
            c10::DataPtr result = allocator->allocate(1024);
            std::cout << "✗ ERROR: Should have thrown exception!\n";
            assert(false);
        } catch (const std::runtime_error& e) {
            std::cout << "✓ Correctly threw exception: " << e.what() << "\n";
        } catch (...) {
            std::cout << "✗ Threw unexpected exception type\n";
        }
        
        std::cout << "\n";
    }
    
    static void testStatisticsUpdates() {
        std::cout << "TEST 5: Statistics Update Verification\n";
        std::cout << "---------------------------------------\n";
        
        if (!at::cuda::is_available()) {  // CORRECTED
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        auto allocator = std::make_unique<GCAllocator>();
        allocator->setWrappedAllocator(c10::cuda::CUDACachingAllocator::get());
        
        auto stats1 = allocator->getStats();
        size_t initial_requests = stats1.getTotalRequests();
        
        std::cout << "Initial requests: " << initial_requests << "\n";
        
        std::vector<c10::DataPtr> allocations;
        
        for (int i = 0; i < 5; i++) {
            size_t size = (i + 1) * 1024 * 1024;
            try {
                auto ptr = allocator->allocate(size);
                if (ptr.get()) {
                    allocations.push_back(std::move(ptr));
                    std::cout << "  Allocated " << size << " bytes\n";
                }
            } catch (...) {
                std::cout << "  Failed to allocate " << size << " bytes\n";
            }
        }
        
        auto stats2 = allocator->getStats();
        std::cout << "\nFinal requests: " << stats2.getTotalRequests() << "\n";
        
        assert(stats2.getTotalRequests() >= initial_requests + 5);
        std::cout << "✓ Request count increased correctly\n\n";
    }
    
    static void testAllocationWithRetry() {
        std::cout << "TEST 6: Allocation with Retry Logic\n";
        std::cout << "------------------------------------\n";
        
        if (!at::cuda::is_available()) {  // CORRECTED
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        auto allocator = std::make_unique<GCAllocator>();
        allocator->setWrappedAllocator(c10::cuda::CUDACachingAllocator::get());
        
        RetryConfig config;
        config.max_retries = 3;
        config.initial_delay = std::chrono::milliseconds(10);
        config.enable_cache_flush = true;
        allocator->configureRetryStrategy(config);
        
        size_t huge_size = 100ULL * 1024 * 1024 * 1024;  // 100 GB
        std::cout << "Attempting huge allocation: " << huge_size << " bytes\n";
        
        try {
            auto ptr = allocator->allocate(huge_size);
            if (ptr.get()) {
                std::cout << "✓ Surprisingly, huge allocation succeeded!\n";
            }
        } catch (const c10::OutOfMemoryError& e) {
            std::cout << "✓ Got expected OOM error\n";
            auto retry_stats = allocator->getRetryStats();
            std::cout << "Retry attempts: " << retry_stats.getTotalRetryAttempts() << "\n";
        } catch (const std::exception& e) {
            std::cout << "Got exception: " << e.what() << "\n";
        }
        
        std::cout << "\n";
    }
    
    static void testDeallocation() {
        std::cout << "TEST 7: Deallocation Tracking\n";
        std::cout << "------------------------------\n";
        
        if (!at::cuda::is_available()) {  // CORRECTED
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        auto allocator = std::make_unique<GCAllocator>();
        allocator->setWrappedAllocator(c10::cuda::CUDACachingAllocator::get());
        
        auto initial_stats = allocator->getStats();
        size_t initial_deallocations = initial_stats.getTotalDeallocations();
        
        {
            auto ptr = allocator->allocate(1024 * 1024);
            std::cout << "Allocated 1 MB at: " << ptr.get() << "\n";
        }
        
        auto final_stats = allocator->getStats();
        size_t final_deallocations = final_stats.getTotalDeallocations();
        
        if (final_deallocations > initial_deallocations) {
            std::cout << "✓ Deallocation was tracked\n";
        } else {
            std::cout << "⚠ Deallocation might not be tracked\n";
        }
        
        std::cout << "\n";
    }
};

int main() {
    std::cout << "GCAllocator Direct Unit Tests\n";
    std::cout << "==============================\n";
    
    if (at::cuda::is_available()) {  // CORRECTED
        std::cout << "CUDA is available\n";
        std::cout << "CUDA device count: " << at::cuda::device_count() << "\n";  // CORRECTED
    } else {
        std::cout << "CUDA is not available\n";
    }
    
    AllocatorDirectTest::runAllTests();
    
    return 0;
}
