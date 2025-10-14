/*
 * Direct unit tests for GCAllocator::allocate method
 * Tests the allocator without going through PyTorch's tensor creation
 */

#include <iostream>
#include <cassert>
#include <memory>
#include "../gcAllocator/src/gc_allocator_core.h"
#include "../gcAllocator/src/allocator_stats.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/DeviceType.h>
#include <ATen/cuda/CUDAContext.h>

using namespace gc_allocator;

class AllocatorDirectTest {
public:
    static void runAllTests() {
        std::cout << "\n========================================\n";
        std::cout << "DIRECT ALLOCATOR UNIT TESTS\n";
        std::cout << "========================================\n\n";
        
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
        
        // Verify stats object is created
        auto stats = allocator->getStats();
        assert(stats.getTotalAllocations() == 0);
        std::cout << "✓ Stats initialized with zero allocations\n";
        
        // Check wrapped allocator is initially null
        auto wrapped = allocator->getWrappedAllocator();
        std::cout << "✓ Initial wrapped allocator: " << wrapped 
                  << " (expected null)\n\n";
    }
    
    static void testWrappedAllocatorSetting() {
        std::cout << "TEST 2: Setting Wrapped Allocator\n";
        std::cout << "-----------------------------------\n";
        
        auto allocator = std::make_unique<GCAllocator>();
        
        // Get the actual CUDA caching allocator
        auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
        std::cout << "Original CUDA allocator: " << cuda_allocator << "\n";
        
        if (cuda_allocator == nullptr) {
            std::cout << "⚠ WARNING: CUDACachingAllocator::get() returned null\n";
            std::cout << "Attempting to initialize CUDA context...\n";
            
            // Initialize CUDA context
            if (at::cuda::is_available()) {
                at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
                c10::cuda::CUDAGuard guard(0);
                cuda_allocator = c10::cuda::CUDACachingAllocator::get();
                std::cout << "After initialization: " << cuda_allocator << "\n";
            }
        }
        
        // Set the wrapped allocator
        allocator->setWrappedAllocator(cuda_allocator);
        
        // Verify it was set correctly
        auto wrapped = allocator->getWrappedAllocator();
        assert(wrapped == cuda_allocator);
        std::cout << "✓ Wrapped allocator set to: " << wrapped << "\n";
        std::cout << "✓ Verification: wrapped == original: " 
                  << (wrapped == cuda_allocator ? "YES" : "NO") << "\n\n";
    }
    
    static void testDirectAllocation() {
        std::cout << "TEST 3: Direct Allocation Call\n";
        std::cout << "-------------------------------\n";
        
        // Initialize CUDA
        if (!at::cuda::is_available()) {
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
        c10::cuda::CUDAGuard guard(0);
        
        // Create and configure allocator
        auto allocator = std::make_unique<GCAllocator>();
        auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
        
        std::cout << "Setting wrapped allocator to: " << cuda_allocator << "\n";
        allocator->setWrappedAllocator(cuda_allocator);
        
        // Enable logging for visibility
        allocator->setLoggingEnabled(true);
        
        // Get initial stats
        auto initial_stats = allocator->getStats();
        std::cout << "Initial allocations: " << initial_stats.getTotalAllocations() << "\n";
        std::cout << "Initial requests: " << initial_stats.getTotalRequests() << "\n";
        
        // Attempt direct allocation
        size_t alloc_size = 1024 * 1024;  // 1 MB
        std::cout << "\nAttempting to allocate " << alloc_size << " bytes...\n";
        
        try {
            c10::DataPtr result = allocator->allocate(alloc_size);
            
            if (result.get() != nullptr) {
                std::cout << "✓ Allocation successful! Pointer: " << result.get() << "\n";
                
                // Check stats were updated
                auto after_stats = allocator->getStats();
                std::cout << "After allocations: " << after_stats.getTotalAllocations() << "\n";
                std::cout << "After requests: " << after_stats.getTotalRequests() << "\n";
                
                assert(after_stats.getTotalRequests() > initial_stats.getTotalRequests());
                std::cout << "✓ Statistics were updated\n";
                
                // Clean up - DataPtr will auto-delete
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
        
        // Don't set wrapped allocator - leave it null
        std::cout << "Wrapped allocator: " << allocator->getWrappedAllocator() 
                  << " (null)\n";
        
        // Try to allocate - should throw
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
        
        if (!at::cuda::is_available()) {
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        // Setup allocator with wrapped CUDA allocator
        auto allocator = std::make_unique<GCAllocator>();
        allocator->setWrappedAllocator(c10::cuda::CUDACachingAllocator::get());
        
        // Record initial state
        auto stats1 = allocator->getStats();
        size_t initial_requests = stats1.getTotalRequests();
        size_t initial_allocations = stats1.getTotalAllocations();
        size_t initial_bytes = stats1.getTotalBytesAllocated();
        
        std::cout << "Initial state:\n";
        std::cout << "  Requests: " << initial_requests << "\n";
        std::cout << "  Allocations: " << initial_allocations << "\n";
        std::cout << "  Bytes allocated: " << initial_bytes << "\n";
        
        // Perform multiple allocations
        std::vector<c10::DataPtr> allocations;
        size_t total_allocated = 0;
        
        for (int i = 0; i < 5; i++) {
            size_t size = (i + 1) * 1024 * 1024;  // 1MB, 2MB, 3MB, etc.
            try {
                auto ptr = allocator->allocate(size);
                if (ptr.get()) {
                    allocations.push_back(std::move(ptr));
                    total_allocated += size;
                    std::cout << "  Allocated " << size << " bytes\n";
                }
            } catch (...) {
                std::cout << "  Failed to allocate " << size << " bytes\n";
            }
        }
        
        // Check final stats
        auto stats2 = allocator->getStats();
        std::cout << "\nFinal state:\n";
        std::cout << "  Requests: " << stats2.getTotalRequests() << "\n";
        std::cout << "  Allocations: " << stats2.getTotalAllocations() << "\n";
        std::cout << "  Bytes allocated: " << stats2.getTotalBytesAllocated() << "\n";
        
        // Verify increases
        assert(stats2.getTotalRequests() >= initial_requests + 5);
        std::cout << "✓ Request count increased by at least 5\n";
        
        if (allocations.size() > 0) {
            assert(stats2.getTotalAllocations() > initial_allocations);
            std::cout << "✓ Allocation count increased\n";
            
            assert(stats2.getTotalBytesAllocated() >= initial_bytes + total_allocated);
            std::cout << "✓ Bytes allocated increased by at least " 
                      << total_allocated << "\n";
        }
        
        std::cout << "\n";
    }
    
    static void testAllocationWithRetry() {
        std::cout << "TEST 6: Allocation with Retry Logic\n";
        std::cout << "------------------------------------\n";
        
        if (!at::cuda::is_available()) {
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        auto allocator = std::make_unique<GCAllocator>();
        allocator->setWrappedAllocator(c10::cuda::CUDACachingAllocator::get());
        
        // Configure retry strategy
        RetryConfig config;
        config.max_retries = 3;
        config.initial_delay = std::chrono::milliseconds(10);
        config.enable_cache_flush = true;
        allocator->configureRetryStrategy(config);
        std::cout << "Configured retry with max_retries=" << config.max_retries << "\n";
        
        // Try to allocate a very large amount (might trigger retry)
        size_t huge_size = 100ULL * 1024 * 1024 * 1024;  // 100 GB (likely to fail)
        std::cout << "Attempting huge allocation: " << huge_size << " bytes\n";
        
        try {
            auto ptr = allocator->allocate(huge_size);
            if (ptr.get()) {
                std::cout << "✓ Surprisingly, huge allocation succeeded!\n";
            }
        } catch (const c10::OutOfMemoryError& e) {
            std::cout << "✓ Got expected OOM error: " << e.what() << "\n";
            
            // Check retry stats
            auto retry_stats = allocator->getRetryStats();
            std::cout << "Retry attempts: " << retry_stats.getTotalRetryAttempts() << "\n";
            std::cout << "Cache flushes: " << retry_stats.getCacheFlushes() << "\n";
        } catch (const std::exception& e) {
            std::cout << "Got other exception: " << e.what() << "\n";
        }
        
        std::cout << "\n";
    }
    
    static void testDeallocation() {
        std::cout << "TEST 7: Deallocation Tracking\n";
        std::cout << "------------------------------\n";
        
        if (!at::cuda::is_available()) {
            std::cout << "⚠ CUDA not available, skipping test\n\n";
            return;
        }
        
        auto allocator = std::make_unique<GCAllocator>();
        allocator->setWrappedAllocator(c10::cuda::CUDACachingAllocator::get());
        
        // Get initial deallocation count
        auto initial_stats = allocator->getStats();
        size_t initial_deallocations = initial_stats.getTotalDeallocations();
        std::cout << "Initial deallocations: " << initial_deallocations << "\n";
        
        // Allocate and explicitly deallocate
        {
            auto ptr = allocator->allocate(1024 * 1024);
            std::cout << "Allocated 1 MB at: " << ptr.get() << "\n";
            // DataPtr destructor should trigger deallocation
        }
        
        // Check deallocation was tracked
        auto final_stats = allocator->getStats();
        size_t final_deallocations = final_stats.getTotalDeallocations();
        std::cout << "Final deallocations: " << final_deallocations << "\n";
        
        if (final_deallocations > initial_deallocations) {
            std::cout << "✓ Deallocation was tracked\n";
        } else {
            std::cout << "⚠ Deallocation might not be tracked properly\n";
        }
        
        std::cout << "\n";
    }
};

int main() {
    std::cout << "GCAllocator Direct Unit Tests\n";
    std::cout << "==============================\n";
    
    // Check CUDA availability
    if (at::cuda::is_available()) {
        std::cout << "CUDA is available - running full test suite\n";
    } else {
        std::cout << "CUDA is not available - some tests will be skipped\n";
    }
    
    // Run all tests
    AllocatorDirectTest::runAllTests();
    
    return 0;
}
